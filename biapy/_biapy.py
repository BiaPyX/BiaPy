import os
import re
import sys
import argparse
import datetime
import ntpath
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from shutil import copyfile
import numpy as np
import importlib
import multiprocessing
from hashlib import sha256

from bioimageio.spec.model.v0_5 import (
    Author, CiteEntry, Doi, HttpUrl, LicenseId, PytorchStateDictWeightsDescr, TorchscriptWeightsDescr, AxisId, BatchAxis, ChannelAxis,
    FileDescr,Identifier,InputTensorDescr,IntervalOrRatioDataDescr,ParameterizedSize,SpaceInputAxis,SpaceOutputAxis,TensorId,
    WeightsDescr, ArchitectureFromFileDescr, ArchitectureFromLibraryDescr, Version, EnvironmentFileDescr)
from bioimageio.spec import save_bioimageio_package
from bioimageio.core import test_model
from bioimageio.spec.utils import download

from biapy.utils.misc import init_devices, is_dist_avail_and_initialized, set_seed, get_rank, setup_for_distributed
from biapy.config.config import Config
from biapy.engine.check_configuration import check_configuration
from biapy.models import get_bmz_model_info
from biapy.utils.util import create_file_sha256sum

class BiaPy():
    def __init__(self, config, result_dir=os.getenv('HOME'), name="unknown_job", run_id=1, gpu=None, world_size=1, 
        local_rank=-1, dist_on_itp=False, dist_url='env://', dist_backend='nccl'):
        """
        Run the main functionality of the job.

        Parameters
        ----------
        config: str,  optional
            Path to the configuration file.

        result_dir: str, optional
            Path to where the resulting output of the job will be stored. Defaults to the home directory.
        
        name: str, optional
            Job name. Defaults to "unknown_job".

        run_id: int, optional
            Run number of the same job. Defaults to 1.

        gpu: str, optional
            GPU number according to 'nvidia-smi' command. Defaults to None.

        world_size: int, optional
            Number of distributed processes. Defaults to 1.

        local_rank: int, optional
            Node rank for distributed training. Necessary for using the torch.distributed.launch utility. Defaults to -1.

        dist_on_itp: bool, optional
            If True, distributed training is performed. Defaults to False.

        dist_url: str, optional
            URL used to set up distributed training. Defaults to 'env://'.

        dist_backend: str, optional
            Backend to use in distributed mode. Should be either 'nccl' or 'gloo'. Defaults to 'nccl'.
        """

        if dist_backend not in ['nccl', 'gloo']:
            raise ValueError("Invalid value for 'dist_backend'. Should be either 'nccl' or 'gloo'.")
    
        self.args = argparse.Namespace(
            config=config,
            result_dir=result_dir,
            name=name,
            run_id=run_id,
            gpu=gpu,
            world_size=world_size,
            local_rank=local_rank,
            dist_on_itp=dist_on_itp,
            dist_url=dist_url,
            dist_backend=dist_backend
        )
        
        ############
        #  CHECKS  #
        ############

        # Job complete name
        self.job_identifier = self.args.name + '_' + str(self.args.run_id)

        # Prepare working dir
        self.job_dir = os.path.join(self.args.result_dir, self.args.name)
        self.cfg_bck_dir = os.path.join(self.job_dir, 'config_files')
        os.makedirs(self.cfg_bck_dir, exist_ok=True)
        head, tail = ntpath.split(self.args.config)
        self.cfg_filename = tail if tail else ntpath.basename(head)
        self.cfg_file = os.path.join(self.cfg_bck_dir,self.cfg_filename)

        if not os.path.exists(self.args.config):
            raise FileNotFoundError("Provided {} config file does not exist".format(self.args.config))
        copyfile(self.args.config, self.cfg_file)

        # Merge conf file with the default settings
        self.cfg = Config(self.job_dir, self.job_identifier)
        self.cfg._C.merge_from_file(self.cfg_file)
        self.cfg.update_dependencies()
        #self.cfg.freeze()

        now = datetime.datetime.now()
        print("Date: {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        print("Arguments: {}".format(self.args))
        print("Job: {}".format(self.job_identifier))
        print("Python       : {}".format(sys.version.split('\n')[0]))
        print("PyTorch: ", torch.__version__)

        # GPU selection
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        opts = []
        if self.args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            self.num_gpus = len(np.unique(np.array(self.args.gpu.strip().split(','))))
            opts.extend(["SYSTEM.NUM_GPUS", self.num_gpus])

        # GPU management
        self.device = init_devices(self.args, self.cfg.get_cfg_defaults())
        self.cfg._C.merge_from_list(opts)
        self.cfg = self.cfg.get_cfg_defaults()

        # Reproducibility
        set_seed(self.cfg.SYSTEM.SEED)
        
        # Number of CPU calculation
        if self.cfg.SYSTEM.NUM_CPUS == -1:
            self.cpu_count = multiprocessing.cpu_count()
        else:
            self.cpu_count = self.cfg.SYSTEM.NUM_CPUS
        if self.cpu_count < 1: self.cpu_count = 1 # At least 1 CPU
        torch.set_num_threads(self.cpu_count)
        self.cfg.merge_from_list(['SYSTEM.NUM_CPUS', self.cpu_count])

        check_configuration(self.cfg, self.job_identifier)
        print("Configuration details:")
        print(self.cfg)

        ##########################
        #       TRAIN/TEST       #
        ##########################
        workflowname = str(self.cfg.PROBLEM.TYPE).lower()
        mdl = importlib.import_module('biapy.engine.'+workflowname)
        names = [x for x in mdl.__dict__ if not x.startswith("_")]
        globals().update({k: getattr(mdl, k) for k in names})
        name = [x for x in names if "Base" not in x and "Workflow" in x][0]

        # Initialize workflow
        print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*")
        print(f"Initializing {name}")
        print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*\n")
        self.workflow = getattr(mdl, name)(self.cfg, self.job_identifier, self.device, self.args)

    def train(self):
        """Call training phase."""
        if self.cfg.TRAIN.ENABLE:
            self.workflow.train()
        else:
            raise ValueError("Train was not enabled ('TRAIN.ENABLE')")

    def test(self):
        """Call test phase."""
        if self.cfg.TEST.ENABLE:
            self.workflow.test()
        else:
            raise ValueError("Test was not enabled ('TEST.ENABLE')")

    def prepare_model(self):
        """Build up the model based on the selected configuration."""
        self.workflow.prepare_model()

    def export_model_to_bmz(self, building_dir, bmz_cfg=None, reuse_original_bmz_config=False):
        """
        Export a model into Bioimage Model Zoo format. 

        Parameters
        ----------
        building_dir : path
            Path to store files and the build of BMZ package. 

        bmz_cfg : BMZ configuration, optional
            BMZ configuration to export the model. Here multiple keys need to be declared:

            model_source : str
                Source file of the model. E.g. "models/unet.py".

            description : str
                Description of the model.

            authors : list of dicts
                Authors of the model. Need to be a list of dicts. E.g. ``[{"name": "Gizmo"}]``. 

            license : str
                A `SPDX license identifier <https://spdx.org/licenses/>`__. E.g. "CC-BY-4.0", "MIT", 
                "BSD-2-Clause".

            tags : List of dicts
                Tags to make models more findable on the website. 
                E.g. ``[{'modality': 'electron-microscopy', 'content': 'mitochondria'}]``.

            cite : List of dicts, optional 
                List of dictionaries of citations associated. E.g. 
                ``[{"text": "Gizmo et al.", "doi": "10.1002/xyzacab123"}]``
                
            model_name : str, optional
                Name of the model. If not set a name based on the selected configuration
                will be created. 

            input_axes : List of str, optional
                Axis order of the input file. E.g. ["bcyx"].
            
            output_axes : List of str, optional
                Axis order of the output file. E.g. ["bcyx"].

            test_input : 3D/4D Torch tensor, optional
                Test input image sample. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

            test_output : 3D/4D Torch tensor, optional
                Test output image sample. E.g. ``(y, x, channels)`` or ``(z, y, x, channels)``.

            covers : List of str, optional
                A list of cover images provided by either a relative path to the model folder, or 
                a hyperlink starting with 'http[s]'. Please use an image smaller than 500KB and an 
                aspect ratio width to height of 2:1. The supported image formats are: 'jpg', 'png', 'gif'.

        reuse_original_bmz_config : bool, optional
            Whether to reuse the original BMZ fields. This option can only be used if the model to export
            was previously loaded from BMZ.

        """
        if reuse_original_bmz_config and 'original_bmz_config' not in self.workflow.bmz_config:
            raise ValueError("The model to export was not previously loaded from BMZ, so there is no config to reuse.")
        if not reuse_original_bmz_config and bmz_cfg is None:
            raise ValueError("'bmz_cfg' arg must be provided if 'reuse_original_bmz_config' is False.")

        # Check keys
        if not reuse_original_bmz_config:
            need_info = ['description', 'authors', 'license', 'tags']
            for x in need_info:
                if x not in bmz_cfg:
                    raise ValueError(f"'{x}' property must be declared in 'bmz_cfg'")

        # Check if BiaPy has been run so some of the variables have been created
        if not self.workflow.model_prepared:
            raise ValueError("You need first to call prepare_model(), train(), test() or run_job() functions so the model can be built")
        if not reuse_original_bmz_config:
            error = False
            if self.workflow.bmz_config['test_input'] is None:
                if 'test_input' not in bmz_cfg:
                    error = True
                elif bmz_cfg['test_input'] is None:
                    error = True
                if error:
                    raise ValueError("No bmz_cfg['test_input'] available. You can: 1) provide it using bmz_config['test_input'] "
                        "or run the training phase, by calling train() or run_job() functions.")
            if self.workflow.bmz_config['test_output'] is None:
                if 'test_output' not in bmz_cfg:
                    error = True
                elif bmz_cfg['test_output'] is None:
                    error = True
                if error:
                    raise ValueError("No bmz_cfg['test_output'] available. You can: 1) provide it using bmz_config['test_output'] "
                        "or run the training phase, by calling train() or run_job() functions.")

        # Check BMZ dictionary keys and values 
        if not reuse_original_bmz_config:
            if bmz_cfg['description'] == "":
                raise ValueError("'bmz_cfg['description']' can not be empty.")
            if not isinstance(bmz_cfg['authors'], list):
                raise ValueError("'bmz_cfg['authors']' needs to be a list of dicts. E.g. [{'name': 'Daniel'}]")
            else:
                if len(bmz_cfg['authors']) == 0:
                    raise ValueError("'bmz_cfg['authors']' can not be empty.")
                for d in bmz_cfg['authors']:
                    if not isinstance(d, dict):
                        raise ValueError("'bmz_cfg['authors']' must be a list of dicts. E.g. [{'name': 'Daniel'}]")
                    else:
                        if len(d.keys()) < 2 or 'name' not in d or 'github_user' not in d:
                            raise ValueError("Author dictionary must have at least 'name' and 'github_user' keys")
                        for k in d.keys():
                            if k not in ['name', 'affiliation', 'email', 'github_user', 'orcid']:
                                raise ValueError("Author dictionary available keys are: ['name', 'affiliation', 'email', "
                                    f"'github_user', 'orcid']. Provided {k}")
            if bmz_cfg['license'] == "":
                    raise ValueError("'bmz_cfg['license']' can not be empty. E.g. 'CC-BY-4.0'")
            if not isinstance(bmz_cfg['tags'], list):
                raise ValueError("'bmz_cfg['tags']' needs to be a list of dicts. E.g. [{'modality': 'electron-microscopy', 'content': 'mitochondria'}]")
            else:
                if len(bmz_cfg['tags']) == 0:
                    raise ValueError("'bmz_cfg['tags']' can not be empty")
                for d in bmz_cfg['tags']:
                    if not isinstance(d, dict):
                        raise ValueError("'bmz_cfg['tags']' must be a list of dicts. E.g. [{'modality': 'electron-microscopy', 'content': 'mitochondria'}]")                    
            if 'cite' in bmz_cfg:
                if not isinstance(bmz_cfg['cite'], list):
                    raise ValueError("'bmz_cfg['cite']' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': '10.1002/xyzacab123'}]")
                else:
                    for d in bmz_cfg['cite']:
                        if not isinstance(d, dict):
                            raise ValueError("'bmz_cfg['cite']' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': '10.1002/xyzacab123'}]")
                        else:
                            if len(d.keys()) < 2 or 'text' not in d:
                                raise ValueError("Cite dictionary must have at least 'text' key")
                            for k in d.keys():
                                if k not in ['text', 'doi', 'url']:
                                    raise ValueError(f"Cite dictionary available keys are: ['text', 'doi', 'url']. Provided {k}")

            if 'input_axes' in bmz_cfg:
                if not isinstance(bmz_cfg['input_axes'], list):
                    raise ValueError("'bmz_cfg['input_axes']' needs to be a list containing just one str. E.g. ['bcyx'].")
                if len(bmz_cfg['input_axes']) != 1: 
                    raise ValueError("'bmz_cfg['input_axes']' needs to be a list containing just one str. E.g. ['bcyx'].")  
                if not isinstance(bmz_cfg['input_axes'][0], str): 
                    raise ValueError("'bmz_cfg['input_axes']' needs to be a list containing just one str. E.g. ['bcyx'].") 
            if 'output_axes' in bmz_cfg:
                if not isinstance(bmz_cfg['output_axes'], list):
                    raise ValueError("'bmz_cfg['output_axes']' needs to be a list containing just one str. E.g. ['bcyx'].")
                if len(bmz_cfg['output_axes']) != 1: 
                    raise ValueError("'bmz_cfg['output_axes']' needs to be a list containing just one str. E.g. ['bcyx'].")  
                if not isinstance(bmz_cfg['output_axes'][0], str): 
                    raise ValueError("'bmz_cfg['output_axes']' needs to be a list containing just one str. E.g. ['bcyx'].") 
            if 'test_input' in bmz_cfg and not torch.is_tensor(bmz_cfg['test_input']):
                raise ValueError("'bmz_cfg['test_input']' needs to be a Tensor") 
            if 'test_output' in bmz_cfg and not torch.is_tensor(bmz_cfg['test_output']):
                raise ValueError("'bmz_cfg['test_output']' needs to be a Tensor") 
            if 'covers' in bmz_cfg:
                if not isinstance(bmz_cfg['covers'], list):
                    raise ValueError("'bmz_cfg['covers']' needs to be a list containing strings.")

        # Save input/output samples
        os.makedirs(building_dir, exist_ok=True)
        test_input_path = os.path.join(building_dir, "test-input.npy")
        test_output_path = os.path.join(building_dir, "test-output.npy")
        if not reuse_original_bmz_config:
            test_input = self.workflow.bmz_config['test_input'] if 'test_input' not in bmz_cfg else bmz_cfg['test_input']
            input_axes = [
                BatchAxis(),
                ChannelAxis(channel_names=[Identifier("raw"+str(i)) for i in range(test_input.shape[-1])])
            ]
            if test_input.ndim == 3:
                np.save(
                    test_input_path, 
                    test_input.permute((2, 0, 1)).unsqueeze(0) if torch.is_tensor(test_input) else np.expand_dims(test_input.transpose((2, 0, 1)),0)
                )
                input_axes += [
                    SpaceInputAxis(id=AxisId('y'), size=ParameterizedSize(min=256, step=8)),
                    SpaceInputAxis(id=AxisId('x'), size=ParameterizedSize(min=256, step=8)),
                ]
                in_axes = ["bcyx"]
            else:
                np.save(
                    test_input_path, 
                    test_input.permute((3, 0, 1, 2)).unsqueeze(0) if torch.is_tensor(test_input) else np.expand_dims(test_input.transpose((3, 0, 1, 2)),0)
                )
                input_axes += [
                    SpaceInputAxis(id=AxisId("z"), size=ParameterizedSize(min=16, step=8)),
                    SpaceInputAxis(id=AxisId('y'), size=ParameterizedSize(min=144, step=72)),
                    SpaceInputAxis(id=AxisId('x'), size=ParameterizedSize(min=144, step=72)),
                ]
                in_axes = ["bczyx"]   
            data_descr = IntervalOrRatioDataDescr(type="float32")

            input_descr = InputTensorDescr(
                id=TensorId("raw"), 
                axes=input_axes, 
                test_tensor=FileDescr(source=test_input_path), 
                data=data_descr
                )
            inputs = [input_descr]

            test_output = self.workflow.bmz_config['test_output'] if 'test_output' not in bmz_cfg else bmz_cfg['test_output']
            output_axes = [
                BatchAxis(),
                ChannelAxis(channel_names=[Identifier("raw"+str(i)) for i in range(test_output.shape[-1])])
            ]
            if test_output.ndim == 3:
                np.save(
                    test_output_path, 
                    test_output.permute((2, 0 ,1)).unsqueeze(0) if torch.is_tensor(test_output) else np.expand_dims(test_output.transpose((2, 0, 1)),0)
                )
                output_axes += [
                    SpaceInputAxis(id=AxisId('y'), size=ParameterizedSize(min=256, step=8)),
                    SpaceInputAxis(id=AxisId('x'), size=ParameterizedSize(min=256, step=8)),
                ]
                in_axes = ["bcyx"]
            else:
                np.save(
                    test_output_path, 
                    test_output.permute((3, 0 ,1, 2)).unsqueeze(0) if torch.is_tensor(test_output) else np.expand_dims(test_output.transpose((3, 0, 1, 2)),0)
                )
                output_axes += [
                    SpaceInputAxis(id=AxisId("z"), size=ParameterizedSize(min=16, step=8)),
                    SpaceInputAxis(id=AxisId('y'), size=ParameterizedSize(min=144, step=72)),
                    SpaceInputAxis(id=AxisId('x'), size=ParameterizedSize(min=144, step=72)),
                ]
                in_axes = ["bczyx"]   
            data_descr = IntervalOrRatioDataDescr(type="float32")

            output_descr = InputTensorDescr(
                id=TensorId("raw"), 
                axes=output_axes, 
                test_tensor=FileDescr(source=test_output_path), 
                data=data_descr
                )
            outputs = [output_descr]
        else:
            inputs = self.workflow.bmz_config['original_bmz_config'].inputs
            outputs = self.workflow.bmz_config['original_bmz_config'].outputs
            
        # Name of the model
        if not reuse_original_bmz_config:
            if 'model_name' in bmz_cfg:
                model_name = bmz_cfg['model_name']
            else:
                if self.cfg.MODEL.SOURCE == "biapy":
                    model_name = "my_"+self.cfg.MODEL.ARCHITECTURE+"_"+self.cfg.PROBLEM.NDIM
                elif self.cfg.MODEL.SOURCE == "torchvision":
                    model_name = "my_"+self.cfg.MODEL.TORCHVISION_MODEL_NAME+"_"+self.cfg.PROBLEM.NDIM
                else:
                    model_name = str("my_BMZ_"+self.cfg.MODEL.BMZ.SOURCE_MODEL_DOI).replace('/','_')
        else:
            model_name = self.workflow.bmz_config['original_bmz_config'].name

        # Preprocessing
        # Actually Torchvision has its own preprocessing but it can not be adapted to BMZ easily, so for now
        # we set it like we were using BiaPy backend
        if self.cfg.MODEL.SOURCE in ["biapy", "torchvision"] or \
            self.cfg.MODEL.SOURCE == "bmz" and len(self.workflow.bmz_config['preprocessing']) == 0:
            if self.cfg.DATA.NORMALIZATION.TYPE == 'div':
                preprocessing = [[{"name": "scale_linear", "kwargs": {"gain": 1/255, "offset": 0}}]]
            elif self.cfg.DATA.NORMALIZATION.TYPE == 'scale_range':
                preprocessing = [[{'name': 'scale_range'}]]
            else:
                if self.cfg.DATA.NORMALIZATION.APPLICATION_MODE == "dataset":
                    if not os.path.exists(self.cfg.PATHS.MEAN_INFO_FILE) or not os.path.exists(self.cfg.PATHS.STD_INFO_FILE):
                        raise FileNotFoundError("Not mean/std files found in {} and {}"
                            .format(self.cfg.PATHS.MEAN_INFO_FILE, self.cfg.PATHS.STD_INFO_FILE))
                    custom_mean = float(np.load(self.cfg.PATHS.MEAN_INFO_FILE))
                    custom_std = float(np.load(self.cfg.PATHS.STD_INFO_FILE))
                    preprocessing = [[{"name": "zero_mean_unit_variance", "kwargs": {
                        "mean": custom_mean, "std": custom_std, 'mode': "per_dataset"
                        }}]]
                else:
                    preprocessing = [[{"name": "zero_mean_unit_variance", 
                        'kwargs': {'axes': 'czyx' if self.cfg.PROBLEM.NDIM == "3D" else 'cyx', 'mode': "per_sample"
                        }}]]
            
            # Add percentile norm
            if self.cfg.DATA.NORMALIZATION.PERC_CLIP:
                min_percentile, max_percentile = 0, 100
                if self.cfg.DATA.NORMALIZATION.PERC_CLIP:
                    min_percentile = self.cfg.DATA.NORMALIZATION.PERC_LOWER
                    max_percentile = self.cfg.DATA.NORMALIZATION.PERC_UPPER
                perc_instructions = {'axes': 'cyx', 'max_percentile': max_percentile, 
                    'min_percentile': min_percentile, 'mode': self.cfg.DATA.NORMALIZATION.APPLICATION_MODE}
                if 'kwargs' not in preprocessing[0][0]:
                    preprocessing[0][0]['kwargs'] = perc_instructions
                else:
                    preprocessing[0][0]['kwargs'].update(perc_instructions)

        else: # BMZ
            # Done as in https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/model_creation.ipynb
            preprocessing = [[{"name": prep.name, "kwargs": prep.kwargs} for prep in inp.preprocessing] for inp in self.workflow.bmz_config['original_bmz_config'].inputs]

        # Post-processing (not clear for now so just output the raw output of the model)
        postprocessing = None
        # if self.cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'DETECTION', "SUPER_RESOLUTION", "SELF_SUPERVISED"]:
        #     postprocessing = [{"name": "binarize", "kwargs": {"threshold": 0.5}}]

        # Configure tags. For now there is no "biapy" tag so we are filling as much as we can
        if not reuse_original_bmz_config:
            tags = bmz_cfg['tags'][0]
            if 'dims' not in tags:
                tags['dims'] = self.cfg.PROBLEM.NDIM.lower()
            tags['framework'] = "pytorch"
            # tags['software'] = "biapy" # {'software': ['ilastik', 'imagej', 'fiji', 'imjoy', 'deepimagej', 'napari']}
            # tags['method'] = # {'method': ['stardist', 'cellpose', 'yolo', 'care', 'n2v', 'denoiseg']},
            # tags['task'] = {'task': ['semantic-segmentation', 'instance-segmentation', 'object-detection', 'image-classification', \
            # 'denoising', 'image-restoration', 'image-reconstruction', 'in-silico-labeling']}
            tags = [tags]
        else:
            tags = [self.workflow.bmz_config['original_bmz_config'].tags]

        # Description 
        if not reuse_original_bmz_config:
            description = bmz_cfg['description']
        else:
            description = self.workflow.bmz_config['original_bmz_config'].description

        # Authors
        authors = [] 
        if not reuse_original_bmz_config:
            for author in bmz_cfg['authors']:
                affiliation = author['affiliation'] if 'affiliation' in author else None
                orcid = author['orcid'] if 'orcid' in author else None
                email = author['email'] if 'email' in author else None
                a = Author(
                    name = author['name'], 
                    affiliation = affiliation, 
                    github_user = author['github_user'],
                    orcid = orcid,
                    email = email
                )
                authors.append(a)
        else:
            authors = self.workflow.bmz_config['original_bmz_config'].authors 

        # License 
        if not reuse_original_bmz_config:
            license = LicenseId(bmz_cfg['license']),
        else:
            license = self.workflow.bmz_config['original_bmz_config'].license

        # Doc 
        if not reuse_original_bmz_config:
            if str(bmz_cfg['doc']).startswith("http"):
                doc = HttpUrl("https://biapy.readthedocs.io/en/latest/")
            else:
                doc = bmz_cfg['doc']
        else:
            doc = str(self.workflow.bmz_config['original_bmz_config'].documentation)

        # Cite 
        citations = []
        if not reuse_original_bmz_config:
            for cite in bmz_cfg['cite']:
                url = cite['url'] if 'url' in cite else None
                doi = Doi(re.sub(r'^.*?10', '10', cite['doi'])) if 'doi' in cite else None
                c = CiteEntry(
                    text = cite['text'], 
                    doi = doi,
                    url = url
                )
                citations.append(c)

            # Add BiaPy citation
            citations.append(
                CiteEntry(
                    text = "BiaPy: A unified framework for versatile bioimage analysis with deep learning", 
                    doi = "10.1101/2024.02.03.576026",
                    url = "https://www.biorxiv.org/content/10.1101/2024.02.03.576026.abstract"
                )
            )
        else:
            citations = self.workflow.bmz_config['original_bmz_config'].cite

        # Cover 
        covers = None
        if not reuse_original_bmz_config and 'covers' in bmz_cfg:
            covers = bmz_cfg['covers']
        elif 'original_bmz_config' in self.workflow.bmz_config and 'covers' in self.workflow.bmz_config['original_bmz_config']:
            covers = self.workflow.bmz_config['original_bmz_config'].covers

        # Change dir as the building process copies to the current directory the files used to create the BMZ model
        cwd = os.getcwd()
        os.chdir(building_dir)

        # Weights + architecture        
        # If it's a BiaPy model
        if not reuse_original_bmz_config and 'model_file' in self.workflow.bmz_config:    
            arch_file_path = re.sub(r':.*', '', self.workflow.bmz_config['model_file'])
            arch_file_sha256 = create_file_sha256sum(arch_file_path)
            pytorch_architecture = ArchitectureFromFileDescr(
                source=arch_file_path,
                sha256=arch_file_sha256,
                callable=self.workflow.bmz_config['model_name'],
                kwargs=self.workflow.bmz_config['model_build_kwargs']
            )
            state_dict_source = "/data/dfranco/exp_results/2d_instance_segmentation/checkpoints/2d_instance_segmentation_1-checkpoint-best.pth"
            # state_dict_sha256 = sha256(str(self.workflow.model.state_dict()).encode('utf-8')).hexdigest()
            state_dict_sha256 = None
        else:
            state_dict_source, state_dict_sha256, \
            pytorch_architecture = get_bmz_model_info(
                self.workflow.bmz_config['original_bmz_config'], 
                self.workflow.bmz_config['original_model_spec_version']
            )                    

        pytorch_state_dict = PytorchStateDictWeightsDescr(
            source = state_dict_source,
            sha256 = state_dict_sha256,
            architecture = pytorch_architecture,
            pytorch_version = Version(torch.__version__),
            dependencies=EnvironmentFileDescr(source="/data/dfranco/BiaPy/environment.yml")
        ),
        # torchscript = TorchscriptWeightsDescr(
        #     source=self.workflow.bmz_config['original_bmz_config'].weights.torchscript.source,
        #     sha256=self.workflow.bmz_config['original_bmz_config'].weights.torchscript.sha256,
        #     pytorch_version=Version(torch.__version__),
        #     parent="pytorch_state_dict", # these weights were converted from the pytorch_state_dict weights ones.
        # ),

        weights_descr = WeightsDescr(
            pytorch_state_dict=pytorch_state_dict,
            # torchscript,
        )

        # Export model to BMZ format 
        args = dict(
            name=model_name,
            description=description,
            authors=authors,
            cite=citations,
            license=license,
            documentation=doc,
            git_repo = HttpUrl("https://github.com/BiaPyX/BiaPy"),
            inputs=inputs,
            outputs=outputs,
            weights=weights_descr,
            tags=tags,
            covers=covers,
            maintainers=authors,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
        )

        print(f"Building BMZ package: {args}")
        model_descr = ModelDescr(**args)
        print(f"Created '{model_descr.name}'")

        # Checking model consistency
        summary = test_model(model_descr)
        summary.display()

        # Saving the model into BMZ format
        model_path = os.path.join(building_dir, model_name+".zip")
        print("Package path:", save_bioimageio_package(model_descr, output_path=Path(model_path)))
        
        # Recover the original working path
        os.chdir(cwd)


    def run_job(self):
        """Run a complete BiaPy workflow."""
        if self.cfg.TRAIN.ENABLE:
            self.train()

        if is_dist_avail_and_initialized():
            print(f"[Rank {get_rank()} ({os.getpid()})] Process waiting (train finished) . . . ")
            dist.barrier()

        if self.cfg.TEST.ENABLE:
            self.test()

        if is_dist_avail_and_initialized():
            print(f"[Rank {get_rank()} ({os.getpid()})] Process waiting (test finished) . . . ")
            setup_for_distributed(self.args.rank == 0)
            dist.barrier()
            dist.destroy_process_group()

        print("FINISHED JOB {} !!".format(self.job_identifier))