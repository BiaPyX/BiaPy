import os
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

from biapy.utils.misc import init_devices, is_dist_avail_and_initialized, set_seed
from biapy.config.config import Config
from biapy.engine.check_configuration import check_configuration

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

    def export_model_to_bmz(self, bmz_cfg):
        """
        Export a model into Bioimage Model Zoo format. 

        Parameters
        ----------
        bmz_cfg : BMZ configuration
            BMZ configuration to export the model. Here multiple keys need to be declared:

            description : str
                Description of the model.

            authors : list of dicts
                Authors of the model. Need to be a list of dicts. E.g. ``[{"name": "Gizmo"}]``. 

            license : str
                License of the model. E.g. "CC-BY-4.0"

            tags : List of str
                Tags to make models more findable on the website. E.g. ``["nucleus-segmentation"]``.

            cite : List of dicts 
                List of dictionaries of citations associated. E.g. 
                ``[{"text": "Gizmo et al.", "doi": "doi:10.1002/xyzacab123"}]``

            doc : path
                Path to a file with a documentation of the model in markdown. E.g. "my-model/doc.md"
                
            build_dir : path
                Path to store files and the build of BMZ package. 

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
        """
        # Check keys
        need_info = ['description', 'authors', 'license', 'tags', 'cite', 'doc', 'build_dir']
        for x in need_info:
            if x not in bmz_cfg:
                raise ValueError(f"'{x}' property must be declared in 'bmz_cfg'")

        # Check if BiaPy has been run so some of the variables have been created
        if not self.workflow.model_prepared:
            raise ValueError("You need first to call prepare_model(), train(), test() or run_job() functions so the model can be built")
        if 'test_input' in bmz_cfg and bmz_cfg['test_input'] is None and self.workflow.bmz_test_input is None:
            raise ValueError("No bmz_cfg['test_input'] available. You can: 1) provide it using bmz_config['test_input'] "
                "or run the training phase, by calling train() or run_job() functions.")
        if 'test_output' in bmz_cfg and bmz_cfg['test_output'] is None and self.workflow.bmz_test_output is None:
            raise ValueError("No bmz_cfg['test_output'] available. You can: 1) provide it using bmz_config['test_output'] "
                "or run the training phase, by calling train() or run_job() functions.")

        # Check BMZ dictionary keys and values 
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
        if not isinstance(bmz_cfg['cite'], list):
            raise ValueError("'bmz_cfg['cite']' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': 'doi:10.1002/xyzacab123'}]")
        else:
            for d in bmz_cfg['cite']:
                if not isinstance(d, dict):
                    raise ValueError("'bmz_cfg['cite']' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': 'doi:10.1002/xyzacab123'}]")
        if bmz_cfg['doc'] == "":
            raise ValueError("'bmz_cfg['doc']' can not be empty. E.g. '/home/user/my-model/doc.md'")
        else:
            if not os.path.exists(bmz_cfg['doc']):
                raise ValueError("'bmz_cfg['doc']' file does not exist!")
        if bmz_cfg['build_dir'] == "":
            raise ValueError("'bmz_cfg['build_dir']' can not be empty.")

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

        from bioimageio.core.build_spec import build_model
        from bioimageio.core.resource_tests import test_model
        from bioimageio.core import load_resource_description

        # Save input/output samples
        os.makedirs(bmz_cfg['build_dir'], exist_ok=True)
        input_sample_path = os.path.join(bmz_cfg['build_dir'], "test-input.npy")
        output_sample_path = os.path.join(bmz_cfg['build_dir'], "test-output.npy")
        test_input = self.workflow.bmz_test_input if 'test_input' not in bmz_cfg else bmz_cfg['test_input']
        test_output = self.workflow.bmz_test_output if 'test_output' not in bmz_cfg else bmz_cfg['test_output']
        if test_input.ndim == 3:
            np.save(input_sample_path, test_input.permute((2, 0 ,1)).unsqueeze(0))
            np.save(output_sample_path, test_output.permute((2, 0 ,1)).unsqueeze(0))
        else:
            np.save(input_sample_path, test_input.permute((3, 0 ,1, 2)).unsqueeze(0))
            np.save(output_sample_path, test_output.permute((3, 0 ,1, 2)).unsqueeze(0))

        # Name of the model
        if 'model_name' in bmz_cfg:
            model_name = bmz_cfg['model_name']
        else:
            if self.cfg.MODEL.SOURCE == "biapy":
                model_name = "my_"+self.cfg.MODEL.ARCHITECTURE+"_"+self.cfg.PROBLEM.NDIM
            elif self.cfg.MODEL.SOURCE == "torchvision":
                model_name = "my_"+self.cfg.MODEL.TORCHVISION_MODEL_NAME+"_"+self.cfg.PROBLEM.NDIM
            else:
                model_name = "my_BMZ_"+self.cfg.MODEL.BMZ.SOURCE_MODEL_DOI

        # Preprocessing
        # Actually Torchvision has its own preprocessing but it can not be adapted to BMZ easily, so for now
        # we set it like we were using BiaPy backend
        if self.cfg.MODEL.SOURCE in ["biapy", "torchvision"]:
            if self.cfg.DATA.NORMALIZATION.TYPE == 'div':
                preprocessing = [[{"name": "scale_linear", "kwargs": {"gain": 1/255, "offset": 0}}]]
            else:
                if not os.path.exists(self.cfg.PATHS.MEAN_INFO_FILE) or not os.path.exists(self.cfg.PATHS.STD_INFO_FILE):
                    raise FileNotFoundError("Not mean/std files found in {} and {}"
                        .format(self.cfg.PATHS.MEAN_INFO_FILE, self.cfg.PATHS.STD_INFO_FILE))
                custom_mean = float(np.load(self.cfg.PATHS.MEAN_INFO_FILE))
                custom_std = float(np.load(self.cfg.PATHS.STD_INFO_FILE))
                preprocessing = [[{"name": "zero_mean_unit_variance", "kwargs": {"mean": custom_mean, "std": custom_std, 'mode': 'per_sample'}}]]
        else: # BMZ
            # Done as in https://github.com/bioimage-io/core-bioimage-io-python/blob/main/example/model_creation.ipynb
            preprocessing = [[{"name": prep.name, "kwargs": prep.kwargs} for prep in inp.preprocessing] for inp in self.workflow.bmz_model_resource.inputs]

        # Post-processing (not clear for now so just output the raw output of the model)
        postprocessing = None
        # if self.cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'DETECTION', "SUPER_RESOLUTION", "SELF_SUPERVISED"]:
        #     postprocessing = [{"name": "binarize", "kwargs": {"threshold": 0.5}}]

        # Configure tags. For now there is no "biapy" tag so we are filling as much as we can
        tags = bmz_cfg['tags'][0]
        if 'dims' not in tags:
            tags['dims'] = self.cfg.PROBLEM.NDIM.lower()
        tags['framework'] = "pytorch"
        # tags['software'] = "biapy" # {'software': ['ilastik', 'imagej', 'fiji', 'imjoy', 'deepimagej', 'napari']}
        # tags['method'] = # {'method': ['stardist', 'cellpose', 'yolo', 'care', 'n2v', 'denoiseg']},
        # tags['task'] = {'task': ['semantic-segmentation', 'instance-segmentation', 'object-detection', 'image-classification', \
        # 'denoising', 'image-restoration', 'image-reconstruction', 'in-silico-labeling']}

        # Change dir as the building process copies to the current directory the files used to create the BMZ model
        cwd = os.getcwd()
        os.chdir(bmz_cfg['build_dir'])

        # Save model's weights 
        torchscript_model = torch.jit.script(self.workflow.model)
        weight_file = os.path.join(bmz_cfg['build_dir'], "weights.pt")
        torchscript_model.save(weight_file)

        # Export model to BMZ format 
        build_model(
            # the weight file and the type of the weights
            weight_uri=weight_file,
            weight_type="torchscript",
            pytorch_version=str(torch.__version__),
            # the test input and output data as well as the description of the tensors
            # these are passed as list because we support multiple inputs / outputs per model
            test_inputs=[input_sample_path],
            test_outputs=[output_sample_path],
            input_axes=["bcyx"] if 'input_axes' not in bmz_cfg else bmz_cfg['input_axes'],
            output_axes=["bcyx"] if 'output_axes' not in bmz_cfg else bmz_cfg['output_axes'],
            # where to save the model zip, how to call the model and a short description of it
            output_path=os.path.join(bmz_cfg['build_dir'], "model.zip"),
            name=model_name,
            description=bmz_cfg['description'],
            # additional metadata about authors, licenses, citation etc.
            authors=bmz_cfg['authors'],
            license=bmz_cfg['license'],
            documentation=bmz_cfg['doc'],
            tags=[tags],
            cite=bmz_cfg['cite'],
            preprocessing=preprocessing,
            postprocessing=postprocessing,
        )

        # Checking model consistency
        my_model = load_resource_description(os.path.join(bmz_cfg['build_dir'], "model.zip") )
        test_model(my_model)

        # Recover the original working path
        os.chdir(cwd)


    def run_job(self):
        """Run a complete BiaPy workflow."""
        if self.cfg.TRAIN.ENABLE:
            self.train()

        if self.cfg.TEST.ENABLE:
            self.test()

        if is_dist_avail_and_initialized():
            dist.barrier()
            dist.destroy_process_group()

        print("FINISHED JOB {} !!".format(self.job_identifier))
        sys.exit(0)