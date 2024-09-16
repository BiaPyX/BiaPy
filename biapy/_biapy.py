import os
from pathlib import Path
import re
import sys
import argparse
import datetime
import ntpath
import torch
import torch.distributed as dist
from shutil import copyfile
import numpy as np
import importlib
import multiprocessing
from typing import (
    Optional,
)
from bioimageio.spec.model.v0_5 import (
    Author,
    Maintainer,
    CiteEntry,
    Doi,
    HttpUrl,
    LicenseId,
    PytorchStateDictWeightsDescr,
    AxisId,
    BatchAxis,
    ChannelAxis,
    FileDescr,
    Identifier,
    InputTensorDescr,
    OutputTensorDescr,
    IntervalOrRatioDataDescr,
    SpaceInputAxis,
    SpaceOutputAxis,
    TensorId,
    WeightsDescr,
    ArchitectureFromFileDescr,
    ModelDescr,
)
from packaging.version import Version
from bioimageio.spec._internal.io_basics import Sha256
from bioimageio.spec import save_bioimageio_package

from biapy.utils.misc import (
    init_devices,
    is_dist_avail_and_initialized,
    set_seed,
    get_rank,
    setup_for_distributed,
)
from biapy.config.config import Config
from biapy.engine.check_configuration import check_configuration
from biapy.models import get_bmz_model_info
from biapy.utils.util import create_file_sha256sum


class BiaPy:
    def __init__(
        self,
        config: str,
        result_dir: Optional[str] = "",
        name: Optional[str] = "unknown_job",
        run_id: Optional[int] = 1,
        gpu: Optional[str] = "",
        world_size: Optional[int] = 1,
        local_rank: Optional[int] = -1,
        dist_on_itp: Optional[bool] = False,
        dist_url: Optional[str] = "env://",
        dist_backend: Optional[str] = "nccl",
    ):
        """
        Run the main functionality of the job.

        Parameters
        ----------
        config: str
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
        result_dir = result_dir if result_dir != "" else str(os.getenv("HOME"))

        if dist_backend not in ["nccl", "gloo"]:
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
            dist_backend=dist_backend,
        )

        ############
        #  CHECKS  #
        ############

        # Job complete name
        self.job_identifier = self.args.name + "_" + str(self.args.run_id)

        # Prepare working dir
        self.job_dir = os.path.join(self.args.result_dir, self.args.name)
        self.cfg_bck_dir = os.path.join(self.job_dir, "config_files")
        os.makedirs(self.cfg_bck_dir, exist_ok=True)
        head, tail = ntpath.split(self.args.config)
        self.cfg_filename = tail if tail else ntpath.basename(head)
        self.cfg_file = os.path.join(self.cfg_bck_dir, self.cfg_filename)

        if not os.path.exists(self.args.config):
            raise FileNotFoundError("Provided {} config file does not exist".format(self.args.config))
        copyfile(self.args.config, self.cfg_file)

        # Merge conf file with the default settings
        self.cfg = Config(self.job_dir, self.job_identifier)
        self.cfg._C.merge_from_file(self.cfg_file)
        self.cfg.update_dependencies()
        # self.cfg.freeze()

        now = datetime.datetime.now()
        print("Date: {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        print("Arguments: {}".format(self.args))
        print("Job: {}".format(self.job_identifier))
        print("Python       : {}".format(sys.version.split("\n")[0]))
        print("PyTorch: ", torch.__version__)

        # GPU selection
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        opts = []
        if self.args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            self.num_gpus = len(np.unique(np.array(self.args.gpu.strip().split(","))))
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
        if self.cpu_count < 1:
            self.cpu_count = 1  # At least 1 CPU
        torch.set_num_threads(self.cpu_count)
        self.cfg.merge_from_list(["SYSTEM.NUM_CPUS", self.cpu_count])

        check_configuration(self.cfg, self.job_identifier)
        print("Configuration details:")
        print(self.cfg)

        ##########################
        #       TRAIN/TEST       #
        ##########################
        workflowname = str(self.cfg.PROBLEM.TYPE).lower()
        mdl = importlib.import_module("biapy.engine." + workflowname)
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

    def export_model_to_bmz(
        self,
        building_dir: str,
        bmz_cfg: Optional[dict] = {},
        reuse_original_bmz_config: Optional[bool] = False,
    ):
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

            model_name : str
                Name of the model. If not set a name based on the selected configuration
                will be created.

            license : str
                A `SPDX license identifier <https://spdx.org/licenses/>`__. E.g. "CC-BY-4.0", "MIT",
                "BSD-2-Clause".

            tags : List of str
                Tags to make models more findable on the website. Only set useful information related to
                the data the model was trained with, as the BiaPy will introduce the rest of the tags for you,
                such as dimensions, software ("biapy" in this case), workflow used etc.
                E.g. ``['electron-microscopy','mitochondria']``.

            maintainers : list of dicts, optional
                Maintainers of the model. Need to be a list of dicts. E.g. ``[{"name": "Gizmo"}]``. If not
                provided the authors will be set as maintainers.

            cite : List of dicts, optional
                List of dictionaries of citations associated. E.g.
                ``[{"text": "Gizmo et al.", "doi": "10.1002/xyzacab123"}]``

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
        if bmz_cfg is None:
            bmz_cfg = {}
        if reuse_original_bmz_config and "original_bmz_config" not in self.workflow.bmz_config:
            raise ValueError("The model to export was not previously loaded from BMZ, so there is no config to reuse.")
        if not reuse_original_bmz_config and len(bmz_cfg) == 0:
            raise ValueError("'bmz_cfg' arg must be provided if 'reuse_original_bmz_config' is False.")

        if "original_bmz_config" in self.workflow.bmz_config:
            original_model_version = Version(self.workflow.bmz_config["original_bmz_config"].format_version)

        # Check keys
        if not reuse_original_bmz_config:
            need_info = [
                "description",
                "authors",
                "license",
                "tags",
                "model_name",
            ]
            for x in need_info:
                if x not in bmz_cfg:
                    raise ValueError(f"'{x}' property must be declared in 'bmz_cfg'")

        # Check if BiaPy has been run so some of the variables have been created
        if not self.workflow.model_prepared:
            raise ValueError(
                "You need first to call prepare_model(), train(), test() or run_job() functions so the model can be built"
            )
        if (
            not reuse_original_bmz_config
            and "model_file" in self.workflow.bmz_config
            and self.workflow.checkpoint_path is None
        ):
            raise ValueError(
                "You need first to call prepare_model(), train(), test() or run_job() functions so the model can be built"
            )
        if not reuse_original_bmz_config:
            error = False
            if self.workflow.bmz_config["test_input"] is None:
                if "test_input" not in bmz_cfg:
                    error = True
                elif bmz_cfg["test_input"] is None:
                    error = True
                if error:
                    raise ValueError(
                        "No bmz_cfg['test_input'] available. You can: 1) provide it using bmz_config['test_input'] "
                        "or run the training phase, by calling train() or run_job() functions."
                    )
            if self.workflow.bmz_config["test_output"] is None:
                if "test_output" not in bmz_cfg:
                    error = True
                elif bmz_cfg["test_output"] is None:
                    error = True
                if error:
                    raise ValueError(
                        "No bmz_cfg['test_output'] available. You can: 1) provide it using bmz_config['test_output'] "
                        "or run the training phase, by calling train() or run_job() functions."
                    )

        # Check BMZ dictionary keys and values
        if not reuse_original_bmz_config:
            if bmz_cfg["description"] == "":
                raise ValueError("'bmz_cfg['description']' can not be empty.")
            if not isinstance(bmz_cfg["authors"], list):
                raise ValueError("'bmz_cfg['authors']' needs to be a list of dicts. E.g. [{'name': 'Daniel'}]")
            else:
                if len(bmz_cfg["authors"]) == 0:
                    raise ValueError("'bmz_cfg['authors']' can not be empty.")
                for d in bmz_cfg["authors"]:
                    if not isinstance(d, dict):
                        raise ValueError("'bmz_cfg['authors']' must be a list of dicts. E.g. [{'name': 'Daniel'}]")
                    else:
                        if len(d.keys()) < 2 or "name" not in d or "github_user" not in d:
                            raise ValueError("Author dictionary must have at least 'name' and 'github_user' keys")
                        for k in d.keys():
                            if k not in [
                                "name",
                                "affiliation",
                                "email",
                                "github_user",
                                "orcid",
                            ]:
                                raise ValueError(
                                    "Author dictionary available keys are: ['name', 'affiliation', 'email', "
                                    f"'github_user', 'orcid']. Provided {k}"
                                )
            if bmz_cfg["license"] == "":
                raise ValueError("'bmz_cfg['license']' can not be empty. E.g. 'CC-BY-4.0'")
            if not isinstance(bmz_cfg["tags"], list):
                raise ValueError(
                    "'bmz_cfg['tags']' needs to be a list of str. E.g. ['electron-microscopy', 'mitochondria']"
                )
            else:
                if len(bmz_cfg["tags"]) == 0:
                    raise ValueError("'bmz_cfg['tags']' can not be empty")
                for d in bmz_cfg["tags"]:
                    if not isinstance(d, str):
                        raise ValueError(
                            "'bmz_cfg['tags']' must be a list of str. E.g. ['electron-microscopy', 'mitochondria']"
                        )
            if "maintainers" in bmz_cfg:
                if len(bmz_cfg["maintainers"]) == 0:
                    raise ValueError("'bmz_cfg['maintainers']' can not be empty.")
                for d in bmz_cfg["maintainers"]:
                    if not isinstance(d, dict):
                        raise ValueError("'bmz_cfg['maintainers']' must be a list of dicts. E.g. [{'name': 'Daniel'}]")
                    else:
                        if len(d.keys()) < 2 or "name" not in d or "github_user" not in d:
                            raise ValueError("Author dictionary must have at least 'name' and 'github_user' keys")
                        for k in d.keys():
                            if k not in [
                                "name",
                                "affiliation",
                                "email",
                                "github_user",
                                "orcid",
                            ]:
                                raise ValueError(
                                    "Author dictionary available keys are: ['name', 'affiliation', 'email', "
                                    f"'github_user', 'orcid']. Provided {k}"
                                )
            else:
                bmz_cfg["maintainers"] = bmz_cfg["authors"]

            if "cite" in bmz_cfg:
                if not isinstance(bmz_cfg["cite"], list):
                    raise ValueError(
                        "'bmz_cfg['cite']' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': '10.1002/xyzacab123'}]"
                    )
                else:
                    for d in bmz_cfg["cite"]:
                        if not isinstance(d, dict):
                            raise ValueError(
                                "'bmz_cfg['cite']' needs to be a list of dicts. E.g. [{'text': 'Gizmo et al.', 'doi': '10.1002/xyzacab123'}]"
                            )
                        else:
                            if len(d.keys()) < 2 or "text" not in d:
                                raise ValueError("Cite dictionary must have at least 'text' key")
                            for k in d.keys():
                                if k not in ["text", "doi", "url"]:
                                    raise ValueError(
                                        f"Cite dictionary available keys are: ['text', 'doi', 'url']. Provided {k}"
                                    )

            if "input_axes" in bmz_cfg:
                if not isinstance(bmz_cfg["input_axes"], list):
                    raise ValueError(
                        "'bmz_cfg['input_axes']' needs to be a list containing just one str. E.g. ['bcyx']."
                    )
                if len(bmz_cfg["input_axes"]) != 1:
                    raise ValueError(
                        "'bmz_cfg['input_axes']' needs to be a list containing just one str. E.g. ['bcyx']."
                    )
                if not isinstance(bmz_cfg["input_axes"][0], str):
                    raise ValueError(
                        "'bmz_cfg['input_axes']' needs to be a list containing just one str. E.g. ['bcyx']."
                    )
            if "output_axes" in bmz_cfg:
                if not isinstance(bmz_cfg["output_axes"], list):
                    raise ValueError(
                        "'bmz_cfg['output_axes']' needs to be a list containing just one str. E.g. ['bcyx']."
                    )
                if len(bmz_cfg["output_axes"]) != 1:
                    raise ValueError(
                        "'bmz_cfg['output_axes']' needs to be a list containing just one str. E.g. ['bcyx']."
                    )
                if not isinstance(bmz_cfg["output_axes"][0], str):
                    raise ValueError(
                        "'bmz_cfg['output_axes']' needs to be a list containing just one str. E.g. ['bcyx']."
                    )
            if "test_input" in bmz_cfg and not torch.is_tensor(bmz_cfg["test_input"]):
                raise ValueError("'bmz_cfg['test_input']' needs to be a Tensor")
            if "test_output" in bmz_cfg and not torch.is_tensor(bmz_cfg["test_output"]):
                raise ValueError("'bmz_cfg['test_output']' needs to be a Tensor")
            if "covers" in bmz_cfg:
                if not isinstance(bmz_cfg["covers"], list):
                    raise ValueError("'bmz_cfg['covers']' needs to be a list containing strings.")

        # Preprocessing
        # Actually Torchvision has its own preprocessing but it can not be adapted to BMZ easily, so for now
        # we set it like we were using BiaPy backend
        if (
            self.cfg.MODEL.SOURCE in ["biapy", "torchvision"]
            or self.cfg.MODEL.SOURCE == "bmz"
        ):
            if self.cfg.DATA.NORMALIZATION.TYPE == "div":
                preprocessing = [
                    {
                        "id": "scale_linear",
                        "kwargs": {"gain": 1 / 255, "offset": 0},
                    }
                ]
            elif self.cfg.DATA.NORMALIZATION.TYPE == "scale_range":
                axes = ["channel"]
                axes += list("zyx") if self.cfg.PROBLEM.NDIM == "3D" else list("yx")
                preprocessing = [{
                    "id": "scale_range", 
                    "kwargs": {
                        "max_percentile": 100,
                        "min_percentile": 0,
                        "axes": axes,
                        }
                    }]

                # Add percentile norm
                if self.cfg.DATA.NORMALIZATION.PERC_CLIP:
                    min_percentile, max_percentile = 0, 100
                    if self.cfg.DATA.NORMALIZATION.PERC_CLIP:
                        min_percentile = self.cfg.DATA.NORMALIZATION.PERC_LOWER
                        max_percentile = self.cfg.DATA.NORMALIZATION.PERC_UPPER
                    perc_instructions = {
                        "axes": axes,
                        "max_percentile": max_percentile,
                        "min_percentile": min_percentile,
                    }
                    preprocessing[0]["kwargs"].update(perc_instructions)
                    
            else:  # custom
                custom_mean = self.cfg.DATA.NORMALIZATION.CUSTOM_MEAN
                custom_std = self.cfg.DATA.NORMALIZATION.CUSTOM_STD

                if custom_mean != -1 and custom_std != -1:
                    preprocessing = [
                        {
                            "id": "fixed_zero_mean_unit_variance",
                            "kwargs": {
                                "mean": custom_mean,
                                "std": custom_std,
                            },
                        }
                    ]
                else:
                    axes = ["channel"]
                    axes += list("zyx") if self.cfg.PROBLEM.NDIM == "3D" else list("yx")
                    preprocessing = [
                        {
                            "id": "zero_mean_unit_variance",
                            "kwargs": {
                                "axes": axes,
                            },
                        }
                    ]

        # BMZ, reusing the preprocessing
        else:
            if original_model_version > Version("0.5.0"):
                preprocessing = self.workflow.bmz_config["original_bmz_config"].inputs[0].preprocessing
            else:
                preprocessing = []
                for prep in self.workflow.bmz_config["original_bmz_config"].inputs[0].preprocessing:
                    p = {}
                    p["id"] = prep.name
                    if "ScaleRangeDescr" in str(type(prep)):
                        p["kwargs"] = {}
                        axes = list(prep.kwargs.axes)
                        axes[axes.index("c")] = "channel"
                        p["kwargs"]["axes"] = axes
                        p["kwargs"]["min_percentile"] = prep.kwargs.min_percentile
                        p["kwargs"]["max_percentile"] = prep.kwargs.max_percentile
                    preprocessing.append(p)

        # Post-processing (not clear for now so just output the raw output of the model)
        # postprocessing = None
        # if self.cfg.PROBLEM.TYPE in ['SEMANTIC_SEG', 'DETECTION', "SUPER_RESOLUTION", "SELF_SUPERVISED"]:
        #     postprocessing = [{"name": "binarize", "kwargs": {"threshold": 0.5}}]

        # Save input/output samples
        os.makedirs(building_dir, exist_ok=True)
        test_input_path = os.path.join(building_dir, "test-input.npy")
        test_output_path = os.path.join(building_dir, "test-output.npy")
        if not reuse_original_bmz_config:
            test_input = (
                self.workflow.bmz_config["test_input"] if "test_input" not in bmz_cfg else bmz_cfg["test_input"]
            )
            input_axes = [
                BatchAxis(),
                ChannelAxis(channel_names=[Identifier("channel" + str(i)) for i in range(test_input.shape[-1])]),
            ]
            if test_input.ndim == 3:
                np.save(
                    test_input_path,
                    (
                        test_input.permute((2, 0, 1)).unsqueeze(0)
                        if torch.is_tensor(test_input)
                        else np.expand_dims(test_input.transpose((2, 0, 1)), 0)
                    ),
                )
                input_axes += [
                    SpaceInputAxis(id=AxisId("y"), size=self.cfg.DATA.PATCH_SIZE[0]),
                    SpaceInputAxis(id=AxisId("x"), size=self.cfg.DATA.PATCH_SIZE[1]),
                ]
            else:
                np.save(
                    test_input_path,
                    (
                        test_input.permute((3, 0, 1, 2)).unsqueeze(0)
                        if torch.is_tensor(test_input)
                        else np.expand_dims(test_input.transpose((3, 0, 1, 2)), 0)
                    ),
                )
                input_axes += [
                    SpaceInputAxis(id=AxisId("z"), size=self.cfg.DATA.PATCH_SIZE[0]),
                    SpaceInputAxis(id=AxisId("y"), size=self.cfg.DATA.PATCH_SIZE[1]),
                    SpaceInputAxis(id=AxisId("x"), size=self.cfg.DATA.PATCH_SIZE[2]),
                ]
            data_descr = IntervalOrRatioDataDescr(type="float32")
            input_descr = InputTensorDescr(
                id=TensorId("input0"),
                axes=input_axes,
                test_tensor=FileDescr(source=Path(test_input_path)),
                data=data_descr,
                preprocessing=preprocessing,  # type: ignore
            )
            inputs = [input_descr]

            test_output = (
                self.workflow.bmz_config["test_output"] if "test_output" not in bmz_cfg else bmz_cfg["test_output"]
            )
            output_axes = [
                BatchAxis(),
                ChannelAxis(channel_names=[Identifier("channel" + str(i)) for i in range(test_output.shape[-1])]),
            ]
            if test_output.ndim == 3:
                np.save(
                    test_output_path,
                    (
                        test_output.permute((2, 0, 1)).unsqueeze(0)
                        if torch.is_tensor(test_output)
                        else np.expand_dims(test_output.transpose((2, 0, 1)), 0)
                    ),
                )
                output_axes += [
                    SpaceOutputAxis(id=AxisId("y"), size=self.cfg.DATA.PATCH_SIZE[0]),
                    SpaceOutputAxis(id=AxisId("x"), size=self.cfg.DATA.PATCH_SIZE[1]),
                ]
            else:
                np.save(
                    test_output_path,
                    (
                        test_output.permute((3, 0, 1, 2)).unsqueeze(0)
                        if torch.is_tensor(test_output)
                        else np.expand_dims(test_output.transpose((3, 0, 1, 2)), 0)
                    ),
                )
                output_axes += [
                    SpaceOutputAxis(id=AxisId("z"), size=self.cfg.DATA.PATCH_SIZE[0]),
                    SpaceOutputAxis(id=AxisId("y"), size=self.cfg.DATA.PATCH_SIZE[1]),
                    SpaceOutputAxis(id=AxisId("x"), size=self.cfg.DATA.PATCH_SIZE[2]),
                ]
            data_descr = IntervalOrRatioDataDescr(type="float32")
            output_descr = OutputTensorDescr(
                id=TensorId("output0"),
                axes=output_axes,
                test_tensor=FileDescr(source=Path(test_output_path)),
                data=data_descr,
                # postprocessing=postprocessing,
            )
            outputs = [output_descr]
        else:
            inputs = self.workflow.bmz_config["original_bmz_config"].inputs
            outputs = self.workflow.bmz_config["original_bmz_config"].outputs

        # Name of the model
        if not reuse_original_bmz_config:
            model_name = bmz_cfg["model_name"]
        else:
            model_name = self.workflow.bmz_config["original_bmz_config"].name

        # Configure tags
        if not reuse_original_bmz_config:
            tags = bmz_cfg["tags"]
            if "2d" not in tags and "3d" not in tags:
                tags += [str(self.cfg.PROBLEM.NDIM.lower())]
            if "pytorch" not in tags:
                tags += ["pytorch"]
            if "biapy" not in tags:
                tags += ["biapy"]
            tags += [self.cfg.PROBLEM.TYPE.lower().replace("_", "-").replace("seg", "segmentation")]
            tags += [self.cfg.MODEL.ARCHITECTURE.lower().replace("_", "-")]
        else:
            tags = self.workflow.bmz_config["original_bmz_config"].tags

        # Description
        if not reuse_original_bmz_config:
            description = bmz_cfg["description"]
        else:
            description = self.workflow.bmz_config["original_bmz_config"].description

        # Authors & maintainers
        authors, maintainers = [], []
        if not reuse_original_bmz_config:
            for author in bmz_cfg["authors"]:
                args = dict(name=author["name"], github_user=author["github_user"])
                if "affiliation" in author:
                    args["affiliation"] = author["affiliation"]
                if "orcid" in author:
                    args["orcid"] = author["orcid"]
                if "email" in author:
                    args["email"] = author["email"]
                authors.append(Author(**args))
                maintainers.append(Maintainer(**args))
        else:
            authors = self.workflow.bmz_config["original_bmz_config"].authors
            maintainers = self.workflow.bmz_config["original_bmz_config"].maintainers

        # License
        if not reuse_original_bmz_config:
            license = LicenseId(bmz_cfg["license"])
        else:
            license = self.workflow.bmz_config["original_bmz_config"].license

        # Doc
        if not reuse_original_bmz_config:
            if str(bmz_cfg["doc"]).startswith("http"):
                doc = HttpUrl("https://biapy.readthedocs.io/en/latest/")
            else:
                doc = bmz_cfg["doc"]
        else:
            doc = self.workflow.bmz_config["original_bmz_config"].documentation

        # Cite
        citations = []
        if not reuse_original_bmz_config:
            for cite in bmz_cfg["cite"]:
                args = dict(text=cite["text"])
                if "url" in cite:
                    args["url"] = cite["url"] 
                if "doi" in cite:
                    args["doi"] = Doi(re.sub(r"^.*?10", "10", cite["doi"]))
                citations.append(CiteEntry(**args))

            # Add BiaPy citation
            citations.append(
                CiteEntry(
                    text="BiaPy: A unified framework for versatile bioimage analysis with deep learning",
                    doi=Doi("10.1101/2024.02.03.576026"),
                    url=HttpUrl("https://www.biorxiv.org/content/10.1101/2024.02.03.576026.abstract"),
                )
            )
        else:
            citations = self.workflow.bmz_config["original_bmz_config"].cite

        # Cover
        covers = []
        if not reuse_original_bmz_config and "covers" in bmz_cfg:
            covers = bmz_cfg["covers"]
        elif (
            "original_bmz_config" in self.workflow.bmz_config
            and "covers" in self.workflow.bmz_config["original_bmz_config"]
        ):
            covers = self.workflow.bmz_config["original_bmz_config"].covers

        # Change dir as the building process copies to the current directory the files used to create the BMZ model
        cwd = os.getcwd()
        os.chdir(building_dir)

        # Weights + architecture
        # If it's a BiaPy model
        if not reuse_original_bmz_config and "model_file" in self.workflow.bmz_config:
            arch_file_path = re.sub(r":.*", "", self.workflow.bmz_config["model_file"])
            arch_file_sha256 = create_file_sha256sum(arch_file_path)
            pytorch_architecture = ArchitectureFromFileDescr(
                source=Path(arch_file_path),
                sha256=Sha256(arch_file_sha256),
                callable=self.workflow.bmz_config["model_name"],
                kwargs=self.workflow.model_build_kwargs,
            )
            state_dict_source = Path(self.workflow.checkpoint_path)
            state_dict_sha256 = None
        else:
            state_dict_source, state_dict_sha256, pytorch_architecture = get_bmz_model_info(
                self.workflow.bmz_config["original_bmz_config"],
                original_model_version,
            )

        # Only exporting in pytorch_state_dict
        pytorch_state_dict = PytorchStateDictWeightsDescr(
            source=state_dict_source,
            sha256=state_dict_sha256,
            architecture=pytorch_architecture,
            pytorch_version=torch.__version__,
        )
        # torchscript = TorchscriptWeightsDescr(
        #     source=self.workflow.bmz_config['original_bmz_config'].weights.torchscript.source,
        #     sha256=self.workflow.bmz_config['original_bmz_config'].weights.torchscript.sha256,
        #     pytorch_version=Version(torch.__version__),
        #     parent="pytorch_state_dict", # these weights were converted from the pytorch_state_dict weights ones.
        # ),

        # Export model to BMZ format
        model_descr = ModelDescr(
            name=model_name,
            description=description,
            authors=authors,
            cite=citations,
            license=license,
            documentation=doc,
            git_repo=HttpUrl("https://github.com/BiaPyX/BiaPy"),
            inputs=inputs,
            outputs=outputs,
            weights=WeightsDescr(
                pytorch_state_dict=pytorch_state_dict,
                # torchscript,
            ),
            tags=tags,
            covers=covers,
            maintainers=maintainers,
        )

        # print(f"Building BMZ package: {args}")
        print(f"Created '{model_descr.name}'")

        # Checking model consistency
        # summary = test_model(model_descr)
        # summary.display()

        # Saving the model into BMZ format
        model_path = os.path.join(building_dir, model_name + ".zip")
        print(
            "Package path:",
            save_bioimageio_package(model_descr, output_path=Path(model_path)),
        )

        # Recover the original working path
        os.chdir(cwd)

        print("FINISHED JOB {} !!".format(self.job_identifier))
        
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
