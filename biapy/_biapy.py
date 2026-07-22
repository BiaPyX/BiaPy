import copy
import io
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
import re
import sys
import argparse
import datetime
import ntpath
import yaml
import torch
import pooch
import torch.distributed as dist
from shutil import copyfile
import numpy as np
import importlib
from yacs.config import CfgNode as CN
from typing import (
    Optional,
    Dict,
    Union,
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
    SpaceOutputAxisWithHalo,
    SizeReference,
    TensorId,
    WeightsDescr,
    ArchitectureFromFileDescr,
    ModelDescr,
    FileDescr_dependencies,
)
from packaging.version import Version
from bioimageio.spec._internal.io_basics import Sha256
from bioimageio.spec import save_bioimageio_package
from bioimageio.spec.dataset.v0_3 import LinkedDataset

import biapy
from biapy.utils.misc import (
    init_devices,
    is_dist_avail_and_initialized,
    set_seed,
    get_rank,
    is_main_process,
    get_world_size,
    setup_for_distributed,
    compute_threads_and_workers,
)
from biapy.models.bmz_utils import (
    create_model_doc,
    create_environment_file_for_model,
    create_model_cover,
    get_bmz_model_info,
)
from biapy.models import adapt_bmz_model_kwargs
from biapy.config.config import Config, update_dependencies
from biapy.engine.check_configuration import (
    check_configuration,
    convert_old_model_cfg_to_current_version,
    diff_between_configs,
)
from biapy.utils.util import create_file_sha256sum
from biapy.data.data_manipulation import ensure_2d_shape, ensure_3d_shape


class _Tee:
    """Write to multiple streams simultaneously (used to mirror stdout/stderr to a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        return False


class BiaPy:
    def __init__(
        self,
        config: Union[str, dict, CN],
        result_dir: Optional[str] = None,
        name: Optional[str] = "unknown_job",
        run_id: Optional[int] = 1,
        gpu: Optional[str] = "",
        world_size: Optional[int] = 1,
        local_rank: Optional[int] = -1,
        dist_on_itp: Optional[bool] = False,
        dist_url: Optional[str] = "env://",
        dist_backend: Optional[str] = "nccl",
        verbose: Optional[bool] = False,
        save_files: Optional[bool] = None,
    ):
        """
        Run the main functionality of the job.

        Parameters
        ----------
        config: str or dict or CfgNode
            Configuration source. It can be a path to a YAML file, a dict of overrides (e.g.
            from :func:`biapy.build_config`), a YACS ``CfgNode``, or a path to a BiaPy ``.pth``
            checkpoint (the configuration is embedded in it, so the workflow is rebuilt for
            inference). The in-memory forms allow running BiaPy without a YAML file on disk.

        result_dir: str, optional
            Path where the job outputs (results, checkpoints, logs, config backup) are stored.
            If omitted, BiaPy runs ephemerally (nothing is written to disk) and only in-memory
            prediction is available; operations that must write then raise a clear error. See
            ``save_files``.

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

        verbose: bool, optional
            Whether to mirror BiaPy's regular output to the console while building the
            workflow. Defaults to ``False`` so that constructing a :class:`BiaPy` through the
            Python API is quiet (everything is still written to the run log file). The CLI
            entry point sets it to ``True``. Regardless of this flag, running the workflow
            (:meth:`train`, :meth:`test`, :meth:`predict`, :meth:`run_job`) mirrors its output
            to the console. Use :meth:`print_config` / ``print(biapy)`` to inspect the
            configuration on demand.

        save_files: bool, optional
            Whether BiaPy is allowed to write files to disk (config backup, run log, ...).
            Defaults to ``None``, meaning it follows ``result_dir``: it writes when a
            ``result_dir`` is given and stays ephemeral (nothing written) when it is not. Pass
            ``True`` to force writing (a ``result_dir`` is then required) or ``False`` to force
            an ephemeral run even when a ``result_dir`` is given.
        """
        self.verbose = bool(verbose)

        # Third instantiation option: a model checkpoint as the configuration source. A BiaPy
        # '.pth' embeds the whole configuration, so the workflow is rebuilt from it (see
        # BiaPy.load_workflow_from_model for BMZ ids). '.safetensors' has no embedded config.
        if isinstance(config, str) and (config.endswith(".pth") or config.endswith(".safetensors")):
            config = self._config_from_checkpoint(config)

        if dist_backend not in ["nccl", "gloo"]:
            raise ValueError("Invalid value for 'dist_backend'. Should be either 'nccl' or 'gloo'.")

        # File writing follows result_dir: with a result_dir we write there, without one we run
        # ephemerally (nothing on disk), which is enough to load a model and predict in memory.
        # save_files can force it either way; forcing writes without a result_dir is an error.
        self._result_dir_provided = result_dir is not None and result_dir != ""
        if save_files is None:
            self.save_files = self._result_dir_provided
        else:
            self.save_files = bool(save_files)
        if self.save_files and not self._result_dir_provided:
            raise ValueError(
                "save_files=True needs a 'result_dir' to write results, checkpoints and logs. "
                "Pass result_dir=..., or omit save_files (it defaults to writing only when a "
                "result_dir is given)."
            )
        if not self._result_dir_provided:
            # Throwaway base for the internal path strings; nothing is ever written there.
            result_dir = os.path.join(tempfile.gettempdir(), "biapy_ephemeral")

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
        if self.save_files:
            os.makedirs(self.cfg_bck_dir, exist_ok=True)
        if isinstance(self.args.config, str):
            head, tail = ntpath.split(self.args.config)
            self.cfg_filename = tail if tail else ntpath.basename(head)
        else:
            # In-memory config (dict/CfgNode): synthesize a filename for logs and backups.
            self.cfg_filename = str(self.args.name) + ".yaml"

        now = datetime.datetime.now()
        self.log_timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        self._stdout_log_file = None
        self._stdout_log_path = None
        self._null_stream = None

        # Config and log share run_id + timestamp so each run has a matching pair.
        cfg_root, cfg_ext = os.path.splitext(self.cfg_filename)
        self.cfg_file = os.path.join(
            self.cfg_bck_dir, f"{cfg_root}_{self.args.run_id}_{self.log_timestamp}{cfg_ext}"
        )

        # Buffer output until we know the rank 0 log file. In quiet mode it is not mirrored to
        # the console, so building the workflow is silent.
        _early_buf = io.StringIO()
        if self.verbose:
            sys.stdout = _Tee(sys.__stdout__, _early_buf)
            sys.stderr = _Tee(sys.__stderr__, _early_buf)
        else:
            sys.stdout = _Tee(_early_buf)
            sys.stderr = _Tee(_early_buf)

        print("Date     : {}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
        print("Arguments: {}".format(self.args))
        print("Job      : {}".format(self.job_identifier))
        print("BiaPy    : {}".format(biapy.__version__))
        print("Python   : {}".format(sys.version.split("\n")[0]))
        print("PyTorch  : {}".format(torch.__version__))

        # Load the raw config from the source (YAML path, dict or CfgNode)
        original_cfg = self._load_raw_config(self.args.config)

        # Back up the config file when a path was given (in-memory configs are backed up later)
        if self.save_files and is_main_process() and isinstance(self.args.config, str):
            copyfile(self.args.config, self.cfg_file)

        # Merge configuration file with the default settings
        cfg_manager = Config(self.job_dir, self.job_identifier)

        # Translates the input config it to current version
        temp_cfg = CN(convert_old_model_cfg_to_current_version(original_cfg))
        if cfg_manager._C.PROBLEM.PRINT_OLD_KEY_CHANGES:
            print("The following changes were made in order to adapt the input configuration:")
            diff_between_configs(original_cfg, temp_cfg)
        del original_cfg
        cfg_manager._C.merge_from_other_cfg(temp_cfg)
        update_dependencies(cfg_manager)

        # GPU selection
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu

        opts = []
        if self.args.gpu != "" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
            self.num_gpus = len(np.unique(np.array(self.args.gpu.strip().split(","))))
            opts.extend(["SYSTEM.NUM_GPUS", self.num_gpus])
        else:
            self.num_gpus = 0
            opts.extend(["SYSTEM.NUM_GPUS", self.num_gpus])

        # GPU management — distributed rank is known after this call
        self.device = init_devices(self.args, cfg_manager.get_cfg_defaults())
        cfg_manager._C.merge_from_list(opts)
        self.cfg: CN = cfg_manager.get_cfg_defaults()

        # Back up in-memory configs once fully resolved (mirrors the copyfile done for paths)
        if self.save_files and is_main_process() and not isinstance(self.args.config, str):
            with open(self.cfg_file, "w", encoding="utf8") as stream:
                stream.write(self.cfg.dump())

        # Rank is settled: on rank 0 open the log file (unless disabled) and flush the buffer
        # into it. In verbose mode the buffer was already shown live on the console.
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if self.save_files and is_main_process():
            logs_dir = self.cfg.LOG.LOG_DIR
            os.makedirs(logs_dir, exist_ok=True)
            self._stdout_log_path = os.path.join(
                logs_dir,
                f"{self.job_identifier}_log_{self.args.run_id}_{self.log_timestamp}.txt",
            )
            self._stdout_log_file = open(self._stdout_log_path, "w", encoding="utf-8", buffering=1)
            self._stdout_log_file.write(_early_buf.getvalue())
            self._stdout_log_file.flush()
        _early_buf.close()
        # Route the rest of construction: silent unless verbose (see _route).
        self._route(mirror_console=self.verbose)

        # Reproducibility
        set_seed(self.cfg.SYSTEM.SEED)

        # Number of CPU calculation
        cpu_budget, cpu_per_rank, main_threads, num_workers = compute_threads_and_workers(
            user_num_cpus=self.cfg.SYSTEM.NUM_CPUS,
            world_size=get_world_size(),
            training_samples=None,
            max_workers_cap=8,  # To avoid too many workers that can lead to memory issues
        )
        # Set threads for the main (rank) process
        torch.set_num_threads(main_threads)
        # 'set_num_interop_threads' can only be set once per process; guard it so several
        # BiaPy instances can be created in the same process (e.g. multi-workflow predict).
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.cfg.merge_from_list(["SYSTEM.NUM_CPUS", cpu_budget])

        print(
            f"CPU budget(total)={cpu_budget} | per_rank={cpu_per_rank} | "
            f"main_threads={main_threads} | num_workers(per_rank)={num_workers} | "
            f"world_size={get_world_size()}"
        )
        
        check_configuration(self.cfg, self.job_identifier)
        print("Configuration details:")
        print(self.cfg)

        # System resources resolved above; kept so the workflow can be rebuilt after a config
        # change (see update_config / _build_workflow) without recomputing them.
        self._system_dict = {
            "cpu_budget": cpu_budget,
            "cpu_per_rank": cpu_per_rank,
            "main_threads": main_threads,
            "num_workers_hint": num_workers,
        }

        ##########################
        #       TRAIN/TEST       #
        ##########################
        self._build_workflow()

        # Quiet (API) mode: give the console back so the interactive session stays clean;
        # each run re-attaches output via _run_output. Verbose (CLI) keeps the tee until run_job.
        if not self.verbose:
            self._restore_std_streams()

    def _build_workflow(self):
        """
        Instantiate the workflow object from the current :attr:`cfg`.

        Resolves the workflow class from ``PROBLEM.TYPE`` and constructs it. This is where much
        config-derived state is decided (dimensionality, resolution, axes order, post-processing
        flags, checkpoint consistency, metrics, ...) and where ``cfg`` gets frozen, so
        :meth:`update_config` calls it again to refresh that state. Expects a defrosted ``cfg``.
        """
        workflowname = str(self.cfg.PROBLEM.TYPE).lower()
        mdl = importlib.import_module("biapy.engine." + workflowname)
        names = [x for x in mdl.__dict__ if not x.startswith("_")]
        globals().update({k: getattr(mdl, k) for k in names})
        name = [x for x in names if "Base" not in x and "Workflow" in x][0]

        print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*")
        print(f"Initializing {name}")
        print("*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*\n")
        self.workflow = getattr(mdl, name)(
            self.cfg,
            self.job_identifier,
            self.device,
            system_dict=self._system_dict,
            args=self.args,
        )
        self.workflow.log_timestamp = self.log_timestamp

    def _null_sink(self):
        """Lazily-opened /dev/null writer used to discard output (keeps a valid fileno)."""
        if self._null_stream is None:
            self._null_stream = open(os.devnull, "w")
        return self._null_stream

    def _route(self, mirror_console: bool):
        """
        Point stdout/stderr where output should go: to the console (plus the log file when
        one exists) if ``mirror_console``, otherwise silently (log file only, or /dev/null
        when no file is written).
        """
        log = self._stdout_log_file
        if mirror_console:
            extra = [log] if log is not None else []
            sys.stdout = _Tee(sys.__stdout__, *extra)
            sys.stderr = _Tee(sys.__stderr__, *extra)
        else:
            target = log if log is not None else self._null_sink()
            sys.stdout = _Tee(target)
            sys.stderr = _Tee(target)

    @staticmethod
    def _restore_std_streams():
        """Give stdout/stderr back to the real console."""
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    @contextmanager
    def _run_output(self, mirror_console: bool = True):
        """
        Route BiaPy's output for the duration of a run, restoring the previous streams after.

        ``mirror_console`` shows the output on the console (and log file); when ``False`` the
        run is silent (log file only, or discarded when no file is written).
        """
        prev_out, prev_err = sys.stdout, sys.stderr
        self._route(mirror_console=mirror_console)
        try:
            yield
        finally:
            if self._stdout_log_file is not None:
                self._stdout_log_file.flush()
            sys.stdout, sys.stderr = prev_out, prev_err

    def print_config(self):
        """
        Print the full resolved configuration of the workflow to the console.

        This is the configuration dump that BiaPy prints automatically when run from the CLI;
        through the Python API construction is quiet, so call this to inspect it on demand.
        Use ``print(biapy)`` (or :meth:`__repr__`) for a short summary instead.
        """
        print("Configuration details:")
        print(self.cfg)

    @staticmethod
    def _render_table(header: str, rows) -> str:
        """Render a ``header`` plus aligned ``(label, value)`` rows as a boxed text block."""
        rows = [(str(label), str(value)) for label, value in rows]
        label_w = max((len(label) for label, _ in rows), default=0)
        value_w = max((len(value) for _, value in rows), default=0)
        width = max(len(header), label_w + 2 + value_w)
        lines = [header, "=" * width]
        lines += ["{}: {}".format(label.ljust(label_w), value) for label, value in rows]
        return "\n".join(lines)

    def _model_desc(self) -> str:
        """Short model description: BMZ id, torchvision or BiaPy architecture (no details)."""
        cfg = self.cfg
        source = str(cfg.MODEL.SOURCE).lower()
        if source == "bmz":
            return "BMZ: {}".format(cfg.MODEL.BMZ.SOURCE_MODEL_ID)
        if source == "torchvision":
            return "torchvision: {}".format(cfg.MODEL.ARCHITECTURE)
        return str(cfg.MODEL.ARCHITECTURE)

    def _enabled_augmentations(self):
        """List the augmentation toggles that are switched on (each has a sibling ``*_PROB``)."""
        aug = self.cfg.AUGMENTOR
        enabled = []
        for key in aug.keys():
            if isinstance(aug[key], bool) and aug[key] and (key + "_PROB") in aug:
                enabled.append(key)
        return enabled

    def _enabled_postprocessing(self):
        """List enabled TEST.POST_PROCESSING methods (top-level bools and nested ``.ENABLE`` groups)."""
        pp = self.cfg.TEST.POST_PROCESSING
        enabled = []
        for key in pp.keys():
            value = pp[key]
            if isinstance(value, bool) and value:
                enabled.append(key)
            elif isinstance(value, CN) and bool(getattr(value, "ENABLE", False)):
                enabled.append(key)
        return enabled

    def _device_desc(self) -> str:
        """Human-readable compute target: GPU count and device, or CPU."""
        num_gpus = int(getattr(self, "num_gpus", 0) or 0)
        device = str(getattr(self, "device", "?"))
        if num_gpus > 0:
            return "{} GPU(s) [{}]".format(num_gpus, device)
        return "CPU [{}]".format(device)

    def _file_rows(self):
        """Rows describing the files being written (empty when file writing is disabled)."""
        if not self.save_files:
            return [("Files", "disabled (save_files=False)")]
        rows = []
        if self._stdout_log_path:
            rows.append(("Log file", str(self._stdout_log_path)))
        if getattr(self, "cfg_file", None):
            rows.append(("Config file", str(self.cfg_file)))
        return rows

    def _require_writable_output(self, action: str):
        """Raise a clear error if ``action`` needs to write to disk but no output dir is available."""
        if self._result_dir_provided and self.save_files:
            return
        reasons = []
        if not self._result_dir_provided:
            reasons.append("no 'result_dir' was provided")
        if not self.save_files:
            reasons.append("file writing is disabled (save_files=False)")
        raise ValueError(
            "{} needs to write files to disk, but {}. Rebuild the BiaPy object with "
            "result_dir=... and save_files=True.".format(action, " and ".join(reasons))
        )

    def _task_requires_disk(self) -> bool:
        """Workflows that must write intermediate files to complete (synapse detection, by-chunks)."""
        cfg = self.cfg
        if cfg.PROBLEM.TYPE == "INSTANCE_SEG" and str(cfg.PROBLEM.INSTANCE_SEG.TYPE) == "synapses":
            return True
        if cfg.TEST.BY_CHUNKS.ENABLE:
            return True
        return False

    def _summary(self) -> str:
        """Build a human-readable, multi-line summary of the configured workflow."""
        cfg = self.cfg

        # Workflow class name (e.g. "Instance_Segmentation_Workflow"), falling back to PROBLEM.TYPE.
        workflow_cls = type(getattr(self, "workflow", None)).__name__
        if not getattr(self, "workflow", None):
            workflow_cls = str(cfg.PROBLEM.TYPE)

        # Enabled phases.
        phases = []
        if cfg.TRAIN.ENABLE:
            phases.append("train")
        if cfg.TEST.ENABLE:
            phases.append("test")
        phases = ", ".join(phases) if phases else "none"

        rows = [
            ("Job", "{} (run {})".format(self.args.name, self.args.run_id)),
            ("Workflow", "{}  [{}]".format(cfg.PROBLEM.TYPE, workflow_cls)),
            ("Dimensions", str(cfg.PROBLEM.NDIM)),
            ("Patch size", str(tuple(cfg.DATA.PATCH_SIZE))),
            ("Classes", str(cfg.DATA.N_CLASSES)),
            ("Model", self._model_desc()),
            ("Phases enabled", phases),
        ]

        # Checkpoint that will be / was loaded (useful after load_workflow_from_model).
        if cfg.MODEL.LOAD_CHECKPOINT and cfg.PATHS.CHECKPOINT_FILE:
            rows.append(("Checkpoint", str(cfg.PATHS.CHECKPOINT_FILE)))

        rows.append(("Device", self._device_desc()))
        rows.append(("BiaPy version", str(biapy.__version__)))
        return self._render_table("BiaPy workflow", rows)

    def print_train_info(self):
        """
        Print a concise overview of the training configuration.

        Shows the model (without architecture details) and its source, the patch size, the
        compute device(s), number of epochs, learning rate, optimizer, batch size and the
        enabled augmentations. Use :meth:`print_config` for the full configuration dump.
        """
        cfg = self.cfg
        augs = self._enabled_augmentations()
        if not cfg.AUGMENTOR.ENABLE:
            augs_desc = "disabled"
        elif augs:
            augs_desc = "{} ({})".format(len(augs), ", ".join(augs))
        else:
            augs_desc = "enabled (none selected)"

        # LR / optimizer are stored as single-element lists in the config.
        lr = cfg.TRAIN.LR[0] if isinstance(cfg.TRAIN.LR, (list, tuple)) and cfg.TRAIN.LR else cfg.TRAIN.LR
        optimizer = (
            cfg.TRAIN.OPTIMIZER[0]
            if isinstance(cfg.TRAIN.OPTIMIZER, (list, tuple)) and cfg.TRAIN.OPTIMIZER
            else cfg.TRAIN.OPTIMIZER
        )
        scheduler = cfg.TRAIN.LR_SCHEDULER.NAME if cfg.TRAIN.LR_SCHEDULER.NAME else "none"

        rows = [
            ("Enabled", str(cfg.TRAIN.ENABLE)),
            ("Model", self._model_desc()),
            ("Model source", str(cfg.MODEL.SOURCE)),
            ("Patch size", str(tuple(cfg.DATA.PATCH_SIZE))),
            ("Device", self._device_desc()),
            ("Epochs", str(cfg.TRAIN.EPOCHS)),
            ("Batch size", str(cfg.TRAIN.BATCH_SIZE)),
            ("Learning rate", str(lr)),
            ("Optimizer", str(optimizer)),
            ("LR scheduler", str(scheduler)),
            ("Patience", str(cfg.TRAIN.PATIENCE)),
            ("Train data", str(cfg.DATA.TRAIN.PATH)),
            ("Augmentations", augs_desc),
        ]
        rows += self._file_rows()
        print(self._render_table("BiaPy training configuration", rows))

    def print_test_info(self):
        """
        Print a concise overview of the inference/test configuration.

        Shows the test data path, whether ground truth is provided, the patch overlap and
        padding used to stitch predictions, and the enabled post-processing methods. Use
        :meth:`print_config` for the full configuration dump.
        """
        cfg = self.cfg
        load_gt = bool(cfg.DATA.TEST.LOAD_GT)
        gt_desc = "yes ({})".format(cfg.DATA.TEST.GT_PATH) if load_gt else "no"

        pp = self._enabled_postprocessing()
        pp_desc = ", ".join(pp) if pp else "none"

        rows = [
            ("Enabled", str(cfg.TEST.ENABLE)),
            ("Test data", str(cfg.DATA.TEST.PATH)),
            ("Ground truth", gt_desc),
            ("Patch size", str(tuple(cfg.DATA.PATCH_SIZE))),
            ("Overlap", str(tuple(cfg.DATA.TEST.OVERLAP))),
            ("Padding", str(tuple(cfg.DATA.TEST.PADDING))),
            ("Device", self._device_desc()),
            ("Post-processing", pp_desc),
        ]
        rows += self._file_rows()
        print(self._render_table("BiaPy test configuration", rows))

    def update_config(
        self,
        updates: Optional[dict] = None,
        check: bool = False,
        rebuild: bool = True,
        **kwargs,
    ):
        """
        Update configuration values on an already-built :class:`BiaPy` instance.

        Values are applied in place to :attr:`cfg` (the workflow shares the same object), so
        this is the way to tweak the configuration after construction, e.g. to enable a phase
        or change training hyper-parameters before calling :meth:`train` / :meth:`test`.

        By default the workflow is **rebuilt** afterwards (see ``rebuild``). This matters
        because the workflow object decides a lot of state from the configuration at
        construction time and then freezes ``cfg`` (dimensionality, resolution, axes order,
        post-processing flags, checkpoint consistency, metrics, ...). Editing ``cfg`` alone
        would leave that cached state stale, so a structural change (``PROBLEM.*``, ``MODEL.*``,
        ``DATA.PATCH_SIZE``, ``DATA.N_CLASSES``, ``TEST.POST_PROCESSING.*``, ...) could crash
        later. Rebuilding re-derives it from the new configuration.

        Parameters
        ----------
        updates : dict, optional
            Mapping of dotted config keys to values, e.g.
            ``{"TRAIN.ENABLE": True, "TRAIN.EPOCHS": 100}``.

        check : bool, optional
            When ``True``, re-run :func:`check_configuration` after applying the changes so
            inconsistencies are reported immediately. Defaults to ``False``. Note that
            rebuilding the workflow (the default) already re-validates a good part of the
            configuration.

        rebuild : bool, optional
            When ``True`` (default), rebuild the workflow object so its construction-time
            cached state reflects the new configuration. Set to ``False`` only for changes you
            know are read fresh at run time (e.g. ``TRAIN.EPOCHS``, ``TRAIN.LR``, data paths)
            to avoid the cost of reconstruction (which also drops any already-prepared model).

        kwargs : optional
            Convenience form where ``__`` in a keyword is turned into ``.`` (e.g.
            ``TRAIN__EPOCHS=100`` is ``"TRAIN.EPOCHS"``).

        Returns
        -------
        CfgNode
            The updated configuration (also available as :attr:`cfg`).

        Notes
        -----
        ``SYSTEM.*`` options (GPUs, CPUs, seed, distributed backend) and the compute device
        are resolved once when the :class:`BiaPy` object is created, before the workflow
        exists, so they cannot be changed here — build a new ``BiaPy(config, gpu=...)`` for
        that. The training/test data pipelines themselves are built lazily inside
        :meth:`train` / :meth:`test`, so enabling a phase and pointing it at the data is
        enough to run it.
        """
        merged = dict(updates) if updates else {}
        for key, value in kwargs.items():
            merged[key.replace("__", ".")] = value
        if not merged:
            return self.cfg

        # SYSTEM.* / device are fixed at object creation and can't be reflected by rebuilding
        # only the workflow, so reject them instead of silently ignoring the change.
        process_keys = [k for k in merged if str(k).split(".")[0] == "SYSTEM"]
        if process_keys:
            raise ValueError(
                "SYSTEM.* options (GPUs, CPUs, seed, distributed backend) are decided when the "
                "BiaPy object is created and cannot be changed with update_config (offending "
                "keys: {}). Build a new BiaPy(config, gpu=..., ...) instead.".format(
                    ", ".join(sorted(process_keys))
                )
            )

        opts = []
        for key, value in merged.items():
            opts.extend([key, value])

        was_frozen = self.cfg.is_frozen()
        self.cfg.defrost()
        try:
            # merge_from_list validates key names/types and raises a clear error on typos.
            self.cfg.merge_from_list(opts)
            update_dependencies(self.cfg)
            if check:
                check_configuration(self.cfg, self.job_identifier)
            if rebuild:
                # Rebuild the workflow so its cached construction-time state matches the new
                # config (quiet unless verbose, like the initial build).
                with self._run_output(mirror_console=self.verbose):
                    self._build_workflow()
        finally:
            # The workflow build usually freezes cfg, but not every workflow does.
            if was_frozen and not self.cfg.is_frozen():
                self.cfg.freeze()
        return self.cfg

    def __repr__(self) -> str:
        """Return a readable summary of the workflow instead of the default object id."""
        return self._summary()

    def __str__(self) -> str:
        """Return the same readable summary as :meth:`__repr__`."""
        return self._summary()

    @staticmethod
    def _load_raw_config(config: Union[str, dict, CN]) -> dict:
        """Return the raw config as a plain dict from a YAML path, a dict or a CfgNode."""
        if isinstance(config, CN):
            return yaml.safe_load(config.dump())
        if isinstance(config, dict):
            return copy.deepcopy(config)
        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError("Provided {} config file does not exist".format(config))
            with open(config, "r", encoding="utf8") as stream:
                try:
                    cfg_content = stream.read()
                    if "\t" in cfg_content:
                        cfg_content = cfg_content.replace("\t", "  ")
                    return yaml.safe_load(cfg_content)
                except yaml.YAMLError as exc:
                    raise ValueError(
                        "Error reading the configuration file. Please check the file format: {}".format(exc)
                    )
        raise TypeError(
            "'config' must be a path to a YAML file (str), a dict or a YACS CfgNode. Provided: {}".format(type(config))
        )

    @classmethod
    def _config_from_checkpoint(cls, source: str) -> dict:
        """
        Rebuild an inference configuration dict from a BiaPy model checkpoint path.

        A ``.pth`` embeds the configuration (``checkpoint["cfg"]``); it is filtered to the
        current schema (dropping runtime-derived keys and stale ``PATHS``) and switched to
        inference (test enabled, training disabled, checkpoint loaded). ``.safetensors`` has
        no embedded config and is rejected.
        """
        if source.endswith(".safetensors"):
            raise ValueError(
                "'.safetensors' checkpoints do not embed the BiaPy configuration, so the "
                "workflow cannot be inferred from them. Build the configuration explicitly with "
                "build_config(...) and pass it to BiaPy(...) (with the checkpoint in "
                "'PATHS.CHECKPOINT_FILE' and 'MODEL.LOAD_CHECKPOINT' enabled)."
            )
        if not os.path.isfile(source):
            raise FileNotFoundError("Checkpoint file not found: {}".format(source))

        from functools import partial as _partial
        from biapy.config.config import Config as _Config

        torch.serialization.add_safe_globals([CN, set, _partial, torch.nn.modules.normalization.LayerNorm])
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        if "cfg" not in checkpoint:
            raise ValueError(
                "Checkpoint '{}' does not embed a BiaPy configuration; the workflow cannot be "
                "rebuilt from it. Build the configuration explicitly with build_config(...) and "
                "pass it to BiaPy(...) instead.".format(source)
            )

        defaults = _Config("load_workflow", "load_workflow").get_cfg_defaults()
        cfg_dict = _filter_cfg_to_schema(cls._load_raw_config(checkpoint["cfg"]), defaults)
        # Drop the embedded PATHS (they point to the original training job) so they are
        # re-derived from the new result_dir/name at construction time.
        cfg_dict.pop("PATHS", None)
        _deep_merge(
            cfg_dict,
            {
                "TRAIN": {"ENABLE": False},
                "TEST": {"ENABLE": True},
                "DATA": {"TEST": {"LOAD_GT": False, "USE_VAL_AS_TEST": False}},
                "MODEL": {"SOURCE": "biapy", "LOAD_CHECKPOINT": True},
                "PATHS": {"CHECKPOINT_FILE": source},
            },
        )
        return cfg_dict

    @classmethod
    def load_workflow_from_model(cls, source: str, **run_kwargs):
        """
        Build a ready-to-infer :class:`BiaPy` from a trained model, inferring the workflow.

        The workflow (and the rest of the configuration) is recovered automatically from the
        ``source``:

        - A BiaPy ``.pth`` checkpoint: the configuration is embedded inside it, so the whole
          workflow is rebuilt from it (handled by the constructor).
        - A BioImage Model Zoo id/nickname: the workflow and dimensionality are inferred from
          the model's RDF via :func:`biapy.models.check_bmz_args`.

        A ``.safetensors`` checkpoint does not embed the configuration; build it explicitly with
        :func:`build_config` and pass it to :class:`BiaPy` instead.

        Parameters
        ----------
        source : str
            Path to a ``.pth`` BiaPy checkpoint, or a BMZ model id/nickname.

        run_kwargs : dict, optional
            Extra run arguments forwarded to the constructor (``result_dir``, ``name``,
            ``gpu``, ...).

        Returns
        -------
        BiaPy
            A test-enabled BiaPy instance ready to run :meth:`predict`.
        """
        # File checkpoints ('.pth' / '.safetensors') are handled by the constructor directly.
        if isinstance(source, str) and (source.endswith(".pth") or source.endswith(".safetensors")):
            return cls(source, **run_kwargs)

        # Otherwise treat 'source' as a BMZ id/nickname and infer the workflow from its RDF.
        from biapy.models import check_bmz_args

        _, _, workflow_info = check_bmz_args(source)
        workflow = workflow_info.get("workflow_type")
        if not workflow:
            raise ValueError(
                "Could not infer the workflow from the BMZ model '{}'. Build the configuration "
                "explicitly with build_config(...) and pass it to BiaPy(...).".format(source)
            )
        cfg_dict = {
            "PROBLEM": {"TYPE": workflow, "NDIM": workflow_info.get("ndim", "2D")},
            "MODEL": {"SOURCE": "bmz", "BMZ": {"SOURCE_MODEL_ID": source}},
            "TRAIN": {"ENABLE": False},
            "TEST": {"ENABLE": True},
            "DATA": {"TEST": {"LOAD_GT": False, "USE_VAL_AS_TEST": False}},
        }
        return cls(cfg_dict, **run_kwargs)

    def train(self):
        """Call training phase."""
        # Training always writes checkpoints and logs, so a writable output dir is mandatory.
        self._require_writable_output("Training")
        if is_dist_avail_and_initialized():
            setup_for_distributed(is_main_process())

        if self.cfg.TRAIN.ENABLE:
            with self._run_output():
                self.workflow.train()
        else:
            raise ValueError(
                "Training is not enabled ('TRAIN.ENABLE' is False), so train() cannot run. "
                "This usually means the instance was built for inference (e.g. via "
                "BiaPy.load_workflow_from_model). Enable training and point it at your data, e.g.:\n"
                "    biapy.update_config({'TRAIN.ENABLE': True, "
                "'DATA.TRAIN.PATH': '/path/to/x', 'DATA.TRAIN.GT_PATH': '/path/to/y'})\n"
                "Then call biapy.train(). See biapy.print_train_info() for the current training "
                "settings."
            )

        if is_dist_avail_and_initialized():
            setup_for_distributed(True)
            print(f"[Rank {get_rank()} ({os.getpid()})] Process waiting (train finished, step 2) . . . ")
            dist.barrier()

    def test(self):
        """Call test phase."""
        # test() writes its results (metrics, output images) to disk, so it needs an output dir.
        self._require_writable_output("Testing")
        if is_dist_avail_and_initialized():
            setup_for_distributed(is_main_process())

        if self.cfg.TEST.ENABLE:
            with self._run_output():
                self.workflow.test()
        else:
            raise ValueError(
                "Testing is not enabled ('TEST.ENABLE' is False), so test() cannot run. "
                "Enable it and point it at your data, e.g.:\n"
                "    biapy.update_config({'TEST.ENABLE': True, 'DATA.TEST.PATH': '/path/to/x'})\n"
                "Then call biapy.test(). See biapy.print_test_info() for the current test settings."
            )

        # if is_dist_avail_and_initialized():
        #     setup_for_distributed(True)
        #     print(f"[Rank {get_rank()} ({os.getpid()})] Process waiting (test finished) . . . ")
        #     self.wait_and_stop_ddp()

    def predict(self, image, gt=None, return_prediction=True, verbose=False):
        """
        Run inference over an in-memory image (NumPy array) without writing it to disk.

        The model is prepared automatically if it has not been loaded yet.

        Parameters
        ----------
        image : NDArray
            Input image to predict over. Reshaped internally to BiaPy's expected layout
            (``(Y, X, C)`` in 2D, ``(Z, Y, X, C)`` in 3D).

        gt : NDArray, optional
            Ground truth associated to ``image``. Only used when metrics are requested
            (``DATA.TEST.LOAD_GT`` enabled).

        return_prediction : bool, optional
            When ``True`` (default), the produced prediction is returned as a NumPy array
            and nothing is written to disk. When ``False``, the normal on-disk outputs are
            produced and ``None`` is returned.

        verbose : bool, optional
            When ``True``, show BiaPy's inference output on the console. Defaults to ``False``
            so repeated predictions stay quiet (output still goes to the log file if enabled).

        Returns
        -------
        NDArray or None
            The predicted output array (batch axis dropped for the single image) when
            ``return_prediction`` is ``True``, otherwise ``None``.
        """
        # Output-dir requirements: writing results to disk (return_prediction=False) or a
        # workflow that must write intermediate files (synapse detection / by-chunks).
        if not return_prediction:
            self._require_writable_output("predict(return_prediction=False)")
        elif self._task_requires_disk():
            self._require_writable_output("This workflow (synapse detection / by-chunks)")

        was_frozen = self.cfg.is_frozen()
        self.cfg.defrost()
        self.cfg.merge_from_list(["TEST.ENABLE", True, "TEST.BY_CHUNKS.ENABLE", False])
        if was_frozen:
            self.cfg.freeze()

        prev_return = self.workflow.return_prediction
        prev_save = self.workflow.save_to_disk
        self.workflow.return_prediction = bool(return_prediction)
        self.workflow.save_to_disk = not bool(return_prediction)
        self.workflow._predictions = []

        try:
            with self._run_output(mirror_console=verbose):
                if not self.workflow.model_prepared:
                    self.workflow.prepare_model()
                self.workflow.test(image=image, gt=gt)
        finally:
            self.workflow.return_prediction = prev_return
            self.workflow.save_to_disk = prev_save

        if not return_prediction:
            return None

        preds = self.workflow._predictions
        if not preds:
            print("WARNING: 'predict' did not capture any prediction to return.")
            return None

        chosen = None
        for role in ("post", "raw"):
            match = [e["data"] for e in preds if e.get("role") == role]
            if match:
                chosen = match[-1]
                break
        if chosen is None:
            chosen = preds[-1]["data"]

        if isinstance(chosen, np.ndarray) and chosen.ndim >= 1 and chosen.shape[0] == 1:
            chosen = chosen[0]
        return chosen

    def export_model_to_bmz(
        self,
        building_dir: str,
        bmz_cfg: Optional[Dict] = {},
        reuse_original_bmz_config: Optional[bool] = False,
    ):
        """
        Export a model into Bioimage Model Zoo format.

        Parameters
        ----------
        building_dir : path
            Path to store files and the build of BMZ package.

        bmz_cfg : dict, optional
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

            data : dict
                Information of the data used to train the model. Expected keys:
                    * ``name``: Name of the dataset.
                    * ``doi``: DOI of the dataset or a reference to find it.
                    * ``image_modality``: image modality of the dataset.
                    * ``dataset_id`` (optional): id of the dataset in `BMZ page <https://bioimage.io/#/?type=dataset>`__.

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

            test_input : 4D/5D Torch tensor, optional
                Test input image sample. It is crucial to give the expected axes as follows:
                E.g. ``(batch, channels, y, x)`` or ``(batch, channels, z, y, x)``.

            test_output : 4D/5D Torch tensor, optional
                Test output image sample. It is crucial to give the expected axes as follows:
                E.g. ``(batch, channels, y, x)`` or ``(batch, channels, z, y, x)``.

            covers : List of str, optional
                A list of cover images provided by either a relative path to the model folder, or
                a hyperlink starting with 'http[s]'. Please use an image smaller than 500KB and an
                aspect ratio width to height of 2:1. The supported image formats are: 'jpg', 'png', 'gif'.

            doc_path : str, optional
                Path to the documentation file.

            version : str, optional
                Version of the model. If not provided it will be ``"0.1.0"``.

        reuse_original_bmz_config : bool, optional
            Whether to reuse the original BMZ fields. This option can only be used if the model to export
            was previously loaded from BMZ.

        """
        # Exporting builds a package on disk, so writing must be enabled and an output dir set.
        self._require_writable_output("Exporting a model to BMZ")
        if not is_main_process():
            return

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
            needed_info = [
                "description",
                "authors",
                "license",
                "tags",
                "model_name",
                "data",
            ]
            for x in needed_info:
                if x not in bmz_cfg:
                    raise ValueError(f"'{x}' property must be declared in 'bmz_cfg'")

        # Check if BiaPy has been run so some of the variables have been created
        if not self.workflow.model_prepared:
            raise ValueError(
                "You need first to call train(), test() or run_job() functions so the model can be built"
            )
        
        if not reuse_original_bmz_config:
            error = False
            # If the model was loaded from BMZ we will still use the file that describes the model
            if "original_bmz_config" not in self.workflow.bmz_config:
                if "collected_sources" not in self.workflow.bmz_config:
                    raise ValueError(
                        "You need first to call train(), test() or run_job() functions so the model can be built"
                    )

                if self.workflow.checkpoint_path is None:
                    raise ValueError(
                        "Seems that you have forgoten to activate 'MODEL.LOAD_CHECKPOINT' to True"
                    )
                
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
            if not isinstance(bmz_cfg["data"], dict):
                raise ValueError("'bmz_cfg['data']' needs to be a dict.")
            else:
                needed_info = [
                    "name",
                    "doi",
                    "image_modality",
                ]
                for x in needed_info:
                    if x not in bmz_cfg["data"]:
                        raise ValueError(f"'{x}' property must be declared in 'bmz_cfg['data']'")

            if "doc_path" in bmz_cfg:
                if not os.path.exists(bmz_cfg["doc_path"]):
                    raise ValueError("Documentation file {} does not exist".format(bmz_cfg["doc_path"]))
                if not bmz_cfg["doc_path"].endswith(".md"):
                    raise ValueError(
                        "Documentation file {} has no .md extension, please change it".format(bmz_cfg["doc_path"])
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
            else:
                if (
                    "cover_raw" not in self.workflow.bmz_config 
                    or self.workflow.bmz_config["cover_raw"] is None 
                    or "cover_gt" not in self.workflow.bmz_config 
                    or self.workflow.bmz_config["cover_gt"] is None
                ):
                    raise ValueError(
                        "There is no information about covers. You can: 1) provide it using bmz_config['covers'] or run the training phase, by calling train() or run_job() functions, so a cover can be generated."
                    )

        # Add percentile norm
        preprocessing = []
        axes = list("zyx") if self.cfg.PROBLEM.NDIM == "3D" else list("yx")
        if self.cfg.DATA.NORMALIZATION.PERC_CLIP.ENABLE:
            min_percentile = max(self.cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_PERC, 0)
            max_percentile = min(self.cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_PERC, 100)
            if min_percentile != 0 or max_percentile != 100:
                preprocessing.append(
                    {
                        "id": "clip",
                        "kwargs": {
                            "max_percentile": max_percentile,
                            "min_percentile": min_percentile,
                            "axes": axes,
                        },
                    }
                )
            else:
                lower_value = self.cfg.DATA.NORMALIZATION.PERC_CLIP.LOWER_VALUE
                upper_value = self.cfg.DATA.NORMALIZATION.PERC_CLIP.UPPER_VALUE
                if lower_value != -1 or upper_value != -1:
                    preprocessing.append(
                        {
                            "id": "clip",
                            "kwargs": {
                                "max_value": upper_value if upper_value != -1 else None,
                                "min_value": lower_value if lower_value != -1 else None,
                                "axes": axes,
                            },
                        }
                    )

        if self.cfg.DATA.NORMALIZATION.TYPE == "div":
            max_val = 255
            if not reuse_original_bmz_config:
                test_input = (
                    self.workflow.bmz_config["test_input"] if "test_input" not in bmz_cfg else bmz_cfg["test_input"]
                )
                if test_input.max() == 1:
                    max_val = 1
                elif test_input.max() > 255:
                    max_val = 65535

            if max_val != 1:
                preprocessing.append(
                    {
                        "id": "scale_linear",
                        "kwargs": {"gain": float(1 / max_val), "offset": 0},
                    }
                )
        elif self.cfg.DATA.NORMALIZATION.TYPE == "scale_range":
            preprocessing.append(
                {
                    "id": "scale_range",
                    "kwargs": {
                        "max_percentile": 100,
                        "min_percentile": 0,
                        "axes": axes,
                    },
                }
            )
        else:  # zero_mean_unit_variance
            custom_mean = self.cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.MEAN_VAL
            custom_std = self.cfg.DATA.NORMALIZATION.ZERO_MEAN_UNIT_VAR.STD_VAL

            if custom_mean != [-1] and custom_std != [-1]:
                preprocessing.append(
                    {
                        "id": "fixed_zero_mean_unit_variance",
                        "kwargs": {
                            "mean": [custom_mean],
                            "std": [custom_std],
                            "axis": "channel",
                        },
                    }
                )
            else:
                preprocessing.append(
                    {
                        "id": "zero_mean_unit_variance",
                        "kwargs": {
                            "axes": axes,
                        },
                    }
                )

        print("Pre-processing: {}".format(preprocessing))

        # Post-processing activated only for old BiaPy models uploaded to BMZ without no "explicit_activations" field
        postprocessing = []
        if "postprocessing" in self.workflow.bmz_config:
            for post in self.workflow.bmz_config["postprocessing"]:
                if post == "sigmoid":
                    postprocessing.append(
                        {
                            "id": "sigmoid",
                        }
                    )
        print("Post-processing: {}".format(postprocessing))

        # Save input/output samples
        os.makedirs(building_dir, exist_ok=True)
        test_input_path = os.path.join(building_dir, "test-input.npy")
        test_output_path = os.path.join(building_dir, "test-output.npy")
        if not reuse_original_bmz_config:
            test_input = (
                self.workflow.bmz_config["test_input"] if "test_input" not in bmz_cfg else bmz_cfg["test_input"]
            )
            input_axes = [
                BatchAxis(size=1),
                ChannelAxis(channel_names=[Identifier("channel" + str(i)) for i in range(test_input.shape[1])]),
            ]
            np.save(test_input_path, test_input)
            if test_input.ndim == 4:
                input_axes += [
                    SpaceInputAxis(id=AxisId("y"), size=self.cfg.DATA.PATCH_SIZE[0]),
                    SpaceInputAxis(id=AxisId("x"), size=self.cfg.DATA.PATCH_SIZE[1]),
                ]
            else:
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
            np.save(test_output_path, test_output)
            # Classification workflow
            if len(test_output.shape) == 2:
                output_axes = [
                    BatchAxis(size=1),
                    SpaceOutputAxis(id=AxisId("z"), size=self.cfg.DATA.N_CLASSES)
                ]
            else:
                output_axes = [
                    BatchAxis(size=1),
                    ChannelAxis(channel_names=[Identifier("channel" + str(i)) for i in range(test_output.shape[1])]),
                ]
                if self.cfg.PROBLEM.NDIM == "3D":
                    if test_output.shape[test_output.ndim-3] >= 20:
                        output_axes += [
                            SpaceOutputAxisWithHalo(
                                halo=(test_output.shape[test_output.ndim-3]//8) & ~1, 
                                    id=AxisId("z"), 
                                    size=SizeReference(
                                        tensor_id='input0', # type: ignore
                                        axis_id='z', # type: ignore
                                        offset=0,
                                    ),
                                scale=float(test_input.shape[test_input.ndim-3]/test_output.shape[test_output.ndim-3]),
                            )
                        ]
                    else:
                        output_axes += [SpaceOutputAxis(id=AxisId("z"), size=test_output.shape[test_output.ndim-3])]
                output_axes += [
                    SpaceOutputAxisWithHalo(
                        halo=(test_output.shape[test_output.ndim-2]//8) & ~1, 
                        id=AxisId("y"), 
                        size=SizeReference(
                            tensor_id='input0', # type: ignore
                            axis_id='y', # type: ignore
                            offset=0,
                        ),
                        scale=float(test_input.shape[test_input.ndim-2]/test_output.shape[test_output.ndim-2]),
                    )
                ]
                output_axes += [
                    SpaceOutputAxisWithHalo(
                        halo=(test_output.shape[test_output.ndim-1]//8) & ~1,  
                        id=AxisId("x"),
                        size=SizeReference(
                            tensor_id='input0', # type: ignore
                            axis_id='x', # type: ignore
                            offset=0,
                        ),
                        scale=float(test_input.shape[test_input.ndim-1]/test_output.shape[test_output.ndim-1]),
                    ),
                ]
            data_descr = IntervalOrRatioDataDescr(type="float32")
            output_descr = OutputTensorDescr(
                id=TensorId("output0"),
                axes=output_axes,
                test_tensor=FileDescr(source=Path(test_output_path)),
                data=data_descr,
                postprocessing=postprocessing,
            )
            outputs = [output_descr]
        else:
            inputs = []
            input_shapes = []
            for i, input in enumerate(self.workflow.bmz_config["original_bmz_config"].inputs):
                if type(input) != InputTensorDescr:
                    # Read tensor
                    test_tensor_local_path = pooch.retrieve(
                        self.workflow.bmz_config["original_bmz_config"].test_inputs[i].absolute(), known_hash=None
                    )
                    test_tensor = np.load(test_tensor_local_path).squeeze()
                    if self.cfg.PROBLEM.NDIM == "2D":
                        test_tensor = ensure_2d_shape(test_tensor, test_tensor_local_path)
                    else:
                        test_tensor = ensure_3d_shape(test_tensor, test_tensor_local_path)
                    input_shapes.append(test_tensor.shape)

                    # Create axes object
                    input_axes = []
                    for letter in input.axes:
                        if letter == "b":
                            input_axes.append(BatchAxis())
                        elif letter == "c":
                            input_axes.append(
                                ChannelAxis(
                                    channel_names=[Identifier("channel" + str(i)) for i in range(test_tensor.shape[-1])]
                                ),
                            )
                        elif letter == "z":
                            input_axes.append(SpaceInputAxis(id=AxisId(str(letter)), size=test_tensor.shape[0]))
                        elif letter == "y":
                            if self.cfg.PROBLEM.NDIM == "2D":
                                input_axes.append(SpaceInputAxis(id=AxisId(str(letter)), size=test_tensor.shape[0]))
                            else:
                                input_axes.append(SpaceInputAxis(id=AxisId(str(letter)), size=test_tensor.shape[1]))
                        elif letter == "x":
                            if self.cfg.PROBLEM.NDIM == "2D":
                                input_axes.append(SpaceInputAxis(id=AxisId(str(letter)), size=test_tensor.shape[1]))
                            else:
                                input_axes.append(SpaceInputAxis(id=AxisId(str(letter)), size=test_tensor.shape[2]))

                    input_descr = InputTensorDescr(
                        id=TensorId(input.name),
                        axes=input_axes,
                        test_tensor=FileDescr(source=Path(test_tensor_local_path)),
                        data=IntervalOrRatioDataDescr(type=input.data_type),
                        preprocessing=preprocessing,  # type: ignore
                    )
                    inputs.append(input_descr)
                else:
                    inputs.append(input)

            outputs = []
            for i, output in enumerate(self.workflow.bmz_config["original_bmz_config"].outputs):
                if type(output) != OutputTensorDescr:
                    # Read tensor
                    test_tensor_local_path = pooch.retrieve(
                        self.workflow.bmz_config["original_bmz_config"].test_outputs[i].absolute(), known_hash=None
                    )
                    test_tensor = np.load(test_tensor_local_path).squeeze()
                    if self.cfg.PROBLEM.NDIM == "2D":
                        test_tensor = ensure_2d_shape(test_tensor, test_tensor_local_path)
                    else:
                        test_tensor = ensure_3d_shape(test_tensor, test_tensor_local_path)

                    input_shape = input_shapes[i] if i < len(input_shapes) else input_shapes[0]
                    # Create axes object
                    output_axes = []
                    for letter in output.axes:
                        if letter == "b":
                            output_axes.append(BatchAxis())
                        elif letter == "c":
                            output_axes.append(
                                ChannelAxis(
                                    channel_names=[Identifier("channel" + str(i)) for i in range(test_tensor.shape[-1])]
                                ),
                            )
                        elif letter == "z":
                            output_axes.append(
                                SpaceOutputAxisWithHalo(
                                    halo=(input_shape.shape[0]//8) & ~1, 
                                        id=AxisId(str(letter)), 
                                        size=SizeReference(
                                            tensor_id='input0', # type: ignore
                                            axis_id='z', # type: ignore
                                            offset=0,
                                        )
                                )
                            )
                        elif letter == "y":
                            output_axes.append(
                                SpaceOutputAxisWithHalo(
                                    halo=(input_shape.shape[1]//8) & ~1, 
                                    id=AxisId(str(letter)), 
                                    size=SizeReference(
                                        tensor_id='input0', # type: ignore
                                        axis_id='y', # type: ignore
                                        offset=0,
                                    )
                                )
                            )
                        elif letter == "x":
                            output_axes.append(
                                SpaceOutputAxisWithHalo(
                                halo=(input_shape.shape[2]//8) & ~1, 
                                id=AxisId(str(letter)), 
                                size=SizeReference(
                                        tensor_id='input0', # type: ignore
                                        axis_id='x', # type: ignore
                                        offset=0,
                                    )
                                )
                            )
                            
                    output_descr = OutputTensorDescr(
                        id=TensorId(output.name),
                        axes=output_axes,
                        test_tensor=FileDescr(source=Path(test_tensor_local_path)),
                        data=IntervalOrRatioDataDescr(type=output.data_type),
                        postprocessing=postprocessing,
                    )
                    outputs.append(output_descr)
                else:
                    outputs.append(output)

        # Name of the model
        if not reuse_original_bmz_config:
            model_name = bmz_cfg["model_name"]
        else:
            model_name = self.workflow.bmz_config["original_bmz_config"].name

        # Configure tags
        workflow = self.cfg.PROBLEM.TYPE.lower().replace("_", "-").replace("seg", "segmentation")
        if not reuse_original_bmz_config:
            tags = bmz_cfg["tags"]
            if "2d" not in tags and "3d" not in tags:
                tags += [str(self.cfg.PROBLEM.NDIM.lower())]
            if "pytorch" not in tags:
                tags += ["pytorch"]
            if "biapy" not in tags:
                tags += ["biapy"]
            if workflow not in tags:
                tags += [workflow]
            arch_used = self.cfg.MODEL.ARCHITECTURE.lower().replace("_", "-")
            if arch_used not in tags:
                tags += [arch_used]
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
            _authors = self.workflow.bmz_config["original_bmz_config"].authors
            for author in _authors:
                if type(author) != Author:
                    args = dict(name=author.name, github_user=author.github_user)
                    if author.affiliation:
                        args["affiliation"] = author.affiliation
                    if author.orcid:
                        args["orcid"] = author.orcid
                    if author.email:
                        args["email"] = author.email
                    authors.append(Author(**args))
                else:
                    authors.append(author)
            _maintainers = self.workflow.bmz_config["original_bmz_config"].maintainers
            for author in _maintainers:
                if type(author) != Maintainer:
                    args = dict(name=author.name, github_user=author.github_user)
                    if author.affiliation:
                        args["affiliation"] = author.affiliation
                    if author.orcid:
                        args["orcid"] = author.orcid
                    if author.email:
                        args["email"] = author.email
                    maintainers.append(Maintainer(**args))
                else:
                    maintainers.append(author)

        # License
        if not reuse_original_bmz_config:
            license = LicenseId(bmz_cfg["license"])
        else:
            license = self.workflow.bmz_config["original_bmz_config"].license

        # Doc
        if not reuse_original_bmz_config:
            if "doc_path" in bmz_cfg:
                doc = bmz_cfg["doc_path"]
            else:
                print("Autogenerating documentation . . .")
                doc = os.path.join(building_dir, "documentation.md")
                create_model_doc(
                    biapy_obj=self,
                    bmz_cfg=bmz_cfg,
                    cfg_file=self.cfg_file,
                    task_description=description,
                    doc_output_path=doc,
                )
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
        else:
            _citations = self.workflow.bmz_config["original_bmz_config"].cite
            for cite in _citations:
                if type(cite) != CiteEntry:
                    args = dict(text=cite.text)
                    if cite.url:
                        args["url"] = cite.url
                    if cite.doi:
                        args["doi"] = Doi(cite.doi)
                    citations.append(CiteEntry(**args))
                else:
                    citations.append(cite)

        # Add BiaPy citation if it's not there
        if not any([True for cite in citations if "biapy" in cite.text.lower()]):
            citations.append(
                CiteEntry(
                    text="BiaPy: accessible deep learning on bioimages",
                    doi=Doi("10.1038/s41592-025-02699-y"),
                    url=HttpUrl("https://www.nature.com/articles/s41592-025-02699-y"),
                )
            )

        # Cover
        covers = []
        if not reuse_original_bmz_config and "covers" in bmz_cfg:
            covers = bmz_cfg["covers"]
        elif (
            "original_bmz_config" in self.workflow.bmz_config
            and "covers" in self.workflow.bmz_config["original_bmz_config"]
        ):
            covers = self.workflow.bmz_config["original_bmz_config"].covers
        else:  # create_model_cover
            cover_path = create_model_cover(
                self.workflow.bmz_config["cover_raw"], 
                self.workflow.bmz_config["cover_gt"], 
                building_dir, 
                is_3d=self.cfg.PROBLEM.NDIM == "3D", 
                workflow=workflow,
            )
            covers.append(Path(cover_path))

        # Weights + architecture
        # If it's a BiaPy model
        env_descriptor = None
        if not reuse_original_bmz_config and "collected_sources" in self.workflow.bmz_config:
            arch_file_path = os.path.join(building_dir, "model.py")
            with open(arch_file_path, "w") as f:
                f.write("# This file was automatically generated by BiaPy ({})\n".format(biapy.__version__))
                f.write("\n")
                f.write("# The files scanned were these:\n")
                for line in self.workflow.bmz_config["scanned_files"]:
                    f.write("#     - " + line + "\n")
                f.write("\n\n")  
                for line in self.workflow.bmz_config["all_import_lines"]:
                    f.write(line + "\n")
                f.write("\n")
                for _, src in self.workflow.bmz_config["collected_sources"].items():
                    f.write(src + "\n\n")

            print("✅ model.py created with {} components.".format(len(self.workflow.bmz_config["collected_sources"])))

            arch_file_sha256 = create_file_sha256sum(arch_file_path)
            model_kwargs = self.workflow.model_build_kwargs.copy()
            model_kwargs = adapt_bmz_model_kwargs(model_kwargs, model_to_consume=False)

            pytorch_architecture = ArchitectureFromFileDescr(
                source=Path(arch_file_path),
                sha256=Sha256(arch_file_sha256),
                callable=self.workflow.bmz_config["callable_model"],
                kwargs=model_kwargs,
            )
            state_dict_source = Path(self.workflow.checkpoint_path)
            state_dict_sha256 = None

            # Isolate pytorch_state_dict from checkpoint
            checkpoint = torch.load(state_dict_source, map_location="cpu", weights_only=True)
            if "model" in checkpoint:
                state_dict_source = os.path.join(building_dir, "checkpoint.pth")
                os.makedirs(building_dir, exist_ok=True)
                torch.save(checkpoint["model"], state_dict_source)

            # Only if timm is required we create the env file
            if any([x for x in self.workflow.bmz_config["all_import_lines"] if "timm" in x]):
                env_file_path = create_environment_file_for_model(building_dir)
                env_descriptor = FileDescr_dependencies(
                    source=Path(env_file_path), sha256=Sha256(create_file_sha256sum(env_file_path))
                )
        else:
            state_dict_source, state_dict_sha256, pytorch_architecture = get_bmz_model_info(
                self.workflow.bmz_config["original_bmz_config"],
                original_model_version,
            )

        # Only exporting in pytorch_state_dict
        pytorch_state_dict = PytorchStateDictWeightsDescr(
            source=state_dict_source,  # type: ignore
            sha256=state_dict_sha256,
            architecture=pytorch_architecture,
            pytorch_version=torch.__version__,  # type: ignore
            dependencies=env_descriptor.model_dump() if env_descriptor else None,  # type: ignore
        )

        # torchscript = TorchscriptWeightsDescr(
        #     source=self.workflow.bmz_config['original_bmz_config'].weights.torchscript.source,
        #     sha256=self.workflow.bmz_config['original_bmz_config'].weights.torchscript.sha256,
        #     pytorch_version=Version(torch.__version__),
        #     parent="pytorch_state_dict", # these weights were converted from the pytorch_state_dict weights ones.
        # ),

        dataset_id = None
        if "data" in bmz_cfg and "dataset_id" in bmz_cfg["data"] and isinstance(bmz_cfg["data"]["dataset_id"], str):
            dataset_id = LinkedDataset(id=bmz_cfg["data"]["dataset_id"])

        version = "0.1.0"
        if "version" in bmz_cfg and isinstance(bmz_cfg["version"], str):
            version = bmz_cfg["version"]

        # Export model to BMZ format
        model_descr = ModelDescr(
            name=model_name,
            description=description,
            authors=authors,
            cite=citations,
            license=license,
            documentation=doc,  # type: ignore
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
            training_data=dataset_id,
            version=version,  # type: ignore
        )

        # print(f"Building BMZ package: {args}")
        print(f"Created '{model_descr.name}'")

        # Checking model consistency
        from bioimageio.core import test_model

        summary = test_model(model_descr, absolute_tolerance=1e-3, relative_tolerance=1e-3)
        summary.display()

        # Saving the model into BMZ format
        model_path = os.path.join(building_dir, model_name + ".zip")
        print(
            "Package path:",
            save_bioimageio_package(model_descr, output_path=Path(model_path)),
        )

        print("FINISHED JOB {} !!".format(self.job_identifier))

    def wait_and_stop_ddp(self):
        if is_dist_avail_and_initialized():
            dist.barrier()
            print(f"[Rank {get_rank()} ({os.getpid()})] stopping DDP . . . ")
            dist.destroy_process_group()

    def run_job(self):
        """Run a complete BiaPy workflow."""
        if self.cfg.TRAIN.ENABLE:
            self.train()

        if self.cfg.TEST.ENABLE:
            self.test()

        if self.cfg.MODEL.BMZ.EXPORT.ENABLE:
            if self.cfg.MODEL.BMZ.EXPORT.REUSE_BMZ_CONFIG:
                self.export_model_to_bmz(self.cfg.PATHS.BMZ_EXPORT_PATH, reuse_original_bmz_config=True)
            else:
                # Create a dict with all BMZ requirements
                bmz_cfg = {}
                bmz_cfg["description"] = self.cfg.MODEL.BMZ.EXPORT.DESCRIPTION
                bmz_cfg["authors"] = self.cfg.MODEL.BMZ.EXPORT.AUTHORS
                bmz_cfg["license"] = self.cfg.MODEL.BMZ.EXPORT.LICENSE
                bmz_cfg["tags"] = self.cfg.MODEL.BMZ.EXPORT.TAGS
                bmz_cfg["cite"] = self.cfg.MODEL.BMZ.EXPORT.CITE
                bmz_cfg["doc"] = self.cfg.MODEL.BMZ.EXPORT.DOCUMENTATION
                bmz_cfg["model_name"] = self.cfg.MODEL.BMZ.EXPORT.MODEL_NAME
                bmz_cfg["data"] = self.cfg.MODEL.BMZ.EXPORT.DATASET_INFO[0]
                bmz_cfg["version"] = self.cfg.MODEL.BMZ.EXPORT.MODEL_VERSION

                self.export_model_to_bmz(self.cfg.PATHS.BMZ_EXPORT_PATH, bmz_cfg=bmz_cfg)

        # # Wait until the main process is done
        # self.wait_and_stop_ddp()

        print("FINISHED JOB {} !!".format(self.job_identifier))

        if self._stdout_log_file is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self._stdout_log_file.flush()
            self._stdout_log_file.close()
            self._stdout_log_file = None
        if self._null_stream is not None:
            self._null_stream.close()
            self._null_stream = None


# Workflows accepted by ``build_config`` (must match the values validated in
# ``biapy.engine.check_configuration``).
VALID_WORKFLOWS = [
    "SEMANTIC_SEG",
    "INSTANCE_SEG",
    "CLASSIFICATION",
    "DETECTION",
    "DENOISING",
    "SUPER_RESOLUTION",
    "SELF_SUPERVISED",
    "IMAGE_TO_IMAGE",
]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` in-place (nested dicts merged) and return ``base``."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _filter_cfg_to_schema(cfg_dict: dict, schema: CN) -> dict:
    """
    Keep only the keys of ``cfg_dict`` that exist in the ``schema`` CfgNode (recursively).

    Used to drop runtime-derived keys (e.g. ``PATHS.TEST_FULL_GT_H5``) that a saved config
    embeds but that are not part of the default configuration schema, so it can be merged
    back cleanly.
    """
    out = {}
    for k, v in cfg_dict.items():
        if k not in schema:
            continue
        sv = schema[k]
        if isinstance(v, dict) and isinstance(sv, CN):
            out[k] = _filter_cfg_to_schema(v, sv)
        else:
            out[k] = v
    return out


def build_config(
    workflow: str,
    dims: str,
    phase: str = "both",
    patch_size: Optional[tuple] = None,
    model: Optional[dict] = None,
    train_data: Optional[dict] = None,
    val_data: Optional[dict] = None,
    test_data: Optional[dict] = None,
    extra_config: Optional[dict] = None,
) -> dict:
    """
    Build a BiaPy config-overrides dict from high-level arguments, ready to pass to :class:`BiaPy`.

    The returned dict only contains the keys derived from the arguments and is merged on top
    of the BiaPy defaults by :class:`BiaPy`. Only the minimal inputs are validated here; the
    full validation still happens at construction time via ``check_configuration``.

    Parameters
    ----------
    workflow : str
        Workflow/problem type (maps to ``PROBLEM.TYPE``). One of ``SEMANTIC_SEG``,
        ``INSTANCE_SEG``, ``CLASSIFICATION``, ``DETECTION``, ``DENOISING``,
        ``SUPER_RESOLUTION``, ``SELF_SUPERVISED``, ``IMAGE_TO_IMAGE`` (case-insensitive).

    dims : str
        Data dimensionality, ``2D`` or ``3D`` (maps to ``PROBLEM.NDIM``).

    phase : str, optional
        Phases to enable: ``train``, ``test`` or ``both`` (default). Maps to
        ``TRAIN.ENABLE`` / ``TEST.ENABLE``.

    patch_size : tuple, optional
        Patch size, e.g. ``(256, 256, 1)`` in 2D. Maps to ``DATA.PATCH_SIZE``.

    model, train_data, val_data, test_data : dict, optional
        Settings whose keys are upper-cased onto ``MODEL``, ``DATA.TRAIN``, ``DATA.VAL`` and
        ``DATA.TEST`` respectively, e.g. ``model={"architecture": "unet"}``.

    extra_config : dict, optional
        Escape hatch of raw (upper-case) config keys, deep-merged last, able to override
        anything set above (e.g. ``{"TRAIN": {"EPOCHS": 100}}``).

    Returns
    -------
    dict
        Configuration overrides ready to be passed to :class:`BiaPy`.
    """
    workflow = str(workflow).upper()
    if workflow not in VALID_WORKFLOWS:
        raise ValueError("'workflow' must be one of {}. Provided: {}".format(VALID_WORKFLOWS, workflow))

    dims = str(dims).upper()
    if dims not in ["2D", "3D"]:
        raise ValueError("'dims' must be either '2D' or '3D'. Provided: {}".format(dims))

    phase = str(phase).lower()
    if phase not in ["train", "test", "both"]:
        raise ValueError("'phase' must be one of ['train', 'test', 'both']. Provided: {}".format(phase))

    def _upper_keys(d: dict) -> dict:
        return {str(k).upper(): v for k, v in d.items()}

    cfg: dict = {
        "PROBLEM": {"TYPE": workflow, "NDIM": dims},
        "TRAIN": {"ENABLE": phase in ("train", "both")},
        "TEST": {"ENABLE": phase in ("test", "both")},
    }

    if patch_size is not None:
        cfg.setdefault("DATA", {})["PATCH_SIZE"] = tuple(patch_size)

    if model:
        cfg["MODEL"] = _upper_keys(model)

    if train_data or val_data or test_data:
        data_node = cfg.setdefault("DATA", {})
        if train_data:
            data_node["TRAIN"] = _upper_keys(train_data)
        if val_data:
            data_node["VAL"] = _upper_keys(val_data)
        if test_data:
            data_node["TEST"] = _upper_keys(test_data)

    if extra_config:
        _deep_merge(cfg, copy.deepcopy(extra_config))

    return cfg
