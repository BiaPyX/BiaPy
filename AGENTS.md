# BiaPy agent guide

## What BiaPy is

BiaPy is a PyTorch library + CLI for deep-learning bioimage analysis. A run is fully described by a
YAML config (yacs). `PROBLEM.TYPE` selects the workflow, `PROBLEM.NDIM` selects 2D/3D.

## Documentation

User documentation: https://biapy.readthedocs.io. For agents, start from its index https://biapy.readthedocs.io/en/latest/llms.txt (one link per page, as plain text); https://biapy.readthedocs.io/en/latest/llms-full.txt has the whole documentation in one file.

## Workflows

`PROBLEM.TYPE` is one of `SEMANTIC_SEG`, `INSTANCE_SEG`, `DETECTION`, `DENOISING`, `SUPER_RESOLUTION`,
`SELF_SUPERVISED`, `CLASSIFICATION`, `IMAGE_TO_IMAGE`. Each one has a 2D and a 3D template in
`biapy/config/templates/<workflow>/` (`2d_*.yaml` / `3d_*.yaml`); their data formats are described in the documentation.

Ground truth expected by each workflow (`DATA.TRAIN.GT_PATH`, and `DATA.TEST.GT_PATH` when `DATA.TEST.LOAD_GT` is set):

- `SEMANTIC_SEG`: mask of the same size, one class value per pixel.
- `INSTANCE_SEG`: mask with a different id per object (0 = background); `PROBLEM.INSTANCE_SEG.TYPE = "synapses"`
  uses another format.
- `DETECTION`: one CSV per image, matched by file name, with columns `axis-0`, `axis-1` (`axis-2` in 3D),
  i.e. (y, x) / (z, y, x), plus `class` when `DATA.N_CLASSES > 2`.
- `CLASSIFICATION`: no masks; one sub-folder per class with its images.
- `SUPER_RESOLUTION`: the same image `PROBLEM.SUPER_RESOLUTION.UPSCALING` times larger.
- `IMAGE_TO_IMAGE`: target image of the same size.
- `DENOISING`: none by default (Noise2Void); a clean image if `PROBLEM.DENOISING.LOAD_GT_DATA` is enabled
  or `LOSS.TYPE` has an adversarial term (`BCE`/`HINGE`).
- `SELF_SUPERVISED`: none.

Check a config without running it: `biapy validate-config my_config.yaml [--no-data-check]`
(exit code 0/1), or `biapy.validate_config(cfg, check_data_paths=False)` from Python.

## Repository layout

| Path | What it contains |
|------|------------------|
| `biapy/engine/` | The workflows: one file per `PROBLEM.TYPE`, named after it in lowercase (`SEMANTIC_SEG` → `semantic_seg.py`; it is imported dynamically by `BiaPy._build_workflow`, so the name must match). Exception: `membrane_repair.py`, used by `INSTANCE_SEG` when `PROBLEM.INSTANCE_SEG.MEMBRANE_REPAIR.ENABLE` is set. Also the shared `base_workflow.py`, the train loop, losses, metrics and `check_configuration.py` (config validation) |
| `biapy/config/` | `config.py`: every config option with its default value |
| `biapy/data/` | Data loading, normalization, pre/post-processing and data generators |
| `biapy/models/` | One file per architecture; `__init__.py` has `build_model()` and BioImage Model Zoo loading |
| `biapy/wizard/` | Logic of the configuration wizard shared by the front-ends (BiaPy-GUI, Fiji plugin): data checks, nnU-Net-like planning and config building. Must import without loading PyTorch |
| `biapy/citations/` | `references.yaml`: works to cite and the config rules that decide when each applies (written as `CITATIONS.md`/`citations.bib` at the end of `run_job()` and used in BMZ exports) |
| `biapy/utils/` | Helpers, callbacks and standalone scripts |
| `biapy/config/templates` | Ready-to-use YAML configs, one folder per workflow (plus `sota_implementations/`, with notebooks). Start here when writing a config |
| `tests/` | Fast CPU tests and several heavy GPU checks (see "Tests" below) |

The Python API (`BiaPy` class with `train()`, `test()`, `predict()`, `load_workflow_from_model()`,
`export_model_to_bmz()`; `build_config()`, `validate_config()`) is in `biapy/_biapy.py`, and the CLI
(`biapy` command and its subcommands: `validate-config`, `check-data`, `list-workflows`, `inspect-image`,
`get-config-template`, `create-config`, `search-bmz-models`) in `biapy/__init__.py`. `biapy/__init__.py` imports the engine
lazily: `import biapy` must not load PyTorch, so don't add top-level imports of torch or `biapy._biapy` there.

## Install

Python >= 3.11. Install PyTorch first, then BiaPy in editable mode:

```bash
# No GPU (e.g. a sandbox): CPU build, enough for the unit tests and the cheap checks
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# With an NVIDIA GPU: the CUDA build instead, see https://pytorch.org/get-started/locally/

pip install -e .
python -c "from biapy import BiaPy, build_config; print('ok')"
```

Without the first step, `pip install -e .` pulls the default (CUDA) torch build, several GB.
If torch is already installed (e.g. an existing BiaPy environment), skip it.

Device selection: `--gpu 0[,1]` (CLI) / `gpu=` (API) uses CUDA if available, `--gpu mps` the Apple Silicon GPU;
otherwise BiaPy uses `SYSTEM.DEVICE` (default `"cpu"`). No flag is needed to run on CPU.

## Running

Paths starting with `/path/to/` are placeholders. The YAML files in `biapy/config/templates` use them for the data
paths (`DATA.TRAIN.PATH`, `DATA.TRAIN.GT_PATH`, ...), so copy a template and replace them before running it.

```bash
# CLI
biapy --config /path/to/my_config.yaml --result_dir /path/to/output --name my_job --run_id 1
```

```python
# Python API
from biapy import BiaPy, build_config
cfg = build_config("SEMANTIC_SEG", "2D", patch_size=(256, 256, 1),
                   train_data={"path": "/path/to/train/raw", "gt_path": "/path/to/train/label"},
                   test_data={"path": "/path/to/test/raw", "gt_path": "/path/to/test/label", "load_gt": True},
                   extra_config={"TRAIN": {"EPOCHS": 1}})
biapy = BiaPy(cfg, result_dir="/path/to/output", name="my_job")  # validates the config
biapy.train()
biapy.test()
```

`build_config` enables train and test by default (`phase="both"`); the test data path is only checked when
`biapy.test()` runs. Without `load_gt` the test ground truth is ignored and no metrics are computed.
Don't reuse the `--result_dir`/`--name`/`--run_id` of an existing job unless asked: its outputs are written over.

## Tests

CPU-only unit tests, synthetic data, no GPU or network (`pip install pytest` first, it is not a BiaPy dependency):

```bash
pytest -q                                  # all of them (tests/test_*.py, ~40 s); CI runs exactly this
pytest tests/test_wizard.py -q             # wizard: data checks, planning, configs built (~15 s)
pytest tests/test_wizard_api.py -q         # wizard interface with the front-ends (see Rules)
pytest tests/test_cli.py -q                # biapy subcommands: --json output, text output and exit codes (~7 s)
pytest tests/test_vit_backbones.py -q      # every ViT backbone/architecture combination, built on 'meta' device
pytest tests/test_patch_sampling.py -q     # foreground oversampling (biapy/data/patch_sampling.py)
pytest tests/test_tta_equivariance.py -q   # instance representations/channels and TTA (data/post_processing/tta.py)
```

Do NOT run these unless explicitly asked; they need a GPU:

- `tests/run_checks.py`: end-to-end training/inference of every workflow, BMZ export and reproducibility
  (`export_bmz_test.py`, `bmz_reproducibility.py`). Downloads datasets from Google Drive (`gdown`).
- `tests/check_api.py`: Python API consistency (`build_config` / `BiaPy` / `predict`). Downloads a dataset and trains.
- `tests/calibrate_unet_memory.py`: refits the VRAM model of `biapy/wizard/unet.py`.

The CI (`.github/workflows/check_code_consistency.yml`, weekly on a self-hosted GPU runner) runs `pytest -q`,
then `run_checks.py` and `check_api.py`. Likewise, don't start trainings or large downloads outside the tests
unless asked.

Cheap checks an agent can always do:

```bash
python -m compileall -q biapy
python -c "import biapy, biapy.engine.check_configuration, biapy.models, biapy.wizard"
```

## Rules when changing code

### Adding or changing a config option
yacs only rejects unknown keys and values whose type differs from the default (e.g. a string where
an int is expected). It does not check values: negative epochs, a list of strings where ints are
expected, a value not in the allowed set or an option used in the wrong workflow are all accepted.
Those checks are done by hand in `check_configuration.py`, and nothing fails if you forget them:
a bad value is accepted and can crash later (e.g. mid-training) with an unclear error.
To see the pattern in existing code, `grep -n EVAL_BORDER_CROP` in these files:
`config.py` (default + explanatory comment), `check_configuration()` (the asserts at the top of the
function) and `convert_old_model_cfg_to_current_version()` (its rename from `DET_IGNORE_POINTS_OUTSIDE_BOX`).

Always:

1. `biapy/config/config.py`: add the key with its default value and a comment above it explaining
   what it does, its allowed values and which workflows use it.
2. `biapy/engine/check_configuration.py`: validate it in `check_configuration()` (type, allowed
   values, 2D/3D length, workflows where it is not applicable), with an error message that names the key.
3. Documentation: it lives in a separate repo, [BiaPyX/BiaPy-doc](https://github.com/BiaPyX/BiaPy-doc),
   and is not generated from `config.py`. Update the workflow page (`source/workflows/<workflow>.rst`).

Only when it applies:

4. Renamed, removed or changed type: update `convert_old_model_cfg_to_current_version()` in
   `check_configuration.py` so old YAML configs and checkpoints keep loading.
5. Option users normally set: add it to the relevant YAML files in `biapy/config/templates`.
6. Option the wizard sets (`biapy/wizard/config_builder.py`, the planners, `instance_seg.REPRESENTATIONS`): run `pytest tests/test_wizard.py -q`, which validates the configs it builds with `check_configuration()`.

### Adding a model architecture
To see the pattern in existing code, `grep -rn -i nafnet biapy/` (or `wavelettention`, `rdbm`).

Always:

1. `biapy/models/<arch>.py`: the model class.
2. `biapy/models/__init__.py`: add an `elif modelname == "<arch>":` branch in `build_model()` that
   builds the model from its `cfg.MODEL.*` options.
3. `biapy/config/config.py`:
   - add the name to the `MODEL.ARCHITECTURE` comment (it lists the architectures available per workflow);
   - add its hyperparameters as a block (e.g. `_C.MODEL.NAFNET.*`). These are config options, so the
     rules of the previous section apply to them too.
4. `biapy/engine/check_configuration.py`: architecture names appear in many separate lists, each one
   meaning "architectures that support X". To find them all, pick the most similar existing
   architecture and run `grep -n '"<similar_arch>"' biapy/engine/check_configuration.py`. For each
   hit, add the new name if it supports that feature, and update the error message next to it if it
   repeats the list as text. The main ones:
   - the global list of valid `MODEL.ARCHITECTURE` values;
   - one list per workflow (which architectures each `PROBLEM.TYPE` accepts);
   - architectures that support 3D (`"For 3D these models are available"`);
   - architectures that support more than 2 classes (`DATA.N_CLASSES > 2`);
   - U-Net-like options and checks: `MODEL.YX_DOWN` / `MODEL.Z_DOWN`, `MODEL.CONV_LAYERS` and
     the patch size divisibility check.

   Also add checks for its own constraints (e.g. NAFNet always needs a discriminator and an adversarial loss).
5. `biapy/citations/references.yaml`: add the reference (with `code:` if the implementation is adapted from a
   repo) and a rule `when: {MODEL.SOURCE: biapy, MODEL.ARCHITECTURE: <arch>}` (see the `nafnet` one).
   Same for new losses, post-processing methods or representations that come from a paper.

Only when it applies:

6. Model split across several files: list the extra files in `extra_model_files` in `build_model()`,
   or BioImage Model Zoo export will miss them.
7. Add a template in `biapy/config/templates/<workflow>/` and a test case in `tests/run_checks.py`.

### Changing the wizard (`biapy/wizard/`)
The front-ends (BiaPy-GUI, the Fiji plugin) call what `biapy.wizard.INTERFACE` lists. `tests/test_wizard_api.py`
compares its signatures and result keys with the snapshot `tests/wizard_api.json`:

- Additions (optional argument, new result key, new function): update the snapshot with
  `python tests/test_wizard_api.py --update`.
- Anything that breaks a front-end (removed/renamed item, argument made required, removed key, or a change in the
  meaning of a value the snapshot can't see): also increase `API_VERSION` in `biapy/wizard/__init__.py`, and say
  that BiaPy-GUI and the Fiji plugin must be updated.

Wizard modules must stay importable without PyTorch: import BiaPy's readers inside the functions that use them.

### Style
- NumPy-style docstrings (`Parameters` / `Returns` sections), as in the rest of the code.
- Comments and docstrings in English.
- Keep backward compatibility: existing YAML configs in `biapy/config/templates` and old checkpoints must keep loading.
- Config errors: follow `check_configuration()`, i.e. `assert` or `raise ValueError` with a message
  that names the config key and says what is allowed.
