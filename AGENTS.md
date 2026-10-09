# BiaPy agent guide

## What BiaPy is

BiaPy is a PyTorch library + CLI for deep-learning bioimage analysis. A run is fully described by a
YAML config (yacs). `PROBLEM.TYPE` selects the workflow, `PROBLEM.NDIM` selects 2D/3D.

## Documentation

User documentation: https://biapy.readthedocs.io. For agents, start from its index https://biapy.readthedocs.io/en/latest/llms.txt (one link per page, as plain text); https://biapy.readthedocs.io/en/latest/llms-full.txt has the whole documentation in one file.

## Using BiaPy

Pick the workflow from the task, prepare the data in its format and start from its template:

| Task | `PROBLEM.TYPE` | Training data (input / ground truth) | 2D template |
|------|----------------|--------------------------------------|-------------|
| Assign a class to every pixel (e.g. mitochondria vs. background) | `SEMANTIC_SEG` | Image / mask of the same size with one class value per pixel | `templates/semantic_segmentation/2d_semantic_segmentation.yaml` |
| Separate every individual object, even when they touch (e.g. nuclei, cells) | `INSTANCE_SEG` | Image / mask with **a different id per object** (0 = background) | `templates/instance_segmentation/2d_instance_segmentation.yaml` |
| Locate objects by their center, without their shape | `DETECTION` | Image / **one CSV per image, with the same file name**, columns `axis-0`, `axis-1` (plus `class` if `DATA.N_CLASSES > 2`) | `templates/detection/2d_detection.yaml` |
| Remove noise | `DENOISING` | Noisy image / **no ground truth** (Noise2Void) | `templates/denoising/2d_denoising.yaml` |
| Increase the resolution (×2, ×4) | `SUPER_RESOLUTION` | Low-resolution image / the same image `UPSCALING` times larger | `templates/super-resolution/2d_super-resolution.yaml` |
| Pretrain a model without labels, to fine-tune it later on another task | `SELF_SUPERVISED` | Image / **no ground truth** | `templates/self-supervised/2d_self-supervised.yaml` |
| Assign one class to each whole image | `CLASSIFICATION` | **One sub-folder per class** with its images / the folder is the label | `templates/classification/2d_classification.yaml` |
| Map an image to another image (e.g. inpainting, colorization, stain transfer) | `IMAGE_TO_IMAGE` | Image / target image of the same size | `templates/image-to-image/2d_image-to-image.yaml` |

Every workflow also has a 3D template with the same name, replacing `2d_` with `3d_`.

## Repository layout

| Path | What it contains |
|------|------------------|
| `biapy/engine/` | The workflows: one file per `PROBLEM.TYPE`, named after it in lowercase (`SEMANTIC_SEG` → `semantic_seg.py`; it is imported dynamically by `BiaPy._build_workflow`, so the name must match). Exception: `membrane_repair.py`, used by `IMAGE_TO_IMAGE` when `PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR.ENABLE` is set. Also the shared `base_workflow.py`, the train loop, losses, metrics and `check_configuration.py` (config validation) |
| `biapy/config/` | `config.py`: every config option with its default value |
| `biapy/data/` | Data loading, normalization, pre/post-processing and data generators |
| `biapy/models/` | One file per architecture; `__init__.py` has `build_model()` and BioImage Model Zoo loading |
| `biapy/utils/` | Helpers, callbacks and standalone scripts |
| `templates/` | Ready-to-use YAML configs, one folder per workflow (plus `sota_implementations/`, with notebooks). Start here when writing a config |
| `tests/` | One fast CPU test and several heavy GPU checks (see "Tests" below) |

The Python API (`BiaPy` class, `build_config()`) is in `biapy/_biapy.py` and the CLI (`biapy` command) in
`biapy/__init__.py`.

## Install (CPU, for development)

Python >= 3.11.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
python -c "from biapy import BiaPy, build_config; print('ok')"
```

Install the CPU build of torch first: otherwise `pip install -e .` pulls the CUDA build (several GB).

Device selection: CUDA is used only if `--gpu` (CLI) / `gpu=` (API) is given and CUDA is available;
otherwise BiaPy uses `SYSTEM.DEVICE` (default `"cpu"`, set it to `"mps"` for Apple Silicon).
No flag is needed to run on CPU.

## Running

Paths starting with `/path/to/` are placeholders. The YAML files in `templates/` use them for the data
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
                   test_data={"path": "/path/to/test/raw", "gt_path": "/path/to/test/label"},
                   extra_config={"TRAIN": {"EPOCHS": 1}})
biapy = BiaPy(cfg, result_dir="/path/to/output", name="my_job")  # validates the config
biapy.train()
biapy.test()
```

`build_config` enables both training and test by default (`phase="both"`). Pass `test_data`, or
`phase="train"` to skip the test: the test data path is not checked when `BiaPy(...)` is built, so a
missing one only fails when `biapy.test()` runs.

## Tests

Fast, CPU-only, no data needed (synthetic, ~10 s). `pytest` is not a BiaPy dependency, so install it
first (`pip install pytest`):

```bash
pytest tests/test_tta_equivariance.py -q
```

Do NOT run these unless explicitly asked; they need a GPU:

- `tests/run_checks.py` — end-to-end training/inference for every workflow. Downloads datasets from
  Google Drive (`gdown`) and takes hours.
- `tests/check_api.py` — Python API (`build_config` / `BiaPy` / `predict`) consistency. Downloads a
  dataset from Google Drive (`gdown`) and trains a model.
- `tests/export_bmz_test.py` — runs the job of a given config and exports the model to BioImage Model
  Zoo format. It is called by `run_checks.py`, not run on its own.

`run_checks.py` and `check_api.py` run weekly on a self-hosted GPU runner
(`.github/workflows/check_code_consistency.yml`).

Cheap checks an agent can always do:

```bash
python -m compileall -q biapy
python -c "import biapy, biapy.engine.check_configuration, biapy.models"
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
5. Option users normally set: add it to the relevant YAML files in `templates/`.

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

Only when it applies:

5. Model split across several files: list the extra files in `extra_model_files` in `build_model()`,
   or BioImage Model Zoo export will miss them.
6. Add a template in `templates/<workflow>/` and a test case in `tests/run_checks.py`.

### Style
- NumPy-style docstrings (`Parameters` / `Returns` sections), as in the rest of the code.
- Comments and docstrings in English.
- Keep backward compatibility: existing YAML configs in `templates/` and old checkpoints must keep loading.
- Config errors: follow `check_configuration()`, i.e. `assert` or `raise ValueError` with a message
  that names the config key and says what is allowed.
