# Sanity-check experiments — *Representation matters* paper

This folder contains everything needed to reproduce the **sanity-check experiments** of the paper,
i.e. the runs used to verify that each instance-segmentation representation implemented in BiaPy
behaves like its original implementation. All of them are trained and evaluated on the **DSB2018**
dataset (2D nuclei).

| Sanity check | Configuration file     | Representation (`PROBLEM.INSTANCE_SEG.DATA_CHANNELS`)      |
| ------------ | ---------------------- | ---------------------------------------------------------- |
| `cellpose`   | `sanity_cellpose.yaml` | `['F', 'Gh', 'Gv']` — foreground + CellPose flows           |
| `omnipose`   | `sanity_omnipose.yaml` | `['Db', 'Gh', 'Gv']` — Omnipose distance + flows            |
| `stardist`   | `sanity_stardist.yaml` | `['Db', 'R']` — object probability + star-convex distances  |
| `embedseg`   | `sanity_embedseg.yaml` | `['E_offset', 'E_sigma', 'E_seediness']` — EmbedSeg         |
| `hovernet`   | `sanity_hovernet.yaml` | `['F', 'H', 'V']` — HoVer-Net horizontal/vertical maps      |

`run_sanity_experiments.py` launches any of them (or all of them in a loop) with BiaPy.

---

## 1. Requirements

* A clone of the [BiaPy repository](https://github.com/BiaPyX/BiaPy) — this script lives inside it
  and finds `main.py` by itself.
* The BiaPy environment installed and **activated**. Follow the
  [installation instructions](https://biapyx.github.io).
* A GPU is strongly recommended (the script accepts `--gpu ""` to fall back to CPU, but training
  on CPU is not practical).

## 2. Download the data and the trained models

The DSB2018 dataset and the checkpoints of every sanity check are published on Zenodo:

> **Zenodo record:** [10.5281/zenodo.21649394](https://doi.org/10.5281/zenodo.21649394)

Download and unpack them anywhere in your machine. Two things are expected:

**a) The dataset**, following the layout BiaPy uses (447 training and 50 test images, `.tif`):

```
dsb2018/
├── train/
│   ├── images/     # raw images
│   └── masks/      # instance labels
└── test/
    ├── images/
    └── masks/
```

**b) The checkpoints**, all of them together in a single directory:

```
checkpoints/
├── sanity_cellpose-checkpoint-best.pth
├── sanity_embedseg-checkpoint-best.pth
├── sanity_hovernet-checkpoint-best.pth
├── sanity_omnipose-checkpoint-best.pth
└── sanity_stardist-checkpoint-best.pth
```

The only requirement for the filenames is that each of them contains the job name of its experiment
(`sanity_cellpose`, `sanity_omnipose`, ...), which is how the script matches a checkpoint to the
sanity check being run. Subdirectories are fine, they are searched recursively.

## 3. Run the sanity checks

The configuration files distributed here point to the paths of the machine where the paper's
experiments were run, so use `--data_dir` to point to *your* copy of DSB2018. The originals are
never modified: every run writes its own configuration copy inside the output directory.

### Only inference (default)

Reproduces the reported numbers using the trained models downloaded from Zenodo:

```bash
conda activate BiaPy_env

python run_sanity_experiments.py \
    --experiment cellpose \
    --output_dir /path/to/output \
    --checkpoint_dir /path/to/checkpoints \
    --data_dir /path/to/dsb2018
```

The experiment name is case-insensitive (`cellpose`, `CellPose`, `sanity_cellpose`, ... all work).
Use `all` to run every sanity check, one after another:

```bash
python run_sanity_experiments.py -e all -o /path/to/output -c /path/to/checkpoints -d /path/to/dsb2018
```

### Training from scratch and then testing

```bash
python run_sanity_experiments.py \
    --experiment all \
    --mode train+test \
    --output_dir /path/to/output \
    --data_dir /path/to/dsb2018
```

Here `--checkpoint_dir` is not needed: the models are trained with the settings of each YAML file
and the resulting checkpoints are written by BiaPy to
`<output_dir>/sanity_<experiment>/checkpoints`. That directory can later be passed to
`--checkpoint_dir` to re-run only the test phase (it is searched recursively, so
`--checkpoint_dir <output_dir>` also works).

> Note that training results may differ slightly from the published ones due to the
> non-determinism of GPU training.

## 4. Output

Everything is written inside `--output_dir`:

```
output/
├── config_files/                  # configuration actually used on each run
│   └── sanity_cellpose_test.yaml
├── logs/                          # full console output of each run
│   └── sanity_cellpose_test.log
└── sanity_cellpose/               # BiaPy job directory (results, metrics, checkpoints, ...)
```

A summary with the state of every experiment is printed at the end, and the script exits with a
non-zero code if any of them failed.

The predicted instances of each experiment are stored inside its BiaPy job directory:

```
output/sanity_cellpose/results/sanity_cellpose_1/
├── per_image_instances/           # instances as created from the predicted representation
└── per_image_post_processing/     # instances after the post-processing set in the YAML file
```

Since all the configuration files here enable `TEST.POST_PROCESSING.INSTANCE_REFINEMENT`, the
results reported in the paper are the ones in **`per_image_post_processing`**.

## 5. Evaluate the predictions

BiaPy already prints the matching metrics at the end of every test run, but they can also be
computed separately from the predicted labels with the script `calculate_instance_metrics.py`,
shipped within BiaPy. This is how the numbers of the paper were obtained, as it allows evaluating
the outputs of BiaPy and those of the original implementations exactly the same way:

```bash
python /path/to/BiaPy/biapy/utils/scripts/calculate_instance_metrics.py \
    /path/to/output/sanity_cellpose/results/sanity_cellpose_1/per_image_post_processing/ \
    /path/to/dsb2018/test/masks \
    --thresholds 0.5 0.55 0.6 0.65 0.70 0.75 0.8 0.85 0.9
```

The two positional arguments are the directory with the **predicted** instance labels and the
directory with the **ground truth** ones. Files are paired by base name, so the extensions of the
predictions and the ground truth may differ. Two optional arguments are available:
`--criterion` (`iou`, the default, `iot` or `iop`) and `--verbose` to also print the statistics of
every single image.

The script reports precision, recall, F1, panoptic quality and the rest of the matching statistics
for each IoU threshold given in `--thresholds`. The range above (0.5 to 0.9 in steps of 0.05) is
the one used in the paper: averaging over those thresholds the `accuracy` value it prints
(`TP / (TP + FP + FN)`, what DSB2018 calls *average precision*) gives the mAP usually reported for
this dataset.