"""Grid-search agglomeration/watershed post-processing thresholds on validation data.

Equivalent of pytorch_connectomics' `--mode tune`: runs a trained checkpoint's raw predictions
once against DATA.TEST.USE_VAL_AS_TEST (works for DATA.VAL.FROM_TRAIN too, not just CROSS_VAL --
see check_configuration.py), caches them via TEST.SAVE_MODEL_RAW_OUTPUT, then re-decodes+scores
them for every point in a threshold grid via TEST.REUSE_PREDICTIONS (no re-inference per point).

Example (membrane repair, agglomeration):
    python scripts/tune_postprocess_thresholds.py \
        --config /cephfs/dfranco/jobs/vast_repair27.yaml \
        --checkpoint /cephfs/dfranco/exp_results/vast_repair27/checkpoints/vast_repair27_1-checkpoint-best.pth \
        --result_dir /cephfs/dfranco/exp_results --name vast_repair27_tune --gpu 0 \
        --grid '{"FRAGMENT_SEED_TH": [0.8, 0.9, 0.95], "FRAGMENT_GROWTH_TH": [0.05, 0.1, 0.2], "MERGE_TH": [0.3, 0.5, 0.7]}'

INSTANCE_SEG uses --method agglomeration|watershed (PROBLEM.INSTANCE_SEG.AGGLOMERATION /
.WATERSHED). Membrane repair also supports both --method values: "agglomeration" grid keys are
FRAGMENT_SEED_TH/FRAGMENT_GROWTH_TH/MERGE_TH/MERGE_QUANTILE; "watershed" grid keys are
WATERSHED_SEED_TH/WATERSHED_GROWTH_TH.
"""
import argparse
import copy
import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile

import yaml

BIAPY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POSTPROCESS_BLOCK = {
    ("IMAGE_TO_IMAGE", "agglomeration"): "PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR.POSTPROCESS",
    ("IMAGE_TO_IMAGE", "watershed"): "PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR.POSTPROCESS",
    ("INSTANCE_SEG", "agglomeration"): "PROBLEM.INSTANCE_SEG.AGGLOMERATION",
    ("INSTANCE_SEG", "watershed"): "PROBLEM.INSTANCE_SEG.WATERSHED",
}


def get_by_path(cfg: dict, dotted_key: str, default=None):
    node = cfg
    for k in dotted_key.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def set_by_path(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def detect_block(cfg: dict, method: str) -> str:
    problem_type = get_by_path(cfg, "PROBLEM.TYPE")
    key = (problem_type, method)
    if key not in POSTPROCESS_BLOCK:
        raise ValueError(f"Unsupported (PROBLEM.TYPE, method) combination for threshold tuning: {key}")
    return POSTPROCESS_BLOCK[key]


def build_grid(grid: dict):
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def build_run_config(base_cfg, block, combo, checkpoint, reuse_predictions, method):
    cfg = copy.deepcopy(base_cfg)
    set_by_path(cfg, "TRAIN.ENABLE", False)
    set_by_path(cfg, "TEST.ENABLE", True)
    set_by_path(cfg, "MODEL.LOAD_CHECKPOINT", True)
    set_by_path(cfg, "PATHS.CHECKPOINT_FILE", checkpoint)
    set_by_path(cfg, "DATA.TEST.USE_VAL_AS_TEST", True)
    set_by_path(cfg, "TEST.REUSE_PREDICTIONS", reuse_predictions)
    set_by_path(cfg, "TEST.SAVE_MODEL_RAW_OUTPUT", True)
    if get_by_path(cfg, "PROBLEM.TYPE") == "IMAGE_TO_IMAGE":
        set_by_path(cfg, "PROBLEM.IMAGE_TO_IMAGE.MEMBRANE_REPAIR.POSTPROCESS.METHOD", method)
    for key, value in combo.items():
        set_by_path(cfg, f"{block}.{key}", value)
    return cfg


def run_one(cfg, result_dir, name, run_id, gpu, dry_run):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(cfg, fh)
        tmp_path = fh.name

    cmd = [sys.executable, "-u", "main.py", "--config", tmp_path, "--result_dir", result_dir,
           "--name", name, "--run_id", str(run_id)]
    if gpu:
        cmd += ["--gpu", gpu]

    csv_path = os.path.join(result_dir, name, "results", f"{name}_{run_id}", "test_results_metrics.csv")
    if dry_run:
        print("  would run:", " ".join(cmd))
        os.unlink(tmp_path)
        return csv_path

    try:
        proc = subprocess.run(cmd, cwd=BIAPY_ROOT, capture_output=True, text=True)
    finally:
        os.unlink(tmp_path)
    if proc.returncode != 0:
        raise RuntimeError(f"main.py failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    return csv_path


def read_metric(csv_path, metric):
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, 0
    vals = [float(r[metric]) for r in rows if r.get(metric) not in (None, "")]
    return (sum(vals) / len(vals) if vals else None), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--name", required=True, help="Tuning job name; gets its own result_dir subtree")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--method", choices=["agglomeration", "watershed"], default="agglomeration")
    parser.add_argument("--grid", required=True, help="JSON dict of threshold_key -> list of values")
    parser.add_argument("--metric", default="0.5 TH (post)f1")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        base_cfg = yaml.safe_load(fh)

    block = detect_block(base_cfg, args.method)
    grid = json.loads(args.grid)
    combos = list(build_grid(grid))
    print(f"{len(combos)} combination(s), block={block}")

    out_path = args.out or os.path.join(args.result_dir, args.name, "threshold_grid_results.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results = []
    for i, combo in enumerate(combos):
        reuse = i > 0
        print(f"[{i + 1}/{len(combos)}] {combo} (reuse_predictions={reuse})")
        cfg = build_run_config(base_cfg, block, combo, args.checkpoint, reuse, args.method)
        csv_path = run_one(cfg, args.result_dir, args.name, args.run_id, args.gpu, args.dry_run)
        if args.dry_run:
            continue
        metric_val, n_files = read_metric(csv_path, args.metric)
        print(f"    -> {args.metric} = {metric_val} (n={n_files})")
        results.append({**combo, args.metric: metric_val, "n_files": n_files})

    if args.dry_run or not results:
        return

    results.sort(key=lambda r: (r[args.metric] is None, -(r[args.metric] or 0)))
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nTop combos by {args.metric}:")
    for r in results[:10]:
        print(r)
    print(f"\nFull grid: {out_path}")


if __name__ == "__main__":
    main()
