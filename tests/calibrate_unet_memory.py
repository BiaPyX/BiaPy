"""
Measures the GPU memory of BiaPy's U-Net training steps for plans made by :mod:`biapy.wizard.unet` and fits the
memory model of that module (VRAM_BYTES_PER_ACTIVATION, VRAM_FIXED_BYTES).

How to run
----------
Needs a CUDA GPU. From the repository root, with the BiaPy environment active::

    python tests/calibrate_unet_memory.py [--gpu 0]
"""
import argparse
import os

import numpy as np

from biapy.wizard import unet as planner

# (spacing, patch_size, batch sizes)
CASES = [
    ((1, 1), (256, 256), (2, 8)),
    ((1, 1), (512, 512), (2, 6)),
    ((1, 1), (128, 640), (2, 6)),
    ((1, 1), (64, 64), (8, 32)),
    ((1, 1, 1), (64, 64, 64), (1, 2)),
    ((1, 1, 1), (96, 96, 96), (1, 2)),
    ((1, 1, 1), (128, 128, 128), (1,)),
    ((4, 1, 1), (32, 160, 160), (1, 2)),
    ((5, 1, 1), (16, 256, 256), (1, 2)),
    ((10, 1, 1), (8, 320, 320), (1, 2)),
]


def measure(spacing, patch, batch, n_classes=2, in_channels=1):
    import torch
    from biapy.models.unet import U_Net

    topo = planner.get_topology(spacing, patch)
    plan = {
        "patch_size": topo["patch_size"],
        "feature_maps": planner.feature_maps(len(topo["conv_kernels"]), len(spacing)),
        "pools": topo["pools"],
        "conv_kernels": topo["conv_kernels"],
    }
    m = planner.plan_to_biapy(plan)
    ndim = len(spacing)
    p = tuple(plan["patch_size"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_reserved()
    model = U_Net(
        image_shape=p + (in_channels,),
        activation=m["ACTIVATION"],
        feature_maps=m["FEATURE_MAPS"],
        drop_values=m["DROPOUT_VALUES"],
        normalization=m["NORMALIZATION"],
        k_size=3,
        upsample_layer=m["UPSAMPLE_LAYER"],
        yx_down=m["YX_DOWN"],
        z_down=m.get("Z_DOWN", [2] * len(m["YX_DOWN"])),
        output_channels=[n_classes],
        output_channel_info=["class"] if n_classes > 2 else ["F"],
        head_activations=["linear"],
        explicit_activations=False,
        isotropy=m.get("ISOTROPY", True),
        conv_layers=m["CONV_LAYERS"],
    ).cuda()
    n_params = sum(t.numel() for t in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn((batch, in_channels) + p, device="cuda")
    y = torch.randint(0, n_classes, (batch,) + p, device="cuda")
    try:
        for _ in range(2):
            opt.zero_grad()
            out = model(x)
            out = out if isinstance(out, torch.Tensor) else out["pred"]
            prob = out.softmax(1)
            onehot = torch.nn.functional.one_hot(y, n_classes).movedim(-1, 1).float()
            dice = 1 - (2 * (prob * onehot).sum() + 1) / (prob.sum() + onehot.sum() + 1)
            loss = torch.nn.functional.cross_entropy(out, y) + dice
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() - base
    except torch.cuda.OutOfMemoryError:
        peak = None
    del model, opt, x, y
    torch.cuda.empty_cache()
    fm = plan["feature_maps"]
    act = planner.count_activations(p, topo, in_channels, n_classes, fm)
    est_params = planner.count_parameters(topo, in_channels, n_classes, fm)
    return {"ndim": ndim, "patch": p, "batch": batch, "act": act, "params": n_params, "est_params": est_params, "peak": peak}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0")
    a = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    import torch

    torch.backends.cudnn.benchmark = False
    rows = []
    for spacing, patch, batches in CASES:
        for b in batches:
            r = measure(spacing, patch, b)
            rows.append(r)
            print(
                "{}D patch={} bs={} levels={} params={} (est {}) act={:.3g} peak={}".format(
                    r["ndim"], r["patch"], b, "", r["params"], r["est_params"], r["act"],
                    "OOM" if r["peak"] is None else "{:.2f} GB".format(r["peak"] / 1024**3),
                ),
                flush=True,
            )
    for ndim in (2, 3):
        data = [r for r in rows if r["ndim"] == ndim and r["peak"] is not None]
        # peak - 16*params = a * 4 * act * batch + c
        A = np.array([[4 * r["act"] * r["batch"], 1.0] for r in data])
        y = np.array([r["peak"] - planner.VRAM_BYTES_PER_PARAMETER * r["params"] for r in data])
        (a_fit, c_fit), *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ np.array([a_fit, c_fit])
        err = (pred - y) / np.array([r["peak"] for r in data])
        print("{}D: bytes per activation = 4 * {:.3f}, fixed = {:.3f} GB, max rel. error = {:.1%}".format(
            ndim, a_fit, c_fit / 1024**3, np.abs(err).max()))


if __name__ == "__main__":
    main()
