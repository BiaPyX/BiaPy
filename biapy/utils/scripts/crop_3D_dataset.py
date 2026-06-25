#!/usr/bin/env python3
"""
Chunk a 3D image/volume (Z, Y, X) into patches of size (pz, py, px).

Example:
  image shape:  (100, 4096, 4096)
  patch size:   (100, 1024, 1024)  in z,y,x
  output:       1 * 4 * 4 = 16 patches

You will provide the actual load/save functions; placeholders are included.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import numpy as np
from biapy.data.data_manipulation import read_img_as_ndarray
from biapy.data.data_manipulation import save_tif

# -----------------------------
# Core logic
# -----------------------------
@dataclass(frozen=True)
class PatchSpec:
    pz: int
    py: int
    px: int


def _compute_grid(shape: Tuple[int, int, int], spec: PatchSpec, mode: str) -> Tuple[int, int, int]:
    z, y, x = shape
    if mode == "drop":
        nz = z // spec.pz
        ny = y // spec.py
        nx = x // spec.px
    elif mode == "pad":
        nz = (z + spec.pz - 1) // spec.pz
        ny = (y + spec.py - 1) // spec.py
        nx = (x + spec.px - 1) // spec.px
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return nz, ny, nx


def _pad_to_grid(arr: np.ndarray, spec: PatchSpec) -> np.ndarray:
    z, y, x = arr.shape
    tz = ((z + spec.pz - 1) // spec.pz) * spec.pz
    ty = ((y + spec.py - 1) // spec.py) * spec.py
    tx = ((x + spec.px - 1) // spec.px) * spec.px

    pad_z = tz - z
    pad_y = ty - y
    pad_x = tx - x

    if pad_z == 0 and pad_y == 0 and pad_x == 0:
        return arr

    # pad at the end of each axis
    return np.pad(arr, pad_width=((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)


def iter_patches(
    arr: np.ndarray,
    spec: PatchSpec,
    mode: str = "drop",
) -> Iterator[Tuple[Tuple[int, int, int], np.ndarray]]:
    """
    Yields ((iz, iy, ix), patch) where indices are patch-grid indices.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array (Z,Y,X). Got shape: {arr.shape}")

    work = _pad_to_grid(arr, spec) if mode == "pad" else arr
    nz, ny, nx = _compute_grid(work.shape, spec, mode="drop" if mode == "drop" else "pad")

    for iz in range(nz):
        z0 = iz * spec.pz
        z1 = z0 + spec.pz
        for iy in range(ny):
            y0 = iy * spec.py
            y1 = y0 + spec.py
            for ix in range(nx):
                x0 = ix * spec.px
                x1 = x0 + spec.px
                patch = work[z0:z1, y0:y1, x0:x1]
                # In "drop" mode, ensure exact patch size (skip edges that don't fit)
                if mode == "drop" and patch.shape != (spec.pz, spec.py, spec.px):
                    continue
                yield (iz, iy, ix), patch


def make_out_name(stem: str, iz: int, iy: int, ix: int, ext: str) -> str:
    return f"{stem}_z{iz:03d}_y{iy:03d}_x{ix:03d}{ext}"


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk a 3D image (Z,Y,X) into patches (pz,py,px).")
    p.add_argument("--input", "-i", required=True, help="Input image path.")
    p.add_argument("--output-dir", "-o", required=True, help="Directory to write patches.")
    p.add_argument(
        "--patch-size",
        "-p",
        required=True,
        nargs=3,
        type=int,
        metavar=("PZ", "PY", "PX"),
        help="Patch size in z,y,x order. Example: --patch-size 100 1024 1024",
    )
    p.add_argument(
        "--mode",
        choices=["drop", "pad"],
        default="drop",
        help="drop: discard incomplete edge patches. pad: pad with zeros to fit full grid.",
    )
    p.add_argument(
        "--ext",
        default=".tif",
        help="Output extension (used for naming only; your save_patch controls actual format). Default: .tif",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pz, py, px = args.patch_size
    if pz <= 0 or py <= 0 or px <= 0:
        raise ValueError("Patch sizes must be positive integers.")

    spec = PatchSpec(pz=pz, py=py, px=px)

    arr = read_img_as_ndarray(args.input, is_3d=True).squeeze()
    if arr.ndim != 3:
        raise ValueError(f"Loaded image must be 3D (Z,Y,X). Got shape {arr.shape}")

    in_name = os.path.basename(args.input)
    stem, _ = os.path.splitext(in_name)

    count = 0
    for (iz, iy, ix), patch in iter_patches(arr, spec, mode=args.mode):
        out_name = make_out_name(stem, iz, iy, ix, args.ext)
        save_tif(np.expand_dims(np.expand_dims(patch, axis=0), axis=-1), args.output_dir, [out_name], verbose=True)
        count += 1

    print(f"Input shape: {arr.shape}")
    print(f"Patch size:  ({spec.pz}, {spec.py}, {spec.px})  (z,y,x)")
    print(f"Mode:        {args.mode}")
    print(f"Wrote {count} patches to: {args.output_dir}")


if __name__ == "__main__":
    main()
