"""Build a StreamlineField .npz from a directory of skeleton paths.

Reproduces the shipped ``standard_transform/data/<dataset>_streamline_field.npz``.

Input is a directory of ``.npy`` files, one per neuron, each an ``(N, 3+)`` array of
points in the dataset's *original nanometer* space, ordered along a single tall
dendritic path (tip to root). Only the first three columns (x, y, z) are used; any
extra columns (e.g. a Strahler index) are ignored. See the README section
"Spatially-varying streamlines" for the method and how the defaults were chosen.

Run from the repo root (uses the project environment):

    uv run python scripts/build_streamline_field.py --paths /path/to/apical_streamlines --dataset v1dd

Add ``--validate 800`` to also hold out 800 paths, rebuild on the rest, and report the
held-out in-band median lateral deviation (the shipped field is always built on ALL
paths; validation builds a separate throwaway field purely to report the number).

The v1dd apical-path dataset used for the shipped field is archived at: <archive URL>
"""

import argparse
import glob
import os
import time

import numpy as np

import standard_transform
from standard_transform import streamline_field_from_paths
from standard_transform.datasets import minnie_transform, v1dd_transform

TRANSFORMS = {"v1dd": v1dd_transform, "minnie": minnie_transform}
DATA_DIR = os.path.join(os.path.dirname(standard_transform.__file__), "data")


def load_paths(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    if not files:
        raise SystemExit(f"No .npy files found in {directory}")
    return [np.load(f)[:, :3].astype(float) for f in files]


def build(paths, tform_fn, bin_size, depth_band):
    return streamline_field_from_paths(
        paths, tform=tform_fn(), bin_size=tuple(bin_size), depth_band=tuple(depth_band)
    )


def in_band_deviation(field, val_paths, tform_fn, depth_band):
    tform = tform_fn()
    lo, hi = depth_band
    devs = []
    for p in val_paths:
        pp = tform.apply(p)
        m = (pp[:, 1] >= lo) & (pp[:, 1] <= hi)
        if m.sum() < 2:
            continue
        fx, fz = field.streamline_at(pp[-1], pp[m, 1])
        devs.append(np.sqrt((fx - pp[m, 0]) ** 2 + (fz - pp[m, 2]) ** 2).mean())
    return float(np.median(devs)), len(devs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--paths", required=True, help="directory of per-neuron .npy files (nm)")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=sorted(TRANSFORMS),
        help="selects the nm transform and the default output filename",
    )
    ap.add_argument("--out", help="output .npz (default: data/<dataset>_streamline_field.npz)")
    ap.add_argument(
        "--bin-size", type=float, nargs=3, default=(30.0, 20.0, 30.0), metavar=("X", "Y", "Z")
    )
    ap.add_argument(
        "--depth-band", type=float, nargs=2, default=(150.0, 700.0), metavar=("LO", "HI")
    )
    ap.add_argument(
        "--validate", type=int, default=0, metavar="N", help="hold out N paths for a QC report"
    )
    args = ap.parse_args()

    tform_fn = TRANSFORMS[args.dataset]
    out = args.out or os.path.join(DATA_DIR, f"{args.dataset}_streamline_field.npz")
    paths = load_paths(args.paths)
    print(f"loaded {len(paths)} paths from {args.paths}")

    t0 = time.time()
    field = build(paths, tform_fn, args.bin_size, args.depth_band)
    field.to_npz(out)
    print(f"built {field} on all paths in {time.time() - t0:.1f}s")
    print(f"saved -> {out}  ({os.path.getsize(out) / 1024:.0f} KB)")

    if args.validate:
        val, train = paths[: args.validate], paths[args.validate :]
        vfield = build(train, tform_fn, args.bin_size, args.depth_band)
        med, n = in_band_deviation(vfield, val, tform_fn, args.depth_band)
        print(f"QC: held-out in-band median lateral deviation {med:.2f} um (n={n} paths)")


if __name__ == "__main__":
    main()
