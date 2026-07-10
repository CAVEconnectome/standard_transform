"""Select the laplace-fit smoothness weight (lambda) by held-out cross-validation.

Sweeps ``laplace_strength`` over a grid, and for each value runs k-fold CV measuring the
per-path in-band lateral deviation of the held-out streamlines (same metric as
compare_field_methods.py). Reports the mean +/- std deviation across folds per lambda,
the CV minimizer, and the "1-SE" choice (largest lambda within one standard error of the
best -- the more regularized, more parsimonious pick). Selecting lambda on the data at
hand also sidesteps the fact that it is defined relative to the median cell precision.

    uv run python scripts/tune_laplace_strength.py --paths /path/to/apical --dataset v1dd
"""

import argparse
import glob
import os
import time

import numpy as np

from standard_transform import streamline_field_from_paths
from standard_transform.datasets import minnie_transform, v1dd_transform

TRANSFORMS = {"v1dd": v1dd_transform, "minnie": minnie_transform}
DEFAULT_BANDS = {"v1dd": (150.0, 700.0), "minnie": (150.0, 650.0)}
DEFAULT_BIN = (30.0, 20.0, 30.0)
DEFAULT_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)


def load_paths(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    if not files:
        raise SystemExit(f"No .npy files found in {directory}")
    return [np.load(f)[:, :3].astype(float) for f in files]


def fold_dev(train, test, tform, lam, bs, band):
    field = streamline_field_from_paths(
        train, tform=tform, transform_points=False, method="laplace-fit",
        bin_size=bs, depth_band=band, laplace_strength=lam,
    )
    devs = []
    for pp in test:
        m = (pp[:, 1] >= band[0]) & (pp[:, 1] <= band[1])
        if m.sum() < 2:
            continue
        fx, fz = field.streamline_at(pp[-1], pp[m, 1])
        devs.append(np.median(np.sqrt((fx - pp[m, 0]) ** 2 + (fz - pp[m, 2]) ** 2)))
    return float(np.median(devs))  # median across held-out paths in this fold


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--paths", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(TRANSFORMS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=float, nargs="+", default=list(DEFAULT_GRID))
    ap.add_argument("--bin-size", type=float, nargs=3, default=DEFAULT_BIN)
    ap.add_argument("--depth-band", type=float, nargs=2, default=None)
    args = ap.parse_args()

    band = tuple(args.depth_band) if args.depth_band else DEFAULT_BANDS[args.dataset]
    bs = tuple(args.bin_size)
    tform = TRANSFORMS[args.dataset]()
    paths = load_paths(args.paths)
    oriented = [tform.apply(p) for p in paths]
    folds = np.array_split(np.random.default_rng(args.seed).permutation(len(paths)), args.folds)
    print(f"loaded {len(paths)} paths; {args.folds}-fold CV over lambda={args.grid}")

    t0 = time.time()
    # per lambda: list of per-fold median deviations
    curve = {lam: [] for lam in args.grid}
    for fi, test_idx in enumerate(folds):
        test = set(test_idx.tolist())
        train = [oriented[i] for i in range(len(paths)) if i not in test]
        held = [oriented[i] for i in test_idx]
        for lam in args.grid:
            curve[lam].append(fold_dev(train, held, tform, lam, bs, band))
        print(f"  fold {fi + 1}/{args.folds} done ({time.time() - t0:.0f}s)")

    means = {lam: float(np.mean(v)) for lam, v in curve.items()}
    stds = {lam: float(np.std(v)) for lam, v in curve.items()}
    best = min(means, key=means.get)
    se_best = stds[best] / np.sqrt(args.folds)
    # 1-SE rule: largest lambda whose mean is within 1 SE of the best mean
    within = [lam for lam in args.grid if means[lam] <= means[best] + se_best]
    one_se = max(within)

    print(f"\n{'lambda':>8} {'CV median dev (mean +/- std over folds)':>40}")
    for lam in args.grid:
        mark = "  <- best" if lam == best else ("  <- 1SE" if lam == one_se else "")
        print(f"{lam:>8.3f}   {means[lam]:>7.3f} +/- {stds[lam]:<7.3f}{mark}")
    print(f"\nCV minimizer: lambda = {best} ({means[best]:.3f} um)")
    print(f"1-SE choice : lambda = {one_se} ({means[one_se]:.3f} um)")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
