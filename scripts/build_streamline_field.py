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

Pick the regularizer with ``--method`` (``diffusion`` default; ``laplace-fit`` fits a
curl-free scalar-potential field). For the Laplace methods, ``--tune-lambda`` runs a
k-fold CV sweep of the smoothness weight and builds at the selected value instead of
hand-setting ``--laplace-strength``. The chosen method and lambda are stamped into the
.npz alongside the transform frame.

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
DEFAULT_LAMBDA_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)


def load_paths(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    if not files:
        raise SystemExit(f"No .npy files found in {directory}")
    return [np.load(f)[:, :3].astype(float) for f in files]


def build(paths, tform_fn, bin_size, depth_band, method="diffusion", laplace_strength=0.05):
    return streamline_field_from_paths(
        paths,
        tform=tform_fn(),
        bin_size=tuple(bin_size),
        depth_band=tuple(depth_band),
        method=method,
        laplace_strength=laplace_strength,
    )


def _fold_dev(train_oriented, held_oriented, tform, lam, bin_size, band):
    field = streamline_field_from_paths(
        train_oriented, tform=tform, transform_points=False, method="laplace-fit",
        bin_size=tuple(bin_size), depth_band=tuple(band), laplace_strength=lam,
    )
    devs = []
    for pp in held_oriented:
        m = (pp[:, 1] >= band[0]) & (pp[:, 1] <= band[1])
        if m.sum() < 2:
            continue
        fx, fz = field.streamline_at(pp[-1], pp[m, 1])
        devs.append(np.median(np.sqrt((fx - pp[m, 0]) ** 2 + (fz - pp[m, 2]) ** 2)))
    return float(np.median(devs))


def tune_lambda(paths, tform_fn, bin_size, band, grid, folds, seed):
    """k-fold CV over the laplace-fit lambda grid; return (best, one_se, curve)."""
    tform = tform_fn()
    oriented = [tform.apply(p) for p in paths]
    idx_folds = np.array_split(np.random.default_rng(seed).permutation(len(paths)), folds)
    curve = {lam: [] for lam in grid}
    t0 = time.time()
    for fi, test_idx in enumerate(idx_folds):
        test = set(test_idx.tolist())
        train = [oriented[i] for i in range(len(paths)) if i not in test]
        held = [oriented[i] for i in test_idx]
        for lam in grid:
            curve[lam].append(_fold_dev(train, held, tform, lam, bin_size, band))
        print(f"  tune fold {fi + 1}/{folds} done ({time.time() - t0:.0f}s)")
    means = {lam: float(np.mean(v)) for lam, v in curve.items()}
    stds = {lam: float(np.std(v)) for lam, v in curve.items()}
    best = min(means, key=means.get)
    se = stds[best] / np.sqrt(folds)
    one_se = max(lam for lam in grid if means[lam] <= means[best] + se)  # most regularized
    print(f"\n  {'lambda':>8}   CV median dev (mean +/- std over folds)")
    for lam in grid:
        tag = "  <- best" if lam == best else ("  <- 1SE" if lam == one_se else "")
        print(f"  {lam:>8.3f}   {means[lam]:.3f} +/- {stds[lam]:.3f}{tag}")
    return best, one_se, means


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
        "--method",
        default="diffusion",
        choices=("diffusion", "laplace-fit", "laplace-bc"),
        help="field regularizer (default diffusion)",
    )
    ap.add_argument(
        "--laplace-strength", type=float, default=0.05, help="lambda for the Laplace methods"
    )
    ap.add_argument(
        "--tune-lambda",
        action="store_true",
        help="CV-select lambda over --lambda-grid and build laplace-fit at the 1-SE value",
    )
    ap.add_argument("--lambda-grid", type=float, nargs="+", default=list(DEFAULT_LAMBDA_GRID))
    ap.add_argument("--tune-folds", type=int, default=5, help="folds for --tune-lambda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--validate", type=int, default=0, metavar="N", help="hold out N paths for a QC report"
    )
    args = ap.parse_args()

    tform_fn = TRANSFORMS[args.dataset]
    out = args.out or os.path.join(DATA_DIR, f"{args.dataset}_streamline_field.npz")
    paths = load_paths(args.paths)
    print(f"loaded {len(paths)} paths from {args.paths}")

    method = args.method
    lam = args.laplace_strength
    if args.tune_lambda:
        # --tune-lambda selects the smoothness weight from the data rather than hand-set,
        # and implies the laplace-fit method it tunes.
        if method != "laplace-fit":
            print(f"--tune-lambda selects lambda for laplace-fit (overriding method={method!r})")
            method = "laplace-fit"
        print(f"tuning lambda by {args.tune_folds}-fold CV over {args.lambda_grid} ...")
        best, one_se, _ = tune_lambda(
            paths, tform_fn, args.bin_size, args.depth_band, args.lambda_grid, args.tune_folds, args.seed
        )
        lam = one_se  # 1-SE rule: most regularized within one SE of the CV minimum
        print(f"selected lambda = {lam} (CV minimizer {best}, 1-SE rule)\n")

    # laplace_strength is only meaningful for the Laplace methods; recorded only for them.
    stamp_lam = lam if method.startswith("laplace") else None

    t0 = time.time()
    field = build(paths, tform_fn, args.bin_size, args.depth_band, method, lam)
    # to_npz stamps the transform frame, method, and lambda so a stale or mismatched
    # field is self-describing (and a wrong-frame attach is caught on load).
    field.to_npz(out, method=method, laplace_strength=stamp_lam)
    print(f"built {field} [{method}] on all paths in {time.time() - t0:.1f}s")
    print(f"stamped: transform frame v{field._transform.version}, method {method}"
          + (f", lambda {lam}" if stamp_lam is not None else ""))
    print(f"saved -> {out}  ({os.path.getsize(out) / 1024:.0f} KB)")

    if args.validate:
        val, train = paths[: args.validate], paths[args.validate :]
        vfield = build(train, tform_fn, args.bin_size, args.depth_band, method, lam)
        med, n = in_band_deviation(vfield, val, tform_fn, args.depth_band)
        print(f"QC: held-out in-band median lateral deviation {med:.2f} um (n={n} paths)")


if __name__ == "__main__":
    main()
