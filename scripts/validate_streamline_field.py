"""Produce held-out validation data for a StreamlineField, for plotting.

Where ``build_streamline_field.py --validate`` prints a single median-deviation
number, this writes a rich, out-of-sample dataset you can load into a notebook and
plot: it does k-fold cross-validation so *every* input path gets a prediction from a
field that never saw it, then records the per-point lateral deviation (for
deviation-vs-depth and predicted-vs-actual trace plots) and a per-path summary (for
histograms and deviation-vs-cortical-position plots).

Input matches build_streamline_field.py: a directory of per-neuron ``.npy`` files,
each an ``(N, 3+)`` array of points in the dataset's original nanometer space, ordered
tip->root along one tall dendritic path. Only x, y, z are used.

Run from the repo root:

    uv run python scripts/validate_streamline_field.py --paths /path/to/apical_streamlines --dataset v1dd

Output is a single ``.npz`` (default ``validation_<dataset>.npz``) with two record
groups plus metadata; see ``save`` below or the printed summary for the fields. Note
that the shipped field is always built on ALL paths -- the k fields built here are
throwaway, used only to generate out-of-sample predictions.
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
# Match the per-dataset defaults used to build the shipped fields (see CLAUDE.md /
# the README method section). v1dd apical data is reliable to ~700um, Minnie to ~650.
DEFAULT_BANDS = {"v1dd": (150.0, 700.0), "minnie": (150.0, 650.0)}
DEFAULT_BIN = (30.0, 20.0, 30.0)
DATA_DIR = os.path.join(os.path.dirname(standard_transform.__file__), "data")


def load_paths(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.npy")))
    if not files:
        raise SystemExit(f"No .npy files found in {directory}")
    return [np.load(f)[:, :3].astype(float) for f in files]


def build(train_paths, tform_fn, bin_size, depth_band):
    return streamline_field_from_paths(
        train_paths,
        tform=tform_fn(),
        bin_size=tuple(bin_size),
        depth_band=tuple(depth_band),
    )


def predict_path(field, pp, depth_band):
    """Predict the field's streamline through a path's root, at each in-band depth.

    Parameters
    ----------
    field : StreamlineField
    pp : (N, 3) array
        One path already in oriented-micron (post-transform) space.
    depth_band : (lo, hi)

    Returns
    -------
    dict or None
        Per-point arrays (y, actual/predicted x & z, lateral deviation) and the
        oriented anchor (root) location, or None if the path has < 2 in-band points.
    """
    lo, hi = depth_band
    m = (pp[:, 1] >= lo) & (pp[:, 1] <= hi)
    if m.sum() < 2:
        return None
    anchor = pp[-1]  # paths are tip->root, so the last point is the root/soma end
    y = pp[m, 1]
    fx, fz = field.streamline_at(anchor, y)
    dev = np.sqrt((fx - pp[m, 0]) ** 2 + (fz - pp[m, 2]) ** 2)
    return {
        "y": y,
        "x": pp[m, 0],
        "z": pp[m, 2],
        "x_pred": np.asarray(fx),
        "z_pred": np.asarray(fz),
        "dev": dev,
        "anchor": anchor,
    }


def clamp_to_grid(field, xyz):
    """Clamp a point into the field's node-center range along each axis.

    ``StreamlineField.coverage_at`` returns 0 outside the grid, and node centers sit
    half a bin inside the depth band, so a soma at (or just outside) the band edge
    would otherwise read 0. Clamping mirrors the field's own tangent lookup.
    """
    out = np.asarray(xyz, dtype=float).copy()
    for a, grid in enumerate((field._x_grid, field._y_grid, field._z_grid)):
        out[a] = min(max(out[a], float(grid[0])), float(grid[-1]))
    return out


def kfold_indices(n, folds, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return np.array_split(order, folds)


def save(out, dataset, args, point, path):
    """Write point-level and per-path records plus metadata to an .npz."""
    np.savez_compressed(
        out,
        # -- point-level: one row per in-band held-out point --
        pt_path_id=point["path_id"].astype(np.int32),
        pt_fold=point["fold"].astype(np.int16),
        pt_y=point["y"],
        pt_x=point["x"],
        pt_z=point["z"],
        pt_x_pred=point["x_pred"],
        pt_z_pred=point["z_pred"],
        pt_dev=point["dev"],
        # -- per-path: one row per neuron with >=2 in-band points --
        path_id=path["path_id"].astype(np.int32),
        path_fold=path["fold"].astype(np.int16),
        path_n=path["n"].astype(np.int32),
        path_anchor=path["anchor"],
        path_median_dev=path["median_dev"],
        path_mean_dev=path["mean_dev"],
        path_max_dev=path["max_dev"],
        path_coverage=path["coverage"],
        # -- metadata --
        meta_dataset=np.array(dataset),
        meta_bin_size=np.asarray(args.bin_size, dtype=float),
        meta_depth_band=np.asarray(args.depth_band, dtype=float),
        meta_folds=np.array(args.folds),
        meta_seed=np.array(args.seed),
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--paths", required=True, help="directory of per-neuron .npy files (nm)")
    ap.add_argument("--dataset", required=True, choices=sorted(TRANSFORMS))
    ap.add_argument("--out", help="output .npz (default: validation_<dataset>.npz)")
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the fold split")
    ap.add_argument(
        "--bin-size", type=float, nargs=3, default=DEFAULT_BIN, metavar=("X", "Y", "Z")
    )
    ap.add_argument(
        "--depth-band",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="default: per-dataset (v1dd 150 700, minnie 150 650)",
    )
    args = ap.parse_args()

    if args.depth_band is None:
        args.depth_band = DEFAULT_BANDS[args.dataset]
    if args.folds < 2:
        raise SystemExit("--folds must be >= 2 for cross-validation")

    tform_fn = TRANSFORMS[args.dataset]
    tform = tform_fn()
    out = args.out or f"validation_{args.dataset}.npz"

    paths = load_paths(args.paths)
    print(f"loaded {len(paths)} paths from {args.paths}")
    # Pre-transform every path once into oriented microns.
    oriented = [tform.apply(p) for p in paths]

    folds = kfold_indices(len(paths), args.folds, args.seed)

    pt = {k: [] for k in ("path_id", "fold", "y", "x", "z", "x_pred", "z_pred", "dev")}
    pa = {
        k: []
        for k in ("path_id", "fold", "n", "anchor", "median_dev", "mean_dev", "max_dev", "coverage")
    }

    t0 = time.time()
    for fi, test_idx in enumerate(folds):
        test_set = set(test_idx.tolist())
        train_paths = [paths[i] for i in range(len(paths)) if i not in test_set]
        field = build(train_paths, tform_fn, args.bin_size, args.depth_band)
        n_pred = 0
        for i in test_idx:
            res = predict_path(field, oriented[i], args.depth_band)
            if res is None:
                continue
            n_pred += 1
            k = len(res["y"])
            pt["path_id"].append(np.full(k, i))
            pt["fold"].append(np.full(k, fi))
            for key in ("y", "x", "z", "x_pred", "z_pred", "dev"):
                pt[key].append(res[key])
            pa["path_id"].append(i)
            pa["fold"].append(fi)
            pa["n"].append(k)
            pa["anchor"].append(res["anchor"])
            pa["median_dev"].append(float(np.median(res["dev"])))
            pa["mean_dev"].append(float(res["dev"].mean()))
            pa["max_dev"].append(float(res["dev"].max()))
            # coverage_at returns 0 outside the grid (no clamp), and grid node centers
            # are inset half a bin from the band edges, so clamp the query into the
            # node-center range (as the tangent lookup does) to report the coverage of
            # the region the cell's streamline actually integrates through.
            pa["coverage"].append(
                float(field.coverage_at(clamp_to_grid(field, res["anchor"]), transform_points=False)[0])
            )
        print(f"  fold {fi + 1}/{args.folds}: {n_pred} held-out paths predicted")

    point = {
        "path_id": np.concatenate(pt["path_id"]),
        "fold": np.concatenate(pt["fold"]),
        "y": np.concatenate(pt["y"]),
        "x": np.concatenate(pt["x"]),
        "z": np.concatenate(pt["z"]),
        "x_pred": np.concatenate(pt["x_pred"]),
        "z_pred": np.concatenate(pt["z_pred"]),
        "dev": np.concatenate(pt["dev"]),
    }
    path = {k: np.asarray(v) for k, v in pa.items()}
    save(out, args.dataset, args, point, path)

    med = np.median(path["median_dev"])
    p90 = np.percentile(path["median_dev"], 90)
    print(f"\ncross-validated {len(path['path_id'])} paths in {time.time() - t0:.1f}s")
    print(f"per-path median lateral deviation: median {med:.2f} um, 90th pct {p90:.2f} um")
    print(f"saved -> {out}  ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
