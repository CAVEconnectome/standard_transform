# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "standard-transform @ file:///Users/caseysm/Work/Code/standard_transform",
#   "caveclient",
# ]
# ///
"""Build a neuroglancer state with a grid of effective streamlines from the tangent field.

The field stores only tangent vectors, so we integrate a streamline through each seed
point of an (x, z) grid, invert back to the dataset's original nanometer space, convert
to voxels, and emit them as connected `line` annotations. Imagery/segmentation are
intentionally omitted -- drop this annotation layer into your own state.

The field is loaded from the *shipped* ``data/<dataset>_streamline_field.npz`` via the
dataset object, so this visualizes the exact artifact the package serves (correct
transform + depth band), not a rebuilt approximation.

Run standalone (writes a self-contained URL-fragment link, no credentials needed):
    uv run make_streamline_grid_state.py --dataset minnie
    uv run make_streamline_grid_state.py --dataset v1dd

Or upload to the CAVE state server for a short link (uses your datastack + auth token):
    uv run make_streamline_grid_state.py --dataset minnie --datastack minnie65_phase3_v1
"""

import argparse
import json
import os
import urllib.parse

import numpy as np

from standard_transform import (
    StreamlineField,
    available_versions,
    minnie_ds,
    v1dd_ds,
)
from standard_transform.datasets import (
    MINNIE_VOXEL_RESOLUTION,
    V1DD_VOXEL_RESOLUTION,
)

NG_HOST = "https://neuroglancer-demo.appspot.com/"

# dataset -> (Dataset object, neuroglancer voxel resolution nm/voxel, default datastack)
DATASETS = {
    "v1dd": (v1dd_ds, V1DD_VOXEL_RESOLUTION, "v1dd"),
    "minnie": (minnie_ds, MINNIE_VOXEL_RESOLUTION, "minnie65_phase3_v1"),
}

# Grid / sampling (post-transform microns)
SEED_SPACING = 50.0   # spacing of streamline seeds in x and z
DEPTH_STEP = 40.0      # sampling along depth for each drawn streamline
HERE = os.path.dirname(__file__)


def load_field(ds, version=None):
    """Return a StreamlineField to grid.

    With ``version`` given, that streamline version is used (and must be a field).
    Otherwise the dataset default is tried first; if the default is not a field (e.g.
    minnie65 defaults to the hand-drawn 1.4 streamline while its field is version 2.0),
    fall back to the latest streamline version that *is* a field.
    """
    if version is not None:
        field = ds.streamline(version=version)
        if not isinstance(field, StreamlineField):
            raise SystemExit(
                f"{ds.name} streamline version {version!r} is not a field "
                f"(got {type(field).__name__})."
            )
        print(f"loaded {field} for {ds.name} (version {version})")
        return field

    field = ds.streamline()  # shipped default, transform already attached
    if isinstance(field, StreamlineField):
        print(f"loaded {field} for {ds.name}")
        return field

    # Default isn't a field -> find the latest streamline version that is one.
    versions = available_versions(ds.name)["streamline"]["versions"]
    for v in sorted(versions, reverse=True):
        cand = ds.streamline(version=v)
        if isinstance(cand, StreamlineField):
            print(
                f"{ds.name} default streamline is a {type(field).__name__}; "
                f"gridding field version {v!r} instead (pass --version to override)."
            )
            return cand
    raise SystemExit(f"{ds.name} has no streamline field version to grid.")


def streamline_lines(field, voxel_resolution):
    """Integrate one streamline per (x, z) seed; return list of voxel-space point arrays."""
    xs = np.arange(field._x_grid.min(), field._x_grid.max() + 1e-6, SEED_SPACING)
    zs = np.arange(field._z_grid.min(), field._z_grid.max() + 1e-6, SEED_SPACING)
    # Extend past the band so the held-constant (clamped) straight extensions show.
    y_lo, y_hi = field._y_range
    depths = np.arange(y_lo - 150.0, y_hi + 150.0 + 1e-6, DEPTH_STEP)

    lines = []
    for x0 in xs:
        for z0 in zs:
            new_x, new_z = field.streamline_at(np.array([x0, 0.0, z0]), depths)
            pts_um = np.column_stack([new_x, depths, new_z])
            pts_nm = field._transform.invert(pts_um)
            pts_vox = np.round(pts_nm / voxel_resolution).astype(int)
            lines.append(pts_vox)
    print(f"{len(lines)} streamlines, {len(depths)} points each")
    return lines


def make_state(lines, voxel_resolution):
    res = np.asarray(voxel_resolution, dtype=float)
    dims = {
        "x": [res[0] * 1e-9, "m"],
        "y": [res[1] * 1e-9, "m"],
        "z": [res[2] * 1e-9, "m"],
    }
    annotations = []
    idx = 0
    for line in lines:
        for a, b in zip(line[:-1], line[1:]):
            annotations.append(
                {
                    "type": "line",
                    "pointA": a.tolist(),
                    "pointB": b.tolist(),
                    "id": f"{idx:024x}",
                }
            )
            idx += 1

    center = np.round(np.vstack(lines).mean(axis=0)).astype(int).tolist()
    state = {
        "dimensions": dims,
        "position": center,
        "crossSectionScale": 4,
        "projectionScale": 60000,
        "layers": [
            {
                "type": "annotation",
                "name": "streamline_grid",
                "source": {
                    "url": "local://annotations",
                    "transform": {"outputDimensions": dims},
                },
                "annotations": annotations,
                "annotationColor": "#ffdd00",
                "tool": "annotateLine",
            }
        ],
        "selectedLayer": {"visible": True, "layer": "streamline_grid"},
        "layout": "xy-3d",
    }
    print(f"{len(annotations)} line annotations")
    return state


def upload_link(state, datastack, ngl_url):
    """Upload the state to the CAVE state server and return a short link.

    ``ngl_url`` should be a CAVE-aware viewer (it must serve the neuroglancer info
    endpoint); pass None to use the datastack's configured default viewer. The public
    demo host (``NG_HOST``) is *not* CAVE-aware and cannot be used here.
    """
    from caveclient import CAVEclient

    client = CAVEclient(datastack)
    state_id = client.state.upload_state_json(state)
    try:
        return client.state.build_neuroglancer_url(state_id, ngl_url)
    except Exception as e:  # noqa: BLE001 - surface a usable link regardless
        print(f"(could not resolve viewer info for a short link: {e})")
        print(f"uploaded state id: {state_id}")
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset", choices=sorted(DATASETS), default="v1dd", help="which field to grid"
    )
    ap.add_argument(
        "--version",
        default=None,
        help="streamline version to grid (must be a field); default auto-selects the "
        "latest field version if the dataset default is not a field",
    )
    ap.add_argument(
        "--datastack",
        nargs="?",
        const="",
        help="CAVE datastack; if set (bare flag uses the dataset default), upload for a short link",
    )
    ap.add_argument(
        "--ngl-url",
        default=None,
        help="neuroglancer host. For --datastack uploads, must be CAVE-aware "
        "(default: the datastack's configured viewer). For the self-contained link, "
        f"defaults to {NG_HOST}.",
    )
    args = ap.parse_args()

    ds, voxel_resolution, default_datastack = DATASETS[args.dataset]
    out_json = os.path.join(HERE, f"{args.dataset}_streamline_grid_state.json")
    out_link = os.path.join(HERE, f"{args.dataset}_streamline_grid_link.txt")

    field = load_field(ds, args.version)
    lines = streamline_lines(field, voxel_resolution)
    state = make_state(lines, voxel_resolution)
    with open(out_json, "w") as f:
        json.dump(state, f)
    print(f"\nstate written to {out_json}")

    if args.datastack is not None:
        datastack = args.datastack or default_datastack
        link = upload_link(state, datastack, args.ngl_url)
        print(f"\nshort link (state server, {datastack}):\n{link}")
    else:
        host = args.ngl_url or NG_HOST
        link = host + "#!" + urllib.parse.quote(json.dumps(state))
        with open(out_link, "w") as f:
            f.write(link)
        print(f"self-contained link ({len(link)} chars) written to {out_link}")


if __name__ == "__main__":
    main()
