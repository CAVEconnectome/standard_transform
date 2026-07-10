# Build a field for a new dataset

A [`StreamlineField`](../concepts/streamlines-vs-field.md) is estimated offline from
a collection of **pia → white matter paths** — one per neuron — and shipped as a
compressed `.npz`. This guide shows how to build one. For the reasoning behind each
step (depth band, precision, diffusion), read
[Streamline field: method](../streamline-field-method.md).

## The input: one tall path per neuron

Each path is an ordered `n x 3` array of points running along the cortical axis —
in practice, a tall single dendritic path from a deep soma up the apical trunk to a
tip in upper cortex, ordered tip → root. Where the paths come from (skeletons,
neuroglancer annotations) does not matter. Paths are in the dataset's **native
nanometers** unless you say otherwise.

## Build from a script (recommended)

The repo ships `scripts/build_streamline_field.py`, a `uv run` script that loads a
directory of per-neuron `.npy` files and writes the field `.npz`:

```bash
uv run python scripts/build_streamline_field.py \
    --paths /path/to/apical_paths \
    --dataset v1dd
```

Arguments:

| flag | meaning |
|---|---|
| `--paths` | directory of per-neuron `.npy` files (native nm; only x,y,z used) |
| `--dataset` | selects the nm transform and default output name (`v1dd` or `minnie`) |
| `--out` | output `.npz` path (default `standard_transform/data/<dataset>_streamline_field.npz`) |
| `--bin-size` | grid spacing X Y Z in microns (default `30 20 30`) |
| `--depth-band` | in-band depth range LO HI in microns (default `150 700`) |
| `--validate N` | hold out `N` paths, rebuild on the rest, and report held-out median lateral deviation as a QC check |

The shipped field is always built on **all** paths; `--validate` only builds a
throwaway field for quality reporting.

## Build programmatically

To build a field in your own code, call `streamline_field_from_paths` directly:

```python
from standard_transform import streamline_field_from_paths, v1dd_transform

field = streamline_field_from_paths(
    paths,                     # list of n_i x 3 arrays, native nm, tip -> root
    tform=v1dd_transform(),    # transform into oriented microns (nm input)
    bin_size=(30.0, 20.0, 30.0),
    depth_band=(150.0, 700.0),
)
field.to_npz("my_field.npz")
```

Useful parameters (see the [API reference](../reference/streamlines.md) for the full
list):

- `weights` — per-path or per-segment weights to down-weight unreliable structure
  (e.g. by Strahler index / inverse branch order).
- `depth_band` — only segments in this depth range inform the field, and the grid
  spans exactly this range; outside it, orientation is held constant.
- `smoothing_passes` / `smoothing_strength` — control the precision-weighted
  diffusion that fills empty cells and denoises.

## Persist and reload

```python
field.to_npz("my_field.npz")

from standard_transform import StreamlineField, v1dd_transform
field = StreamlineField.from_npz("my_field.npz", tform=v1dd_transform())
```

The grid is stored in **unit-agnostic post-transform microns**, so a single `.npz`
serves both the nm and voxel variants of a dataset — only the attached transform
differs. The transform is not saved in the file; you reattach one on load.

## Wire it into a dataset

Once a field `.npz` exists in `standard_transform/data/`, add a row to the streamline
version registry in `datasets.py` pointing at it and mark it latest — the dataset's
`streamline()` then returns it by default, while the prior `version="1.4"` hand-drawn
curve stays reachable. This mirrors how v1dd is wired; see
[Versioning & reproducibility](../concepts/versioning.md).
