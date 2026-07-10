# standard_transform

Orient and scale points in EM connectomics datasets consistently and easily.

📖 **Full documentation: https://caveconnectome.github.io/standard_transform/**

When working with EM data, the orientation of the dataset often does not match the
orientation you want to reason in. For cortical data you usually want "down" to mean
the direction orthogonal to the pial surface, in microns, with the pia at `y ≈ 0`.
`standard_transform` provides pre-baked affine transforms for two datasets
(Minnie65 and V1dd) that map voxel or nanometer coordinates into that consistent
oriented frame, plus **streamlines** — a curvilinear depth axis — for separating
cortical depth from lateral (radial) distance even where cortex curves.

## Install

```bash
pip install standard-transform
```

## Quickstart

```python
from standard_transform import minnie_ds, v1dd_ds

# n x 3 points -> oriented microns (y = pia->white matter, pia at y≈0)
# resolution defaults to "nm"; pass "vx" or an [x, y, z] list for other units.
pts_um = minnie_ds.transform().apply(xyz_nm)

# just the cortical depth (microns below pia)
depth = minnie_ds.transform().apply_project("y", xyz_nm)

# radial (in-plane) distance following the local streamline
d = v1dd_ds.streamline().radial_distance(xyz0_nm, xyz1_nm)
```

`minnie_ds` and `v1dd_ds` bundle a transform and a streamline and are the
recommended entry points. Both `.transform(resolution="nm", version=None)` and
`.streamline(resolution="nm", version=None)` take the input resolution (`"nm"`,
`"vx"`, or an `[x, y, z]` list) and an optional version.

## What's in the box

- **Transforms** — `apply`, `apply_project`, `apply_dataframe`, and `invert`. Point
  units are selected with `resolution=` (`"nm"`, `"vx"`, or `[x, y, z]`). Accept
  `n x 3` arrays, pandas Series, and split `_x/_y/_z` dataframe columns.
- **Streamlines** — depth-along, radial distance, and cylindrical-like remapping
  along a curvilinear pia-to-white-matter axis.

> **Note:** transforming MeshParty morphology *objects* (skeletons/meshworks) is
> deprecated and moving to [Ossify](https://csdashm.com/ossify/), where you pass the
> transformation to the object. `standard_transform` focuses on coordinate arrays and
> dataframes.

## Streamline fields (breaking data change)

The v1dd and Minnie65 streamlines are now **data-derived, spatially-varying
`StreamlineField`s** by default, replacing the old single hand-drawn curves. This
changes computed *results* but not the interface — `StreamlineField` is a drop-in
subclass of `Streamline`. To recover the previous behavior:

```python
from standard_transform import minnie_ds, v1dd_ds
sl     = v1dd_ds.streamline()                 # data-derived field (latest, default)
sl_old = v1dd_ds.streamline(version="1.4")    # original hand-drawn streamline
sl_m   = minnie_ds.streamline()               # Minnie65 field (latest, default)
```

See the docs for the
[concept](https://caveconnectome.github.io/standard_transform/concepts/streamlines-vs-field/),
the [per-neuron workflow](https://caveconnectome.github.io/standard_transform/guides/per-neuron-streamline/),
and [how the field is built](https://caveconnectome.github.io/standard_transform/streamline-field-method/).
Older definitions stay reachable via `version=` — see
[Versioning & reproducibility](https://caveconnectome.github.io/standard_transform/concepts/versioning/).

## Documentation

The full guide — concepts, task-oriented tutorials, the streamline-field method, and
the API reference — lives at
**https://caveconnectome.github.io/standard_transform/**.

## Development

```bash
uv run poe test          # run tests with coverage
uv run poe doc-preview   # serve the docs locally with live reload
```
