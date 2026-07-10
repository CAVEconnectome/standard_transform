# Datasets

A **`Dataset`** bundles a transform and a streamline together and is the easiest way
to get started. Two are provided as ready-to-use singletons: `minnie_ds` and
`v1dd_ds`.

```python
from standard_transform import minnie_ds, v1dd_ds

v1dd_ds.transform().apply(xyz_nm)                   # transform points (nm default)
v1dd_ds.streamline().radial_distance(xyz0, xyz1)    # streamline query
```

Each dataset exposes two methods (plus the module-level factories `minnie_transform`,
`v1dd_transform`, `minnie_streamline`, `v1dd_streamline`, which take the same
arguments):

- `transform(resolution="nm", version=None)` — a [transform](concepts/oriented-frame.md).
- `streamline(resolution="nm", version=None)` — a
  [streamline](concepts/depth-and-radial-distance.md) (built lazily and cached).

`resolution` is `"nm"` (default), `"vx"` (native voxel size), or an explicit
`[x, y, z]`. `version` pins a historical definition; see
[Versioning & reproducibility](concepts/versioning.md).

## Minnie65

Native voxel resolution `[4, 4, 40]` nm.

- `minnie_ds.transform()` — nanometers → oriented microns (pia flat in x,z at y≈0).
- `minnie_ds.transform("vx")` — native voxels; `minnie_ds.transform([8, 8, 40])` for
  a different scale.
- `minnie_ds.streamline(...)` — returns a data-derived, spatially-varying
  [`StreamlineField`](concepts/streamlines-vs-field.md) **by default** (depth band
  `[150, 650]` µm). Pass `version="1.4"` for the original hand-drawn single streamline.

## V1dd

Native voxel resolution `[9, 9, 45]` nm.

- `v1dd_ds.transform(...)` — same as Minnie65 with the v1dd recipe and `[9, 9, 45]`
  native voxels.
- `v1dd_ds.streamline(...)` — returns a data-derived, spatially-varying
  [`StreamlineField`](concepts/streamlines-vs-field.md) **by default** (depth band
  `[150, 700]` µm). Pass `version="1.4"` for the original hand-drawn single streamline.

## Streamline field status

| dataset | default streamline |
|---|---|
| V1dd | data-derived `StreamlineField` (pass `version="1.4"` for the hand-drawn curve) |
| Minnie65 | data-derived `StreamlineField` (pass `version="1.4"` for the hand-drawn curve) |

Both datasets now default to a data-derived, spatially-varying field. The
field-building procedure is dataset-agnostic and described in
[Streamline field: method](streamline-field-method.md).

## Identity

- `identity_transform` — returns the input unchanged (optionally axis-projected).
  Useful for compatibility when a code path expects a transform object.
