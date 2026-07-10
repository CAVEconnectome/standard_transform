# Upgrading from standard_transform 1.x

`2.0` changes **results**, not the interface. Code written against 1.x keeps running —
the only hard renames are behind deprecation warnings, not errors — but the numbers it
produces change, and some of them change by design. This page lists what moved and how to
reproduce 1.x output exactly.

## TL;DR

- **Two things produce different numbers in 2.0:** the v1dd *transform* (a revised pia
  point) and the *streamlines* for both datasets (hand-drawn curve → data-derived field).
- **Nothing breaks at the API level:** the field is a drop-in subclass of `Streamline`,
  and the old `*_nm` / `*_vx` factory names still work (with a `DeprecationWarning`).
- **To reproduce 1.x results exactly, pin `version="1.4"`** on both the transform and the
  streamline.

## Reproducing 1.x results exactly

```python
from standard_transform import v1dd_ds, minnie_ds

# v1dd: pin BOTH tracks to 1.4 (its affine changed in 2.0, see below)
tf = v1dd_ds.transform(version="1.4")
sl = v1dd_ds.streamline(version="1.4")

# minnie65: the affine is unchanged, so only the streamline needs pinning
tf_m = minnie_ds.transform()                 # identical to 1.x
sl_m = minnie_ds.streamline(version="1.4")   # the old hand-drawn curve
```

`available_versions("v1dd")` (or `dataset.available_versions()`) lists every version and
which one is latest. See [Versioning & reproducibility](concepts/versioning.md).

## What changed in results

### v1dd transform (affine) — new default frame

The v1dd affine advanced to `2.0`: a **revised pia point**, which shifts oriented depth by
~39 µm. So `v1dd_ds.transform()` / `v1dd_transform()` now returns the 2.0 frame by default,
and any depth or transformed coordinate differs from 1.x by that offset. Pin
`version="1.4"` to recover the old frame. **Minnie65's affine did not change** — its
transformed points are identical to 1.x.

### Streamlines — now data-derived fields (both datasets)

`v1dd_ds.streamline()` and `minnie_ds.streamline()` now return a **`StreamlineField`** — a
spatially-varying, curl-free field fit from thousands of apical dendrites — instead of the
single hand-drawn curve. It is a subclass of `Streamline`, so every method
(`streamline_at`, `radial_distance`, `depth_along`, `radial_points`, `transformer`) works
unchanged; only the values differ (and are more accurate — see
[Streamline field: method](streamline-field-method.md)). Pin `version="1.4"` for the old
curve.

If you work with single neurons, note the intended field workflow: integrate the field
once at the soma and apply that one curve across the arbor, rather than re-deriving the
axis per vertex —

```python
sl = v1dd_ds.streamline().streamline_at_point(soma_xyz)  # a plain Streamline
verts_straight = sl.radial_points(soma_xyz, skel.vertices)
```

See the [per-neuron workflow](guides/per-neuron-streamline.md).

## Renamed factory API (`resolution=`)

The `*_nm` / `*_vx` factories and `Dataset` accessors are replaced by a single
`resolution=` argument (`"nm"`, `"vx"`, or an explicit `[x, y, z]`). The old names still
work but emit a `DeprecationWarning` and will be removed in a future release.

| 1.x | 2.0 |
|---|---|
| `minnie_transform_nm()` | `minnie_transform("nm")` |
| `minnie_transform_vx()` | `minnie_transform("vx")` |
| `v1dd_transform_nm()` / `_vx()` | `v1dd_transform("nm")` / `("vx")` |
| `minnie_streamline_nm()` / `_vx()` | `minnie_streamline("nm")` / `("vx")` |
| `v1dd_streamline_nm()` / `_vx()` | `v1dd_streamline("nm")` / `("vx")` |
| `minnie_ds.transform_nm` / `.transform_vx` | `minnie_ds.transform("nm")` / `("vx")` |
| `minnie_ds.transform_res(res)` | `minnie_ds.transform(res)` |
| `minnie_ds.streamline_nm` / `.streamline_vx` | `minnie_ds.streamline("nm")` / `("vx")` |
| `minnie_ds.streamline_res(res)` | `minnie_ds.streamline(res)` |

An explicit voxel resolution still works: `minnie_transform([8, 8, 40])`.

## Morphology object transforms are deprecated

Transforming MeshParty **objects** in place (`tform.apply_skeleton`,
`tform.apply_meshwork_vertices`, `streamline.transform_skeleton_vertices`,
`transform_meshwork_annotations`, …) is deprecated. That capability is moving to
[Ossify](https://csdashm.com/ossify/), where the transformation is passed to the object.
`standard_transform` now focuses on **coordinate arrays and dataframes** — transform the
vertex/annotation arrays directly (e.g. with `radial_points`) instead. The object methods
still run but warn; see [Morphology objects](guides/skeletons-meshworks.md).

## Recommended entry points

If you are still importing the module-level factories directly, the `Dataset` singletons
are now the recommended surface and bundle a transform + streamline together:

```python
from standard_transform import minnie_ds, v1dd_ds

pts_um = minnie_ds.transform().apply(xyz_nm)
d      = v1dd_ds.streamline().radial_distance(xyz0_nm, xyz1_nm)
```
