# Versioning & reproducibility

## Outputs are tied to the package version

Everything `standard_transform` produces is defined by two things: the **rigid affine
numbers** (rotation, pia point, scaling) that build the oriented frame, and the
**streamline data**. Both are baked into the package. If either changes between
releases — as the v1dd streamline does when it moves from a hand-drawn curve to a
data-derived field — then every depth, radial distance, and transformed coordinate
changes too.

That makes results **version-dependent**: a figure made with one release is only
directly comparable to another if they used the same definitions. To keep old work
reproducible and to compare across releases, past definitions stay reachable by name.

## Two independent version tracks

The affine transform and the streamline are versioned **separately**, because they
change on different schedules:

- the **transform** track — the affine numbers;
- the **streamline** track — the depth-axis data.

For example, in the `2.0` release both datasets' streamlines change (hand-drawn → field).
Minnie65's affine numbers do not change, so its transform stays at `1.4` while its
streamline advances to `2.0`. v1dd's affine *does* change in `2.0` (a revised pia point,
a ~39 µm depth shift), so its transform advances to `2.0` too. The tracks move
independently — but they are not unrelated, which is why streamlines pin a transform (see
below).

## Streamlines pin the transform they were built in

A streamline's geometry lives in the oriented frame of a *specific* transform version —
a field's tangent grid is baked in that frame, and the hand curves were traced in it. So
each streamline version **pins** the transform version it belongs to, and is always built
against that transform, never merely "whatever is latest". This keeps a streamline valid
even when the affine changes: v1dd's `2.0` field pins transform `2.0` (its new frame),
while its `1.4` hand curve still pins transform `1.4`; Minnie65's field pins transform
`1.4` because its affine never changed. Field `.npz` files also stamp the transform frame
(plus the build method and λ) as provenance, and loading a field against a mismatched
transform warns. `available_versions` exposes each streamline version's pinned transform
under `transform_versions`.

## Package-version-anchored labels

A version label **is the release in which that definition became the default**. So
`version="1.4"` means "the definition shipped as default through the 1.4 line," and
`version="2.0"` means "the definition that became default in 2.0." Asking for a label
reproduces exactly what that release produced.

| dataset | track | version | pins transform | what it is |
|---|---|---|---|---|
| v1dd | transform | `1.4` | — | original affine frame |
| v1dd | transform | `2.0` (latest) | — | revised pia point (~39 µm depth shift) |
| v1dd | streamline | `1.4` | `1.4` | original hand-drawn single streamline |
| v1dd | streamline | `2.0` (latest) | `2.0` | data-derived spatially-varying field |
| minnie65 | transform | `1.4` (latest) | — | the affine frame (unchanged) |
| minnie65 | streamline | `1.4` | `1.4` | original hand-drawn single streamline |
| minnie65 | streamline | `2.0` (latest) | `1.4` | data-derived spatially-varying field |

## Pinning a version

Pass `version=` to `transform()` / `streamline()` (or the module factories). Omitting
it — or `version=None` — resolves to the latest for that track:

```python
from standard_transform import v1dd_ds

sl_latest = v1dd_ds.streamline()               # latest: the field (2.0)
sl_old    = v1dd_ds.streamline(version="1.4")  # reproduce the hand-drawn streamline

tf        = v1dd_ds.transform(version="1.4")   # pin the affine frame
```

An unknown label raises a `ValueError` listing the valid versions.

## Recording provenance

Every transform and streamline carries a `.version` attribute, so an analysis can
record exactly which definition produced its numbers:

```python
tf = v1dd_ds.transform()
tf.version          # -> "2.0"
sl = v1dd_ds.streamline()
sl.version                 # -> "2.0"  (streamline version)
sl.built_transform_version # -> "2.0"  (the transform frame it was built in)
```

## Inspecting what's available

`available_versions(dataset)` (also `dataset.available_versions()`) lists both tracks,
their versions, and which release introduced each, plus the current latest:

```python
from standard_transform import available_versions

available_versions("v1dd")
# {
#   "transform":  {"latest": "2.0", "versions": {"1.4": "1.4", "2.0": "2.0"}},
#   "streamline": {"latest": "2.0",
#                  "versions": {"1.4": "1.4", "2.0": "2.0"},
#                  "transform_versions": {"1.4": "1.4", "2.0": "2.0"}},
# }
```
