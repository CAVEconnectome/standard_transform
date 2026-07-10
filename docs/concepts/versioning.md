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

For example, in the `2.0` release the v1dd and Minnie65 streamlines change (hand-drawn
→ field) but the affine numbers do not, so each transform stays at its `1.4` version
while the streamline advances to `2.0`.

## Package-version-anchored labels

A version label **is the release in which that definition became the default**. So
`version="1.4"` means "the definition shipped as default through the 1.4 line," and
`version="2.0"` means "the definition that became default in 2.0." Asking for a label
reproduces exactly what that release produced.

| dataset | track | version | what it is |
|---|---|---|---|
| v1dd | transform | `1.4` (latest) | the affine frame (unchanged) |
| v1dd | streamline | `1.4` | original hand-drawn single streamline |
| v1dd | streamline | `2.0` (latest) | data-derived spatially-varying field |
| minnie65 | transform | `1.4` (latest) | the affine frame |
| minnie65 | streamline | `1.4` | original hand-drawn single streamline |
| minnie65 | streamline | `2.0` (latest) | data-derived spatially-varying field |

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
tf.version          # -> "1.4"
v1dd_ds.streamline().version   # -> "2.0"
```

## Inspecting what's available

`available_versions(dataset)` (also `dataset.available_versions()`) lists both tracks,
their versions, and which release introduced each, plus the current latest:

```python
from standard_transform import available_versions

available_versions("v1dd")
# {
#   "transform":  {"latest": "1.4", "versions": {"1.4": "1.4"}},
#   "streamline": {"latest": "2.0", "versions": {"1.4": "1.4", "2.0": "2.0"}},
# }
```
