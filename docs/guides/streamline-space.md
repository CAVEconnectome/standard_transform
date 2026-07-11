# Straightened streamline space

A [`StreamlineField`](../concepts/streamlines-vs-field.md) can map points into a single
**global straightened coordinate system** and back. This "unfolds" the cortical curvature
so that streamlines become vertical lines, while every point keeps its *absolute* place in
one shared frame — which is what you want to **plot many cells at different cortical
locations together**.

This is different from the [per-neuron workflow](per-neuron-streamline.md)
(`radial_points` / `transformer`), which recenters *each* cell on its own soma and reports
distance-from-that-soma. That's right for comparing one cell's arbor to its own axis, but
it collapses every cell onto its own origin, so you can't see cells' relative positions.
Streamline space keeps them.

## The coordinate system

```python
field = v1dd_ds.streamline()               # a StreamlineField

uyw   = field.to_streamline_space(pts)      # (x, y, z) -> (u, y, w)
back  = field.from_streamline_space(uyw)    # exact inverse
```

Each point maps to `(u, y, w)`:

- `u, w` — the lateral position of the **streamline through the point**, evaluated at a
  fixed `reference_depth` (default the pial plane, `y = 0`). All points on one streamline
  share the same `(u, w)`, so streamlines become vertical lines.
- `y` — the depth, passed through unchanged.

`to_streamline_space` and `from_streamline_space` round-trip to sub-micron accuracy. This
is a well-defined *global* chart because the data-derived field is **curl-free** —
streamlines foliate the volume without crossing — so there is exactly one streamline
through each point and the unfolding is invertible. (It works on a `version="1.4"` hand
streamline too, but the interesting case is the field.)

## Plotting a cohort in one frame

```python
import matplotlib.pyplot as plt
from standard_transform import v1dd_ds

field = v1dd_ds.streamline()
fig, ax = plt.subplots()
for cell in cells:                                  # cell.vertices in native nm
    ss = field.to_streamline_space(cell.vertices)   # shared unfolded frame
    ax.scatter(ss[:, 0], ss[:, 1], s=1)             # unfolded x  vs  depth
ax.set_aspect("equal")
ax.invert_yaxis()                                   # pia at top
ax.set_xlabel("unfolded x (µm)")
ax.set_ylabel("depth (µm)")
```

Every cell lands at its true relative position with curvature straightened out — no
per-cell recentering. Pass `transform_points=False` if your points are already in oriented
microns rather than native nm.

## Anchoring to a non-negative frame

By default the unfolded lateral coordinates are in oriented microns and can be negative.
Pass `anchor=True` to shift them so the **smallest `x`/`z` in the data sit at 0**, giving a
non-negative frame with the data's minimum corner at `(0, pia, 0)` — convenient for
gridding or array-indexed heatmaps. Depth is left unchanged (pia stays the `y` origin, so
`y` is distance-from-pia). Use the same `anchor` for the inverse.

```python
ss   = field.to_streamline_space(pts, anchor=True)     # u, w >= 0; min corner at 0
back = field.from_streamline_space(ss, anchor=True)     # exact inverse
field.streamline_space_origin()                         # the (u_min, 0, w_min) offset
```

The origin is a **fixed, data-derived reference**: the shipped fields bake it from the
minimum unfolded position of the paths that built them (`field.build_data_anchor`, stored
in the `.npz`), so the anchored coordinate system is the same across sessions and does not
depend on the query points. Fields without a stored anchor (older files, or one built
without `compute_data_anchor=True`) fall back to the field's grid extent.

## Options and notes

- **`reference_depth`** — the depth at which streamlines are labeled (default `0.0`, the
  pia intercept). Any fixed value works; the forward and inverse maps must use the **same**
  one to round-trip. Choose a depth inside your region of interest if you'd rather label
  streamlines there than at pia.
- **Performance** — the mapping is vectorized (all points marched through the field
  together), so it is exact — no interpolation approximation — and fast enough to throw a
  whole cohort's vertices at in one call (tens of thousands of points in a fraction of a
  second).
- **Edges** — in data-poor margins the field is extrapolated (held constant), so the
  unfolding there is approximate; `field.coverage_at(pts)` flags those regions (≈0 where
  there is no data). See [Streamline field: method](../streamline-field-method.md).
