# Streamlines vs. fields

There are two ways `standard_transform` represents the curvilinear depth axis. They
share an interface but differ in how much they can adapt to the tissue.

## A fixed `Streamline`

A `Streamline` stores **one curve** and applies its shape everywhere: to find the
streamline through any point, it simply **translates** that single curve so it
passes through the point. The shape is identical across the whole volume — only its
position shifts.

This is exactly right when the pia → white matter direction is roughly uniform
across the region you care about. It is what the original, hand-drawn streamlines
provided, and it is still a good approximation over a limited area.

Its limitation: a single curve cannot represent a *spatially-varying* axis. If the
cortical "down" direction genuinely differs between one side of the dataset and the
other (curvature, or regional alignment drift), one fixed shape is accurate on one
side and progressively wrong on the other.

## A spatially-varying `StreamlineField`

A `StreamlineField` (a subclass of `Streamline`, so it is a drop-in replacement)
instead stores a **3D grid of local orientation vectors** — the pia → white matter
direction at every location — estimated from many neurons. The streamline through a
point is obtained by **integrating that field**, so its shape *varies across the
volume*. A field that happens to be uniform in x and z reduces exactly to a single
`Streamline`.

Because the field is estimated directly from data, it captures local curvature and
absorbs regional alignment drift that no single fixed curve can. How it is built and
validated is described in [Streamline field: method](../streamline-field-method.md).

## What changed (and how to opt out)

As of the current release, both the v1dd and Minnie65 streamline accessors return a
`StreamlineField` **by default**:

```python
from standard_transform import minnie_ds, v1dd_ds

sl     = v1dd_ds.streamline()              # data-derived tangent field (latest, default)
sl_old = v1dd_ds.streamline(version="1.4") # original hand-drawn single streamline
sl_m   = minnie_ds.streamline()            # Minnie65 field (latest, default)
```

This is a **breaking change in results** — computed depths and radial distances
differ from the old hand-drawn streamline — but **not a breaking change in
interface**. Every method (`streamline_at`, `radial_distance`, `depth_along`,
`radial_points`) works unchanged, because `StreamlineField` inherits them. Pin
`version="1.4"` to recover the previous behavior exactly — see
[Versioning & reproducibility](versioning.md).

!!! note "Which datasets have a field?"
    Both shipped datasets — v1dd and Minnie65 — default to a data-derived field. The
    accessors always return a `Streamline`-compatible object regardless, so your code
    does not need to branch. See [Datasets](../datasets.md) for the current status of
    each.

## The per-neuron workflow

The intended way to use a field for a single neuron is **not** to re-derive the
axis at every vertex. Instead, integrate the field **once at the cell body** to get
the one streamline appropriate to that soma, then apply that fixed curve across the
whole arbor:

```python
sl = v1dd_ds.streamline().streamline_at_point(soma_xyz)  # a plain Streamline
verts_straight = sl.radial_points(soma_xyz, skel.vertices)
```

This measures the arbor *against its soma's cortical axis* without assuming the
dendrites themselves follow the field — which they often don't (for example,
inverted layer-6 apical dendrites). See the
[per-neuron guide](../guides/per-neuron-streamline.md) for the full walkthrough.
