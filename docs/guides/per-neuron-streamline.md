# Per-neuron streamline workflow

When your streamline is a [`StreamlineField`](../concepts/streamlines-vs-field.md),
the recommended workflow for a single neuron is to **integrate the field once at the
cell body**, then apply that one fixed curve across the entire arbor.

!!! tip "Comparing cells to each other?"
    This workflow recenters each cell on its own soma. To place *multiple* cells in one
    shared, curvature-straightened frame — keeping their relative positions — use
    [Straightened streamline space](streamline-space.md) instead.

## Why not use the field directly per vertex?

The field describes the *typical* pia → white matter direction at each location,
estimated from many neurons. An individual neuron's dendrites do not necessarily
follow it — inverted layer-6 apical dendrites are the classic counterexample. If you
measured every vertex against the local field direction, you would be assuming the
arbor tracks the field, and you would lose the ability to see where it *departs* from
the local axis.

Instead, you want to measure the arbor **against a single reference axis: the
cortical axis at its soma.** That is what `streamline_at_point` gives you.

## The workflow

```python
from standard_transform import v1dd_ds

field = v1dd_ds.streamline()                    # a StreamlineField (default)
sl    = field.streamline_at_point(soma_xyz)     # a plain Streamline through the soma

# Now use `sl` like any fixed streamline across the whole neuron's coordinates:
verts_straight = sl.radial_points(soma_xyz, skel.vertices)   # straighten the arbor
depths         = sl.depth_along(skel.vertices)               # depth of each vertex
```

`streamline_at_point` also **remembers the soma** as the streamline's anchor, so you
can hand the straightening straight to a container that expects a single-argument
transform (e.g. Ossify's `Cell.transform`) with no wrapper:

```python
cell.transform(sl.transformer(), inplace=True)   # anchor = the stored soma
```

`streamline_at_point` integrates the field at the soma and returns an ordinary
`Streamline` (fixed shape), carrying the field's transform. From there everything in
[Depth & radial distance](depth-radial.md) applies unchanged.

!!! note "Transforming morphology objects"
    The examples above transform **coordinate arrays**. If you have a MeshParty
    object, note that the object-transform helpers are deprecated in favor of
    [Ossify](https://csdashm.com/ossify/) — see
    [Morphology objects](skeletons-meshworks.md).

## Checking confidence

The field is better determined in some regions than others (it depends on how much
data informed each grid cell). Query the per-node confidence near a location with:

```python
c = field.coverage_at(soma_xyz)
```

Low coverage is a flag that the streamline through that point rests on little data —
useful for filtering cells in poorly-sampled regions. See
[Streamline field: method](../streamline-field-method.md) for what "confidence"
means and how it is computed.
