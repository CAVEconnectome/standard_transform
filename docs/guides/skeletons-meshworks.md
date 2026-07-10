# Transforming morphology objects (deprecated)

!!! warning "Deprecated — moving to Ossify"
    The methods that transform [MeshParty](https://github.com/sdorkenw/MeshParty/)
    `Skeleton` and `Meshwork` objects in place — `apply_skeleton`,
    `apply_meshwork_vertices`, `apply_meshwork_annotations`, and the streamline
    `transform_skeleton_vertices` / `transform_meshwork_vertices` /
    `transform_meshwork_annotations` — are **deprecated** and will be removed in a
    future release. They emit a `DeprecationWarning`.

    Transforming morphology objects is moving to **[Ossify](https://csdashm.com/ossify/)**.
    The design there is inverted: instead of a transform reaching into an object and
    mutating it, you **pass the transformation to the object** and it transforms its
    own linked representations (mesh, skeleton, graph, annotations) consistently.

## What still works

`standard_transform` remains the right tool for **coordinate arrays**. Nothing about
point/dataframe transforms is deprecated:

```python
# Affine reorientation of a vertex array
verts_um = tform.apply(skel.vertices)

# Streamline straightening of a coordinate array (radial distance vs. depth)
verts_straight = sl.radial_points(soma_xyz, skel.vertices)
depths         = sl.depth_along(skel.vertices)
```

To transform a morphology object, compute the transformed coordinates this way and
hand them to your object — or use Ossify, which manages the object and its linked
layers for you.

## Ossify integration

Ossify's `Cell.transform` takes a single-argument callable `f(points) -> points`.
`Streamline.transformer(anchor)` returns exactly that — the streamline straightening
with the anchor bound in — so no `partial`/`lambda` is needed:

```python
cell.transform(sl.transformer(cell.s.root_location), inplace=True)
```

For the per-neuron field workflow, `streamline_at_point` remembers the soma, so the
anchor can be omitted:

```python
sl = v1dd_ds.streamline().streamline_at_point(soma_xyz)   # stores the anchor
cell.transform(sl.transformer(), inplace=True)
```

The transformer coerces its input to a float array, so it is robust to the
object-dtype arrays some containers hand out.
