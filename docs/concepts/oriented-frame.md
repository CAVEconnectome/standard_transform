# The oriented frame

## The problem

An EM dataset is a volume of tissue imaged on a microscope. Its native coordinate
system — voxels, or nanometers derived from them — is aligned to the imaging setup,
not to the brain. Two things are usually "off":

- **Units and scale.** Coordinates come in voxels (with anisotropic voxel sizes,
  e.g. `[4, 4, 40]` nm for Minnie65) or in nanometers. Neither is convenient for
  thinking about distances in tissue.
- **Orientation.** The pial surface — the top of cortex — is not axis-aligned. The
  cortical "down" direction (pia → white matter) points at some arbitrary angle
  relative to the volume's axes, and it can be slightly different from one dataset
  to the next.

If you want to ask a simple anatomical question — *how deep below the pia is this
synapse?* — you cannot just read off a coordinate, because no single native axis
corresponds to depth.

## What the transform does

`standard_transform` applies a fixed sequence of **affine operations**
(rotation → translation → scaling) that lands every dataset in the same frame:

- **microns**, so distances are directly interpretable;
- the **y-axis oriented from pia to white matter** — increasing `y` means deeper
  into cortex;
- the **pial surface at `y ≈ 0`**, so `y` reads directly as cortical depth.

The x and z axes span the plane parallel to the pial surface (the laminar plane).
Because every dataset is mapped into this same convention, analyses and figures
built on top of it are comparable across datasets without per-dataset bookkeeping.

The recipe for each dataset is pre-baked — you don't fit anything. Each is a small
`TransformSequence` of primitives (a rotation derived from the dataset's measured
"up" vector, a translation that puts pia at `y = 0`, and a scaling to microns).
See [Datasets](../datasets.md) for the specific constants.

## Input units: the `resolution` argument

You tell the transform what units your points are in with `resolution`:

- `resolution="nm"` (the default) — **nanometers**.
- `resolution="vx"` — **voxels**, using the dataset's native voxel size.
- `resolution=[x, y, z]` — an explicit voxel resolution; the transform scales to
  nanometers first, then applies the shared recipe.

```python
minnie_ds.transform()                    # nm (default)
minnie_ds.transform("vx")                # native voxels
minnie_ds.transform([8, 8, 40])          # explicit resolution
```

All choices produce the same oriented microns for equivalent points, so results are
unit-invariant — the only difference is what you put in.

## It's invertible

A `TransformSequence` runs its primitives forward under `apply` and in reverse
(each inverted) under `invert`. So you can always take a point you've computed in
the oriented frame and recover its original native coordinate — useful for, say,
turning an analysis result back into a location you can look up in the dataset.

```python
pts_um   = tform.apply(pts_nm)     # native → oriented microns
pts_back = tform.invert(pts_um)    # oriented microns → native
```

## Next

The oriented frame gives you a straight, global depth axis. But cortex curves, and
a single straight axis is not enough to measure depth and lateral distance
accurately everywhere — that's what [streamlines](depth-and-radial-distance.md)
are for.
