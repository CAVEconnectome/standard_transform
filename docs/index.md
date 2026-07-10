# standard_transform

**Orient and scale points in EM connectomics datasets consistently and easily.**

When you work with electron-microscopy (EM) reconstructions of cortex, the raw
coordinate frame of the dataset rarely matches the frame you want to reason in.
Voxel or nanometer coordinates are tied to how the tissue happened to sit on the
microscope, not to the anatomy. `standard_transform` provides **pre-baked
transforms** that move points from a dataset's native space into a single
consistent frame:

- units are **microns**,
- the **y-axis runs from pia to white matter**, and
- the **pial surface sits at approximately `y = 0`**.

On top of that frame it also provides **streamlines** — a curvilinear depth axis
that follows the natural pia-to-white-matter direction as it bends across the
volume — so you can separate *cortical depth* from *lateral (radial) distance*
even where cortex is curved.

## Who this is for

Anyone analyzing the Minnie65 or V1dd EM datasets (or similar CAVE data) who wants
points and dataframes expressed in an anatomically meaningful, dataset-independent
frame — for comparing depths, measuring radial distances, or straightening a
neuron's coordinates along the cortical axis.

## Install

```bash
pip install standard-transform
```

## Quickstart

```python
from standard_transform import minnie_ds, v1dd_ds

# Transform an n x 3 array of points into oriented microns.
# resolution defaults to "nm"; pass "vx" or an [x, y, z] list for other units.
pts_um = minnie_ds.transform().apply(xyz_nm)

# Just the cortical depth (microns below pia)
depth = minnie_ds.transform().apply_project("y", xyz_nm)

# Radial (in-plane) distance between two points, following the local streamline
d = v1dd_ds.streamline().radial_distance(xyz0_nm, xyz1_nm)
```

`minnie_ds` and `v1dd_ds` bundle a transform and a streamline together and are the
recommended entry points. Both `.transform(resolution="nm", version=None)` and
`.streamline(resolution="nm", version=None)` take the input **resolution** (`"nm"`,
`"vx"`, or an `[x, y, z]` list) and an optional **version**. See
[Datasets](datasets.md) for what each bundle contains and
[Versioning & reproducibility](concepts/versioning.md) for `version=`.

## Where to go next

- **New here?** Start with [Concepts → The oriented frame](concepts/oriented-frame.md)
  to build the mental model, then [Depth vs. radial distance](concepts/depth-and-radial-distance.md).
- **Have a task in mind?** Jump to the [Guides](guides/transform-points.md).
- **Using streamlines?** For both v1dd and Minnie65 the default is now a data-derived,
  spatially-varying field — read [Streamlines vs. fields](concepts/streamlines-vs-field.md)
  for what changed and why, and the [per-neuron workflow](guides/per-neuron-streamline.md)
  for the recommended way to apply one to a single cell.
- **Reference:** the full [API reference](reference/transforms.md) is generated from
  the source docstrings.
