# standard_transforms

Orient and scale points in EM datasets the same way!

When working with EM data, often the orientation of the dataset does not match the desired orientation in space. For example, in cortical data you might want "down" to correspond to the direction orthogonal to the pial surface. This package includes prebaked affine transforms for two datasets, Minnie65 and v1dd, to convert from voxel or nanometer coordinates to a consistent oriented frame in microns.

Install via `pip install standard-transform`.

## Usage

At its simplest, we import the transform we want, initialize and object, and then are ready to rotate, scale, and translate away!
Let's start with going from nanometer space in Minnie to the oriented space.
We can make the pre-baked transform by importing one of the generating functions, in this case `minnie_transform_nm`.

```python
from standard_transform import minnie_transform_nm

tform_nm = minnie_transform_nm()
```

There are two main useful functions, `apply` and `apply_project`.
Both functions transform an `n x 3` array or pandas series with 3-element vectors to points in the new space, with the y-axis oriented along axis from pia to white matter and the units in microns.
Using `apply` alone returns another `n x 3` array, while `apply_project` takes both an axis and points and returns just the values along that axis.
For example, if you have skeleton vertices in nm, you can produce transformed ones with:

```python
new_vertices = tform_nm.apply(sk.vertices)
```

while if you just want the depth:

```python
sk_depth = tform_nm.apply_project('y', sk.vertices)
```

## Available transforms

There are four transforms currently available, two for each dataset.

### Minnie65

* `minnie_transform_nm` : Transform from nanometer units in the original Minnie65 space to microns in a space where the pial surface is flat in x and z along y=0.

* `minnie_transform_vx` : Transform from voxel units in the original Minnie65 space to microns in a space where the pial surface is flat in x and z along y=0. By default, `minnie_transform_vx()` assumes a voxel size of `[4,4,40]` nm/voxel, but specifying a voxel resolution (for example, with `minnie_transform_vx(voxel_resolution=[8,8,40])`) will use a different scale.

Both functions will also take dataframe columns, for example `tform.apply(df['pt_position])`.

### V1dd

* `v1dd_transform_nm` : Transform from nanometer units in the original V1dd space to microns in a space where the pial surface is flat in x and z along y=0.

* `v1dd_transform_vx` : Transform from voxel units in the original Minnie65 space to microns in a space where the pial surface is flat in x and z along y=0. By default, `v1dd_transform_vx()` assumes a voxel size of `[9,9,45]` nm/voxel, but specifying a voxel resolution (for example, with `minnie_transform_vx(voxel_resolution=[8,8,40])`) will use a different scale.
