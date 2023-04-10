# standard_transforms

Orient and scale points in EM datasets consistently and easily.

When working with EM data, often the orientation of the dataset does not match the desired orientation in space. For example, in cortical data you might want "down" to correspond to the direction orthogonal to the pial surface. This package includes prebaked affine transforms for two datasets, Minnie65 and V1dd, to convert from voxel or nanometer coordinates to a consistent oriented frame in microns.

Install via `pip install standard-transform`.

# Usage

## Transforms

At its simplest, we import the transform we want, initialize and object, and then are ready to rotate, scale, and translate away!
Let's start with going from nanometer space in Minnie to the oriented space.
We can make the pre-baked transform by importing one of the generating functions, in this case `minnie_transform_nm`.

```python
from standard_transform import minnie_transform_nm

tform_nm = minnie_transform_nm()
```

There are three main useful functions, `apply`, `apply_project`, and `apply_dataframe`.
All functions transform an `n x 3` array or pandas series with 3-element vectors to points in a new space, with the y-axis oriented along axis from pia to white matter and the units in microns and the pial surface at approximately y=0.
Using `apply` alone returns another `n x 3` array, while `apply_project` takes both an axis and points and returns just the values along that axis.
For example, if you have skeleton vertices in nm, you can produce transformed ones with:

```python
new_vertices = tform_nm.apply(sk.vertices)
```

while if you just want the depth:

```python
sk_depth = tform_nm.apply_project('y', sk.vertices)
```

These two functions can take either 3-element points, `n x 3` arrays, or a column of a pandas dataframe with 3-element vectors in each row.

The third function is specifically for use with dataframes, but offers a bit more flexibility. Instead of passing the series, you pass the column name and the dataframe itself.

```python
pts_out = tform_nm.apply_dataframe(column_name, df)
```

Why is this useful when you can just use `tform.apply(df[column_name])`?
It is often handy to work with dataframes with split position columns, where x, y, and z coordinates are in three separate columns.
If they are named `{column_name}_x`, `{column_name}_y`, and `{column_name}_z`, then the `apply_dataframe` function will autodetect this split position situation and act seamlessly to generate points out of them.
To get the projection functionality with the `apply_dataframe` method, pass it as an additional argument. e.g.
```python
pts_out_x = tform_nm.apply_dataframe(column_name, df, projection='x')
```

Finally, if you have points in the transformed space and want to back them out to the original coordinates, you can invert the transform of any TransformSequence.

```python
pts_orig = tform.invert(pts_transformed)
```

## Streamlines

Because mammalian cortex is strongly organized in a laminar manner, it's often useful to distinguish radial distance from distance along the depth axis.
However, when comparing the radial distance of points at different depths, the situation is complicated because the orientation of the depth axis, which we call a "streamline" is curvilinear, and changes with depth.
The Streamline class helps with computing radial distances between arbitrary locations, as well as to store particular streamlines in a consistent way.

Currently we just offer streamlines for the v1dd dataset, but a minnie65 one is forthcoming.
Like transforms, you must specify if the original data is in nm or voxels, and they are associated with a particular transform in order to use its y-axis orientation.
```python
from standard_transform import v1dd_streamline_nm, v1dd_streamline_vx
```

A Streamline object has a few key functions.
To get the x,z locations of a streamline passing through a core point `xyz` at a depth or depths `y` in the post-transform space:
```python
streamline.streamline_at(xyz, y)
```
The argument `return_as_point` can be set to `True` to return an nx3-array instead of the x, z components separately. 

To translate the points defining the streamline to intersect with a point xyz in the pre-transform space:
```python
streamline.streamline_points_tform(xyz)
```

To compute the radial distance in the post-transform space from two points in the pre-transform space:
```python
streamline.radial_distance(xyz0, xyz1)
``` 

One can also get the curvilinear distance, or depth along the streamline of a point in the pre or post-transform space and y=0 in the post-transform space:
```python
streamline.depth_along(xyz, depth_from=0)
```

Similarly, you can put all of these together to get a point in the curvilienar space of the streamline, akin to cylindrical coordinates, where the x axis is radial distance and the y axis is depth along the streamline. For each point, an equivalent location in x,z space is found with the same radial distance and angle as the original x and z.
```python
xyz_rad = streamline.radial_points(xyz0, pts)
```

## Datasets

Datasets are a convenient way to store a transform and streamline together, and are the easiest way to get started.
There are two datasets currently available, `minnie65` and `v1dd`.
To get a dataset, simply import it and initialize it.
Datasets have a transform and streamline, and can be used to transform points or get streamline information.
For example, to transform a particular location in v1dd 
```python
from standard_transform import minnie65_ds, v1dd_ds

v1dd_ds.transform_nm.apply(xyz)
```

Or to get the streamline at a particular point:
```python
v1dd_ds.streamline_nm.radial_distance(xyz0, xyz1)
```

A particular resolution can be used by passing it as an argument to `dataset.transform_res(voxel_resolution)` or `dataset.streamline_res(voxel_resolution)`.

Note: Currently a data-driven streamline is only available for v1dd.
Minnie65 is using a placeholder "identity streamline" that is just a straight line in the y direction.

### Minnie65

* `minnie_transform_nm` : Transform from nanometer units in the original Minnie65 space to microns in a space where the pial surface is flat in x and z along y=0.

* `minnie_transform_vx` : Transform from voxel units in the original Minnie65 space to microns in a space where the pial surface is flat in x and z along y=0. By default, `minnie_transform_vx()` assumes a voxel size of `[4,4,40]` nm/voxel, but specifying a voxel resolution (for example, with `minnie_transform_vx(voxel_resolution=[8,8,40])`) will use a different scale.

Both functions will also take dataframe columns, for example `tform.apply(df['pt_position])`.

### V1dd

* `v1dd_transform_nm` : Transform from nanometer units in the original V1dd space to microns in a space where the pial surface is flat in x and z along y=0.

* `v1dd_transform_vx` : Transform from voxel units in the original Minnie65 space to microns in a space where the pial surface is flat in x and z along y=0. By default, `v1dd_transform_vx()` assumes a voxel size of `[9,9,45]` nm/voxel, but specifying a voxel resolution (for example, with `v1dd_transform_vx(voxel_resolution=[1000, 1000, 1000])`) will use a different scale.

### Identity

* `identity_transform` : This transform returns the input data unchanged (although perhaps axis-projected), but can be useful for compatability purposes.
