# Transform points & dataframes

This guide covers moving raw coordinates into the [oriented frame](../concepts/oriented-frame.md).

## Get a transform

Use a [dataset](../datasets.md) bundle (recommended) and pass the `resolution` of your
input points — `"nm"` (default), `"vx"`, or an explicit `[x, y, z]`:

```python
from standard_transform import minnie_ds

tform = minnie_ds.transform()          # nm input (default)
tform = minnie_ds.transform("vx")      # native voxel input
```

Or import the module-level factory, which takes the same arguments:

```python
from standard_transform import minnie_transform
tform = minnie_transform(resolution="nm", version=None)
```

## `apply` — transform points

`apply` takes an `n x 3` array (or a single 3-vector, or a pandas Series of
3-vectors) and returns points in oriented microns:

```python
new_vertices = tform.apply(sk.vertices)   # n x 3 array in -> n x 3 array out
```

A single point returns a 1-D array; an `n x 3` array stays 2-D.

## `apply_project` — extract one axis

When you only want one dimension of the result — most commonly depth — use
`apply_project` with `"x"`, `"y"`, or `"z"` (equivalently `0`, `1`, `2`):

```python
depth = tform.apply_project("y", sk.vertices)   # cortical depth, microns below pia
```

For a single input point this returns a plain Python `float`; for many points it
returns an `n`-length array.

## `apply_dataframe` — work directly with dataframes

If your data is in a pandas dataframe, pass the **column name** and the dataframe
instead of slicing it yourself:

```python
pts_out = tform.apply_dataframe("pt_position", df)
```

Why not just `tform.apply(df["pt_position"])`? Because `apply_dataframe` also
handles **split position columns**. Connectomics dataframes often store a point as
three separate columns `pt_position_x`, `pt_position_y`, `pt_position_z`. Given the
prefix `"pt_position"`, `apply_dataframe` auto-detects the split layout and
reassembles the points seamlessly — you use the same call either way.

To project a single axis, pass `projection`:

```python
depth = tform.apply_dataframe("pt_position", df, projection="y")
```

!!! tip "Equivalent input forms"
    `apply` and `apply_dataframe` accept, and produce equal results for: an
    `n x 3` array, a pandas Series of 3-vectors, a single dataframe column of
    3-vectors, and a split `_x/_y/_z` column set. Use whichever your data is
    already in.

## `invert` — go back to native coordinates

Any transform round-trips. Given points in the oriented frame, recover the original
coordinates:

```python
pts_orig = tform.invert(pts_transformed)
```

## Arbitrary voxel resolution

Pass an explicit `[x, y, z]` resolution instead of `"nm"`/`"vx"`:

```python
minnie_ds.transform([8, 8, 40]).apply(pts)
# equivalently: minnie_transform([8, 8, 40])
```
