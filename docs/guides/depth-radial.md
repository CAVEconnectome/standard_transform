# Cortical depth & radial distance

This guide covers the streamline measurements. For the *why*, see
[Depth vs. radial distance](../concepts/depth-and-radial-distance.md).

## Get a streamline

```python
from standard_transform import v1dd_ds
sl = v1dd_ds.streamline()          # nm input (default)
sl = v1dd_ds.streamline("vx")      # voxel input
```

Streamlines are tied to a transform (they use its y-axis orientation), so like
transforms they take a `resolution` (`"nm"`, `"vx"`, or an `[x, y, z]` list) telling
them what units the points you pass in are in.

All methods below take a `transform_points` flag (default `True`): inputs are
assumed to be in the **original pre-transform coordinates** and are transformed
first. Pass `transform_points=False` if your inputs are already in oriented microns.

## Depth along the streamline

Straight `y` is only an approximation of depth where cortex curves. To measure depth
as path length *down the curving axis* from the pia:

```python
d = sl.depth_along(xyz, depth_from=0)     # depth of each point along the streamline
```

To measure the depth *between* two points along the streamline:

```python
d = sl.depth_between(xyz0, xyz1)
```

## Radial (in-plane) distance

To measure how far apart two points are within the laminar plane, using the
streamline as the `d = 0` reference:

```python
d = sl.radial_distance(xyz0, xyz1)                       # xyz1 may be n x 3
d, angle = sl.radial_distance(xyz0, xyz1, return_angle=True)
```

`xyz0` is a single anchor point; `xyz1` can be many points. With `return_angle=True`
you also get the in-plane angle (radians from the x-axis).

## The streamline at a location

To get the x, z position of the streamline passing through an anchor `xyz` at one or
more depths `y` (in oriented microns):

```python
x, z = sl.streamline_at(xyz, y)
pts  = sl.streamline_at(xyz, y, return_as_point=True)   # n x 3 instead
```

## Cylindrical-like coordinates

`radial_points` maps points into a coordinate system where one axis is radial
distance from an anchor and the other is depth-along-the-streamline, preserving each
point's radial distance and angle:

```python
xyz_rad = sl.radial_points(xyz0, pts)
```

This is the operation used to "straighten" a neuron's coordinates along the cortical
axis. (Transforming MeshParty *objects* directly is deprecated in favor of
[Ossify](https://csdashm.com/ossify/) — see
[Morphology objects](skeletons-meshworks.md).)

## Unit invariance

The nm, voxel, and arbitrary-resolution variants all produce the same results for
equivalent inputs:

```python
root_nm = np.array([817335.0, 611523.0, 336240.0])
a = v1dd_ds.streamline("nm").radial_points(root_nm, pts_nm)
b = v1dd_ds.streamline("vx").radial_points(root_nm / [9, 9, 45], pts_vx)
# a == b
```
