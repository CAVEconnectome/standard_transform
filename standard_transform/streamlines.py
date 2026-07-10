import numpy as np
from scipy import interpolate
from typing import Union
from .base import identity_transform
from .utils import is_list_like, warn_object_transform_deprecated


def _asfloat(a):
    """Coerce input to a float ndarray.

    Streamline methods feed coordinates to scipy interpolators, which reject
    object-dtype arrays. Callers (and tools like Ossify that pass ``DataFrame.values``)
    may hand us object arrays, lists, or pandas objects, so coerce defensively.
    """
    return np.asarray(a, dtype=float)


class Streamline(object):
    def __init__(self, points, tform=None, transform_points=True, anchor=None):
        """Build a streamline object to determine distances from a curving pia-to-white matter axis

        Parameters
        ----------
        points : nx3 array
            Set of points making up the streamline
        tform : TransformSequence, optional
            A transform sequence, by default the identity transform.
        transform_points : bool, optional
            If points are in the pre-transform coordinates use True, if in the post-transform coordinates use False. By default True.
        anchor : 3-element array, optional
            A default anchor point (pre-transform coordinates) for :meth:`transformer`.
            Usually set for you by :meth:`StreamlineField.streamline_at_point`. By default None.
        """
        if tform is None:
            tform = identity_transform()
        self._transform = tform
        self._anchor = anchor

        if transform_points:
            self._points = self._transform.apply(points)
        else:
            self._points = points

        curve_x = self._points[:, 0]
        curve_y = self._points[:, 1]
        curve_z = self._points[:, 2]
        self.x_interp = interpolate.interp1d(
            curve_y, curve_x, kind="linear", fill_value="extrapolate"
        )
        self.z_interp = interpolate.interp1d(
            curve_y, curve_z, kind="linear", fill_value="extrapolate"
        )

    def streamline_at(
        self, xyz0, y1, return_as_point=False
    ) -> Union[np.ndarray, tuple]:
        """Location of the streamline passing through point xyz0 at depth y1 in the post-transform space (usually microns)

        Parameters
        ----------
        xyz0 : 3-element array
            Anchor point through which the streamline passes
        y1 : float or array
            Depth or depths at which to find the streamline x and z points.
        return_as_point : bool, optional
            If True, return the x and z points as tuple, if False returns as an xyz point, by default False.

        Returns
        -------
        new_x, new_z : float or array
            X and z coordinate values for the streamline at the given depths.
        """
        xyz0 = _asfloat(xyz0)
        if xyz0.ndim != 1:
            raise ValueError(
                "streamline_at expects a single anchor point (a 3-element vector) for "
                "xyz0; to transform many points use radial_points() or depth_along()."
            )
        y1 = _asfloat(y1)
        y0 = xyz0[1]
        new_x = xyz0[0] + (self.x_interp(y1) - self.x_interp(y0))
        new_z = xyz0[2] + (self.z_interp(y1) - self.z_interp(y0))
        if return_as_point:
            return np.vstack([new_x, y1, new_z]).T
        else:
            return new_x, new_z

    def streamline_points_tform(self, xyz0_raw) -> np.ndarray:
        """Returns the original streamline points at a location that passes through the given point in the pre-transform space.

        Parameters
        ----------
        xyz0_raw : 3-element array
            Anchor point which the streamline passes through in the pre-transform space.

        Returns
        -------
        nx3 array
            Original streamline points transformed to pass through the given point.
        """
        xyz0 = self._transform.apply(_asfloat(xyz0_raw))
        y1 = self._points[:, 1]
        new_x, new_z = self.streamline_at(xyz0, y1)
        new_xyz = np.vstack([new_x, y1, new_z]).T
        return self._transform.invert(new_xyz)

    def radial_distance(
        self, xyz0, xyz1, transform_points=True, return_angle=False
    ) -> Union[np.ndarray, tuple]:
        """Find the distance between two points along the x-z plane using the streamline as d=0.

        Parameters
        ----------
        xyz0 : 3-element array
            First point coordinates
        xyz1 : nx3 element array
            One or more coordinates to find the radial distance to.
        transform_points : bool, optional
            If points are in the pre-transform coordinates use True, if in the post-transform coordinates use False. By default True.
        return_angle : bool, optional
            If True, return the angle between the two points, by default False. The angle is in radians and is measured from the x-axis.

        Returns
        -------
        float
            Distance in the x-z plane after accounting for streamline curvature.
        """
        xyz0 = _asfloat(xyz0)
        xyz1 = _asfloat(xyz1)
        if transform_points:
            xyz0 = self._transform.apply(xyz0)
            xyz1 = self._transform.apply(xyz1)
        if xyz0.ndim != 1:
            raise ValueError("xyz0 must be a single point")
        new_x, new_z = self.streamline_at(xyz0, xyz1[:, 1])
        d = np.sqrt((new_x - xyz1[:, 0]) ** 2 + (new_z - xyz1[:, 2]) ** 2)
        if return_angle:
            return d, np.arctan2(new_z - xyz1[:, 2], new_x - xyz1[:, 0]) + np.pi
        else:
            return d

    def radial_points(
        self,
        xyz0,
        xyz1,
        transform_points=True,
        depth_along_streamline=True,
        depth_from=0,
        delta=0.1,
    ) -> np.ndarray:
        """Returns points along the streamline at a given distance from an anchor point.

        Parameters
        ----------
        xyz0 : np.ndarray
            3 element point, the anchor of the radial distance measure.
        xyz1 : np.ndarray
            Nx3 array of points to transform.
        transform_points : bool, optional
            Choose whether to transform points before computing radial distances and depths, by default True
        depth_along_streamline : bool, optional
            If True, use path length along streamline as depth rather than the raw y coordindate, by default True
        depth_from : int, optional
            Sets the post-transform y coordinate to use for zero depth, by default 0.
        delta : float, optional
            Sets the resolution of the depth measurement. Smaller values are more accurate, by default 0.1

        Returns
        -------
        np.array
            Nx3 array of points with the same radial distance relative to xyz0 and depth as the input points.
            The relative angle of x and z is maintained from the original coordinates.
        """
        xyz0 = _asfloat(xyz0)
        xyz1 = _asfloat(xyz1)
        d, angle = self.radial_distance(
            xyz0,
            xyz1,
            transform_points=transform_points,
            return_angle=True,
        )
        if depth_along_streamline:
            y = self.depth_along(
                xyz1,
                transform_points=transform_points,
                delta=delta,
                depth_from=depth_from,
            )
        else:
            y = xyz1[:, 1]
        return (
            np.atleast_2d(d).T
            * np.vstack([np.cos(angle), np.zeros(len(y)), np.sin(angle)]).T
            + np.vstack([np.zeros(len(y)), y, np.zeros(len(y))]).T
        )

    def depth_between(self, xyz0, xyz1, delta=0.1, transform_points=True) -> np.ndarray:
        """Find the distance between two points along the depth axis using the streamline.

        Parameters
        ----------
        xyz0 : 3-element array
            First point coordinates
        xyz1 : 3-element array
            Second point coordinates
        delta : float, optional
            Step size for the integration in post-transform coordinates, by default 0.1.
        transform_points : bool, optional
            If points are in the pre-transform coordinates use True, if in the post-transform coordinates use False. By default True.

        Returns
        -------
        float
            Distance in the y-axis after accounting for streamline curvature.
        """
        xyz0 = _asfloat(xyz0)
        xyz1 = _asfloat(xyz1)
        if transform_points:
            xyz0 = self._transform.apply(xyz0)
            xyz1 = self._transform.apply(xyz1)

        if xyz0.ndim != 1:
            raise ValueError("xyz0 must be a single point")
        if xyz1.ndim != 2:
            xyz1 = np.atleast_2d(xyz1)

        all_ys = np.concatenate([[xyz0[1]], xyz1[:, 1]])
        ys = np.linspace(
            np.min(all_ys) + delta,
            np.max(all_ys),
            (np.floor(np.max(all_ys) - np.min(all_ys)) / delta).astype(int),
        )
        ycc = np.concatenate([all_ys, ys])
        yorder = np.argsort(ycc)
        xs, zs = self.streamline_at(xyz0, ycc[yorder])

        intermediate_pts = np.vstack([xs, ycc[yorder], zs]).T
        ds = np.cumsum(
            np.linalg.norm(intermediate_pts[0:-1, :] - intermediate_pts[1:, :], axis=1)
        )  # Cumulative distance of ordered points along the streamline
        ds = np.concatenate([[0], ds])

        base_d = ds[yorder.argsort()[0]]
        return ds[yorder.argsort()[1 : xyz1.shape[0] + 1]] - base_d

    def depth_along(
        self, xyz, depth_from=0, delta=0.1, transform_points=True
    ) -> np.ndarray:
        """Find the depth from pia along the streamline.

        Parameters
        ----------
        xyz : nx3 array
            Point coordinates
        delta : float, optional
            Step size for the integration in post-transform coordinates, by default 0.1.
        transform_points : bool, optional
            If points are in the pre-transform coordinates use True, if in the post-transform coordinates use False. By default True.

        Returns
        -------
        float
            Distance in the y-axis after accounting for streamline curvature.
        """
        xyz = _asfloat(xyz)
        if transform_points:
            xyz = self._transform.apply(xyz)
        xm = np.mean(xyz[:, 0])
        zm = np.mean(xyz[:, 2])
        base_pt = np.array([xm, depth_from, zm])
        sl_pts = self.streamline_at(base_pt, xyz[:, 1], return_as_point=True)
        depths = self.depth_between(base_pt, sl_pts, transform_points=False)
        return depths

    def transformer(
        self,
        anchor=None,
        *,
        transform_points=True,
        depth_along_streamline=True,
        depth_from=0,
        delta=0.1,
    ):
        """Return a single-argument callable that straightens points along this streamline.

        The returned function maps an ``(n, 3)`` array of points to an ``(n, 3)`` array
        (via :meth:`radial_points`) with the anchor bound in. That fits transform
        protocols that pass only the point array — e.g. Ossify's ``Cell.transform`` — so
        no ``functools.partial``/``lambda`` is needed at the call site:

        >>> cell.transform(sl.transformer(cell.s.root_location), inplace=True)

        For the per-neuron field workflow the anchor is remembered for you:

        >>> sl = field.streamline_at_point(soma)      # stores soma as the anchor
        >>> cell.transform(sl.transformer(), inplace=True)

        Parameters
        ----------
        anchor : 3-element array, optional
            Origin the straightening is measured from (typically a soma/root), in the
            same coordinate space as the points passed to the returned function. If
            omitted, uses the anchor stored on the streamline (e.g. set by
            :meth:`StreamlineField.streamline_at_point`); if neither is available a
            ``ValueError`` is raised.
        transform_points, depth_along_streamline, depth_from, delta
            Passed through to :meth:`radial_points`.

        Returns
        -------
        callable
            ``f(points) -> points`` mapping an ``(n, 3)`` array to an ``(n, 3)`` array.
        """
        if anchor is None:
            anchor = self._anchor
        if anchor is None:
            raise ValueError(
                "No anchor available for transformer(). Pass transformer(anchor=...), "
                "or build the streamline with a stored anchor via "
                "StreamlineField.streamline_at_point()."
            )
        anchor = _asfloat(anchor)

        def _straighten(points):
            return self.radial_points(
                anchor,
                points,
                transform_points=transform_points,
                depth_along_streamline=depth_along_streamline,
                depth_from=depth_from,
                delta=delta,
            )

        return _straighten

    def transform_skeleton_vertices(
        self, sk, root_loc=None, depth_from=0, delta=0.1, inplace=False
    ) -> "meshparty.skeleton.Skeleton":
        """Transforms skeleton vertices to the post-transform coordinate system via the radial points function.

        .. deprecated::
            Transforming morphology objects is moving to Ossify
            (https://csdashm.com/ossify/), where the transformation is passed to the
            object. Use :meth:`radial_points` on coordinate arrays instead.

        Parameters
        ----------
        sk : trimesh.skeleton.Skeleton
            Skeleton to transform
        root_loc : np.ndarray, optional
            3 element array of the root location, by default None. If none, selects the skeleton root.
        depth_from : numeric, optional
            Sets the post-transform y coordinate to use for zero depth, by default 0.
        delta : float, optional
            Sets the resolution of the depth measurement. Smaller values are more accurate, by default 0.1
        inplace : bool, optional
            If True, transform the vertices in place, by default False
        """
        warn_object_transform_deprecated("Streamline.transform_skeleton_vertices")
        if not inplace:
            sk = sk.copy()
        if root_loc is None:
            root_loc = sk.root_position
        verts_all = sk._rooted.vertices
        sk.vertices = self.radial_points(
            root_loc, verts_all, depth_from=depth_from, delta=delta
        )
        return sk

    def transform_meshwork_vertices(
        self, nrn, root_loc=None, depth_from=0, delta=0.1, inplace=False
    ) -> "meshparty.meshwork.Meshwork":
        """Transforms meshwork vertices to the post-transform coordinate system.

        .. deprecated::
            Transforming morphology objects is moving to Ossify
            (https://csdashm.com/ossify/), where the transformation is passed to the
            object. Use :meth:`radial_points` on coordinate arrays instead.

        Parameters
        ----------
        root_loc : np.ndarray, optional
            3 element array of the root location, by default None
        inplace : bool, optional
            If True, transform the vertices in place, by default False

        Returns
        -------
        np.ndarray
            Nx3 array of transformed vertices
        """
        warn_object_transform_deprecated("Streamline.transform_meshwork_vertices")
        if not inplace:
            nrn = nrn.copy()
        curr_mask = nrn.mesh.node_mask
        if root_loc is None:
            root_loc = nrn.skeleton.root_position
        nrn.mesh.vertices = self.radial_points(
            root_loc, nrn.mesh.vertices, depth_from=depth_from, delta=delta
        )
        nrn.skeleton.vertices = self.radial_points(
            root_loc, nrn.skeleton.vertices, depth_from=depth_from, delta=delta
        )
        nrn.apply_mask(curr_mask)
        return nrn

    def transform_meshwork_annotations(
        self, nrn, anno_dict, root_loc=None, depth_from=0, delta=0.1, inplace=False
    ) -> "meshparty.meshwork.Meshwork":
        """Transforms meshwork annotations to the post-transform coordinate system via the radial points function.

        .. deprecated::
            Transforming morphology objects is moving to Ossify
            (https://csdashm.com/ossify/), where the transformation is passed to the
            object. Use :meth:`radial_points` on coordinate arrays instead.

        Parameters
        ----------
        nrn : meshwork.Meshwork
            File to transform
        anno_dict : dict
            Dictionary whose keys are annotation tables in the meshwork and whose values are the columns to transform (string or list of strings)
        root_loc : array-like, optional
            location of the root for computing radial points, by default None. If None, uses the root of the meshwork skeleton.
        depth_from : int, optional
            Sets the post-transform y coordinate to use for zero depth, by default 0.
        delta : float, optional
            Sets the resolution of the depth measurement. Smaller values are more accurate, by default 0.1
        inplace : bool, optional
            If True, transform the vertices in place, by default False

        Returns
        -------
        meshwork
            Object with transformed annotation positions.
        """
        warn_object_transform_deprecated("Streamline.transform_meshwork_annotations")
        if not inplace:
            nrn = nrn.copy()

        if root_loc is None:
            root_loc = nrn.skeleton.root_position
        for tbl in anno_dict:
            vs = anno_dict[tbl]
            if not is_list_like(vs):
                vs = [vs]
            for v in vs:
                nrn.anno[tbl]._data[v] = self.radial_points(
                    root_loc,
                    np.vstack(nrn.anno[tbl]._data[v].values),
                    depth_from=depth_from,
                    delta=delta,
                ).tolist()
        return nrn


class StreamlineField(Streamline):
    def __init__(
        self,
        x_grid,
        y_grid,
        z_grid,
        tangents,
        confidence=None,
        tform=None,
        integration_step=None,
    ):
        """A spatially-varying streamline defined by a 3D grid of tangent vectors.

        Unlike :class:`Streamline`, which stores a single curve and translates it to
        pass through any anchor point, a ``StreamlineField`` stores the local direction
        of the pia-to-white-matter axis at every location. The streamline through a
        point is obtained by integrating that field, so the shape varies across the
        volume. A field that is uniform in x and z reproduces a single ``Streamline``.

        Grid values and all query points are in the post-transform (oriented micron)
        space. The local tangent is stored as ``(dx/dy, dz/dy)``: the volume is
        parametrized along depth ``y``, so the y-component of the tangent is implicitly 1.

        Parameters
        ----------
        x_grid, y_grid, z_grid : 1d arrays
            Strictly-ascending grid node coordinates along each axis, in post-transform
            microns. Each must have length >= 2.
        tangents : array, shape (nx, ny, nz, 2)
            Local tangent ``(dx/dy, dz/dy)`` at each grid node. No node may be empty
            (see :func:`streamline_field_from_paths`, which fills sparse cells).
        confidence : array, shape (nx, ny, nz), optional
            Per-node confidence, e.g. precision = count / variance, retained as a
            coverage signal (see :meth:`coverage_at`). By default None.
        tform : TransformSequence, optional
            Transform mapping input coordinates into the grid's post-transform space,
            by default the identity transform.
        integration_step : float, optional
            Step size in y (microns) used when integrating streamlines through the
            field. By default None, which uses the grid's own y-spacing (the field is
            piecewise-linear between nodes, so a finer step adds no accuracy).
        """
        if tform is None:
            tform = identity_transform()
        self._transform = tform
        self._anchor = None
        # Provenance, set from the .npz stamps by from_npz (None for an in-memory field):
        # the transform version whose oriented frame the grid was built in, and how the
        # field was regularized (method and, for the Laplace methods, the lambda used).
        self.built_transform_version = None
        self.build_method = None
        self.build_laplace_strength = None

        self._x_grid = np.asarray(x_grid, dtype=float)
        self._y_grid = np.asarray(y_grid, dtype=float)
        self._z_grid = np.asarray(z_grid, dtype=float)
        self._tangents = np.asarray(tangents, dtype=float)
        self._confidence = None if confidence is None else np.asarray(confidence, dtype=float)
        # The field is piecewise-linear between nodes, so integrating at the grid's
        # own y-spacing is exactly as accurate as a finer step and keeps each
        # streamline to a couple dozen field evaluations.
        if integration_step is None and len(self._y_grid) > 1:
            integration_step = float(np.median(np.diff(self._y_grid)))
        self._integration_step = integration_step if integration_step else 1.0
        self._y_range = (float(self._y_grid.min()), float(self._y_grid.max()))
        self._x_lo, self._x_hi = float(self._x_grid.min()), float(self._x_grid.max())
        self._z_lo, self._z_hi = float(self._z_grid.min()), float(self._z_grid.max())

        for name, grid in (("x", self._x_grid), ("y", self._y_grid), ("z", self._z_grid)):
            if len(grid) < 2:
                raise ValueError(
                    f"{name}_grid must have at least 2 nodes to interpolate; "
                    "widen the bounds or coarsen the bin size."
                )

        grid_pts = (self._x_grid, self._y_grid, self._z_grid)
        self._tx_interp = interpolate.RegularGridInterpolator(
            grid_pts, self._tangents[..., 0], bounds_error=False, fill_value=None
        )
        self._tz_interp = interpolate.RegularGridInterpolator(
            grid_pts, self._tangents[..., 1], bounds_error=False, fill_value=None
        )
        if self._confidence is not None:
            self._conf_interp = interpolate.RegularGridInterpolator(
                grid_pts, self._confidence, bounds_error=False, fill_value=0.0, method="nearest"
            )

    def __repr__(self):
        shape = self._tangents.shape[:3]
        return f"Tangent-field streamline on a {shape[0]}x{shape[1]}x{shape[2]} grid"

    def _tangent_at(self, x, y, z):
        # Clamp into the grid so orientation is held constant beyond the sampled band
        # (above pia / below white matter) rather than linearly extrapolated.
        x = min(max(x, self._x_lo), self._x_hi)
        y = min(max(y, self._y_range[0]), self._y_range[1])
        z = min(max(z, self._z_lo), self._z_hi)
        pt = np.array([[x, y, z]])
        return float(self._tx_interp(pt)[0]), float(self._tz_interp(pt)[0])

    def _march(self, x, y, z, y_target, step):
        """Integrate the field in y from (x, y, z) toward y_target (RK2 midpoint)."""
        xs, ys, zs = [x], [y], [z]
        if step == 0:
            return ys, xs, zs
        going_up = step > 0
        while (going_up and y < y_target) or (not going_up and y > y_target):
            dy = step
            if (going_up and y + dy > y_target) or (not going_up and y + dy < y_target):
                dy = y_target - y
            tx, tz = self._tangent_at(x, y, z)
            txm, tzm = self._tangent_at(x + tx * dy / 2.0, y + dy / 2.0, z + tz * dy / 2.0)
            x, z, y = x + txm * dy, z + tzm * dy, y + dy
            xs.append(x)
            ys.append(y)
            zs.append(z)
        return ys, xs, zs

    def _streamline_interp(self, xyz0, y_lo, y_hi):
        """Build x(y), z(y) interpolators for the streamline through xyz0 over [y_lo, y_hi]."""
        x0, y0, z0 = float(xyz0[0]), float(xyz0[1]), float(xyz0[2])
        step = self._integration_step
        ys_up, xs_up, zs_up = self._march(x0, y0, z0, y_hi, step)
        ys_dn, xs_dn, zs_dn = self._march(x0, y0, z0, y_lo, -step)
        ys = np.concatenate([ys_dn[::-1], ys_up[1:]])
        xs = np.concatenate([xs_dn[::-1], xs_up[1:]])
        zs = np.concatenate([zs_dn[::-1], zs_up[1:]])
        ys, uniq = np.unique(ys, return_index=True)
        xs, zs = xs[uniq], zs[uniq]
        x_interp = interpolate.interp1d(ys, xs, kind="linear", fill_value="extrapolate")
        z_interp = interpolate.interp1d(ys, zs, kind="linear", fill_value="extrapolate")
        return x_interp, z_interp

    def streamline_at(self, xyz0, y1, return_as_point=False):
        """Location of the streamline through xyz0 at depth y1, integrated through the field.

        Parameters
        ----------
        xyz0 : 3-element array
            Anchor point (post-transform space) through which the streamline passes.
        y1 : float or array
            Depth or depths at which to evaluate the streamline x and z.
        return_as_point : bool, optional
            If True return an nx3 array, else return (new_x, new_z), by default False.

        Returns
        -------
        new_x, new_z : float or array
            X and z coordinate values for the streamline at the given depths.
        """
        xyz0 = np.reshape(np.asarray(xyz0, dtype=float), 3)
        scalar = np.ndim(y1) == 0
        y1arr = np.atleast_1d(np.asarray(y1, dtype=float))
        step = self._integration_step
        y_lo = min(xyz0[1], y1arr.min()) - step
        y_hi = max(xyz0[1], y1arr.max()) + step
        x_interp, z_interp = self._streamline_interp(xyz0, y_lo, y_hi)
        new_x, new_z = x_interp(y1arr), z_interp(y1arr)
        if return_as_point:
            return np.vstack([new_x, y1arr, new_z]).T
        if scalar:
            return new_x[0], new_z[0]
        return new_x, new_z

    def streamline_points_tform(self, xyz0_raw) -> np.ndarray:
        """Streamline points passing through a pre-transform point, in pre-transform space.

        Parameters
        ----------
        xyz0_raw : 3-element array
            Anchor point which the streamline passes through in the pre-transform space.

        Returns
        -------
        nx3 array
            Streamline points (sampled at the grid depths) in the pre-transform space.
        """
        xyz0 = np.reshape(self._transform.apply(np.asarray(xyz0_raw, dtype=float)), 3)
        y_lo, y_hi = self._y_range
        x_interp, z_interp = self._streamline_interp(xyz0, y_lo, y_hi)
        ys = self._y_grid
        new_xyz = np.vstack([x_interp(ys), ys, z_interp(ys)]).T
        return self._transform.invert(new_xyz)

    def streamline_at_point(self, xyz, transform_points=True) -> Streamline:
        """Extract the single streamline through a point (e.g. a cell body) as a Streamline.

        This is the intended way to work with a neuron: integrate the field once at the
        soma, get back an ordinary fixed-shape :class:`Streamline`, and apply that one
        curve across the whole arbor. It deliberately does *not* re-derive the axis per
        vertex — every vertex is measured against the streamline appropriate to its cell
        body. That makes no assumption that the arbor's lateral extent follows the field
        (it generally does not).

        Parameters
        ----------
        xyz : 3-element array
            Anchor point, typically the soma location.
        transform_points : bool, optional
            If True, transform the anchor into the grid space first, by default True.

        Returns
        -------
        Streamline
            A fixed-shape streamline through the anchor, carrying this field's transform.
        """
        xyz_in = np.reshape(_asfloat(xyz), 3)
        if transform_points:
            # Anchor supplied in pre-transform coordinates; keep it as the stored
            # anchor so the returned streamline's transformer() (transform_points=True)
            # sees the same space.
            anchor_raw = xyz_in
            xyz_grid = np.reshape(self._transform.apply(xyz_in), 3)
        else:
            # Anchor already in grid (post-transform) space; recover a pre-transform
            # anchor for the returned streamline's transformer default.
            xyz_grid = xyz_in
            anchor_raw = np.reshape(self._transform.invert(xyz_in), 3)
        y_lo, y_hi = self._y_range
        x_interp, z_interp = self._streamline_interp(xyz_grid, y_lo, y_hi)
        ys = self._y_grid
        pts = np.vstack([x_interp(ys), ys, z_interp(ys)]).T
        return Streamline(
            pts, tform=self._transform, transform_points=False, anchor=anchor_raw
        )

    def coverage_at(self, xyz, transform_points=True) -> np.ndarray:
        """Per-node confidence (precision) that informed the field near each query point.

        Useful for flagging low-confidence regions (e.g. dataset edges) where the
        streamline rests on little or noisy data. Higher means better determined.

        Parameters
        ----------
        xyz : nx3 array
            Query points.
        transform_points : bool, optional
            If True, transform points into the grid space first, by default True.

        Returns
        -------
        np.ndarray
            Nearest-node confidence for each query point (zeros if none stored).
        """
        if self._confidence is None:
            return np.zeros(np.atleast_2d(xyz).shape[0])
        if transform_points:
            xyz = self._transform.apply(xyz)
        return self._conf_interp(np.atleast_2d(xyz))

    def to_npz(self, path, transform_version=None, method=None, laplace_strength=None) -> None:
        """Save the field grid to a compressed .npz.

        The transform itself is not stored (reattach one via :meth:`from_npz`), but the
        transform *version* the grid was built in is stamped as provenance so a stale
        file -- one attached to a different transform frame later -- can be detected on
        load. The build ``method`` and, for the Laplace methods, the ``laplace_strength``
        used are stamped too. By default the transform version is read from the build
        transform (``self._transform.version``).

        Parameters
        ----------
        path : str
            Output .npz path.
        transform_version : str, optional
            Version label of the transform whose oriented frame this grid lives in. By
            default read from the attached transform, or omitted if it has no version.
        method : str, optional
            The regularizer used to build the field (e.g. ``"laplace-fit"``).
        laplace_strength : float, optional
            The lambda used, for the Laplace methods (e.g. the CV-selected value).
        """
        if transform_version is None:
            transform_version = getattr(self._transform, "version", None)
        np.savez_compressed(
            path,
            x_grid=self._x_grid,
            y_grid=self._y_grid,
            z_grid=self._z_grid,
            tangents=self._tangents,
            confidence=(self._confidence if self._confidence is not None else np.empty(0)),
            integration_step=np.array([self._integration_step], dtype=float),
            transform_version=np.array("" if transform_version is None else str(transform_version)),
            build_method=np.array("" if method is None else str(method)),
            build_laplace_strength=np.array(
                np.nan if laplace_strength is None else float(laplace_strength)
            ),
        )

    @classmethod
    def from_npz(cls, path, tform=None) -> "StreamlineField":
        """Load a field saved by :meth:`to_npz`, attaching the given transform.

        The grid is in post-transform microns and is therefore unit-agnostic: the same
        file serves both nm and voxel variants -- only the attached transform differs.
        Any stamped provenance is exposed as ``built_transform_version``,
        ``build_method``, and ``build_laplace_strength`` (None for older, unstamped
        files).
        """
        with np.load(path) as dat:
            conf = dat["confidence"]
            field = cls(
                dat["x_grid"],
                dat["y_grid"],
                dat["z_grid"],
                dat["tangents"],
                confidence=conf if conf.size else None,
                tform=tform,
                integration_step=float(dat["integration_step"][0]),
            )
            if "transform_version" in dat:
                field.built_transform_version = str(dat["transform_version"]) or None
            if "build_method" in dat:
                field.build_method = str(dat["build_method"]) or None
            if "build_laplace_strength" in dat:
                val = float(dat["build_laplace_strength"])
                field.build_laplace_strength = None if np.isnan(val) else val
        return field


def _precision_smooth(mean_t, prec, n_passes, lam):
    """Precision-weighted diffusion that both fills empty nodes and denoises the field.

    Each pass replaces a node with a blend of itself and its 6 neighbors, weighted by
    precision (count / variance). Well-determined nodes barely move and anchor the
    result; noisy or empty nodes (low/zero precision) are pulled toward their better-
    determined neighbors. Precision itself diffuses so empty regions fill over passes.

    Parameters
    ----------
    mean_t : array, shape (nx, ny, nz, 2)
        Per-node mean tangent (zeros where empty).
    prec : array, shape (nx, ny, nz)
        Per-node precision (zero where empty).
    n_passes : int
        Number of diffusion passes.
    lam : float
        Neighbor coupling strength; larger smooths more.

    Returns
    -------
    (mean_t, prec) : smoothed tangents and diffused precision.
    """
    m = mean_t.copy().astype(float)
    p = prec.copy().astype(float)
    for _ in range(n_passes):
        nbr_pm = np.zeros_like(m)  # sum of neighbor precision * neighbor tangent
        nbr_p = np.zeros(p.shape)  # sum of neighbor precision
        for axis in range(3):
            for shift in (-1, 1):
                m_sh = np.roll(m, shift, axis=axis)
                p_sh = np.roll(p, shift, axis=axis).copy()
                edge = [slice(None)] * 3
                edge[axis] = 0 if shift == 1 else -1
                p_sh[tuple(edge)] = 0.0  # don't let np.roll wrap across the boundary
                nbr_pm += p_sh[..., None] * m_sh
                nbr_p += p_sh
        den = p + lam * nbr_p
        upd = den > 0
        m_new = m.copy()
        m_new[upd] = (p[upd][:, None] * m[upd] + lam * nbr_pm[upd]) / den[upd][:, None]
        m = m_new
        p = p + lam * nbr_p
    return m, p


def _hold_fill(field, data_mask, smooth_passes=8):
    """Extrapolate a fitted tangent field into data-poor cells by holding the trend.

    Each non-data cell is set to the tangent of its nearest in-data cell (a nearest-
    neighbour / Voronoi continuation), so the field *holds* the boundary trend outward
    rather than decaying toward vertical -- which is what the harmonic potential does at
    the margins and what collapses depth-integrated quantities there. (A Laplace
    relaxation would give the same "hold" in the limit but needs far more iterations than
    is worthwhile; nearest continuation is exact and cheap.) A few edge-safe neighbour-
    averaging passes then soften the Voronoi seams without reintroducing any decay --
    everything outside the data is already at a held value, so averaging preserves it.
    Data cells are never modified, so the curl-free fit is untouched where it is informed.

    Parameters
    ----------
    field : (nx, ny, nz, 2) array
        Tangent field; data cells carry the fitted values to be held.
    data_mask : (nx, ny, nz) bool array
        True where the cell had data (held fixed).
    smooth_passes : int
        Neighbour-averaging passes over the extrapolated cells to soften seams.

    Returns
    -------
    (nx, ny, nz, 2) array
        Field with data cells unchanged and non-data cells filled by held extrapolation.
    """
    from scipy.ndimage import distance_transform_edt

    empty = ~data_mask
    # nearest in-data cell index for every cell -> continue that tangent outward
    idx = distance_transform_edt(empty, return_indices=True)[1]
    f = field.copy()
    f[empty] = field[idx[0], idx[1], idx[2]][empty]

    for _ in range(smooth_passes):
        num = np.zeros_like(f)
        cnt = np.zeros(f.shape[:3])
        for axis in range(3):
            for shift in (-1, 1):
                sh = np.roll(f, shift, axis=axis)
                keep = np.ones(f.shape[:3])
                sl = [slice(None)] * 3
                sl[axis] = 0 if shift == 1 else -1
                keep[tuple(sl)] = 0.0  # rolled-in boundary plane wrapped -> drop it
                num += keep[..., None] * sh
                cnt += keep
        upd = empty & (cnt > 0)
        f[upd] = num[upd] / cnt[upd][:, None]
    return f


def _laplace_field(
    mean_t, prec, bs, mode="fit", lam_rel=0.3, deep_layers=1, bc_weight=1e3,
    bc_shallow_empirical=False, edge_extrapolation="hold",
):
    """Recover a tangent field as the gradient of a scalar depth potential (a PDE fit).

    Models the streamline field as the horizontal gradient of a depth potential
    ``phi = y + psi``: to first order the tangent ``(dx/dy, dz/dy)`` equals
    ``(psi_x, psi_z)``, an approximation that holds in the small-tangent regime here
    (apical axes are near-vertical). Solving for the single scalar ``psi`` -- rather than
    smoothing the two tangent components independently as :func:`_precision_smooth` does
    -- enforces integrability: the field is curl-free, so streamlines derive from one
    consistent depth function and cannot cross, and a Laplacian term couples across depth
    so the solution is harmonic where data is sparse.

    Both modes minimize the same energy
    ``sum_edges w_e (dpsi/dl - t_e)^2 + lam * sum_edges (dpsi/dl)^2`` over grid edges;
    only the data-weight mask ``w_e`` differs:

    - ``mode="fit"``  -- precision-weighted data term in *every* cell: a Laplace-
      *regularized* fit that tracks interior orientation and fills gaps harmonically.
    - ``mode="bc"``   -- data only on the deep boundary (empirical, precision-gated) with
      a flat shallow face (tangent 0, inherited from the pia flattening); pure Laplace in
      the interior. The parsimonious, boundary-driven model.

    Parameters
    ----------
    mean_t : (nx, ny, nz, 2) array
        Per-cell mean tangent ``(dx/dy, dz/dy)`` (zeros where empty).
    prec : (nx, ny, nz) array
        Per-cell precision (zero where empty); the data weight.
    bs : 3-tuple
        Grid spacing (hx, hy, hz) in microns.
    mode : {"fit", "bc"}
    lam_rel : float
        Smoothness weight relative to the median cell precision (``fit`` mode). Larger
        smooths more; too large washes out real depth structure (empty cells fill
        harmonically regardless, since their data weight is zero).
    deep_layers : int
        Number of deep (and shallow) y-layers used as boundaries in ``bc`` mode.
    bc_weight : float
        Boundary data weight in ``bc`` mode (>> interior, approximating a hard BC).
    edge_extrapolation : {"hold", "harmonic"}
        How the field behaves in data-poor cells (``fit`` mode only). ``"harmonic"`` keeps
        the raw potential solution, whose natural boundary condition relaxes the tangent
        toward vertical (zero) beyond the data -- which collapses depth-integrated
        quantities like displacement to ~0 at the margins. ``"hold"`` (default) instead
        continues the nearest in-data tangent outward (see :func:`_hold_fill`), leaving
        the data region's curl-free fit untouched but giving an approximately-right,
        non-collapsing extrapolation in data-poor regions. The extrapolated cells are no
        longer curl-free, but they carry no data anyway (flag them with ``coverage_at``).

    Returns
    -------
    (nx, ny, nz, 2) array
        The recovered tangent field ``(psi_x, psi_z)``.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    shape = tuple(prec.shape)
    nx, ny, nz = shape
    N = nx * ny * nz
    hx, hy, hz = float(bs[0]), float(bs[1]), float(bs[2])
    tx, tz = mean_t[..., 0].ravel(), mean_t[..., 1].ravel()
    pflat = prec.ravel()

    def fwd(axis, h):
        idx = np.arange(N).reshape(shape)
        if axis == 0:
            a, b = idx[:-1].ravel(), idx[1:].ravel()
        elif axis == 1:
            a, b = idx[:, :-1].ravel(), idx[:, 1:].ravel()
        else:
            a, b = idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel()
        e = a.size
        rows = np.concatenate([np.arange(e), np.arange(e)])
        cols = np.concatenate([b, a])
        vals = np.concatenate([np.full(e, 1.0 / h), np.full(e, -1.0 / h)])
        return sp.csr_matrix((vals, (rows, cols)), shape=(e, N)), a, b

    Gx, ax, bx = fwd(0, hx)
    Gy, _, _ = fwd(1, hy)
    Gz, az, bz = fwd(2, hz)

    def edge_wt_tgt(wc, tgt, a, b):
        wa, wb = wc[a], wc[b]
        denom = wa + wb
        we = 0.5 * denom
        te = np.divide(wa * tgt[a] + wb * tgt[b], denom, out=np.zeros_like(we), where=denom > 0)
        return we, te

    if mode == "fit":
        wc, tgt_x, tgt_z = pflat, tx, tz
        lam = lam_rel * (np.median(pflat[pflat > 0]) if np.any(pflat > 0) else 1.0)
    elif mode == "bc":
        jj = np.broadcast_to(np.arange(ny)[None, :, None], shape).ravel()
        wc = np.zeros(N)
        tgt_x, tgt_z = np.zeros(N), np.zeros(N)
        deep = (jj >= ny - deep_layers) & (pflat > 0)  # empirical, only where data exists
        wc[deep] = bc_weight
        tgt_x[deep], tgt_z[deep] = tx[deep], tz[deep]
        if bc_shallow_empirical:
            top = (jj < deep_layers) & (pflat > 0)  # empirical shallow face too
            wc[top] = bc_weight
            tgt_x[top], tgt_z[top] = tx[top], tz[top]
        else:
            top = jj < deep_layers  # flat shallow face (tangent 0)
            wc[top] = bc_weight
        lam = 1.0  # interior is pure Laplace; scale is arbitrary since bc_weight >> lam
    else:
        raise ValueError(f"mode must be 'fit' or 'bc', got {mode!r}.")

    wex, tex = edge_wt_tgt(wc, tgt_x, ax, bx)
    wez, tez = edge_wt_tgt(wc, tgt_z, az, bz)
    Wx, Wz = sp.diags(wex), sp.diags(wez)
    lap = Gx.T @ Gx + Gy.T @ Gy + Gz.T @ Gz
    A = (Gx.T @ Wx @ Gx + Gz.T @ Wz @ Gz + lam * lap).tocsr()
    rhs = Gx.T @ (wex * tex) + Gz.T @ (wez * tez)
    # Tiny Tikhonov term fixes the constant-potential null space; negligible on gradients.
    A = A + (1e-6 * A.diagonal().max()) * sp.identity(N, format="csr")
    psi = spsolve(A, rhs).reshape(shape)

    out = np.zeros_like(mean_t)
    out[..., 0] = np.gradient(psi, hx, axis=0)
    out[..., 1] = np.gradient(psi, hz, axis=2)

    if mode == "fit" and edge_extrapolation == "hold":
        # Replace the potential's relax-to-vertical extrapolation in data-poor cells with
        # a held continuation of the nearest fitted tangents. Data cells are unchanged, so
        # the curl-free fit is preserved exactly where it is informed by data.
        data_mask = prec > 0
        if data_mask.any() and not data_mask.all():
            out = _hold_fill(out, data_mask)
    return out


def streamline_field_from_paths(
    paths,
    weights=None,
    tform=None,
    transform_points=True,
    bin_size=(75.0, 50.0, 75.0),
    bounds=None,
    depth_band=(150.0, 700.0),
    min_dy=1e-3,
    normalize_per_path=False,
    method="laplace-fit",
    smoothing_passes=None,
    smoothing_strength=0.5,
    laplace_strength=0.05,
    bc_deep_layers=1,
    edge_extrapolation="hold",
    integration_step=None,
) -> StreamlineField:
    """Build a :class:`StreamlineField` from a collection of pia-to-white-matter paths.

    Each path (e.g. the principal axis of one neuron, from a skeleton) contributes the
    local tangent ``(dx/dy, dz/dy)`` of each of its segments to the grid cell containing
    that segment's midpoint. Cell tangents are the mean over all contributing segments,
    so noise in individual paths averages out. The input is a plain list of point
    arrays; how those paths were produced (neuroglancer, skeletons, etc.) is immaterial.

    Parameters
    ----------
    paths : sequence of (n_i x 3) arrays
        Ordered point arrays, one per path, in the pre-transform coordinate space
        (unless ``transform_points`` is False). Points should be ordered along the
        neurite; skeleton paths already are.
    weights : sequence of arrays, optional
        Per-path weights matching ``paths``. Each entry may be length ``n_i``
        (per-point; a segment's weight is the mean of its endpoints) or ``n_i - 1``
        (per-segment). Use to down-weight unreliable structure, e.g. terminal tips via
        Strahler number or inverse branch order. By default all segments weigh equally.
    tform : TransformSequence, optional
        Transform into the post-transform oriented-micron space, by default identity.
    transform_points : bool, optional
        If True, transform each path before binning, by default True.
    bin_size : 3-tuple, optional
        Grid spacing (x, y, z) in post-transform microns, by default (75, 50, 75).
    bounds : ((xmin, xmax), (ymin, ymax), (zmin, zmax)), optional
        Grid extent in post-transform microns. If None, taken from the data (the y
        extent is overridden by ``depth_band`` when that is set).
    depth_band : (y_lo, y_hi) or None, optional
        Only segments within this post-transform depth range inform the field, and the
        grid spans exactly this range, by default (150, 700). Outside the band the field
        holds the edge orientation constant (see :class:`StreamlineField`). This drops
        the noisy near-pia tuft and deep tails where tangents are unreliable.
    min_dy : float, optional
        Segments with ``|dy|`` below this (near-horizontal, unreliable direction) are
        discarded, by default 1e-3. Raise it to filter noisier segments.
    normalize_per_path : bool, optional
        If True, scale each path's segment weights to sum to 1 so every neuron
        contributes equally regardless of how many segments (branches) it has. By
        default False. Combine with a single tall path per cell, or with weights.
    method : {"laplace-fit", "diffusion", "laplace-bc"}, optional
        How the binned per-cell tangents are regularized into a full field, by default
        ``"laplace-fit"``: fit a curl-free scalar depth potential with a precision-
        weighted data term in every cell (see :func:`_laplace_field`), which also fills
        gaps harmonically. ``"diffusion"`` is the older precision-weighted diffusion of
        the two tangent components independently (no integrability constraint).
        ``"laplace-bc"`` uses only the deep boundary as an empirical condition with a
        flat shallow face and pure Laplace in between.
    smoothing_passes : int, optional
        Precision-weighted diffusion passes (``method="diffusion"``). By default None,
        which uses enough passes to fill the grid.
    smoothing_strength : float, optional
        Neighbor coupling for the diffusion, by default 0.5. Larger smooths more.
    laplace_strength : float, optional
        Smoothness weight (relative to the median cell precision) for the Laplace
        methods, by default 0.05. Larger smooths more.
    bc_deep_layers : int, optional
        Number of deep/shallow y-layers used as boundaries for ``method="laplace-bc"``,
        by default 1.
    edge_extrapolation : {"hold", "harmonic"}, optional
        Behavior in data-poor cells for ``method="laplace-fit"``, by default ``"hold"``:
        continue the nearest in-data tangent outward rather than letting the potential
        relax toward vertical (which collapses depth-integrated quantities at the
        margins). The data region's curl-free fit is unchanged either way. See
        :func:`_laplace_field`.
    integration_step : float, optional
        Passed through to the resulting field. By default None (uses the y bin size).

    Returns
    -------
    StreamlineField
    """
    if tform is None:
        tform = identity_transform()

    if weights is None:
        weights = [None] * len(paths)

    mids_all, tangents_all, weights_all = [], [], []
    for path, wp in zip(paths, weights):
        p = np.asarray(path, dtype=float)
        if transform_points:
            p = tform.apply(p)
        if len(p) < 2:
            continue
        d = np.diff(p, axis=0)
        mid = (p[:-1] + p[1:]) / 2.0

        if wp is None:
            wseg = np.ones(len(d))
        else:
            wp = np.asarray(wp, dtype=float)
            if len(wp) == len(p):
                wseg = (wp[:-1] + wp[1:]) / 2.0
            elif len(wp) == len(p) - 1:
                wseg = wp
            else:
                raise ValueError(
                    "Each weights entry must have length n_i (per-point) or "
                    "n_i - 1 (per-segment) to match its path."
                )

        good = np.abs(d[:, 1]) > min_dy
        if depth_band is not None:
            good &= (mid[:, 1] >= depth_band[0]) & (mid[:, 1] <= depth_band[1])
        d, mid, wseg = d[good], mid[good], wseg[good]
        if len(d) == 0:
            continue
        if normalize_per_path and wseg.sum() > 0:
            wseg = wseg / wseg.sum()
        mids_all.append(mid)
        tangents_all.append(np.column_stack([d[:, 0] / d[:, 1], d[:, 2] / d[:, 1]]))
        weights_all.append(wseg)

    if not mids_all:
        raise ValueError("No usable segments found in the provided paths.")
    mids = np.vstack(mids_all)
    tangents = np.vstack(tangents_all)
    seg_weights = np.concatenate(weights_all)

    bs = np.asarray(bin_size, dtype=float)
    if bounds is None:
        lo = mids.min(axis=0)
        hi = mids.max(axis=0)
    else:
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
    if depth_band is not None:
        lo[1], hi[1] = depth_band
    n = np.maximum(np.ceil((hi - lo) / bs).astype(int), 1)

    idx = np.clip(((mids - lo) / bs).astype(int), 0, n - 1)
    ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]

    shape = (n[0], n[1], n[2])
    wt_sum = np.zeros(shape + (2,))  # weighted sum for the mean tangent
    w_sum = np.zeros(shape)
    counts = np.zeros(shape)
    t_sum = np.zeros(shape + (2,))  # unweighted sums for the variance estimate
    t_sq = np.zeros(shape + (2,))
    np.add.at(wt_sum, (ix, iy, iz), tangents * seg_weights[:, None])
    np.add.at(w_sum, (ix, iy, iz), seg_weights)
    np.add.at(counts, (ix, iy, iz), 1)
    np.add.at(t_sum, (ix, iy, iz), tangents)
    np.add.at(t_sq, (ix, iy, iz), tangents**2)

    filled = w_sum > 0
    mean_t = np.zeros_like(wt_sum)
    mean_t[filled] = wt_sum[filled] / w_sum[filled][..., None]

    # Per-cell precision = count / (variance + prior). Variance combines both tangent
    # components; the prior (typical dense-cell variance) shrinks single-sample cells.
    mean_u = np.zeros_like(t_sum)
    mean_u[filled] = t_sum[filled] / counts[filled][..., None]
    var = np.clip(t_sq - counts[..., None] * mean_u**2, 0, None).sum(axis=-1)
    var[filled] /= counts[filled]
    dense = counts >= 5
    if dense.any():
        var0 = float(np.median(var[dense]))
    elif filled.any():
        var0 = float(np.median(var[filled]))
    else:
        var0 = 1.0
    var0 = max(var0, 1e-9)
    prec = np.zeros(shape)
    prec[filled] = counts[filled] / (var[filled] + var0)

    if method == "diffusion":
        if smoothing_passes is None:
            smoothing_passes = int(max(shape)) + 5
        mean_t, diffused_p = _precision_smooth(
            mean_t, prec, smoothing_passes, smoothing_strength
        )
        # Any node diffusion never reached (fully disconnected) -> global mean.
        unreached = diffused_p <= 0
        if filled.any() and unreached.any():
            mean_t[unreached] = mean_t[filled].mean(axis=0)
    elif method in ("laplace-fit", "laplace-bc"):
        mode = "fit" if method == "laplace-fit" else "bc"
        mean_t = _laplace_field(
            mean_t, prec, bs, mode=mode, lam_rel=laplace_strength,
            deep_layers=bc_deep_layers, edge_extrapolation=edge_extrapolation,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r}; use 'diffusion', 'laplace-fit', or 'laplace-bc'."
        )

    centers = [lo[a] + (np.arange(n[a]) + 0.5) * bs[a] for a in range(3)]
    return StreamlineField(
        centers[0],
        centers[1],
        centers[2],
        mean_t,
        confidence=prec,
        tform=tform,
        integration_step=integration_step,
    )


identity_streamline = Streamline(points=np.array([[0, 0, 0], [0, 1000, 0]]))
