import numpy as np
from scipy import interpolate
from .base import identity_transform


class Streamline(object):
    def __init__(self, points, tform=None, transform_points=True):
        """Build a streamline object to determine distances from a curving pia-to-white matter axis

        Parameters
        ----------
        points : nx3 array
            Set of points making up the streamline
        tform : TransformSequence, optional
            A transform sequence, by default the identity transform.
        transform_points : bool, optional
            If points are in the pre-transform coordinates use True, if in the post-transform coordinates use False. By default True.
        """
        if tform is None:
            tform = identity_transform()
        self._transform = tform

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

    def streamline_at(self, xyz0, y1, return_as_point=False):
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
        y0 = xyz0[1]
        new_x = xyz0[0] + (self.x_interp(y1) - self.x_interp(y0))
        new_z = xyz0[2] + (self.z_interp(y1) - self.z_interp(y0))
        if return_as_point:
            return np.vstack([new_x, y1, new_z]).T
        else:
            return new_x, new_z

    def streamline_points_tform(self, xyz0_raw):
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
        xyz0 = self._transform.apply(xyz0_raw)
        y1 = self._points[:, 1]
        new_x, new_z = self.streamline_at(xyz0, y1)
        new_xyz = np.vstack([new_x, y1, new_z]).T
        return self._transform.invert(new_xyz)

    def radial_distance(self, xyz0, xyz1, transform_points=True, return_angle=False):
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
    ):
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

    def depth_between(self, xyz0, xyz1, delta=0.1, transform_points=True):
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

    def depth_along(self, xyz, depth_from=0, delta=0.1, transform_points=True):
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
        if transform_points:
            xyz = self._transform.apply(xyz)
        xm = np.mean(xyz[:, 0])
        zm = np.mean(xyz[:, 2])
        base_pt = np.array([xm, depth_from, zm])
        sl_pts = self.streamline_at(base_pt, xyz[:, 1], return_as_point=True)
        depths = self.depth_between(base_pt, sl_pts, transform_points=False)
        return depths


identity_streamline = Streamline(points=np.array([[0, 0, 0], [0, 1000, 0]]))
