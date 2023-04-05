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

    def streamline_at(self, xyz0, y1):
        """Location of the streamline passing through point xyz0 at depth y1 in the post-transform space (usually microns)

        Parameters
        ----------
        xyz0 : 3-element array
            Anchor point through which the streamline passes
        y1 : float or array
            Depth or depths at which to find the streamline x and z points.

        Returns
        -------
        new_x, new_z : float or array
            X and z coordinate values for the streamline at the given depths.
        """
        y0 = xyz0[1]
        new_x = xyz0[0] + (self.x_interp(y1) - self.x_interp(y0))
        new_z = xyz0[2] + (self.z_interp(y1) - self.z_interp(y0))
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

    def radial_distance(self, xyz0, xyz1, transform_points=True):
        """Find the distance between two points along the x-z plane using the streamline as d=0.

        Parameters
        ----------
        xyz0 : 3-element array
            First point coordinates
        xyz1 : 3-element array
            Second point coordinates
        transform_points : bool, optional
            If points are in the pre-transform coordinates use True, if in the post-transform coordinates use False. By default True.

        Returns
        -------
        float
            Distance in the x-z plane after accounting for streamline curvature.
        """
        if transform_points:
            xyz0 = self._transform.apply(xyz0)
            xyz1 = self._transform.apply(xyz1)
        new_x, new_z = self.streamline_at(xyz0, xyz1[:,1])
        return np.sqrt((new_x - xyz1[:, 0]) ** 2 + (new_z - xyz1[:, 2]) ** 2)

identity_streamline = Streamline(
    points=np.array([[0, 0, 0], [0, 1000, 0]])
)