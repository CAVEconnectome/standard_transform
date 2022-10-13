from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
from collections.abc import Iterable

class ScaleTransform(object):
    def __init__(self, scaling):
        if not isinstance(scaling, Iterable):
            scaling = np.array(3 * [scaling]).reshape(1, 3)
        else:
            if len(scaling) != 3:
                raise ValueError("Scaling must be single number or have three elements")
            scaling = np.array(scaling).reshape(1, 3)
        self._scaling = scaling

    def apply(self, pts):
        return np.atleast_2d(pts) * self._scaling

    def __repr__(self):
        return f"Scale by {self._scaling}"


class TranslateTransform(object):
    def __init__(self, translate):
        if not isinstance(translate, Iterable):
            raise ValueError("Translate must be a three element vector")
        if len(translate) != 3:
            raise ValueError("Translate must be a three element vector")
        self._translate = np.array(translate)

    def apply(self, pts):
        return np.atleast_2d(pts) + self._translate

    def __repr__(self):
        return f"Translate by {self._translate}"


class RotationTransform(object):
    def __init__(self, *params, **param_kwargs):
        self._params = params
        self._param_kwargs = param_kwargs
        self._transform = R.from_euler(*self._params, **self._param_kwargs)

    def apply(self, pts):
        return self._transform.apply(np.atleast_2d(pts))

    def __repr__(self):
        return f"Rotate with params {self._params} and {self._param_kwargs}"


class TransformSequence(object):
    def __init__(self):
        self._transforms = []

    def __repr__(self):
        return "Transformation Sequence:\n\t" + "\n\t".join(
            [t.__repr__() for t in self._transforms]
        )

    def add_transform(self, transform):
        self._transforms.append(transform)

    def add_scaling(self, scaling):
        self.add_transform(ScaleTransform(scaling))

    def add_translation(self, translate):
        self.add_transform(TranslateTransform(translate))

    def add_rotation(self, *rotation_params, **rotation_kwargs):
        self.add_transform(RotationTransform(*rotation_params, **rotation_kwargs))

    def apply(self, pts, as_int=False):
        if isinstance(pts, pd.Series):
            return self.column_apply(pts, as_int=as_int)
        else:
            return self.list_apply(pts, as_int=as_int)

    def list_apply(self, pts, as_int=False):
        pts = np.array(pts).copy()
        for t in self._transforms:
            pts = t.apply(pts)
        if as_int:
            return pts.astype(int)
        else:
            return pts

    def column_apply(self, col, return_array=False, as_int=False):
        pts = np.vstack(col)
        out = self.apply(pts)
        if return_array:
            return self.apply(pts, as_int=as_int)
        else:
            return self.apply(pts, as_int=as_int).tolist()

    def apply_project(self, projection, pts, as_int=False):
        """Apply transform and extract one dimension (e.g. depth)

        Parameters
        ----------
        projection : str or int
            Which dimension to project out of the transformed data. One of "x","y", or "z" (or 0,1,2 equivalently).
        pts : np.ndarray or pd.Series
            Either an n x 3 array or pandas Series object with 3-element arrays as elements.
        as_int : bool, optional
            Return locations as integers, by default False

        Returns
        -------
        np.array
            N-length array 
        """
        proj_map = {
            "x": 0,
            "y": 1,
            "z": 2,
            0: 0,
            1: 1,
            2: 2,
        }

        if projection not in proj_map:
            raise ValueError('Projection must be one of "x", "y", or "z"')
        return np.array(self.apply(pts, as_int=as_int))[:, proj_map.get(projection)]


