from __future__ import annotations

from collections.abc import Iterable
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from .utils import get_dataframe_points, is_list_like


class ScaleTransform(object):
    def __init__(self, scaling):
        if not isinstance(scaling, Iterable):
            scaling = np.array(3 * [scaling]).reshape(1, 3)
        else:
            if len(scaling) != 3:
                raise ValueError("Scaling must be single number or have three elements")
            scaling = np.array(scaling).reshape(1, 3)
        self._scaling = scaling

    def apply(self, pts) -> np.ndarray:
        return np.atleast_2d(pts) * self._scaling

    def invert(self, pts) -> np.ndarray:
        return np.atleast_2d(pts) / self._scaling

    def __repr__(self):
        return f"Scale by {self._scaling}"


class TranslateTransform(object):
    def __init__(self, translate):
        if not isinstance(translate, Iterable):
            raise ValueError("Translate must be a three element vector")
        if len(translate) != 3:
            raise ValueError("Translate must be a three element vector")
        self._translate = np.array(translate)

    def apply(self, pts) -> np.ndarray:
        return np.atleast_2d(pts) + self._translate

    def invert(self, pts) -> np.ndarray:
        return np.atleast_2d(pts) - self._translate

    def __repr__(self):
        return f"Translate by {self._translate}"


class RotationTransform(object):
    def __init__(self, *params, **param_kwargs):
        self._params = params
        self._param_kwargs = param_kwargs
        self._transform = R.from_euler(*self._params, **self._param_kwargs)

    def apply(self, pts) -> np.ndarray:
        return self._transform.apply(np.atleast_2d(pts))

    def invert(self, pts) -> np.ndarray:
        return self._transform.inv().apply(np.atleast_2d(pts))

    def __repr__(self):
        return f"Rotate with params {self._params} and {self._param_kwargs}"


class TransformSequence(object):
    def __init__(self):
        self._transforms = []

    def __repr__(self):
        return "Transformation Sequence:\n\t" + "\n\t".join(
            [t.__repr__() for t in self._transforms]
        )

    def add_transform(self, transform) -> None:
        self._transforms.append(transform)

    def add_scaling(self, scaling) -> None:
        self.add_transform(ScaleTransform(scaling))

    def add_translation(self, translate) -> None:
        self.add_transform(TranslateTransform(translate))

    def add_rotation(self, *rotation_params, **rotation_kwargs):
        self.add_transform(RotationTransform(*rotation_params, **rotation_kwargs))

    def apply(self, pts, as_int=False) -> np.ndarray:
        if isinstance(pts, pd.Series):
            return self.column_apply(pts, as_int=as_int)
        else:
            return self.list_apply(pts, as_int=as_int)

    def invert(self, pts_tf, as_int=False) -> np.ndarray:
        """Invert points post-transform back into the original coordinate system

        Parameters
        ----------
        pts_tf : array-like
            Points in the post-transform coordinate system
        as_int : bool, optional
            Return locations as integers, by default False

        Returns
        -------
        array-like
            Points in the original coordinate system
        """
        if isinstance(pts_tf, pd.Series):
            return self.column_invert(pts_tf, as_int=as_int)
        else:
            return self.list_invert(pts_tf, as_int=as_int)

    def list_apply(self, pts, as_int=False) -> np.ndarray:
        pts = np.array(pts)
        orig_shape = pts.shape
        for t in self._transforms:
            pts = t.apply(pts)
        pts = np.reshape(pts, orig_shape)
        if as_int:
            return pts.astype(int)
        else:
            return pts

    def list_invert(self, pts, as_int=False) -> np.ndarray:
        pts = np.array(pts)
        orig_shape = pts.shape
        for t in self._transforms[::-1]:
            pts = t.invert(pts)
        pts = np.reshape(pts, orig_shape)
        if as_int:
            return pts.astype(int)
        else:
            return pts

    def column_apply(
        self, col, return_array=False, as_int=False
    ) -> "Union[np.ndarray, list]":
        pts = np.vstack(col)
        if return_array:
            return self.apply(pts, as_int=as_int)
        else:
            return self.apply(pts, as_int=as_int).tolist()

    def column_invert(
        self, col, return_array=False, as_int=False
    ) -> "Union[np.ndarray, list]":
        pts = np.vstack(col)
        out = self.apply(pts)
        if return_array:
            return self.invert(pts, as_int=as_int)
        else:
            return self.invert(pts, as_int=as_int).tolist()

    def apply_project(self, projection, pts, as_int=False) -> np.ndarray:
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
        # If a single point is passed, then return a single number not an array.
        if len(np.array(pts).shape) == 1 and not isinstance(pts, pd.Series):

            def output_fn(x):
                return x[0]
        else:

            def output_fn(x):
                return x

        if projection not in proj_map:
            raise ValueError('Projection must be one of "x", "y", or "z"')
        return output_fn(
            np.array(np.atleast_2d(self.apply(pts, as_int=as_int)))[
                :, proj_map.get(projection)
            ]
        )

    def apply_dataframe(
        self, col, df, projection=None, return_array=False, as_int=False
    ) -> "Union[np.ndarray, list]":
        """Apply transformation on a dataframe position column (or prefix of a split position column).

        Parameters
        ----------
        col : str
            Column name or prefix of a split position column. e.g. `pt_position` for `pt_position_x`, `pt_position_y`, `pt_position_z`.
            Whether or not column is split is auto-determined.
        df : pd.DataFrame
            Dataframe with data
        projection : str, optional
            If specified as 'x', 'y', or 'z' return only one element, by default None.
        as_int : bool, optional
            If True, cast values to integers, by default False
        """
        pts = get_dataframe_points(pt_col=col, df=df)
        if projection:
            out = self.apply_project(projection, pts, as_int=as_int)
            if return_array:
                return out
            else:
                return out.tolist()
        else:
            out = self.apply(pts, as_int)
            if return_array:
                return out
            else:
                return out.tolist()

    def apply_skeleton(self, sk, inplace=False) -> "meshparty.skeleton.Skeleton":
        """Apply transformation to a meshparty Skeleton

        Parameters
        ----------
        sk : meshparty.Skeleton
            Skeleton to transform
        inplace : bool, optional
            If True, transform vertices in place, by default False

        Returns
        -------
        Skeleton
            Skeleton with transformed vertices
        """
        if not inplace:
            sk = sk.copy()
        sk.vertices = self.apply(sk._rooted.vertices)
        return sk

    def apply_meshwork_vertices(
        self, nrn, inplace=False
    ) -> "meshparty.meshwork.Meshwork":
        """Apply transformation to mesh and skeleton vertices of a meshparty Meshwork

        Parameters
        ----------
        nrn : meshparty.meshwork.Meshwork
            Object to transform
        inplace : bool, optional
            If True, transform vertices in place, by default False

        Returns
        -------
        Skeleton
            Skeleton with transformed vertices
        """

        if not inplace:
            nrn = nrn.copy()
        curr_mask = nrn.mesh.node_mask
        nrn.reset_mask()
        nrn.mesh.vertices = self.apply(nrn.mesh.vertices)
        nrn.skeleton.vertices = self.apply(nrn.skeleton.vertices)
        nrn.apply_mask(curr_mask)
        return nrn

    def apply_meshwork_annotations(
        self, nrn, anno_dict, inplace=False
    ) -> "meshparty.meshwork.Meshwork":
        """Apply transformations to annotations in a meshwork

        Parameters
        ----------
        nrn : meshwork.Meshwork
            Object to transform
        anno_dict : dict
            Dictionary whose keys are annotation tables in the meshwork and whose values are the columns to transform (string or list of strings)
        inplace : bool, optional
            If True, transform vertices in place, by default False

        Returns
        -------
        meshwork
            Object with transformed annotation positions.
        """
        if not inplace:
            nrn = nrn.copy()
        for tbl in anno_dict:
            vs = anno_dict[tbl]
            if not is_list_like(vs):
                vs = [vs]
            for v in vs:
                nrn.anno[tbl]._data[v] = self.apply(nrn.anno[tbl]._data[v])
        return nrn


def identity_transform() -> TransformSequence:
    "Returns the same points provided"
    return TransformSequence()
