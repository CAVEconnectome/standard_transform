import json
import os
import warnings

import numpy as np

from .base import R, TransformSequence, identity_transform
from .streamlines import Streamline, StreamlineField, identity_streamline


def _get_data_path(filename):
    return os.path.join(os.path.dirname(__file__), "data", filename)


MINNIE_VOXEL_RESOLUTION = np.array([4, 4, 40])
V1DD_VOXEL_RESOLUTION = np.array([9, 9, 45])

_DATASET_VOXEL_RESOLUTION = {
    "minnie65": MINNIE_VOXEL_RESOLUTION,
    "v1dd": V1DD_VOXEL_RESOLUTION,
}


#### VERSION REGISTRIES
#
# Transform outputs are tied to the package version: the rigid affine numbers and the
# streamline data define the output frame, so changing either changes every downstream
# result. These registries are the single source of truth for both, versioned on two
# independent tracks. Version labels are package-version-anchored: a label is the
# standard-transform release in which that definition became the default. Requesting a
# past label reproduces the exact frame that release produced.
#
# NOTE: the "2.0" label assumes the v1dd tangent field ships as the next major release.
# Confirm/adjust the labels here at tag time; they live only in this file.

_TRANSFORM_VERSIONS = {
    "minnie65": {
        "1.4": {
            "pia_point_nm":  np.array([183013, 83535, 21480]) * [4, 4, 45],
            "z_angle_deg": 5,
            "introduced_in": "1.4",
        },
    },
    "v1dd": {
        "1.4": {
            "pia_point_nm": np.array([101249, 32249, 9145]) * [9, 9, 45],
            "up_vector": np.array([-0.00497765, 0.96349375, 0.26768454]),
            "introduced_in": "1.4",
        },
        "2.0": {
            "pia_point_nm": np.array([84255, 27946, 10871]) * [9, 9, 45],
            "up_vector": np.array([-0.00497765, 0.96349375, 0.26768454]),
            "introduced_in": "2.0",
        },
    },
}
# minnie65 affine is unchanged, so "1.4" stays latest. v1dd 2.0 revises the pia point
# (a ~39um depth shift), so it becomes latest and streamlines must be rebuilt in it.
_TRANSFORM_LATEST = {"minnie65": "1.4", "v1dd": "2.0"}

# Every streamline is defined in the oriented-micron frame of a *specific* transform
# version: the tangent-field grid is baked in that frame, and the hand-drawn curves
# were traced in it. So each entry pins its ``transform_version`` and the streamline is
# always built against that transform -- never against whatever transform happens to be
# latest. When the affine changes (as v1dd 2.0 does), a field built in the old frame is
# no longer valid: rebuild it against the new transform (scripts/build_streamline_field.py)
# and add a new streamline version pinning that transform version.
_STREAMLINE_VERSIONS = {
    "minnie65": {
        "1.4": {
            "kind": "hand",
            "file": "minnie_um_streamline.json",
            "introduced_in": "1.4",
            "transform_version": "1.4",
        },
        "2.0": {
            "kind": "field",
            "file": "minnie_streamline_field.npz",
            "introduced_in": "2.0",
            "transform_version": "1.4",
        },
    },
    "v1dd": {
        "1.4": {
            "kind": "hand",
            "file": "v1dd_um_streamline.json",
            "introduced_in": "1.4",
            "transform_version": "1.4",
        },
        "2.0": {
            "kind": "field",
            "file": "v1dd_streamline_field.npz",
            "introduced_in": "2.0",
            "transform_version": "2.0",
        },
    },
}
# Defaults align each dataset to a single version: minnie65 stays on 1.4 (its affine is
# unchanged and it defaults to the hand streamline; the field is reachable via "2.0"),
# while v1dd advances to 2.0 (new pia-point transform + rebuilt field).
_STREAMLINE_LATEST = {"minnie65": "2.0", "v1dd": "2.0"}


def _resolve(registry, latest, dataset, version, kind_label):
    if dataset not in registry:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {sorted(registry)}."
        )
    versions = registry[dataset]
    resolved = latest[dataset] if version is None else version
    if resolved not in versions:
        raise ValueError(
            f"Unknown {kind_label} version {resolved!r} for dataset {dataset!r}. "
            f"Available: {sorted(versions)} (latest {latest[dataset]!r})."
        )
    return versions[resolved], resolved


def available_versions(dataset):
    """List the available transform and streamline versions for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name, one of ``"minnie65"`` or ``"v1dd"``.

    Returns
    -------
    dict
        ``{"transform": {"latest": <label>, "versions": {<label>: <introduced_in>}},
        "streamline": {"latest": ..., "versions": {...}, "transform_versions":
        {<label>: <transform version the streamline is built in>}}}``. The two tracks
        are versioned independently -- each label is the package release in which that
        definition became the default -- but every streamline pins the transform version
        whose oriented frame its data lives in.
    """
    if dataset not in _TRANSFORM_VERSIONS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {sorted(_TRANSFORM_VERSIONS)}."
        )
    return {
        "transform": {
            "latest": _TRANSFORM_LATEST[dataset],
            "versions": {
                v: e["introduced_in"] for v, e in _TRANSFORM_VERSIONS[dataset].items()
            },
        },
        "streamline": {
            "latest": _STREAMLINE_LATEST[dataset],
            "versions": {
                v: e["introduced_in"] for v, e in _STREAMLINE_VERSIONS[dataset].items()
            },
            # Each streamline version is built in a specific transform version's frame.
            "transform_versions": {
                v: e["transform_version"]
                for v, e in _STREAMLINE_VERSIONS[dataset].items()
            },
        },
    }


def _resolution_vector(resolution, dataset):
    """Map a ``resolution`` argument to a voxel-resolution vector, or None for nm.

    ``resolution`` may be the string ``"nm"`` (no pre-scaling), ``"vx"`` (the dataset's
    native voxel size), or an explicit ``[x, y, z]`` nm/voxel resolution.
    """
    if isinstance(resolution, str):
        if resolution == "nm":
            return None
        if resolution == "vx":
            return _DATASET_VOXEL_RESOLUTION[dataset]
        raise ValueError(
            f"resolution string must be 'nm' or 'vx', got {resolution!r}. "
            f"Alternatively pass an [x, y, z] resolution."
        )
    return np.asarray(resolution)


def _warn_renamed(old, new):
    warnings.warn(
        f"{old} is deprecated and will be removed in a future release; use {new} "
        f"instead.",
        DeprecationWarning,
        stacklevel=3,
    )


#### TRANSFORMS


def _rotation_from_up_vector(up):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rot, _ = R.align_vectors(np.array([[0, 1, 0]]), [up])
    return rot


def _minnie_transforms(tform, pia_point, params) -> TransformSequence:
    tform.add_rotation("z", params["z_angle_deg"], degrees=True)
    tform.add_translation([0, -tform.apply_project("y", pia_point), 0])
    tform.add_scaling(1 / 1000)
    return tform


def _v1dd_transforms(tform, pia_point, params) -> TransformSequence:
    rot = _rotation_from_up_vector(params["up_vector"])
    angles = rot.as_euler("xyz", degrees=True)

    for ind, ang in zip(["x", "y", "z"], angles):
        tform.add_rotation(ind, ang, degrees=True)

    tform.add_translation([0, -tform.apply_project("y", pia_point), 0])
    tform.add_scaling(1 / 1000)
    return tform


_RECIPES = {"minnie65": _minnie_transforms, "v1dd": _v1dd_transforms}


def _build_transform(dataset, version, voxel_resolution) -> TransformSequence:
    params, resolved = _resolve(
        _TRANSFORM_VERSIONS, _TRANSFORM_LATEST, dataset, version, "transform"
    )
    tform = TransformSequence()
    if voxel_resolution is None:
        pia_point = np.array(params["pia_point_nm"])
    else:
        tform.add_scaling(voxel_resolution)
        pia_point = np.array(params["pia_point_nm"]) / voxel_resolution
    tform = _RECIPES[dataset](tform, pia_point, params)
    tform.version = resolved
    return tform


def minnie_transform(resolution="nm", version=None) -> TransformSequence:
    """Transform for the minnie65 dataset into oriented microns.

    Parameters
    ----------
    resolution : str or array-like, optional
        Units of the input points: ``"nm"`` (default), ``"vx"`` (native voxel size),
        or an explicit ``[x, y, z]`` resolution.
    version : str, optional
        Transform version; by default the latest. See :func:`available_versions`.
    """
    return _build_transform("minnie65", version, _resolution_vector(resolution, "minnie65"))


def v1dd_transform(resolution="nm", version=None) -> TransformSequence:
    """Transform for the v1dd dataset into oriented microns. See :func:`minnie_transform`."""
    return _build_transform("v1dd", version, _resolution_vector(resolution, "v1dd"))


#### STREAMLINES


def _build_streamline(dataset, resolution, version) -> Streamline:
    entry, resolved = _resolve(
        _STREAMLINE_VERSIONS, _STREAMLINE_LATEST, dataset, version, "streamline"
    )
    # A streamline lives in a specific transform version's oriented frame, so build that
    # transform (at the requested resolution) rather than the current latest. The
    # resolution only pre-scales the input; it does not affect the oriented frame the
    # streamline data is baked in.
    tform_version = entry["transform_version"]
    tform = _build_transform(
        dataset, tform_version, _resolution_vector(resolution, dataset)
    )
    path = _get_data_path(entry["file"])
    if entry["kind"] == "field":
        sl = StreamlineField.from_npz(path, tform=tform)
        built = sl.built_transform_version
        if built is not None and built != tform_version:
            warnings.warn(
                f"The {dataset} streamline field ({entry['file']}) was built in the "
                f"transform-{built} frame, but streamline version {resolved!r} pins it "
                f"to transform {tform_version!r}; depths will be misaligned. Rebuild "
                f"the field against transform {tform_version!r} with "
                f"scripts/build_streamline_field.py.",
                UserWarning,
                stacklevel=2,
            )
    else:
        with open(path, "r") as f:
            points = np.array(json.load(f))
        sl = Streamline(points, tform=tform, transform_points=False)
    sl.version = resolved
    sl.transform_version = tform_version
    return sl


def minnie_streamline(resolution="nm", version=None) -> Streamline:
    """Streamline for the minnie65 dataset.

    Parameters
    ----------
    resolution : str or array-like, optional
        Units of the points you will pass in: ``"nm"`` (default), ``"vx"``, or an
        explicit ``[x, y, z]`` resolution.
    version : str, optional
        Streamline version; by default the latest. See :func:`available_versions`.
    """
    return _build_streamline("minnie65", resolution, version)


def v1dd_streamline(resolution="nm", version=None) -> Streamline:
    """Streamline for the v1dd dataset.

    By default (``version=None``) returns the latest definition — the data-derived,
    spatially-varying tangent field (a :class:`StreamlineField`, a drop-in for
    :class:`Streamline`). Pass ``version="1.4"`` for the original hand-drawn single
    streamline. See :func:`minnie_streamline` for ``resolution`` and
    :func:`available_versions`.
    """
    return _build_streamline("v1dd", resolution, version)


#### DEPRECATED ALIASES (superseded by the resolution= API above)


def minnie_transform_nm(version=None) -> TransformSequence:
    _warn_renamed("minnie_transform_nm(...)", "minnie_transform(resolution='nm', ...)")
    return minnie_transform("nm", version=version)


def minnie_transform_vx(
    voxel_resolution=MINNIE_VOXEL_RESOLUTION, version=None
) -> TransformSequence:
    _warn_renamed("minnie_transform_vx(...)", "minnie_transform(resolution=<res>, ...)")
    return minnie_transform(voxel_resolution, version=version)


def v1dd_transform_nm(version=None) -> TransformSequence:
    _warn_renamed("v1dd_transform_nm(...)", "v1dd_transform(resolution='nm', ...)")
    return v1dd_transform("nm", version=version)


def v1dd_transform_vx(
    voxel_resolution=V1DD_VOXEL_RESOLUTION, version=None
) -> TransformSequence:
    _warn_renamed("v1dd_transform_vx(...)", "v1dd_transform(resolution=<res>, ...)")
    return v1dd_transform(voxel_resolution, version=version)


def minnie_streamline_nm(version=None) -> Streamline:
    _warn_renamed("minnie_streamline_nm(...)", "minnie_streamline(resolution='nm', ...)")
    return minnie_streamline("nm", version=version)


def minnie_streamline_vx(
    voxel_resolution=MINNIE_VOXEL_RESOLUTION, version=None
) -> Streamline:
    _warn_renamed("minnie_streamline_vx(...)", "minnie_streamline(resolution=<res>, ...)")
    return minnie_streamline(voxel_resolution, version=version)


def v1dd_streamline_nm(version=None) -> Streamline:
    _warn_renamed("v1dd_streamline_nm(...)", "v1dd_streamline(resolution='nm', ...)")
    return v1dd_streamline("nm", version=version)


def v1dd_streamline_vx(voxel_resolution=V1DD_VOXEL_RESOLUTION, version=None) -> Streamline:
    _warn_renamed("v1dd_streamline_vx(...)", "v1dd_streamline(resolution=<res>, ...)")
    return v1dd_streamline(voxel_resolution, version=version)


## Dataset object
class Dataset(object):
    """Bundle of a transform and a streamline for a single dataset.

    The recommended entry points are the exported ``minnie_ds`` and ``v1dd_ds``
    singletons. Use :meth:`transform` and :meth:`streamline`, each of which takes a
    ``resolution`` (``"nm"`` (default), ``"vx"``, or an ``[x, y, z]`` list) and an
    optional ``version``.

    Parameters
    ----------
    name : str
        Dataset name; must match a key in the version registries.
    transform_fn, streamline_fn : callable
        Factories with signature ``(resolution="nm", version=None)`` returning the
        transform / streamline for this dataset.
    """

    def __init__(self, name, transform_fn, streamline_fn):
        self.name = name
        self._transform_fn = transform_fn
        self._streamline_fn = streamline_fn
        # Streamlines may load a data file (the tangent field), so cache built ones.
        self._streamline_cache = {}

    def transform(self, resolution="nm", version=None) -> TransformSequence:
        """Transform into oriented microns for input at ``resolution``.

        Parameters
        ----------
        resolution : str or array-like, optional
            ``"nm"`` (default), ``"vx"``, or an explicit ``[x, y, z]`` resolution.
        version : str, optional
            Transform version; by default the latest. See :meth:`available_versions`.
        """
        return self._transform_fn(resolution, version=version)

    def streamline(self, resolution="nm", version=None) -> Streamline:
        """Streamline for input at ``resolution`` (data-derived field where available).

        Parameters
        ----------
        resolution : str or array-like, optional
            ``"nm"`` (default), ``"vx"``, or an explicit ``[x, y, z]`` resolution.
        version : str, optional
            Streamline version; by default the latest. See :meth:`available_versions`.
        """
        key = (self._res_key(resolution), version)
        if key not in self._streamline_cache:
            self._streamline_cache[key] = self._streamline_fn(resolution, version=version)
        return self._streamline_cache[key]

    def available_versions(self):
        """Available transform and streamline versions. See :func:`available_versions`."""
        return available_versions(self.name)

    @staticmethod
    def _res_key(resolution):
        if isinstance(resolution, str):
            return resolution
        return tuple(np.asarray(resolution).tolist())

    # -- deprecated accessors (superseded by transform()/streamline()) --

    @property
    def transform_nm(self) -> TransformSequence:
        _warn_renamed("Dataset.transform_nm", "Dataset.transform('nm')")
        return self.transform("nm")

    @property
    def transform_vx(self) -> TransformSequence:
        _warn_renamed("Dataset.transform_vx", "Dataset.transform('vx')")
        return self.transform("vx")

    def transform_res(self, resolution, version=None) -> TransformSequence:
        _warn_renamed("Dataset.transform_res(res)", "Dataset.transform(res)")
        return self.transform(resolution, version=version)

    @property
    def streamline_nm(self) -> Streamline:
        _warn_renamed("Dataset.streamline_nm", "Dataset.streamline('nm')")
        return self.streamline("nm")

    @property
    def streamline_vx(self) -> Streamline:
        _warn_renamed("Dataset.streamline_vx", "Dataset.streamline('vx')")
        return self.streamline("vx")

    def streamline_res(self, resolution, version=None) -> Streamline:
        _warn_renamed("Dataset.streamline_res(res)", "Dataset.streamline(res)")
        return self.streamline(resolution, version=version)


v1dd_ds = Dataset("v1dd", v1dd_transform, v1dd_streamline)
minnie_ds = Dataset("minnie65", minnie_transform, minnie_streamline)
