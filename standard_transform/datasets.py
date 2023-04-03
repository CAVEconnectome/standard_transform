from .base import TransformSequence, R, identity_transform
from .streamlines import Streamline, identity_streamline
import numpy as np

V1DD_STREAMLINE_POINT_FILE = 'data/v1dd_um_streamline.json'

MINNIE_PIA_POINT_NM = np.array([183013, 83535, 21480]) * [4,4,45]
V1DD_PIA_POINT_NM = np.array([101249, 32249, 9145]) * [9,9,45]


def _minnie_transforms( tform, pia_point ):
    angle_offset = 5
    
    tform.add_rotation("z", angle_offset, degrees=True)
    tform.add_translation(
        [0, -tform.apply_project("y", pia_point), 0]
    )
    tform.add_scaling(1 / 1000)
    return tform

def _v1dd_transforms( tform, pia_point):
    up = np.array([-0.00497765, 0.96349375, 0.26768454])
    rot, _ = R.align_vectors(np.array([[0, 1, 0]]), [up])

    angles = rot.as_euler("xyz", degrees=True)

    for ind, ang in zip(["x", "y", "z"], angles):
        tform.add_rotation(ind, ang, degrees=True)

    tform.add_translation([0, -tform.apply_project("y", pia_point), 0])
    tform.add_scaling(1 / 1000)
    return tform

def minnie_transform_vx(voxel_resolution=[4,4,40]):
    "Transform for minnie65 dataset from voxels to oriented microns"
    column_transform = TransformSequence()
    column_transform.add_scaling(voxel_resolution)
    minnie_pia_point_vx = np.array(MINNIE_PIA_POINT_NM) / voxel_resolution
    return _minnie_transforms(column_transform, minnie_pia_point_vx)

def minnie_transform_nm():
    "Transform for minnie65 dataset from nanometers to oriented microns"
    column_transform = TransformSequence()
    return _minnie_transforms(column_transform, MINNIE_PIA_POINT_NM)

def v1dd_transform_vx(voxel_resolution=[9,9,45]):
    "Transform for v1dd dataset from voxelsto oriented microns"
    v1dd_transform = TransformSequence()
    v1dd_transform.add_scaling(voxel_resolution)
    v1dd_pia_point_vx = np.array(V1DD_PIA_POINT_NM) / voxel_resolution
    return _v1dd_transforms(v1dd_transform, v1dd_pia_point_vx)

def v1dd_transform_nm():
    "Transform for v1dd dataset from nanometers to oriented microns"
    v1dd_transform = TransformSequence()
    return _v1dd_transforms(v1dd_transform, V1DD_PIA_POINT_NM)

#### STREAMLINES


def v1dd_streamline_nm():
    "Streamline for v1dd dataset for nm coordinates"
    import json
    with open(V1DD_STREAMLINE_POINT_FILE, 'r') as f:
        points = np.array(json.load(f))
    return Streamline(points, tform=v1dd_transform_nm, transform_points=False)

def v1dd_streamline_vx():
    "Streamline for v1dd dataset for voxel coordinates"
    import json
    with open(V1DD_STREAMLINE_POINT_FILE, 'r') as f:
        points = np.array(json.load(f))
    return Streamline(points, tform=v1dd_transform_vx, transform_points=False)

def minnie_streamline_nm():
    "Streamline for minnie65 dataset for nm coordinates"
    return Streamline(
        np.array([[0, 0, 0], [0, 1, 0]]),
        tform=minnie_transform_nm(),
        transform_points=False,
    )

def minnie_streamline_vx():
    "Streamline for minnie65 dataset for voxel coordinates"
    return Streamline(
        np.array([[0, 0, 0], [0, 1, 0]]),
        tform=minnie_transform_vx(),
        transform_points=False,
    )

## Dataset object
class Dataset(object):
    def __init__(
        self,
        name,
        transform_nm,
        transform_vx,
        streamline_nm,
        streamline_vx,
    ):
        self.name = name
        self._transform_arbitrary = transform_vx
        self.transform_nm = transform_nm()
        self.trasnform_vx = transform_vx()

        self.streamline_nm = streamline_nm()
        self.streamline_vx = streamline_vx() 
    
    def transform_from(self, resolution):
        """Transform from arbitrary resolution to oriented microns

        Parameters
        ----------
        resolution : list-like
            Resolution of original data

        Returns
        -------
        Transform object
        """
        return self._transform_arbitrary(resolution)


v1dd_ds = Dataset(
    'v1dd',
    v1dd_transform_nm,
    v1dd_transform_vx,
    v1dd_streamline_nm,
    v1dd_streamline_vx,
)

minnie_ds = Dataset(
    'minnie65',
    minnie_transform_nm,
    minnie_transform_vx,
    minnie_streamline_nm,
    minnie_streamline_vx,
)