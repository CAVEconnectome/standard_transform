from .base import TransformSequence, R
import numpy as np

MINNIE_PIA_POINT_NM = np.array([183013, 83535, 21480]) * [4,4,45]
V1DD_PIA_POINT_NM = np.array([101249, 32249, 9145]) * [9,9,45]

def identity_transform():
    "Returns the same points provided"
    return TransformSequence()

def _minnie_transforms( tform, pia_point ):
    angle_offset = 5
    
    tform.add_rotation("z", angle_offset, degrees=True)
    tform.add_translation(
        [0, -tform.apply_project("y", pia_point), 0]
    )
    tform.add_scaling(1 / 1000)
    return tform

def _v1dd_transforms( tform, pia_point):
    # ctr = np.array([[910302.55274889, 273823.89004458, 411543.78900446]])
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
