import pytest
import pandas as pd
import numpy as np
from standard_transform import v1dd_streamline_nm, v1dd_streamline_vx


@pytest.fixture()
def v1dd_sl_nm():
    return v1dd_streamline_nm()


@pytest.fixture()
def v1dd_sl_vx():
    return v1dd_streamline_vx()


@pytest.fixture()
def v1dd_nm_pts():
    return np.array(
        [
            [339412.7, 785224.7, 326385.0],
            [340227.5, 785612.7, 326610.0],
            [341187.8, 786359.6, 326655.0],
            [343244.2, 785612.7, 327375.0],
            [342798.0, 785942.5, 327150.0],
        ]
    )

@pytest.fixture()
def v1dd_vx_pts():
    return v1dd_nm_pts() / np.array([9,9,45])

def test_streamline_nm(v1dd_sl_nm, v1dd_nm_pts):
    new_pts = v1dd_sl_nm._transform.apply(v1dd_nm_pts)
    sl_x, sl_z = v1dd_sl_nm.streamline_at(new_pts[0], [0, 1000, 2000])
    sl_pts = v1dd_sl_nm.streamline_at(new_pts[0], [0, 1000, 2000], return_as_point=True)
    assert np.all(sl_pts[:,0] == sl_x)

def test_streamline_tform(v1dd_sl_nm, v1dd_sl_vx):
    pts_nm = np.array([1000, 1000, 1000])
    pts_vx = pts_nm / np.array([9,9,45])
    sl_nm = v1dd_sl_nm.streamline_points_tform(pts_nm)