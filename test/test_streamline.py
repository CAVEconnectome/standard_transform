import pytest
import pandas as pd
import numpy as np
from standard_transform import v1dd_ds


@pytest.fixture()
def pts_arb():
    return np.array(
        [
            [93970, 80981, 6503],
            [121308, 87759, 7414],
            [85876, 86346, 8505],
            [85907, 80487, 9338],
            [79356, 82186, 8454],
        ]
    )

@pytest.fixture()
def pts_vx(pts_arb):
    return pts_arb * np.array([9.7,9.7,45]) / np.array([9,9,45])

@pytest.fixture()
def pts_nm(pts_arb):
    return pts_arb * np.array([9.7,9.7,45])

@pytest.fixture()
def root_location():
    return np.array([817335., 611523., 336240.])

def test_v1dd_streamline(pts_arb, pts_vx, pts_nm, root_location):

    sl_pts_arb = v1dd_ds.streamline_res([9.7, 9.7, 45]).radial_points( root_location/[9.7,9.7,45], pts_arb)
    sl_pts_vx = v1dd_ds.streamline_vx.radial_points( root_location/[9,9,45], pts_vx)
    sl_pts_nm = v1dd_ds.streamline_nm.radial_points( root_location, pts_nm)

    assert np.all(
        np.isclose(sl_pts_nm, sl_pts_vx)
    )

    assert np.all(
        np.isclose(sl_pts_nm, sl_pts_arb)
    )


def test_v1dd_streamline_inverse(root_location):
    sl_pts_0 = v1dd_ds.streamline_nm.streamline_points_tform(root_location)
    sl_pts_1 = v1dd_ds.streamline_res([9.7,9.7,45]).streamline_points_tform(root_location / [9.7, 9.7, 45]) * [9.7,9.7,45]
    assert np.all(np.isclose(sl_pts_0, sl_pts_1))