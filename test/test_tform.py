import pytest
import pandas as pd
import numpy as np
from standard_transform import v1dd_transform_nm, v1dd_transform_vx, minnie_transform_nm, minnie_transform_vx

@pytest.fixture()
def minnie_tform_vx():
    return minnie_transform_vx()

@pytest.fixture()
def v1dd_tform_vx():
    return v1dd_transform_vx()

@pytest.fixture()
def minnie_tform_nm():
    return minnie_transform_nm()

@pytest.fixture()
def v1dd_tform_nm():
    return v1dd_transform_nm()

@pytest.fixture()
def vector_df():
    return pd.DataFrame(
        {
            'pt_position': [
                [59769, 60738, 9145],
                [111838, 68905, 10214],
                [141854, 104048, 8827],
            ],
        }
    )

@pytest.fixture()
def split_df():
    return pd.DataFrame(
        {
            'pt_position_x': [59769, 111838, 141854],
            'pt_position_y': [60738, 68905, 104048],
            'pt_position_z': [9145, 10214, 8827],
        }
    )

@pytest.fixture()
def split_df_t2():
    return pd.DataFrame(
        {
            'pt_position_x_soma': [59769, 111838, 141854],
            'pt_position_y_soma': [60738, 68905, 104048],
            'pt_position_z_soma': [9145, 10214, 8827],
        }
    )


def test_convert_minnie(vector_df,minnie_tform_vx, minnie_tform_nm):
    pts = np.vstack(vector_df['pt_position'].values)
    pts_post_vx = minnie_tform_vx.apply(pts)
    pts_post_nm = minnie_tform_nm.apply(pts * [4,4,40])
    assert np.all(pts_post_vx.ravel()==pts_post_nm.ravel())


def test_convert_v1dd(vector_df,v1dd_tform_vx, v1dd_tform_nm):
    pts = np.vstack(vector_df['pt_position'].values)
    pts_post_vx = v1dd_tform_vx.apply(pts)
    pts_post_nm = v1dd_tform_nm.apply(pts * [9,9,45])
    assert np.all(pts_post_vx.ravel()==pts_post_nm.ravel())

def test_alternative_voxel_res(vector_df):
    pts = np.vstack(vector_df['pt_position'].values)
    pts_nm = pts * [4,4,40]
    pts_mic = pts_nm / np.array([1000, 1000, 1000])

    tform_vx = minnie_transform_vx()
    tform_um = minnie_transform_vx([1000, 1000, 1000])
    pts_vx = tform_vx.apply(pts)
    pts_um = tform_um.apply(pts_mic)
    assert np.all(pts_vx==pts_um)

def test_equivalent_inputs(vector_df, split_df, split_df_t2, minnie_tform_vx):
    pts_arr = minnie_tform_vx.apply(  np.vstack(vector_df['pt_position'].values) )
    pts_ser = minnie_tform_vx.apply(vector_df['pt_position'])
    pts_df = minnie_tform_vx.apply_dataframe('pt_position', vector_df)
    pts_split = minnie_tform_vx.apply_dataframe('pt_position', split_df)
    pts_split_t2 = minnie_tform_vx.apply_dataframe('pt_position_soma', split_df_t2)

    assert np.all(pts_arr==pts_ser)
    assert np.all(pts_arr==pts_df)
    assert np.all(pts_arr==pts_split)
    assert np.all(pts_arr==pts_split_t2)


def test_equivalent_inputs_projection(vector_df, split_df, split_df_t2, minnie_tform_vx):
    pts_arr = minnie_tform_vx.apply_project('x', np.vstack(vector_df['pt_position'].values) )
    pts_ser = minnie_tform_vx.apply_project('x', vector_df['pt_position'])
    pts_df = minnie_tform_vx.apply_dataframe('pt_position', vector_df, projection='x')
    pts_split = minnie_tform_vx.apply_dataframe('pt_position', split_df, projection='x')
    pts_split_t2 = minnie_tform_vx.apply_dataframe('pt_position_soma', split_df_t2, projection='x')

    assert np.all(pts_arr==pts_ser)
    assert np.all(pts_arr==pts_df)
    assert np.all(pts_arr==pts_split)
    assert np.all(pts_arr==pts_split_t2)

def test_single_point(vector_df, v1dd_tform_vx):
    pt_in = vector_df.iloc[0]['pt_position']
    pt_out = v1dd_tform_vx.apply(pt_in)
    assert len(pt_out.shape) == 1
    assert len(v1dd_tform_vx.apply(np.atleast_2d(pt_in)).shape) == 2

    el_out = v1dd_tform_vx.apply_project('z', pt_in)
    assert isinstance(el_out, float)