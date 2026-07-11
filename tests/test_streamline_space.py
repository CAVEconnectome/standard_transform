import numpy as np
import pytest

from standard_transform import v1dd_ds


@pytest.fixture(scope="module")
def field():
    return v1dd_ds.streamline()  # the data-derived (curl-free) StreamlineField


def _inband(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(-150, 150, n), rng.uniform(250, 600, n), rng.uniform(-150, 150, n)]
    )


def test_roundtrip_oriented(field):
    pts = _inband(80)
    ss = field.to_streamline_space(pts, transform_points=False)
    back = field.from_streamline_space(ss, transform_points=False)
    assert np.abs(back - pts).max() < 1.0  # sub-micron round-trip


def test_depth_is_preserved(field):
    pts = _inband(40)
    ss = field.to_streamline_space(pts, transform_points=False)
    assert np.allclose(ss[:, 1], pts[:, 1])  # y (depth) passes through unchanged


def test_single_streamline_shares_lateral(field):
    # every point on one streamline must map to the same (u, w) -- the chart is global
    anchor = np.array([50.0, 400.0, -30.0])
    curve = field.streamline_at(anchor, np.linspace(250, 640, 8), return_as_point=True)
    ss = field.to_streamline_space(curve, transform_points=False)
    assert ss[:, 0].std() < 0.5 and ss[:, 2].std() < 0.5


def test_lateral_order_preserved(field):
    us = [
        field.to_streamline_space(np.array([[x, 400.0, 0.0]]), transform_points=False)[0, 0]
        for x in (-150.0, 0.0, 150.0)
    ]
    assert us[0] < us[1] < us[2]  # absolute lateral order kept (not recentered per cell)


def test_reference_depth_consistency(field):
    # forward and inverse must use the same reference_depth to round-trip
    pts = _inband(30)
    for ref in (0.0, 300.0):
        ss = field.to_streamline_space(pts, reference_depth=ref, transform_points=False)
        back = field.from_streamline_space(ss, reference_depth=ref, transform_points=False)
        assert np.abs(back - pts).max() < 1.0


def test_anchor_puts_data_min_at_zero(field):
    gx, gy, gz = np.meshgrid(field._x_grid, field._y_grid, field._z_grid, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    ss = field.to_streamline_space(nodes, transform_points=False, anchor=True)
    assert ss[:, 0].min() == pytest.approx(0.0, abs=1e-6)  # smallest x -> 0
    assert ss[:, 2].min() == pytest.approx(0.0, abs=1e-6)  # smallest z -> 0
    assert ss[:, 0].min() >= -1e-9 and ss[:, 2].min() >= -1e-9  # non-negative
    assert np.allclose(ss[:, 1], nodes[:, 1])  # depth unchanged (pia stays y=0)


def test_anchor_roundtrip(field):
    pts = _inband(60)
    a = field.to_streamline_space(pts, transform_points=False, anchor=True)
    back = field.from_streamline_space(a, transform_points=False, anchor=True)
    assert np.abs(back - pts).max() < 1.0


def test_transform_points_plumbing(field):
    # default transform_points=True: pre-transform (nm) in and out
    nm = field._transform.invert(_inband(20))
    ss = field.to_streamline_space(nm)
    back = field.from_streamline_space(ss)
    assert np.abs(back - nm).max() < 2000.0  # ~sub-µm oriented error scaled back to nm
