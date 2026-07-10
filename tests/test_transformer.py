import numpy as np
import pytest

from standard_transform import v1dd_ds, v1dd_streamline


@pytest.fixture()
def root():
    return np.array([817335.0, 611523.0, 336240.0])


@pytest.fixture()
def pts(root):
    return root + np.array(
        [[0.0, 0.0, 0.0], [50000.0, 20000.0, 10000.0], [-30000.0, 60000.0, -5000.0]]
    )


def test_object_dtype_inputs_are_coerced(root, pts):
    # Emulates what Ossify passes: DataFrame.values with an object-dtype column.
    pts_obj = pts.astype(object)
    root_obj = root.astype(object)
    expected = v1dd_ds.streamline().radial_points(root, pts)
    got = v1dd_ds.streamline().radial_points(root_obj, pts_obj)
    assert got.dtype == float
    assert np.allclose(got, expected)


def test_streamline_at_rejects_multiple_anchors(pts):
    sl = v1dd_streamline(version="1.4")  # plain Streamline (base streamline_at)
    with pytest.raises(ValueError, match="single anchor point"):
        sl.streamline_at(pts, pts[:, 1])


def test_transformer_explicit_anchor(root, pts):
    sl = v1dd_ds.streamline()
    fn = sl.transformer(root)
    assert callable(fn)
    assert np.allclose(fn(pts), sl.radial_points(root, pts))
    # accepts object-dtype input too (the Ossify path)
    assert np.allclose(fn(pts.astype(object)), sl.radial_points(root, pts))


def test_transformer_stored_anchor_from_field(root, pts):
    field = v1dd_ds.streamline()
    sl = field.streamline_at_point(root)          # remembers root as the anchor
    fn = sl.transformer()                          # no anchor needed
    assert np.allclose(fn(pts), sl.radial_points(root, pts))


def test_transformer_without_anchor_raises():
    sl = v1dd_streamline(version="1.4")            # no stored anchor
    with pytest.raises(ValueError, match="No anchor available"):
        sl.transformer()
