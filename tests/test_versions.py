import numpy as np
import pytest

from standard_transform import (
    Streamline,
    StreamlineField,
    available_versions,
    minnie_ds,
    minnie_streamline,
    v1dd_ds,
    v1dd_streamline,
    v1dd_transform,
    minnie_transform,
)


@pytest.fixture()
def sample_pts():
    return np.array(
        [
            [817335.0, 611523.0, 336240.0],
            [800000.0, 500000.0, 300000.0],
            [900000.0, 700000.0, 400000.0],
        ]
    )


def test_available_versions_shape():
    av = available_versions("v1dd")
    assert av["transform"]["latest"] == "1.4"
    assert av["streamline"]["latest"] == "2.0"
    assert set(av["streamline"]["versions"]) == {"1.4", "2.0"}
    assert v1dd_ds.available_versions() == av
    minnie_sl = available_versions("minnie65")["streamline"]
    assert minnie_sl["latest"] == "2.0"
    assert set(minnie_sl["versions"]) == {"1.4", "2.0"}


def test_unknown_version_raises():
    with pytest.raises(ValueError, match="Unknown streamline version"):
        v1dd_streamline(version="nope")
    with pytest.raises(ValueError, match="Unknown transform version"):
        v1dd_transform(version="9.9")
    with pytest.raises(ValueError, match="Unknown dataset"):
        available_versions("not_a_dataset")


def test_bad_resolution_string_raises():
    with pytest.raises(ValueError, match="resolution string must be"):
        v1dd_transform("microns")


def test_transform_latest_matches_explicit(sample_pts):
    for factory in (v1dd_transform, minnie_transform):
        default = factory()
        pinned = factory(version="1.4")
        assert default.version == "1.4"
        assert pinned.version == "1.4"
        assert np.allclose(default.apply(sample_pts), pinned.apply(sample_pts))


def test_resolution_equivalence(sample_pts):
    # nm points and the same points in native voxels must transform identically.
    res = np.array([9, 9, 45])
    nm = v1dd_transform("nm").apply(sample_pts)
    vx = v1dd_transform("vx").apply(sample_pts / res)
    explicit = v1dd_transform(res).apply(sample_pts / res)
    assert np.allclose(nm, vx)
    assert np.allclose(nm, explicit)


@pytest.mark.parametrize("streamline_fn", [v1dd_streamline, minnie_streamline])
def test_streamline_version_kinds(streamline_fn):
    default = streamline_fn()
    hand = streamline_fn(version="1.4")
    assert isinstance(default, StreamlineField)
    assert default.version == "2.0"
    assert isinstance(hand, Streamline) and not isinstance(hand, StreamlineField)
    assert hand.version == "1.4"
    assert isinstance(minnie_ds.streamline(), StreamlineField)


@pytest.mark.parametrize("streamline_fn", [v1dd_streamline, minnie_streamline])
def test_versioned_streamline_results_differ(streamline_fn, sample_pts):
    root = sample_pts[0]
    field_out = streamline_fn().radial_points(root, sample_pts)
    hand_out = streamline_fn(version="1.4").radial_points(root, sample_pts)
    assert not np.allclose(field_out, hand_out)


def test_deprecated_aliases_warn_and_work(sample_pts):
    from standard_transform import v1dd_transform_nm, v1dd_streamline_nm

    with pytest.warns(DeprecationWarning):
        legacy_tform = v1dd_transform_nm()
    assert np.allclose(legacy_tform.apply(sample_pts), v1dd_transform("nm").apply(sample_pts))

    with pytest.warns(DeprecationWarning):
        v1dd_streamline_nm()

    with pytest.warns(DeprecationWarning):
        _ = v1dd_ds.streamline_nm
