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
    assert av["transform"]["latest"] == "2.0"
    assert set(av["transform"]["versions"]) == {"1.4", "2.0"}
    assert av["streamline"]["latest"] == "2.0"
    assert set(av["streamline"]["versions"]) == {"1.4", "2.0"}
    # each streamline pins the transform frame its data lives in
    assert av["streamline"]["transform_versions"] == {"1.4": "1.4", "2.0": "2.0"}
    assert v1dd_ds.available_versions() == av
    minnie_sl = available_versions("minnie65")["streamline"]
    assert minnie_sl["latest"] == "1.4"
    assert set(minnie_sl["versions"]) == {"1.4", "2.0"}
    # minnie affine is unchanged, so both streamlines live in the 1.4 frame
    assert minnie_sl["transform_versions"] == {"1.4": "1.4", "2.0": "1.4"}


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
    for factory, dataset in ((v1dd_transform, "v1dd"), (minnie_transform, "minnie65")):
        latest = available_versions(dataset)["transform"]["latest"]
        default = factory()
        pinned = factory(version=latest)
        assert default.version == latest
        assert pinned.version == latest
        assert np.allclose(default.apply(sample_pts), pinned.apply(sample_pts))


def test_streamline_pins_its_transform_version(sample_pts):
    # A streamline must be built against the transform version whose oriented frame its
    # data lives in -- never merely the current latest -- and expose that pin.
    v_latest = v1dd_streamline()
    assert v_latest.transform_version == "2.0"
    assert v_latest._transform.version == "2.0"

    v_hand = v1dd_streamline(version="1.4")
    assert v_hand.transform_version == "1.4"
    assert v_hand._transform.version == "1.4"

    # minnie affine is unchanged, so its field lives in the 1.4 frame
    assert minnie_streamline().transform_version == "1.4"
    assert minnie_streamline()._transform.version == "1.4"


def test_streamline_transform_pin_independent_of_transform_latest(sample_pts):
    # v1dd transform latest is 2.0, but the hand streamline (pinned to 1.4) must still
    # carry the 1.4 frame, i.e. pinning is not silently overridden by the latest label.
    root = sample_pts[0]
    hand = v1dd_streamline(version="1.4")
    tform_14 = v1dd_transform(version="1.4")
    # the streamline's own transform reproduces the pinned-version transform exactly
    assert np.allclose(hand._transform.apply(sample_pts), tform_14.apply(sample_pts))
    # and differs from the latest (2.0) transform, which has a shifted pia point
    assert not np.allclose(
        hand._transform.apply(sample_pts), v1dd_transform().apply(sample_pts)
    )


def test_resolution_equivalence(sample_pts):
    # nm points and the same points in native voxels must transform identically.
    res = np.array([9, 9, 45])
    nm = v1dd_transform("nm").apply(sample_pts)
    vx = v1dd_transform("vx").apply(sample_pts / res)
    explicit = v1dd_transform(res).apply(sample_pts / res)
    assert np.allclose(nm, vx)
    assert np.allclose(nm, explicit)


@pytest.mark.parametrize(
    "streamline_fn,dataset",
    [(v1dd_streamline, "v1dd"), (minnie_streamline, "minnie65")],
)
def test_streamline_version_kinds(streamline_fn, dataset):
    # The default follows the dataset's streamline latest (v1dd -> field, minnie -> hand),
    # but both a hand (1.4) and a field (2.0) are always reachable by explicit version.
    latest = available_versions(dataset)["streamline"]["latest"]
    default = streamline_fn()
    assert default.version == latest

    field = streamline_fn(version="2.0")
    assert isinstance(field, StreamlineField)
    hand = streamline_fn(version="1.4")
    assert isinstance(hand, Streamline) and not isinstance(hand, StreamlineField)
    assert hand.version == "1.4"

    assert isinstance(v1dd_ds.streamline(), StreamlineField)  # v1dd default is the field
    assert not isinstance(minnie_ds.streamline(), StreamlineField)  # minnie default is hand


@pytest.mark.parametrize("streamline_fn", [v1dd_streamline, minnie_streamline])
def test_versioned_streamline_results_differ(streamline_fn, sample_pts):
    root = sample_pts[0]
    field_out = streamline_fn(version="2.0").radial_points(root, sample_pts)
    hand_out = streamline_fn(version="1.4").radial_points(root, sample_pts)
    assert not np.allclose(field_out, hand_out)


def _tiny_field(tform=None):
    x = np.array([0.0, 100.0])
    y = np.array([200.0, 400.0])
    z = np.array([0.0, 100.0])
    return StreamlineField(x, y, z, np.zeros((2, 2, 2, 2)), tform=tform)


def test_field_npz_stamps_and_reads_transform_version(tmp_path):
    # to_npz stamps the build transform's version; from_npz reads it back.
    field = _tiny_field(tform=v1dd_transform())  # .version == "2.0"
    p = str(tmp_path / "f.npz")
    field.to_npz(p)
    assert StreamlineField.from_npz(p).built_transform_version == "2.0"
    # explicit override wins over the attached transform's version
    field.to_npz(p, transform_version="1.4")
    assert StreamlineField.from_npz(p).built_transform_version == "1.4"


def test_field_npz_without_stamp_loads_as_none(tmp_path):
    # Older files predate the stamp; loading must not error and reports None.
    p = str(tmp_path / "legacy.npz")
    np.savez_compressed(
        p,
        x_grid=np.array([0.0, 1.0]),
        y_grid=np.array([0.0, 1.0]),
        z_grid=np.array([0.0, 1.0]),
        tangents=np.zeros((2, 2, 2, 2)),
        confidence=np.empty(0),
        integration_step=np.array([1.0]),
    )
    assert StreamlineField.from_npz(p).built_transform_version is None


def test_stale_field_frame_warns(tmp_path, monkeypatch):
    # A field stamped in a different frame than the streamline version pins must warn.
    import standard_transform.datasets as ds

    stale = tmp_path / "v1dd_streamline_field.npz"
    _tiny_field(tform=ds.v1dd_transform(version="2.0")).to_npz(
        str(stale), transform_version="1.4"  # stamped 1.4, but v1dd sl 2.0 pins 2.0
    )
    real_get = ds._get_data_path
    monkeypatch.setattr(
        ds,
        "_get_data_path",
        lambda fn: str(stale) if fn == "v1dd_streamline_field.npz" else real_get(fn),
    )
    with pytest.warns(UserWarning, match="transform-1.4 frame"):
        ds.v1dd_streamline()


def test_deprecated_aliases_warn_and_work(sample_pts):
    from standard_transform import v1dd_transform_nm, v1dd_streamline_nm

    with pytest.warns(DeprecationWarning):
        legacy_tform = v1dd_transform_nm()
    assert np.allclose(legacy_tform.apply(sample_pts), v1dd_transform("nm").apply(sample_pts))

    with pytest.warns(DeprecationWarning):
        v1dd_streamline_nm()

    with pytest.warns(DeprecationWarning):
        _ = v1dd_ds.streamline_nm
