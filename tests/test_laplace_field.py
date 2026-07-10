import numpy as np
import pytest

from standard_transform import StreamlineField, streamline_field_from_paths
from standard_transform.streamlines import _laplace_field


def _grid(nx=10, ny=12, nz=10, bs=(30.0, 20.0, 30.0)):
    xs = (np.arange(nx) + 0.5) * bs[0]
    ys = (np.arange(ny) + 0.5) * bs[1]
    zs = (np.arange(nz) + 0.5) * bs[2]
    return np.meshgrid(xs, ys, zs, indexing="ij"), bs


def test_laplace_fit_recovers_gradient_field():
    (X, Y, Z), bs = _grid()
    # a known potential; its horizontal gradient is the "true" tangent field
    psi = 2e-4 * X * Y + 1e-4 * Z * Y
    truth = np.stack([np.gradient(psi, bs[0], axis=0), np.gradient(psi, bs[2], axis=2)], -1)
    rec = _laplace_field(truth, np.ones(X.shape), bs, mode="fit", lam_rel=0.02)
    scale = np.sqrt((truth**2).sum(-1)).mean()
    assert np.sqrt(((rec - truth) ** 2).sum(-1)).mean() < 0.1 * scale


def test_laplace_fit_is_curl_free():
    (X, Y, Z), bs = _grid()
    # corrupt a gradient field with a non-integrable (rotational) component
    noisy = np.stack([1e-4 * Z, -1e-4 * X], axis=-1)
    rec = _laplace_field(noisy, np.ones(X.shape), bs, mode="fit", lam_rel=0.02)
    curl = np.gradient(rec[..., 0], bs[2], axis=2) - np.gradient(rec[..., 1], bs[0], axis=0)
    assert np.abs(curl[1:-1, 1:-1, 1:-1]).mean() < 1e-12


def _toy_paths(n=40, seed=0):
    rng = np.random.default_rng(seed)
    yg = np.linspace(80.0, 760.0, 120)
    out = []
    for _ in range(n):
        x0, z0 = rng.uniform(-150, 150), rng.uniform(-150, 150)
        x = x0 + 0.1 * yg + rng.normal(0, 2, yg.size)
        z = z0 + 0.03 * yg + rng.normal(0, 2, yg.size)
        out.append(np.column_stack([x, yg, z])[::-1])  # tip -> root, already oriented
    return out


@pytest.mark.parametrize("method", ["laplace-fit", "laplace-bc"])
def test_streamline_field_from_paths_laplace_methods(method):
    field = streamline_field_from_paths(
        _toy_paths(), transform_points=False, method=method, depth_band=(150.0, 700.0)
    )
    assert isinstance(field, StreamlineField)
    # a streamline through a point returns finite x, z across the band
    fx, fz = field.streamline_at(np.array([0.0, 400.0, 0.0]), np.linspace(200.0, 650.0, 10))
    assert np.all(np.isfinite(fx)) and np.all(np.isfinite(fz))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        streamline_field_from_paths(_toy_paths(), transform_points=False, method="nope")


def test_edge_extrapolation_holds_vs_collapses():
    # Data only in a central x-block with a consistent tilt; the margins are empty.
    from standard_transform.streamlines import _laplace_field

    nx, ny, nz, bs = 30, 20, 30, (30.0, 20.0, 30.0)
    prec = np.zeros((nx, ny, nz))
    prec[8:22] = 5.0
    mean_t = np.zeros((nx, ny, nz, 2))
    mean_t[8:22, :, :, 0] = 0.20

    harmonic = _laplace_field(mean_t, prec, bs, mode="fit", lam_rel=0.05, edge_extrapolation="harmonic")
    hold = _laplace_field(mean_t, prec, bs, mode="fit", lam_rel=0.05, edge_extrapolation="hold")

    # data region unchanged by the extrapolation choice
    assert np.allclose(harmonic[8:22, :, :, 0].mean(), hold[8:22, :, :, 0].mean(), atol=0.02)
    # harmonic relaxes the edge tangent toward vertical (0); hold continues the trend
    assert abs(harmonic[0, :, :, 0].mean()) < 0.03
    assert hold[0, :, :, 0].mean() > 0.15


def test_build_provenance_roundtrips(tmp_path):
    field = streamline_field_from_paths(
        _toy_paths(), transform_points=False, method="laplace-fit", depth_band=(150.0, 700.0)
    )
    p = str(tmp_path / "f.npz")
    field.to_npz(p, transform_version="2.0", method="laplace-fit", laplace_strength=0.05)
    loaded = StreamlineField.from_npz(p)
    assert loaded.built_transform_version == "2.0"
    assert loaded.build_method == "laplace-fit"
    assert loaded.build_laplace_strength == 0.05


def test_build_provenance_absent_is_none(tmp_path):
    # A field saved without method/lambda (e.g. diffusion) reports None, not an error.
    field = streamline_field_from_paths(_toy_paths(), transform_points=False, depth_band=(150.0, 700.0))
    p = str(tmp_path / "f.npz")
    field.to_npz(p)
    loaded = StreamlineField.from_npz(p)
    assert loaded.build_method is None
    assert loaded.build_laplace_strength is None
