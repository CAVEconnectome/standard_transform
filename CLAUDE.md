# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`standard_transform` provides pre-baked affine transforms and curvilinear "streamlines" for orienting points from EM connectomics datasets (Minnie65 and V1dd) into a consistent frame: microns, with the y-axis running pia→white matter and the pial surface at approximately y=0. See README.md for the full user-facing API and worked examples.

## Development commands

This project uses `uv` and [poethepoet](https://poethepoet.natn.io/) (`poe`) tasks defined in `pyproject.toml`.

- **Run tests with coverage:** `uv run poe test` (equivalent to `uv run pytest --cov=standard_transform tests`)
- **Run a single test:** `uv run pytest tests/test_tform.py::test_name` or `uv run pytest -k pattern`
- **Lint:** `uv run ruff check` — note the ruff config only selects a minimal set of error codes (`E9,F63,F7,F82`); pre-commit additionally runs import sorting (`I`) and `ruff-format`. Notebooks are excluded from ruff.
- **Version bump (also git-tags and commits):** `uv run poe bump patch|minor|major` (dry run: `uv run poe drybump patch`). Bumping updates `__init__.py` and `pyproject.toml`, runs `uv sync`, and tags `v{version}`.
- **Docs preview:** `uv run poe doc-preview` (mkdocs)
- **Profiling:** `uv run poe profile` (pyinstrument) or `uv run poe profile-all` (scalene)

Note: tests live in `tests/` (the empty `test/` dir is unused). The `build/` and `dist/` dirs contain build artifacts, not source — never edit them.

## Architecture

The package is small and layered. `standard_transform/__init__.py` re-exports only the dataset-specific factory functions and `Dataset` instances — that is the public surface.

- **`base.py`** — the transform engine. A `TransformSequence` holds an ordered list of primitive transforms (`ScaleTransform`, `TranslateTransform`, `RotationTransform`, each with `apply`/`invert`). `apply` runs them forward; `invert` runs them in reverse order with each primitive inverted. All the input-shape polymorphism lives here: `apply`/`invert` dispatch on whether input is a `pd.Series` (of 3-vectors) vs an array; `apply_project` extracts a single axis (x/y/z ↔ 0/1/2); `apply_dataframe` accepts a column name and auto-detects split position columns; `apply_skeleton`/`apply_meshwork_vertices`/`apply_meshwork_annotations` transform MeshParty objects (imported lazily/duck-typed — MeshParty is not a dependency).

- **`streamlines.py`** — the `Streamline` class models the curvilinear depth axis. It stores streamline points in *post-transform* space and builds 1-D interpolators giving x and z as functions of depth y. Every method takes a `transform_points` flag: `True` (default) means inputs are in the original pre-transform coordinates and get transformed first. Key methods: `streamline_at` (x,z of the streamline at a depth), `radial_distance` (in-plane distance accounting for curvature), `depth_along`/`depth_between` (path length along the curve), `radial_points` (map points into cylindrical-like radial/depth coordinates), and the `transform_*` methods that straighten skeletons/meshworks along the streamline (relocating root to origin).
  - `StreamlineField(Streamline)` is the spatially-varying version: instead of one curve it holds a 3D grid of local tangent vectors `(dx/dy, dz/dy)` and integrates the streamline through any anchor, so the shape varies across the volume (a field uniform in x,z reduces to a plain `Streamline`). It overrides only `__init__`, `streamline_at` (now integrates the field), and `streamline_points_tform`; everything else is inherited. `streamline_at_point(soma)` returns a plain `Streamline` for a neuron — the intended per-cell workflow (integrate once at the soma, apply that fixed curve across the arbor). Query coords are clamped into the grid, so orientation is held constant outside the sampled depth band. `to_npz`/`from_npz` persist the grid (transform not stored — reattached on load). Build with `streamline_field_from_paths`, which bins per-segment tangents, restricts to a reliable `depth_band`, weights cells by precision `count/(var+prior)`, and regularizes via a precision-weighted diffusion (`_precision_smooth`). The method rationale (how the band and precision/diffusion were chosen) is written up in README.md under "Spatially-varying streamlines".
  - **The field is the default streamline.** `v1dd_streamline_nm/_vx()` (and `v1dd_ds.streamline_nm/_vx`, which are lazy properties) return the `StreamlineField` loaded from `data/v1dd_streamline_field.npz`; pass `legacy=True` to the module functions for the old hand-drawn json streamline. This was an invisible-API but breaking-results change. Both v1dd and Minnie now default to a field (each `data/<dataset>_streamline_field.npz`, registry version "2.0"); `version="1.4"` recovers each dataset's hand json. The field npz is generated offline from ~17-18k skeleton apical paths (tall single dendritic paths, tip→root, in nm), not rebuilt at runtime. Reproduce it with `uv run python scripts/build_streamline_field.py --paths <dir> --dataset v1dd|minnie` (add `--validate N` for a held-out QC deviation). Depth bands differ per dataset — chosen from where per-cell tangent std rises: v1dd `[150, 700]`, Minnie `[150, 650]` (Minnie apical data thins out below ~650µm). `make_streamline_grid_state.py` (repo root, a `uv run` script) emits a neuroglancer QC visualization of the streamline grid.

- **`datasets.py`** — dataset-specific constants and factories. Defines voxel resolutions and pia reference points, the per-dataset rotation/translation/scaling recipes (`_minnie_transforms`, `_v1dd_transforms`), and the public factory functions (`minnie_transform_nm/vx`, `v1dd_transform_nm/vx`, `*_streamline_nm/vx`). The `Dataset` class bundles a transform + streamline for each dataset; the exported singletons `minnie_ds` and `v1dd_ds` are the recommended entry points. Streamline point data is loaded from JSON in `standard_transform/data/`.

- **`utils.py`** — dataframe helpers, chiefly split-position-column detection. Supports two naming conventions: type 1 (`pt_position_x/y/z`) and type 2 (`pt_position_x_suffix`); ambiguity between them raises.

### Conventions when extending

- Every transform primitive must implement both `apply` and `invert` so `TransformSequence.invert` round-trips.
- Transforms and streamlines come in paired `_nm` (nanometer input) and `_vx` (voxel input, with a `voxel_resolution` argument) variants; a `_vx` factory scales by resolution first, then delegates to the shared transform recipe. Follow this pattern when adding a dataset.
- Pia points are stored in nm; voxel variants derive their pia point by dividing by resolution.
