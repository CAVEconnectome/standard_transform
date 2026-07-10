# Streamline field: method

This page explains how a [`StreamlineField`](concepts/streamlines-vs-field.md) is
estimated from data and why each step is there. It is written to be
**dataset-agnostic** — the procedure is the same regardless of which dataset the
paths come from; only the numbers change. The concrete figures below are from v1dd
apical dendrites and are illustrative, not universal.

## Input

The field is estimated from a large set of **tall single paths** — one per neuron,
running from a deep soma up the apical trunk to a tip in upper cortex, ordered along
the neurite. Each path traces, noisily, the local pia → white matter direction.
Individual dendrites wander; the axis they *collectively* trace does not.

The construction has four steps.

## 1. Local tangents

Each path is transformed into oriented microns, and every segment contributes its
local orientation — the tangent `(dx/dy, dz/dy)`, the direction of the cortical axis
at that point — binned onto a 3D grid by the segment's midpoint. A grid cell's
tangent is the **mean of the segments that land in it**, so per-path noise averages
out.

Near-horizontal segments (with `|dy|` below a small threshold, `min_dy`) are
discarded: a nearly flat segment says little about the up direction and its
`(dx/dy, dz/dy)` blows up numerically.

## 2. Depth band

Tangent reliability is strongly **depth-dependent**. Measuring it directly — as the
spread (standard deviation) of tangents within each grid cell versus depth — shows a
reliable middle band flanked by noisy ends. For v1dd apicals:

| depth (post-transform µm) | median per-cell tangent std |
|---|---|
| y < 150 (tuft / near pia) | ~6.1 |
| 150–700 (apical trunk)    | ~0.8 |
| y > 700 (deep tail)       | ~3.1 |

The ends are 5–8× noisier than the middle: near the pia the apical tuft splays out
and no longer points "up," and below the somata there is little apical data. So only
segments **within the band** inform the field, and the grid spans exactly the band.

The band is dataset-specific. Repeating this per-cell-std measurement on the Minnie65
apical paths gives a flat reliable floor (median per-cell tangent std ≈ 0.4–0.6) from
150 µm down to **650 µm**, above which both the spread rises and the number of paths
reaching that depth collapses (≈2000 → ≈900 per 50 µm slab). So Minnie65 ships a
slightly shallower band, `[150, 650]` — its apical data simply does not extend as deep
as v1dd's. Always re-derive the band from the data rather than reusing another
dataset's numbers.

Outside the band the field **holds the edge orientation constant** — a query depth
beyond the band is clamped to the band edge — so streamlines simply continue
*straight* where the data can no longer be trusted, rather than extrapolating a
drifting curve. The band is the `depth_band` parameter; choose it per dataset by
looking at where the per-cell spread rises.

## 3. Precision

Not all in-band cells are equally trustworthy: some rest on few segments, some on
segments that disagree. Each cell gets a **precision**

```
precision = count / (variance + prior)
```

combining how much data it saw (`count`) with how much that data agreed
(`variance`). The `prior` — roughly the typical variance of well-sampled cells —
shrinks single-sample cells toward modest confidence instead of false certainty.

This precision is what `coverage_at` reports, and it drives the next step.

## 4. Precision-weighted diffusion

Finally the mean field is regularized by a **diffusion** in which each cell is
repeatedly replaced by a precision-weighted blend of itself and its 6 grid
neighbors. Three things happen at once:

- **well-determined cells barely move** and anchor the result;
- **noisy cells are pulled toward their better-determined neighbors**;
- **empty cells** (grid corners with no data) **fill in** from the nearest real data
  as precision diffuses outward.

`smoothing_strength` sets the neighbor coupling and `smoothing_passes` how many
passes run (default: enough to propagate across the grid).

## Validation

On held-out paths, each field tracks real dendrites better than the dataset's old
hand-drawn streamline over the same region — while additionally adapting to local axis
curvature and absorbing regional alignment drift that a single fixed streamline cannot
represent. The `--validate N` flag of the build script reproduces these held-out checks
(it holds out `N` paths, rebuilds on the rest, and reports the in-band median lateral
deviation):

| dataset | field median deviation | hand-drawn streamline, same paths |
|---|---|---|
| v1dd    | ~11 µm | ~13 µm |
| Minnie65 | ~7.5 µm | ~10.2 µm |

## Shipped fields

Both fields were built with `scripts/build_streamline_field.py` at `bin_size =
(30, 20, 30)` µm from ~17–18k apical paths, and are stored as
`standard_transform/data/<dataset>_streamline_field.npz` (grids in unit-agnostic
post-transform microns — the same file serves the nm and voxel variants).

| dataset | depth band (µm) | grid (x × y × z) | # paths |
|---|---|---|---|
| v1dd    | `[150, 700]` | 40 × 28 × 28 | ~17k |
| Minnie65 | `[150, 650]` | 35 × 25 × 17 | ~18k |

Reproduce:

```bash
uv run python scripts/build_streamline_field.py --paths <dir> --dataset minnie \
    --depth-band 150 650 --validate 800
```

## Choosing parameters for a new dataset

- **`depth_band`** — plot per-cell tangent std vs. depth and keep the reliable
  middle. v1dd `[150, 700]` and Minnie65 `[150, 650]` bracket a sensible range, but
  neither is universal — re-derive it from the new dataset.
- **`bin_size`** — small enough to resolve real curvature, large enough that cells
  see multiple segments. The shipped fields use `(30, 20, 30)` µm.
- **`prior` / smoothing** — increase regularization when paths are sparse or noisy.

## Visualizing a field

`make_streamline_grid_state.py` (repo root, a `uv run` script) integrates one
streamline per `(x, z)` seed across a field and emits them as neuroglancer `line`
annotations for QC — a quick way to eyeball the curves against real apicals:

```bash
uv run make_streamline_grid_state.py --dataset minnie   # self-contained link
uv run make_streamline_grid_state.py --dataset minnie --datastack minnie65_phase3_v1
```

See [Build a field for a new dataset](guides/build-a-field.md) for the mechanics.
