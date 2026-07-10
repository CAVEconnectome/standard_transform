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

## 4. Regularization: a curl-free potential fit

The binned mean tangents are noisy and have gaps, so they are regularized into a smooth
field. The default method (`method="laplace-fit"`) models the field as the horizontal
gradient of a scalar **depth potential** and solves for that potential by a
precision-weighted least squares with a Laplacian smoothness term. Three properties
follow:

- **Fits the data.** Where a cell has data the fitted field matches its binned tangent,
  weighted by the cell's precision, so well-sampled cells dominate and noisy ones are
  pulled toward their better-determined neighbors.
- **Curl-free by construction.** Because the field is the gradient of a single scalar it
  is integrable: streamlines derive from one consistent depth function and cannot cross
  or form caustics — a guarantee the older independent-component smoothing could not make.
- **Harmonic gap-fill.** Where a cell has no data the fit reduces to Laplace's equation,
  so empty cells fill by the smoothest (minimal-energy) interpolation of their neighbors.

The smoothness weight `laplace_strength` (λ) trades data-fidelity against smoothness, and
it is **not hand-set**: `--tune-lambda` selects it by held-out cross-validation (a k-fold
λ sweep, taking the most-regularized value within one standard error of the CV minimum).
Both shipped datasets select λ ≈ 0.05.

### Edge behavior

There are two distinct "outside the data" cases:

- **Beyond the depth band (in `y`):** query depths past the band are clamped to the band
  edge, so streamlines continue *straight* rather than extrapolating a drifting curve
  (see step 2).
- **Data-poor cells laterally (in `x, z`):** by default (`edge_extrapolation="hold"`) the
  field **continues the nearest in-data orientation** into unsampled margins instead of
  letting the harmonic solution relax toward vertical. Relaxing to vertical would collapse
  depth-integrated quantities (e.g. lateral displacement) to ~0 at the margins; holding
  the trend keeps them approximately right, so edge cells need no special-casing. Those
  cells are no longer curl-free, but they carry no data — `coverage_at` reads ~0 there.

The earlier method, `method="diffusion"`, instead smooths the two tangent components
independently by a precision-weighted diffusion (no integrability constraint). It stays
available for reproducing fields built that way.

## Validation

On held-out paths the field tracks real dendrites better than both the dataset's old
hand-drawn streamline and the earlier diffusion field, while adapting to local curvature
and absorbing regional alignment drift a single fixed streamline cannot represent. The
`--tune-lambda` sweep and the `--validate N` flag reproduce these held-out checks. For
v1dd (5-fold cross-validation, ~17k apical paths), per-path median in-band lateral
deviation:

| method | median deviation |
|---|---|
| `laplace-fit` (default) | ~8.7 µm |
| `diffusion` | ~9.7 µm |
| hand-drawn streamline | higher still |

So the potential fit is both **more accurate** than the diffusion field and **curl-free**
(zero streamline crossings). Re-run the sweep per dataset — the optimum λ and the exact
numbers differ.

## Shipped fields

Both fields were built with `scripts/build_streamline_field.py` using `method="laplace-fit"`
(the default) at `bin_size = (30, 20, 30)` µm and CV-selected `laplace_strength ≈ 0.05`
from ~17–18k apical paths, and are stored as
`standard_transform/data/<dataset>_streamline_field.npz` (grids in unit-agnostic
post-transform microns — the same file serves the nm and voxel variants). Each `.npz`
also stamps its build provenance: the transform frame it was built in, the method, and λ.

| dataset | depth band (µm) | grid (x × y × z) | transform frame | # paths |
|---|---|---|---|---|
| v1dd    | `[150, 700]` | 40 × 28 × 28 | `2.0` | ~17k |
| Minnie65 | `[150, 650]` | 35 × 25 × 17 | `1.4` | ~18k |

The build always builds in the transform version its streamline registry entry **pins**
(not merely the latest transform), so a field always matches its registry slot's frame.

Reproduce (λ chosen by CV, edges held by default):

```bash
uv run python scripts/build_streamline_field.py --paths <dir> --dataset minnie \
    --depth-band 150 650 --tune-lambda --validate 800
```

## Choosing parameters for a new dataset

- **`depth_band`** — plot per-cell tangent std vs. depth and keep the reliable
  middle. v1dd `[150, 700]` and Minnie65 `[150, 650]` bracket a sensible range, but
  neither is universal — re-derive it from the new dataset.
- **`bin_size`** — small enough to resolve real curvature, large enough that cells
  see multiple segments. The shipped fields use `(30, 20, 30)` µm.
- **`laplace_strength` (λ)** — the fit's one smoothness knob; don't guess it, let
  `--tune-lambda` pick it by cross-validation on the dataset at hand (≈0.05 for both
  shipped datasets, but re-tune).
- **`edge_extrapolation`** — `"hold"` (default) continues the trend into data-poor
  margins; `"harmonic"` recovers the old relax-to-vertical behavior.

## Visualizing a field

`make_streamline_grid_state.py` (repo root, a `uv run` script) integrates one
streamline per `(x, z)` seed across a field and emits them as neuroglancer `line`
annotations for QC — a quick way to eyeball the curves against real apicals:

```bash
uv run make_streamline_grid_state.py --dataset minnie   # self-contained link
uv run make_streamline_grid_state.py --dataset minnie --datastack minnie65_phase3_v1
```

See [Build a field for a new dataset](guides/build-a-field.md) for the mechanics.
