# Design: cortical layer support

**Status:** draft / design only (no implementation yet)
**Scope of this doc:** how `standard_transform` would represent, fit, store, version, and
query cortical layer boundaries.

## Motivation

Cortical layer boundaries (pia, L1/L2-3, … , L6/white-matter) are surfaces that

1. **live in the oriented (transformed) coordinate frame** — "depth" only means anything
   after the affine transform puts the pial surface near `y = 0` with y running
   pia→white-matter, and
2. **vary in space** — the depth of a given boundary changes with lateral position
   (e.g. layer thicknesses differ across VISp vs the HVAs).

Because of (1) a layer definition is meaningless outside this package's transform, and
because of (2) it is the same kind of *spatially-varying scalar field in the oriented
frame* that we already build for streamlines. So layers belong here, versioned against
the transform they were defined in — not as a standalone external annotation. This is
the same "doesn't exist outside the tooling → version it against its dependencies"
argument that motivated pinning streamlines to their transform frame.

## Decisions already made

| Decision | Choice | Consequence |
|---|---|---|
| Boundary representation | **Raw-y surfaces** `y_b(x, z)` in oriented microns | Cheap 2-D lookup + y-threshold to classify; **no dependency on the streamline field** |
| Data source | **Fit from layer-labeled cells** | Spatially-varying by construction; needs labeled soma data |
| Frame dependency | **Transform only** | A layer set pins `transform_version`; no streamline pin |

The raw-y choice means layer assignment thresholds a point's vertical `y` against the
interpolated boundary surfaces at its `(x, z)`. It deliberately **ignores streamline
tilt** near boundaries (a point is classified by vertical depth, not depth-along-
streamline). This is an accepted approximation: layer bands are thick relative to the
lateral error introduced by tilt, and it keeps layers decoupled from the streamline
field. (Defining boundaries as level sets of depth-along-streamline was the more faithful
alternative; see *Alternatives*.)

## Data model — `data/<dataset>_layers.npz`

A layer set is a stack of `B` ordered boundary surfaces defining `B − 1` layers.

| Key | Shape | Meaning |
|---|---|---|
| `x_grid`, `z_grid` | `(nx,)`, `(nz,)` | Ascending oriented-micron node coords over the (x,z) plane |
| `boundary_depths` | `(B, nx, nz)` | Raw-y depth of each boundary at each column |
| `boundary_names` | `(B,)` | e.g. `["pia","L1/L23","L23/L4","L4/L5","L5/L6","L6/WM"]` |
| `layer_names` | `(B-1,)` | Labels between consecutive boundaries, e.g. `["L1","L2/3","L4","L5","L6"]` |
| `confidence` | `(B, nx, nz)` | Per-node fit precision (coverage signal; optional) |
| `transform_version` | scalar str | Provenance: the transform frame the surfaces live in |
| `build_*` | scalars | Fit metadata (n_cells, estimator, bin size) |

The grid is in post-transform microns and is therefore unit-agnostic — the same file
serves nm and voxel variants; only the attached transform differs (mirrors
`StreamlineField.from_npz`). One `RegularGridInterpolator((x_grid, z_grid), …)` per
boundary supplies depths at arbitrary `(x, z)`.

`boundary_depths` includes the two end boundaries and pia, but see *Boundary taxonomy*:
pia is supplied by the transform (flat y ≈ 0), not fit, so its row may be a constant
plane (or the pia name may be omitted and treated implicitly). The outer edges (top of
L2/3, bottom of L6) will carry systematically lower `confidence` than the internal ones.

## Class `LayerSet`

Parallels `StreamlineField`: constructed with the grid + surfaces, carries a transform,
persisted via `to_npz`/`from_npz` with provenance stamps.

```python
ls = minnie_ds.layers()                     # versioned LayerSet, transform attached

ls.layer_at(xyz)                            # -> layer label (or index) per point
ls.boundary_depth("L4/L5", xyz)             # -> that boundary's raw-y depth at each (x,z)
ls.thickness("L4", xyz)                     # -> y_below - y_above at each (x,z)
ls.coverage_at(xyz)                         # -> per-point fit confidence (edges ~0)
ls.boundaries()                             # -> surfaces, for plotting / neuroglancer
```

**`layer_at` algorithm** (the whole classifier):
1. `xyz → oriented (x, y, z)` via the attached transform.
2. Interpolate the `B` boundary depths at each point's `(x, z)` → an increasing stack.
3. `searchsorted` the point's `y` into that stack → layer index.
4. `y` above the pia boundary → "above pia"/`None`; below the last boundary → "WM".

No streamline integration; a batched 2-D interpolation + threshold.

## Boundary taxonomy (not all boundaries are the same)

Layers here are defined by **excitatory** neuron populations, and that makes the two end
boundaries structurally different from the internal ones, because L1 has essentially no
excitatory neurons and white matter has no neuronal somata. So a boundary is one of three
kinds:

- **Pia (top of L1) — *given by the frame*, not fit.** The transform already flattens pia
  to y ≈ 0, and no cell population defines it. So pia is a flat y ≈ 0 plane (or a small
  offset), and L1's extent is simply `[pia, top-of-L2/3]`. In an excitatory-defined
  scheme, **L1 is a definitional band with nothing inside it to fit** — its lower bound
  (top of L2/3) is real, but its interior is empty of the cells we're using.
- **Outer edges (L1/L2-3 = top of L2/3, and L6/WM = bottom of L6) — *one-sided*.** Cells
  on one side, empty on the other. These are **density-onset / edge** estimates on a
  *single* population, not separations between two.
- **Internal interfaces (L2-3/L4, L4/L5, L5/L6) — *two-sided*.** A labeled population on
  each side; the well-posed 2-class separation.

**Unification (one code path):** treat pia and WM as empty "background" pseudo-populations
and run the *same* per-column separator on every adjacent pair. When one side is empty the
separator degenerates to an onset detector, so the edge cases fall out of the internal
case rather than needing special handling. Pia stays special only in that it's supplied by
the transform rather than estimated.

**Confidence asymmetry:** the outer edges are inherently noisier — an edge/onset of one
distribution has far higher variance than a 50% crossing between two, and it is sensitive
to stragglers and to the shape of the density falloff. So L1/L2-3 and L6/WM get more
smoothing and lower `coverage`, and consumers should treat them as softer than the
internal boundaries. This is consistent with the coverage story: they are the
least-determined surfaces.

## Fitter `fit_layers_from_cells`

Structurally the **same pipeline as the streamline-field fit**, estimating a *scalar
depth* per column instead of a tangent — so it reuses the binning, precision-weighted
smoothing / harmonic fill, and provenance machinery in `streamlines.py`.

Input: labeled somata — positions (nm or oriented) + a per-cell layer label.

1. **Transform + bin** somata over the `(x, z)` grid.
2. **Per column, per adjacent pair**, estimate the interface depth `y_b` with a single
   separator that handles all three boundary kinds (see *Boundary taxonomy*): a 2-class
   crossing for internal interfaces, degenerating to a one-sided density-onset for the
   outer edges (empty side), with pia taken from the transform rather than estimated.
   Recommended separator: logistic `P(deeper | y) = 0.5` crossing.
3. **Precision-weighted smooth + harmonic fill** the per-column estimates into a surface
   (the existing `_precision_smooth` / Laplace-fit code on a 2-D scalar grid). Confidence
   = column count × separation sharpness, so the noisier outer edges are down-weighted and
   smoothed more.
4. **Enforce monotonic ordering** per column (`y_{b+1} ≥ y_b`) so surfaces cannot cross
   where data is thin.

Because it reuses the field fitter, the same CV-based smoothness selection
(`--tune-lambda` analog) applies if we use the Laplace-fit smoother.

If L1 needs interior definition (it has no excitatory cells), the only way to anchor it
from data is to bring in **inhibitory** neuron layer calls — which changes the definition
from "excitatory-defined layers" to a mixed one. That's a deliberate scope choice, not a
default; see *Open questions*.

## Versioning & `Dataset` integration

- New registry `_LAYER_VERSIONS[dataset][label] = {file, transform_version, introduced_in}`,
  parallel to `_STREAMLINE_VERSIONS` but pinning **only** `transform_version`.
- `Dataset.layers(resolution="nm", version=None)` mirrors `transform()` / `streamline()`:
  build the transform at the pinned version, load the npz, attach.
- `available_versions(dataset)` gains a `"layers"` track (with `transform_versions`).
- Same stamp-and-guard as the field: the npz records its build transform frame, and
  loading against a mismatched transform warns.

Dependency graph after this: **transform → {streamline, layers}** (layers a sibling of
streamlines, both children of the transform; no layer↔streamline edge under the raw-y
choice).

## Build & visualization

- `scripts/build_layers.py --cells <labeled_cells> --dataset <name>` — fit and write the
  npz, with an optional held-out CV report (per-boundary median depth error) mirroring
  `build_streamline_field.py --validate` / `--tune-lambda`.
- Extend the neuroglancer script (`make_streamline_grid_state.py` sibling) to emit
  boundary surfaces (as meshes or gridded line/point annotations) and/or layer-colored
  slabs, in the same self-contained-link + CAVE-upload modes.

## Open questions / decisions still needed

1. **Per-column estimator.** Logistic `P=0.5` crossing (recommended, robust) vs midpoint
   of adjacent extents vs max-margin threshold — for the *internal* interfaces.
2. **Outer-edge estimator.** How to place the one-sided boundaries (top of L2/3, bottom
   of L6): density-onset relative to the column's peak density, or a robust depth
   quantile of the bounding population (which percentile?). Higher-variance than the
   internal case, so the choice matters more.
3. **Pia.** Take pia as flat y ≈ 0 (from the transform), a fitted small offset, or fit
   it from an external pial-surface mesh if one is available? And is L1 left as a
   definitional `[pia, top-of-L2/3]` band, or anchored using **inhibitory** layer calls
   (which changes the definition to mixed excitatory+inhibitory)?
4. **Layer scheme & names, per dataset.** Which boundaries/labels (e.g. is L2/3 split?
   WM as a terminal half-open band?). Likely differs between minnie65 and v1dd.
5. **Labeled-cell input format.** Assume a dataframe with split position columns +
   a layer-label column, reusing `utils` split-column detection? Which label vocabulary?
6. **Grid resolution over (x,z).** Layers are smoother than the tangent field, so a
   coarser bin than the field's (30, ·, 30) is probably right.
7. **Coverage semantics at edges.** Same story as the field: surfaces are harmonically
   extended into data-free columns and flagged by `coverage_at ≈ 0`; confirm we surface
   (not mask) and let callers threshold.
8. **Monotonicity enforcement method.** Simple per-column clip (`y_b = max(y_b, y_{b-1})`)
   vs isotonic regression across boundaries.

## Alternatives considered (not chosen)

- **Streamline-depth boundaries** (level sets of depth-along-streamline): more faithful
  to curvature and more parsimonious (a flat-in-depth boundary auto-curves via the
  field), but couples layers to the streamline field and needs per-point streamline
  integration to classify. Rejected in favor of the simpler raw-y surfaces; revisit if
  tilt-induced misclassification near boundaries proves material.
- **3-D voxel label field**: heavier, non-parametric, no smooth surface to reason about.

## Future / related

- The same variational fitter generalizes to **arbitrary boundary surfaces**, not just
  laminar cortex — e.g. **nucleus/structure boundaries**, where you *have* the bounding
  surface and want interior structure. That's the boundary-term direction discussed for
  the Laplace fit, populated from the other side.
- If equivolumetric normalization is ever wanted, it would layer on top as a depth
  reparametrization, independent of this layer-boundary representation.
