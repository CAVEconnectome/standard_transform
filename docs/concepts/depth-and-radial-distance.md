# Depth vs. radial distance in laminar cortex

## Why a straight axis isn't enough

Mammalian cortex is strongly **laminar**: neurons, connectivity, and cell types are
organized in layers stacked from the pial surface down to white matter. Because of
that organization, two very different questions constantly come up:

- **Depth:** how far *along the pia → white matter direction* is a point? (Which
  layer is it in?)
- **Radial distance:** how far apart are two points *within the laminar plane*,
  i.e. tangential to the layers?

The [oriented frame](oriented-frame.md) gives you a single global depth axis (`y`).
That works where cortex is flat. But cortex **curves** — the true "down" direction
tilts and bends as you move across the volume. If you keep using one straight axis,
then "depth" and "in-plane distance" get mixed together: a point that is actually
directly *below* another (same lamina position, deeper) can look laterally
displaced, and vice versa. The error grows with curvature and with distance from
wherever the straight axis happened to be correct.

## The streamline

A **streamline** is a curvilinear depth axis: a curve that follows the local
pia → white matter direction as it bends through the tissue. Think of it as the
path a plumb line would trace if "down" were always the cortical down at each point.

With a streamline you can measure the two quantities *separately and correctly*:

- **Depth along the streamline** — path length down the curve from the pia, rather
  than a raw `y` difference. This is what `depth_along` and `depth_between` compute.
- **Radial distance** — in-plane distance measured relative to the streamline as the
  `d = 0` reference, accounting for the fact that the curve is tilted. This is what
  `radial_distance` computes.

You can also remap points into a cylindrical-like coordinate system — radial
distance as one axis, depth-along-the-streamline as the other — with
`radial_points`. This is the basis for "straightening" a neuron along the cortical
axis (see [Skeletons & meshworks](../guides/skeletons-meshworks.md)).

## A worked intuition

Imagine two synapses. In raw oriented `y` they differ by 50 µm, and they also
differ in `x`. Are they at the same lamina depth but offset sideways, or genuinely
at different depths? If the cortex is curved there, the straight `y` answer is
biased. Measuring depth *along the streamline* answers the anatomical question:
how far down the cortical column each one really is.

## Next

There are two ways a streamline can be defined: a single fixed curve applied
everywhere, or a curve that varies with location. That distinction — and why the
v1dd default recently changed — is covered in
[Streamlines vs. fields](streamlines-vs-field.md).
