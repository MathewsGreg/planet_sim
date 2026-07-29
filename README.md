# planet_sim

A from-scratch, home-PC-scale simulation of ocean and atmosphere circulation
on a randomly generated Earth-like planet — geodesic grid, procedural
continents, and a rotating shallow-water dynamical core, built to understand
*why* global circulation looks the way it does (Hadley cells, jet streams,
western boundary currents, the thermohaline "conveyor belt") rather than to
reproduce Earth specifically. Inspired by Helen Czerski's *The Blue Machine*.

The core question this project is chasing: if you gave a planet the same
rotation, the same rough ocean/land proportions, and pointed a sun at it, but
scrambled the continents, would the same broad circulation patterns
re-emerge? That's testable once the ocean layer and randomized-continent
ensembles are in place.

## Status

- [x] Geodesic grid (icosahedral subdivision) with full edge/neighbor topology
- [x] Procedural continents + latitude/elevation-driven biomes
- [x] Single-layer rotating shallow-water atmosphere (C-grid, energy-conserving)
- [ ] Multi-layer atmosphere (Hadley/Ferrel/polar cell structure)
- [ ] Ocean layer (wind-driven surface + buoyancy-driven deep/thermohaline)
- [ ] Atmosphere-ocean coupling, seasonal cycle, ice-albedo feedback
- [ ] Web visualization (interactive globe, controls, analytics)
- [ ] Randomized-continent ensemble runs

## Layout

| File | Purpose |
|---|---|
| `grid.py` | Geodesic sphere: icosahedron subdivision, per-cell area/lat-lon, cell-neighbor and global edge topology |
| `terrain.py` | Procedural elevation (random low-degree spherical harmonics → continent-scale blobs, not noise), land/ocean split, latitude+elevation biome classification |
| `atmosphere.py` | Rotating shallow-water solver on the grid — the physics core |
| `visualize.py` | Renders grid + terrain as a 3D globe (pyvista) |

Each module's `__main__` block runs a self-contained sanity check
(`python grid.py`, `python terrain.py`, `python atmosphere.py`, ...).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin/pip on macOS/Linux
```

## The grid

Standard climate-model construction: start from an icosahedron (20
triangular faces) and recursively subdivide each triangle into 4,
reprojecting new vertices onto the unit sphere. This gives a mesh with no
pole singularity and roughly uniform cell size — the same base mesh
MPAS/ICON-class models use (they typically take the Voronoi *dual* of it for
hexagonal cells; this project works directly with the triangular primal
mesh). Subdivision level trades resolution for cost: level 4 = 5,120 cells,
level 5 = 20,480, level 6 = 81,920 — level 4-5 is the practical range for
fast iteration on a home PC.

## The atmosphere solver, and what broke building it

The physics is a single-layer rotating shallow-water model:

```
Du/Dt = -g*grad(h) - f*(rhat x u) + diffusion - u/tau_fric
dh/dt + div(h*u) = -(h - h_eq(lat, season)) / tau_rad
```

`h` (layer thickness) stands in for pressure; the `h_eq(lat, season)`
relaxation term is a proxy for solar heating (no real radiative transfer) —
warm the tropics, cool the poles, let the pressure-gradient force and
Coriolis organize the resulting flow into circulation. This is a shallow-water
cousin of the Held–Suarez relaxation used in real GCM benchmarks.

Three real, sequential bugs surfaced while getting this stable, each one a
fairly canonical numerical-methods lesson:

1. **Collocated-grid checkerboard instability.** The first version stored
   both `h` and velocity at cell centers. Pressure-gradient and mass-divergence
   weren't discrete adjoints of each other, so a resting state didn't stay at
   rest — it blew up exponentially, independent of any physical forcing. Fix:
   moved velocity onto edges as a single scalar (the edge-normal component),
   an Arakawa C-grid / TRiSK-style staggering, with `divergence()` and
   `gradient_normal()` built so `sum(A*h*div(u)) == -sum(L*d*u*grad(h))`
   exactly (verified to float precision).
2. **RK2 time-integrator instability.** Gravity waves are undamped
   oscillators, and explicit RK2's stability region only touches the
   imaginary axis at the origin — any purely oscillatory mode amplifies every
   step, regardless of step size. Fix: switched to RK4, whose stability
   region actually covers part of the imaginary axis (damps rather than
   amplifies oscillatory modes within range).
3. **Non-energy-conserving Coriolis term.** Coriolis does zero physical work
   (`f*(rhat x u)` is always perpendicular to `u`), and a discrete scheme has
   to preserve that or it becomes a spurious energy source. The edge-tangential
   velocity needed for the Coriolis term isn't a prognostic variable, and the
   first (least-squares) reconstruction used to estimate it had no such
   guarantee — an isolated Coriolis-only test (no forcing, no diffusion) grew
   5.7x in energy over ~400 steps. Fix: replaced it with Perot's (2000) exact
   geometric reconstruction, which reconstructs the zero-net-work property to
   ~1e-8 relative precision (empirically verified, see `atmosphere.py`).

With all three fixed, a 150-day run with full insolation forcing stays
bounded, with wind speeds settling into a physically plausible ~24-25 m/s
range (jet-stream scale) rather than diverging.

## Roadmap

Next: bring the ocean layer online (2-layer wind + buoyancy-driven, same
grid/edge machinery), couple it to the atmosphere, add seasons and
ice-albedo feedback, then run randomized-continent ensembles to see whether
Earth-like patterns (western boundary intensification, subtropical gyres,
the thermohaline overturning loop) re-emerge independent of geography.

Once there's something worth watching move, this is meant to live on a web
page — an interactive globe with time controls and analytics (energy
budgets, circulation diagnostics), not just static renders.
