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
- [x] Wind-driven ocean surface layer (reduced-gravity, one-way wind forcing) -- stable and checkpointable; biharmonic diffusion now in production data, gyre structure much improved; see below
- [ ] Buoyancy-driven deep/thermohaline ocean layer (temperature + salinity, sea-ice/brine rejection)
- [ ] Atmosphere-ocean coupling (currently one-way, atmosphere -> ocean only), seasonal cycle, ice-albedo feedback
- [x] Web visualization — interactive globe (terrain/wind/currents layers, wind animation with time scrubber); analytics still minimal
- [ ] Randomized-continent ensemble runs

## Layout

| File | Purpose |
|---|---|
| `grid.py` | Geodesic sphere: icosahedron subdivision, per-cell area/lat-lon, cell-neighbor and global edge topology |
| `terrain.py` | Procedural elevation (random low-degree spherical harmonics → continent-scale blobs, not noise), land/ocean split, latitude+elevation biome classification |
| `atmosphere.py` | Rotating shallow-water solver on the grid — the physics core |
| `ocean.py` | Wind-driven reduced-gravity ocean surface layer — reuses `atmosphere.py`'s C-grid geometry/operators directly, one-way forced by the atmosphere's time-mean wind |
| `checkpoint.py` | Save/load a full terrain + atmosphere + ocean snapshot to one .npz, so a multi-minute-to-half-hour spin-up doesn't have to be re-run from scratch every time |
| `visualize.py` | Renders grid + terrain as a 3D globe (pyvista) |
| `map_view.py` | 2D equirectangular map rendering — flat/colormapped fills, wind quivers, current streamlines, coastline overlay |
| `run_ocean_currents.py` | Spins up the atmosphere, time-averages its wind over a full year, spins up the ocean under it, plots current speed + streamlines, saves a checkpoint |
| `continue_ocean_spinup.py` | Resumes an ocean checkpoint with changed params (e.g. viscosity) and/or more simulated time, without re-running the atmosphere |
| `export_web_data.py` | Runs a fresh terrain + animated atmosphere + ocean spin-up and packs it into `web_data.json` for the web globe |
| `globe_template.html` + `build_globe.py` | Self-contained interactive globe (vanilla Canvas 2D, no dependencies) — `build_globe.py` inlines `web_data.json` into the template to produce the final `globe.html` |

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

## The ocean solver, and what's still rough

`ocean.py` is a wind-driven, reduced-gravity single-layer shallow-water
model — a stand-in for the wind-mixed surface ocean sitting over a deep,
quiescent abyss — reusing `atmosphere.py`'s C-grid geometry and operators
unmodified. It's one-way forced: `run_ocean_currents.py` spins the
atmosphere up, time-averages its wind over a full year, and feeds that in
as a fixed body force; the ocean doesn't push back on the atmosphere yet.

Getting a stable run took three more of the same kind of lesson as the
atmosphere:

1. **Inertial-period CFL violation.** Reduced gravity makes ocean gravity
   waves ~22x slower than the atmosphere's, so the gravity-wave CFL alone
   allowed a 14-hour timestep — far too coarse to resolve Coriolis inertial
   oscillations, which need `f·dt` well inside RK4's stability region.
   Slow blowup to NaN by ~day 110. Fix: `cfl_dt()` also bounds `dt` by the
   inertial period, which ends up the binding constraint here (the
   atmosphere never needed this bound explicitly, since its gravity-wave
   CFL was already far smaller by comparison).
2. **Non-conservative coastal handling.** The first version hard-reset
   land cells to `h0` and coastal edges to zero *after* each RK4 step —
   invisible to the integrator, and on real jagged coastlines it silently
   deleted/injected mass at every land cell, every step (the no-land
   self-test never caught it, since that code path never ran there). Fix:
   replaced with smooth relaxation terms inside `tendency()`, the same way
   `atmosphere.py` already treats land.
3. **Wind-forcing grid noise.** Even after fixing #2, a real atmosphere-
   derived wind field still blew the ocean up by ~day 180. Time-averaging
   removes transient (day-to-day) noise but not spatially-persistent
   grid-scale structure in the wind field. Fix: `smooth_edge_field()`
   smooths the forcing the same way `atmosphere.py` already smooths spiky
   single-cell terrain slopes.

With all three fixed, a full checkpointed run (atmosphere spin-up + a
full year of wind-averaging + a multi-year ocean spin-up) is stable, but
the resulting current field is a rough first pass, not a clean one:

- **Hemisphere asymmetry turned out to be real, not a bug.** This world's
  random continents (seed=1) put only ~6% land in the southern high
  latitudes vs. ~35% in the north — an unobstructed band supports a
  strong, continuous zonal jet the way Earth's real Antarctic
  Circumpolar Current does; the land-cluttered north can't build the same
  speed. Checked directly via land fraction by latitude band, not assumed.
- **The western boundary layer is structurally under-resolved.** Munk
  (1950) theory puts the boundary-layer width at `(nu/beta)^(1/3)` —
  ~100km at this grid's original `nu`, against ~316km cells. That's why
  currents came out turbulent and gyre-less rather than organized into a
  loop: the boundary current has nowhere to exist at that resolution.
  Raising `nu` to match the grid (`munk_matched_nu()`) fixed the
  *structure* — streamlines went from noisy scribble to smooth, coherent
  flow, with recirculation loops appearing for the first time — but at a
  real cost: peak current speed collapsed ~30x (0.13-0.17 m/s → ~0.004
  m/s). Harmonic (∇²) viscosity isn't scale-selective — its damping rate
  scales with wavenumber², so a coefficient large enough to fix a ~100km
  problem also heavily damps the ~5,000km gyre-scale flow, not just the
  coast.

**Biharmonic (∇⁴) diffusion: implemented, tuned, not yet in production
data.** `ocean.py` now has a real hyperviscosity term (`nu4`, applied as
`edge_laplacian` twice with a sign flip -- that flip isn't optional, since
squaring a proper diffusion operator's eigenvalues makes them a *source*
without it) plus its own CFL bound and a `munk_matched_nu4()` sizing
helper, with `nu` left back at its original small value. Real ocean
models (MOM, POP, MITgcm) use biharmonic rather than harmonic viscosity
for exactly this job — its damping rate scales with wavenumber⁴ instead
of wavenumber², concentrating it much more tightly at the grid scale and
leaving basin-scale flow largely alone.

Swept `target_cells` (1.0/1.5/2.0) against a cached wind-forcing field
(no atmosphere re-run needed) and checked each visually, not just for
stability:
- 1.0 was too weak to even bind the timestep -- indistinguishable from
  no biharmonic term at all.
- 1.5 showed marginal improvement, still mostly turbulent scribble.
- **2.0** was a real qualitative jump: streamlines went from noisy
  scribble to smooth, coherent flow, with a genuine closed loop visible
  near a landmass for the first time. Cost: dt drops ~5.5x (2745s ->
  500s), so a full ocean spin-up is meaningfully more expensive now.

`target_cells=2.0` is wired into `export_web_data.py`, and the fresh
production data (real terrain + full atmosphere coupling, not the cached
test forcing) has now been generated end to end: a full 4-year ocean
spin-up, stable throughout, with `nu4=1.98e18`. The qualitative payoff
carried through to the real run, not just the cached-forcing sweep --
traced current streamlines dropped from ~28k short, noisy segments to
7,450 coherent ones. Globe rebuilt and republished from this data.

## The web globe

`globe.html` (built from `globe_template.html` + `export_web_data.py`'s
output) is a self-contained, dependency-free interactive globe: vanilla
Canvas 2D, not WebGL/Three.js, since a strict CSP on published artifacts
blocks fetching any external library. Rendering exploits one fact about a
sphere: it's convex, so simple backface culling (skip any triangle facing
away from the viewer) gives fully correct visibility with no z-sorting
needed, which is what makes a plain 2D canvas workable for a rotatable
"3D" globe at all. Three toggleable layers (terrain, animated wind with a
day scrubber, ocean currents), all reusing the exact `l_min=2` terrain and
the same fresh spin-up data described above — deliberately not the older
checkpoints, so every layer agrees on where the coastlines are.

Also live: a dashed equator ring + N/S pole markers on every layer (an
orientation aid, not data), a reset-view button that tweens back to a
default north-up/slightly-tilted orientation along the shortest rotational
path, and direction -- not just magnitude -- on both flow layers. Wind
gets animated arrow glyphs (cheap to redraw every frame, since it's just
reading that frame's already-computed direction). Currents get actual
traced streamlines instead: `export_web_data.py` reuses `map_view.py`'s
existing matplotlib-streamplot tracer once (currents are a static layer;
no need to re-integrate every frame), converts the resulting lon/lat path
segments to xyz, and the page just rotates + redraws those precomputed
segments each frame. The streamlines now reflect the post-biharmonic
current field (7,450 segments, smooth coherent loops rather than the
earlier ~28k-segment noisy scribble).

## Roadmap

Immediate next step: a buoyancy-driven deep/thermohaline layer (temperature + salinity,
sea-ice formation with brine rejection at the poles) under the wind-driven
surface layer, two-way atmosphere-ocean coupling, a seasonal cycle,
ice-albedo feedback, then randomized-continent ensembles to see whether
Earth-like patterns (western boundary intensification, subtropical gyres,
the thermohaline overturning loop) re-emerge independent of geography.
