"""
Wind-driven ocean surface layer: a reduced-gravity shallow-water model on
the same geodesic grid and C-grid edge machinery as atmosphere.py.

Scope (deliberately v1): a single active layer standing in for the wind-
mixed surface ocean, sitting over a deep, quiescent abyss. It's forced by
wind stress (one-way -- atmosphere pushes the ocean, ocean doesn't push
back yet) and otherwise governed by exactly the same physics shape as the
atmosphere solver:

    d(u_n)/dt = -g' * d(eta)/dn - f * u_tangential + wind_forcing
                + diffusion - u_n / tau_fric(coast)
    d(eta)/dt + div(eta * u) = diffusion

`eta` here is the active-layer thickness (m), started at h0 and evolving
by mass divergence only -- there's no radiative-relaxation analogue for
the ocean layer, and there doesn't need to be: wind-driven Ekman pumping
*is* div(u) changing eta, which then sets up the geostrophically-balanced
gyre circulation via grad(eta). That's the actual mechanism, not a
simplification of it.

Why "reduced gravity": g' = g * (delta_rho/rho) is the full gravitational
acceleration discounted by the (tiny, ~0.1-0.3%) density contrast across
the main thermocline, since that's the only density jump this single
active layer feels. It's ~1/500th of atmosphere.py's effective g*h0, which
is exactly why ocean gravity waves/Rossby radii are so much smaller and
slower than atmospheric ones, and why the ocean's CFL-stable timestep can
be much larger even though real ocean currents are much slower than winds.

Grid/edges/operators are all reused unmodified from atmosphere.py
(CGridGeometry) -- same Div/Grad discrete-adjoint pair, same Perot
Coriolis reconstruction, same edge-graph-Laplacian diffusion for grid-scale
noise. None of that machinery is atmosphere-specific; it's just "shallow
water on this grid," which is exactly what the ocean also is.

Land handling (v1, deliberately simple, and revised once already -- see
below): rather than solve on a masked sub-domain, land gets treated the
same way atmosphere.py treats it -- smooth relaxation terms blended into
the tendency itself, not a post-step mask. Edges touching land get a short
friction timescale (tau_fric_coast); land cells get a short relaxation of
h back toward h0 (tau_land_h). Both are ordinary terms RK4 integrates
along with everything else, so nothing discontinuous happens between
stages.

That smoothness matters, not just style: the first version of this instead
hard-reset land h to h0 and coastal-edge u to 0 *after* each full RK4 step
(outside the integrator). It was invisible to the 4 RK stages -- and,
worse, on real jagged coastlines it silently deleted/injected mass at
every land cell every step (the "no land" self-test never caught this,
since with is_land all False the reset path never ran and volume came out
conserved to 1e-6; real terrain hits it constantly). That non-conservative,
discontinuous forcing is what was destabilizing multi-month runs -- stable
for ~2 weeks, NaN by ~5-6 months. Relaxation terms integrated smoothly
inside tendency() don't have either problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid import Grid, build_grid
from atmosphere import CGridGeometry, OMEGA, RADIUS, OBLIQUITY, AtmosState, ShallowWaterAtmosphere


@dataclass
class OceanParams:
    h0: float = 300.0             # mean active-layer (wind-driven / thermocline) depth, m
    g_reduced: float = 0.02       # g * delta_rho/rho, m/s^2 -- ~1/500 of full g (see module
                                   # docstring): the density contrast across the thermocline,
                                   # not full seawater density, is what restores this layer
    tau_fric: float = 200 * 86400.0        # open-ocean Rayleigh friction / spin-down
                                            # timescale, s -- stands in for interfacial drag
                                            # at the base of the wind-driven layer; real
                                            # gyres spin down over months, not days
    tau_fric_coast: float = 0.25 * 86400.0  # short friction timescale on edges touching
                                             # land -- approximates a no-flow coastal wall
    tau_land_h: float = 0.5 * 86400.0       # short relaxation timescale pulling land cells'
                                             # h back toward h0 -- land isn't part of the
                                             # active ocean layer, but (unlike atmosphere.py's
                                             # land, which has its own real h_eq physics) there's
                                             # no equivalent equilibrium to relax *to* other than
                                             # "inert," so this is a numerical device, not a
                                             # physical timescale -- kept short so it doesn't
                                             # meaningfully affect adjacent ocean cells but long
                                             # enough to stay a smooth term RK4 can integrate
                                             # (see module docstring: a hard post-step reset,
                                             # what this replaced, doesn't conserve volume and
                                             # was the actual source of a slow coastal blowup)
    nu: float = 2.0e4              # velocity diffusion coeff, m^2/s (grid-noise damping,
                                    # same role as atmosphere.py's nu) -- deliberately left at
                                    # its original small value now that nu4 (below) is doing
                                    # the boundary-layer job; see module docstring's "harmonic
                                    # vs biharmonic" section for why cranking *this* up instead
                                    # was the wrong lever (it isn't scale-selective, so a
                                    # coefficient large enough to resolve a ~100km boundary
                                    # layer also flattens the ~5,000km gyre-scale flow)
    nu4: float = 0.0                # biharmonic (nabla^4) velocity hyperviscosity, m^4/s --
                                    # 0 by default (opt-in via munk_matched_nu4()); this is
                                    # the scale-selective replacement for over-cranking nu
    kappa_h: float = 2.0e4         # thickness diffusion coeff, m^2/s (checkerboard-mode
                                    # damping, same role as atmosphere.py's kappa_h)
    rho_air: float = 1.2           # kg/m^3, for the bulk wind-stress formula
    rho0: float = 1025.0           # kg/m^3, reference seawater density
    Cd: float = 1.3e-3             # dimensionless bulk drag coefficient
    radius: float = RADIUS
    obliquity: float = OBLIQUITY

    # -- sea surface temperature: a passive tracer (advected + relaxed +
    # diffused), same shape as atmosphere.py's moisture q -- doesn't feed
    # back into h/u dynamics yet (that's the buoyancy-coupling stage still
    # to come). Its only job so far is giving sea ice a real, physical
    # location instead of ice being a land-only, latitude-band biome. --
    T_mean: float = 10.0        # degC, global-mean equilibrium SST
    delta_T_eq: float = 18.0    # degC, equator-to-pole equilibrium SST swing --
                                 # combined with T_mean gives a -8..28degC range.
                                 # First pass used 13/15 (-2..28degC): technically
                                 # correct (equilibrium fixed to land exactly in
                                 # that range, see sst_equilibrium docstring) but
                                 # empirically useless -- the pole target sat right
                                 # at the -1.8degC freezing line, so real ocean
                                 # cells mostly stayed just *above* it and sea ice
                                 # covered ~0.1% of the planet, functionally
                                 # invisible. Checked in a full production run,
                                 # not just the idealized self-test -- this wider
                                 # swing is itself still a first pass, not
                                 # re-validated yet
    tau_sst: float = 30 * 86400.0  # relaxation timescale toward T_eq, s -- a
                                    # numerical stand-in for net air-sea heat
                                    # exchange (there's no real coupled heat flux
                                    # yet), same role as tau_land_h. 30 days is a
                                    # guess at mixed-layer thermal inertia, not
                                    # measured -- flagged for empirical check
    kappa_T: float = 2.0e4      # heat diffusion coeff, m^2/s -- same order as
                                 # kappa_h, reuses its (unenforced but so-far-fine)
                                 # CFL margin rather than deriving a new bound
    T_freeze: float = -1.8      # degC, real seawater freezing point (lower than
                                 # fresh water's due to salinity) -- below this,
                                 # an ocean cell counts as sea ice


@dataclass
class OceanState:
    h: np.ndarray   # (n_cells,) m, active-layer thickness
    u: np.ndarray   # (n_edges,) m/s, edge-normal component
    T: np.ndarray   # (n_cells,) degC, sea surface temperature (passive tracer)


def wind_stress_forcing_edge(
    atmos: ShallowWaterAtmosphere, atmos_state: AtmosState, p: OceanParams
) -> np.ndarray:
    """
    Edge-normal wind-stress body-force (m/s^2) from a snapshot of the
    atmosphere's state, to drive the ocean momentum equation.

    Bulk formula: tau = rho_air * Cd * |U_air| * U_air (vector), then
    divided by rho0*h0 to become an acceleration on the ocean layer.
    Projected onto the edge normal: since |U|*U is just U scaled by its
    own magnitude, (|U|*U)-dot-n = |U| * U_n exactly -- so we only need
    the wind *speed* (scalar, reconstructed at cells and averaged to
    edges) times the edge-normal wind velocity that's already the
    atmosphere's prognostic variable on this same edge grid.
    """
    geo = atmos.geo
    vec = geo.reconstruct_cell_vector(atmos_state.u)
    speed_cell = np.linalg.norm(vec, axis=1)
    speed_edge = 0.5 * (speed_cell[geo.cell_left] + speed_cell[geo.cell_right])
    tau_edge_normal = p.rho_air * p.Cd * speed_edge * atmos_state.u
    return tau_edge_normal / (p.rho0 * p.h0)


def smooth_edge_field(geo: CGridGeometry, field: np.ndarray, passes: int = 4) -> np.ndarray:
    """
    Repeated neighbor-averaging on an edge-defined field -- same technique
    atmosphere.py already uses to smooth single-cell topography spikes
    before they blow up the orographic-rain term. A real atmosphere's
    time-mean wind still carries grid-scale structure (jets, coastlines,
    residual numerical noise the atmosphere's own diffusion keeps small
    but not zero) that time-averaging over many steps doesn't remove,
    since it's spatially persistent, not transient. Feeding that directly
    into the ocean as a body force excited exactly the grid-scale
    instability this smooths away (confirmed: without it, a real
    atmosphere-forced ocean run went to NaN by ~day 180; the idealized,
    analytically smooth wind forcing used in this module's own self-test
    never showed the problem, which is what pointed at forcing roughness
    rather than the land-coupling as the actual cause).
    """
    for _ in range(passes):
        neighbor_mean = field[geo.edge_neighbors].mean(axis=1)
        field = 0.5 * field + 0.5 * neighbor_mean
    return field


class WindDrivenOcean:
    def __init__(
        self,
        grid: Grid,
        params: OceanParams | None = None,
        terrain=None,
        geo: CGridGeometry | None = None,
        wind_forcing_edge: np.ndarray | None = None,
    ):
        self.grid = grid
        self.p = params or OceanParams()
        self.geo = geo if geo is not None else CGridGeometry(grid, self.p.radius)
        self.t = 0.0

        if terrain is not None:
            self.is_land = terrain.is_land
        else:
            self.is_land = np.zeros(grid.n_cells, dtype=bool)

        land_l = self.is_land[self.geo.cell_left]
        land_r = self.is_land[self.geo.cell_right]
        self.coast_edge = land_l | land_r  # any edge touching land at all

        fric_rate_ocean = 1.0 / self.p.tau_fric
        fric_rate_coast = 1.0 / self.p.tau_fric_coast
        self.fric_rate_edge = np.where(self.coast_edge, fric_rate_coast, fric_rate_ocean)
        self.land_h_relax_rate = np.where(self.is_land, 1.0 / self.p.tau_land_h, 0.0)

        n_edges = self.geo.edges.n_edges
        self.wind_forcing_edge = (
            wind_forcing_edge if wind_forcing_edge is not None else np.zeros(n_edges)
        )

    def set_wind_forcing(self, forcing_edge: np.ndarray) -> None:
        self.wind_forcing_edge = forcing_edge

    def initial_state(self, perturb: float = 0.0, seed: int = 0) -> OceanState:
        h = np.full(self.grid.n_cells, self.p.h0)
        if perturb:
            rng = np.random.default_rng(seed)
            h = h + perturb * rng.standard_normal(self.grid.n_cells)
        u = np.zeros(self.geo.edges.n_edges)
        T = np.full(self.grid.n_cells, self.p.T_mean)  # starts uniform, relaxes
        # toward sst_equilibrium(t) the same way h starts at h0 and relaxes
        # toward h_eq in atmosphere.py
        return OceanState(h=h, u=u, T=T)

    def sst_equilibrium(self, t: float) -> np.ndarray:
        """Latitude+season equilibrium SST (degC) -- the ocean's own insolation
        proxy, same shape of forcing as atmosphere.py's h_equilibrium (warm
        tropics, cold poles, seasonal wobble via obliquity) but independent of
        it -- different units/physics, no shared state, deliberately not
        wired to the atmosphere's actual h_eq (that coupling is a later
        stage).

        Normalizes the warm and cold excursions from the mean *separately*,
        not by a single insolation_norm.max() the way atmosphere.h_equilibrium
        does -- coszen is asymmetric (floored at 0 on the polar/night side,
        up to 1 at the subsolar point), so the below-mean excursion at the
        poles is proportionally larger than the above-mean excursion at the
        equator. atmosphere.py gets away with a single max() because its
        delta_h_eq (400) is small next to h0 (3000); this module's
        delta_T_eq isn't small relative to anything, and dividing both signs
        by the same (equator-side) max blew the pole target out to -36degC
        against an intended -2degC floor -- caught by an isolated t=0 check
        of this function alone, before any dynamics ran. Normalizing each
        side by its own extremum instead guarantees the equilibrium field
        stays within exactly [T_mean-delta_T_eq, T_mean+delta_T_eq]."""
        lat = np.radians(self.grid.lat)
        year = 365.25 * 86400.0
        decl = self.p.obliquity * np.sin(2 * np.pi * t / year)
        coszen = np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl), 0.0, None)
        insolation_norm = coszen - coszen.mean()
        pos_scale = max(insolation_norm.max(), 1e-9)
        neg_scale = max(-insolation_norm.min(), 1e-9)
        norm = np.where(
            insolation_norm >= 0, insolation_norm / pos_scale, insolation_norm / neg_scale
        )
        return self.p.T_mean + self.p.delta_T_eq * norm

    def cfl_dt(self, safety: float = 0.4) -> float:
        """
        Three independent stability limits, take the tightest:

        1. Gravity-wave CFL (dx / wave speed), same as atmosphere.py. With
           reduced gravity this alone allows a *huge* step -- g'*h0 is
           ~1/500th of the atmosphere's g*h0, so wave_speed drops ~22x and
           this bound alone would let dt run into the tens of thousands
           of seconds.
        2. Inertial-period limit: f*dt must stay well inside RK4's
           stability region on the imaginary axis (Coriolis is a pure
           rotation -- an undamped-oscillator term, same shape as gravity
           waves). The atmosphere solver never had to think about this
           bound explicitly because its gravity-wave CFL was already far
           smaller than the inertial period at this grid's resolution.
           Here it's the other way around: the slow reduced-gravity waves
           mean this bound binds instead, and skipping it produces a slow
           (~months-long) exponential blowup from under-resolved inertial
           oscillations -- confirmed empirically (NaN by ~day 110 without
           this term, using the gravity-wave dt alone).
        3. Diffusion stability: nu*dt/dx^2 has its own explicit-scheme
           bound (unlike gravity waves/Coriolis, diffusion sits on the
           *real*, not imaginary, axis, so RK4's stability boundary there
           is a completely different number -- roughly 2.8 for RK4's real
           axis vs ~2.8 total extent on the imaginary axis, but eigenvalues
           of a graph Laplacian can approach -4*nu/dx^2, hence the extra
           safety margin below). Not binding at nu's current (small,
           original) value.
        4. Biharmonic stability: same idea, but edge_laplacian applied
           twice has eigenvalues that are the *square* of the single-
           application ones, i.e. up to (2/dx^2)^2 = 4/dx^4 in magnitude
           -- so this bound scales with dx^4, not dx^2, and gets tight
           fast as nu4 grows. This is usually the tightest of the four
           once nu4 is turned on (see munk_matched_nu4()) -- confirmed
           empirically while tuning it, see ocean.py's __main__.
        """
        dx = np.sqrt(self.geo.cell_area.mean())
        wave_speed = np.sqrt(self.p.g_reduced * self.p.h0)
        dt_gravity = safety * dx / wave_speed

        f_max = np.abs(self.geo.f_edge).max()
        dt_inertial = safety / f_max if f_max > 0 else np.inf

        dt_diffusion = safety * dx ** 2 / (4.0 * self.p.nu) if self.p.nu > 0 else np.inf
        dx4 = (dx ** 2) ** 2
        dt_biharmonic = safety * dx4 / (4.0 * self.p.nu4) if self.p.nu4 > 0 else np.inf

        return min(dt_gravity, dt_inertial, dt_diffusion, dt_biharmonic)

    def tendency(self, s: OceanState, t: float) -> OceanState:
        p, geo = self.p, self.geo

        h_edge = 0.5 * (s.h[geo.cell_left] + s.h[geo.cell_right])
        flux = h_edge * s.u
        dhdt = (
            -geo.divergence(flux)
            + p.kappa_h * geo.cell_laplacian(s.h)
            - self.land_h_relax_rate * (s.h - p.h0)
        )

        # SST: passive tracer, relax toward equilibrium, diffuse -- but
        # advected as a *material* derivative (-u.grad(T)), not flux
        # divergence (-div(T*u)) the way moisture q is. q is extensive (a
        # column-integrated depth of water, "stuff" that a flux genuinely
        # transports), so flux form is the right equation for it. T is
        # intensive -- flux form silently adds a T*div(u) compressibility
        # term (div(Tu) = T*div(u) + u.grad(T)) that has no business being
        # in a temperature equation, and it isn't harmless: this ocean
        # layer's h/div(u) never equilibrates in the no-relaxation idealized
        # self-test below, so that spurious term compounded every step and
        # drove T to -38degC against a T_eq floor of ~-2degC (caught by
        # Stage 2's sanity check, first pass had this bug). Recovering
        # u.grad(T) = div(T*u) - T*div(u) removes the compressibility term
        # exactly -- pure advection + a linear relaxation + diffusion all
        # individually obey a maximum principle (can't manufacture a new
        # extreme beyond T_eq's own range), which flux form didn't.
        T_edge = 0.5 * (s.T[geo.cell_left] + s.T[geo.cell_right])
        div_u = geo.divergence(s.u)
        advect = geo.divergence(T_edge * s.u) - s.T * div_u  # = u.grad(T)
        T_eq = self.sst_equilibrium(t)
        dTdt = (
            -advect
            + (T_eq - s.T) / p.tau_sst
            + p.kappa_T * geo.cell_laplacian(s.T)
        )

        grad_eta = geo.gradient_normal(s.h)
        u_tan = geo.tangential_at_edges(s.u)
        coriolis = geo.f_edge * u_tan
        diffusion = p.nu * geo.edge_laplacian(s.u)

        # Biharmonic (nabla^4) hyperviscosity: apply the same Laplacian
        # twice, with a MINUS sign. That sign is not optional -- edge_laplacian's
        # eigenvalues are <= 0 (it's a proper diffusion operator), so its
        # eigenvalues squared (what applying it twice produces) are >= 0,
        # which would be a source, not a sink, without the flip. Physically,
        # this is nu4 * (-nabla^2)^2 u, the standard hyperviscosity form.
        # Why bother when nu already provides diffusion: nu's damping rate
        # scales with wavenumber^2, nu4's with wavenumber^4 -- nu4 can be
        # tuned to strongly damp grid-scale noise (and sharpen the western
        # boundary layer) while barely touching gyre-scale flow, which a
        # single harmonic coefficient structurally cannot do (see
        # munk_matched_nu() vs munk_matched_nu4() below).
        biharmonic = -p.nu4 * geo.edge_laplacian(geo.edge_laplacian(s.u))

        dudt = (
            -p.g_reduced * grad_eta
            - coriolis
            + diffusion
            + biharmonic
            - self.fric_rate_edge * s.u
            + self.wind_forcing_edge
        )
        return OceanState(h=dhdt, u=dudt, T=dTdt)

    def step(self, s: OceanState, dt: float) -> OceanState:
        """RK4, same rationale as atmosphere.py: gravity waves here are
        just as undamped-oscillator-shaped, so RK2 would amplify them.
        t is threaded through each RK stage (self.t + 0/0.5/0.5/1 * dt),
        same as atmosphere.py's step(), so sst_equilibrium()'s seasonal
        cycle evolves within a step rather than freezing at the step's
        start time."""

        def add(a: OceanState, b: OceanState, scale: float) -> OceanState:
            return OceanState(h=a.h + scale * b.h, u=a.u + scale * b.u, T=a.T + scale * b.T)

        k1 = self.tendency(s, self.t)
        k2 = self.tendency(add(s, k1, 0.5 * dt), self.t + 0.5 * dt)
        k3 = self.tendency(add(s, k2, 0.5 * dt), self.t + 0.5 * dt)
        k4 = self.tendency(add(s, k3, dt), self.t + dt)

        h_new = s.h + (dt / 6.0) * (k1.h + 2 * k2.h + 2 * k3.h + k4.h)
        u_new = s.u + (dt / 6.0) * (k1.u + 2 * k2.u + 2 * k3.u + k4.u)
        T_new = s.T + (dt / 6.0) * (k1.T + 2 * k2.T + 2 * k3.T + k4.T)
        self.t += dt

        return OceanState(h=h_new, u=u_new, T=T_new)

    def energy(self, s: OceanState) -> float:
        """Diagnostic total energy (KE + available potential energy)."""
        h_edge = 0.5 * (s.h[self.geo.cell_left] + s.h[self.geo.cell_right])
        ke = 0.5 * np.sum(h_edge * s.u ** 2 * self.geo.length * self.geo.d_center)
        ape = 0.5 * self.p.g_reduced * np.sum((s.h - self.p.h0) ** 2 * self.geo.cell_area)
        return ke + ape


def munk_matched_nu(geo: CGridGeometry, target_cells: float = 3.0, lat_deg: float = 30.0) -> float:
    """
    Pick a horizontal viscosity nu so the Munk (1950) western-boundary-
    layer width delta_M = (nu/beta)^(1/3) spans about `target_cells` grid
    cells at `lat_deg` -- large enough for a western boundary current to
    resolve as an actual jet rather than alias into grid-scale noise.

    Checked empirically for this project's default grid (level 4, ~316km
    cells) and default nu=2e4: delta_M comes out to ~100km, under a third
    of a single grid cell. That's not "a bit coarse," it's structurally
    unresolvable -- the boundary current has nowhere to exist -- and it's
    the leading suspect for why a real spin-up produced turbulent,
    gyre-less currents instead of an organized loop. Real coarse-resolution
    ocean models do exactly this: raise horizontal viscosity so it matches
    what the grid can represent, rather than what the real ocean's
    boundary layer actually measures -- an explicit numerical-eddy-
    viscosity parameterization of the unresolved mesoscale mixing, not
    an arbitrary knob.

    beta = df/dy = 2*Omega*cos(lat)/R, evaluated at a representative
    mid-latitude since it varies with latitude (the boundary layer is
    correspondingly narrower near the poles, wider near the equator --
    this just picks *a* resolved scale, not a uniformly-resolved one).
    """
    dx = np.sqrt(geo.cell_area.mean())
    beta = 2.0 * OMEGA * np.cos(np.radians(lat_deg)) / geo.radius
    delta_target = target_cells * dx
    return beta * delta_target ** 3


def munk_matched_nu4(geo: CGridGeometry, target_cells: float = 3.0, lat_deg: float = 30.0) -> float:
    """
    Biharmonic analogue of munk_matched_nu(): for hyperviscous (nabla^4)
    friction, the theoretical western-boundary-layer width scales as
    delta_M4 = (nu4/beta)^(1/5) instead of (nu/beta)^(1/3) -- one more
    factor of the friction term's derivative order (biharmonic brings 4
    derivatives into the vorticity balance instead of 2, so closing the
    balance against beta needs one more power of nu4 relative to length).

    Flagged honestly: that exponent is taken from the standard boundary-
    layer scaling argument, not fit to this grid. The bigger uncertainty
    is target_cells itself -- biharmonic friction is *supposed* to be far
    more scale-selective than harmonic, so it likely doesn't need as many
    "cells across" to avoid flattening gyre-scale flow the way the
    harmonic version did. Treat this function's output as a reasonable
    starting order of magnitude, not a derived answer -- ocean.py's
    __main__ sweeps a few target_cells values against a cheap idealized
    test before committing to one, the same empirical-first approach
    every other stability constant in this module got.
    """
    dx = np.sqrt(geo.cell_area.mean())
    beta = 2.0 * OMEGA * np.cos(np.radians(lat_deg)) / geo.radius
    delta_target = target_cells * dx
    return beta * delta_target ** 5


def idealized_zonal_wind_forcing(grid: Grid, geo: CGridGeometry, p: OceanParams, amplitude: float = 6.0) -> np.ndarray:
    """
    Crude standalone stand-in for the atmosphere's time-mean wind, for fast
    testing of the ocean model alone (no atmosphere spin-up needed): three
    alternating latitude bands per hemisphere (easterlies / westerlies /
    polar easterlies), loosely matching Earth's real zonal-mean surface
    wind pattern. `amplitude` is a wind-speed scale in m/s; converted to a
    wind-stress body force the same way wind_stress_forcing_edge() does,
    just skipping the "run the shallow-water atmosphere" step.
    """
    lat_edge = np.degrees(np.arcsin(np.clip(geo.rhat_edge[:, 2], -1.0, 1.0)))
    u_wind = amplitude * np.sin(3.0 * np.radians(lat_edge))  # zonal (eastward) wind speed
    # Project the zonal (eastward) wind vector onto the edge normal: build
    # the true 3D eastward unit vector at each edge and dot it with n_edge.
    lon_edge = np.degrees(np.arctan2(geo.rhat_edge[:, 1], geo.rhat_edge[:, 0]))
    lon_r = np.radians(lon_edge)
    east3 = np.stack([-np.sin(lon_r), np.cos(lon_r), np.zeros_like(lon_r)], axis=1)
    u_edge_normal = u_wind * np.einsum("ij,ij->i", east3, geo.normal)
    speed = np.abs(u_wind)
    tau_edge_normal = p.rho_air * p.Cd * speed * u_edge_normal
    return tau_edge_normal / (p.rho0 * p.h0)


if __name__ == "__main__":
    g = build_grid(4)

    print("=== Stage 0: pure gravity wave (no Coriolis/forcing/diffusion/friction) ===")
    quiet = WindDrivenOcean(g, OceanParams(tau_fric=1e18, nu=0.0))
    quiet.geo.f_edge = quiet.geo.f_edge * 0.0  # disable Coriolis for this isolated test
    dt = quiet.cfl_dt()
    print(f"cells={g.n_cells}  edges={quiet.geo.edges.n_edges}  dt={dt:.1f}s  "
          f"wave speed={np.sqrt(quiet.p.g_reduced*quiet.p.h0):.3f} m/s")
    s = quiet.initial_state(perturb=2.0, seed=0)
    e0 = quiet.energy(s)
    for i in range(300):
        s = quiet.step(s, dt)
        if i % 50 == 0 or i == 299:
            print(f"  step {i:4d}  h[min,max]=[{s.h.min():8.3f},{s.h.max():8.3f}]  "
                  f"|u|max={np.abs(s.u).max():8.5f}  E/E0={quiet.energy(s)/e0:8.4f}  "
                  f"finite={np.isfinite(s.h).all()}")

    print("\n=== Stage 1: idealized zonal wind forcing, no land (zonal jets) ===")
    model = WindDrivenOcean(g)
    model.set_wind_forcing(idealized_zonal_wind_forcing(g, model.geo, model.p))
    dt = model.cfl_dt()
    s = model.initial_state()
    n_days = 3 * 365
    n_steps = int(n_days * 86400 / dt)
    print(f"dt={dt:.0f}s ({dt/3600:.2f}h)  steps={n_steps}  simulating {n_days} days")
    for i in range(n_steps):
        s = model.step(s, dt)
        if i % max(1, n_steps // 10) == 0 or i == n_steps - 1:
            speed = np.abs(s.u)
            print(f"  t={model.t/86400:6.1f}d  h[min,max]=[{s.h.min():7.2f},{s.h.max():7.2f}]  "
                  f"|u| mean/max={speed.mean():.4f}/{speed.max():.4f} m/s  "
                  f"T[min,max]={s.T.min():5.1f},{s.T.max():5.1f}degC  "
                  f"finite={np.isfinite(s.h).all() and np.isfinite(s.T).all()}")

    print("\n=== Stage 2: SST sanity check (from the Stage 1 spin-up above) ===")
    # No land in this self-test (WindDrivenOcean(g) with no terrain), so
    # T_eq's latitude shape is all this run has to reproduce -- checking it
    # lands in a plausible range and that ice forms poleward, not equatorward,
    # is the cheap version of the "check empirically" this project's other
    # constants got, before wiring this into a real terrain-coupled run.
    abs_lat = np.abs(g.lat)
    pole_T = s.T[abs_lat > 80].mean()
    equator_T = s.T[abs_lat < 10].mean()
    ice_frac = (s.T <= model.p.T_freeze).mean()
    print(f"  mean T: poles(|lat|>80)={pole_T:.1f}degC  equator(|lat|<10)={equator_T:.1f}degC")
    print(f"  sea ice fraction (T<={model.p.T_freeze}degC): {ice_frac:.3f}")
    print(f"  monotonic pole<equator: {pole_T < equator_T}")
