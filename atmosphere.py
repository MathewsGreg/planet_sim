"""
Single-layer rotating shallow-water model on the geodesic Grid, using a
C-grid (Arakawa C / TRiSK-family) staggering: layer thickness h lives at
cell centers, velocity lives on edges as a single scalar -- the component
normal to that edge.

Why staggered, not collocated: an earlier version stored both h and the
full velocity vector at cell centers. That's a "collocated" grid, and it
has a classic failure mode -- the discrete pressure-gradient operator and
the discrete mass-divergence operator aren't adjoints of each other, so
the scheme has no guarantee pressure work integrates to zero. It pumped
energy into a grid-scale checkerboard mode and blew up exponentially even
with zero Coriolis/forcing/diffusion (pure gravity-wave test). Staggering
fixes this structurally: with h on cells and u_normal on edges, Grad and
Div below are built as exact discrete adjoints (see the identity in the
`CGridGeometry` docstring), so the linear gravity-wave problem is neutrally
stable by construction, the same reason real ocean/atmosphere models never
collocate velocity and pressure/height.

    d(u_n)/dt = -g * d(h + b)/dn - f * u_tangential + diffusion - u_n/tau_fric(land)
    dh/dt + div(h * u) = -(h - h_eq(lat, t, land)) / tau_rad(land)

Coriolis needs the *tangential* velocity at each edge, which isn't a
prognostic variable here. We reconstruct a full 2D velocity vector at each
cell center by least-squares fitting the 3 surrounding edge-normal values,
then average adjacent cells' reconstructions to each edge. This is an
approximation (real TRiSK uses an exact, mimetic tangential-reconstruction
operator with provable energy/enstrophy properties) -- but it only feeds
Coriolis, which was never the source of the instability above, so a smooth
least-squares reconstruction is a reasonable "TRiSK-lite" compromise.

Optional Terrain coupling (see ShallowWaterAtmosphere.__init__): land
elevation enters the momentum equation as static bottom topography `b` --
the pressure-gradient term uses grad(h+b) instead of grad(h), so a static
"free-surface height" gradient permanently pushes flow around/over
mountains, the standard shallow-water-with-topography formulation. Land
cells also get shorter friction and radiative-relaxation timescales than
ocean cells, standing in for land's much lower heat capacity and higher
surface roughness -- continents both deflect the flow mechanically and
heat/cool faster than the ocean next to them, which is what actually
generates weather rather than just perturbing it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid import Grid, build_edge_topology, build_grid

# ---- physical constants (Earth-like defaults, all overridable) ----
OMEGA = 7.292e-5       # rad/s, planetary rotation rate
G = 9.81                # m/s^2
RADIUS = 6.371e6        # m
OBLIQUITY = np.radians(23.4)


@dataclass
class AtmosParams:
    h0: float = 3000.0          # mean/equivalent layer depth (m) -- reduced well
                                 # below the real ~10km scale height so that the
                                 # gravity-wave speed sqrt(g*h0) stays low enough
                                 # for a tractable CFL time step at this resolution
    delta_h_eq: float = 400.0   # equator-to-pole equilibrium thickness contrast (m)
    tau_rad: float = 8 * 86400.0    # radiative relaxation timescale over ocean (s)
    tau_fric: float = 12 * 86400.0  # linear drag timescale over ocean (s)
    tau_rad_land: float = 2.5 * 86400.0   # land has much lower heat capacity than
                                            # ocean -- it tracks its (more extreme,
                                            # see land_amplify) equilibrium far faster
    tau_fric_land: float = 3 * 86400.0     # land is rougher (terrain, vegetation) --
                                            # more surface drag than open ocean
    land_amplify: float = 1.6   # land's h_eq deviates from h0 this much more than
                                 # ocean's at the same latitude/season -- continental
                                 # climates swing hotter/colder than maritime ones
    topo_scale: float = 700.0   # m of bottom topography per unit of terrain's
                                 # (unitless, ~N(0,1)) elevation field above sea level

    # -- moisture: a passive tracer (doesn't feed back into h/u dynamics) --
    q_sat: float = 40.0             # mm precipitable water, ocean evaporation ceiling
    q_crit_frac: float = 0.5        # convective rain triggers above q_crit_frac*q_sat
    tau_evap: float = 4 * 86400.0   # ocean evaporation relaxation timescale (s)
    tau_precip: float = 1.5 * 86400.0  # convective rain-out timescale (s)
    oro_coeff: float = 3.0e-3       # orographic rain coefficient, 1/m (upslope-wind-
                                     # driven rain-out rate = oro_coeff * ascent(m/s) * q)
    kappa_q: float = 5.0e4          # moisture diffusion coeff, m^2/s (same role as kappa_h)
    conv_sensitivity: float = 2.5e7  # mm*s -- shifts the convective threshold down where
                                      # wind converges (rising motion, real rain-belt cause)
                                      # and up where it diverges (subsidence, real desert-
                                      # belt cause, e.g. the subtropical highs behind Earth's
                                      # deserts) -- reuses the existing divergence operator
    tau_div_avg: float = 15 * 86400.0  # smoothing timescale for the divergence field used
                                        # by precipitation(): instantaneous div(u) is noise-
                                        # dominated at the grid scale (checked empirically --
                                        # its point-to-point std exceeded the coherent
                                        # latitude-band signal), so a running time-average is
                                        # what actually drives the desert-belt modulation
    nu: float = 5.0e4           # velocity diffusion coeff, m^2/s (eddy-diffusivity scale)
    kappa_h: float = 5.0e4      # thickness diffusion coeff, m^2/s -- central-difference
                                 # flux averaging (h_edge = avg(h_left,h_right)) is blind to
                                 # a checkerboard mode in h (an alternating pattern averages
                                 # away to the smooth value at every face, so it feels zero
                                 # restoring flux); this term is what damps that mode
    omega: float = OMEGA
    g: float = G
    radius: float = RADIUS
    obliquity: float = OBLIQUITY


class CGridGeometry:
    """
    Precomputed geometry + the Div/Grad operator pair.

    Adjoint identity these are built to satisfy exactly:
        sum_i A_i * h_i * Div(u)_i  ==  -sum_e L_e * d_e * u_e * Grad(h)_e
    which holds whenever Div(u)_i = (1/A_i) sum_e sign(i,e) L_e u_e and
    Grad(h)_e = (h_right - h_left)/d_e, with sign(left,e)=+1, sign(right,e)=-1
    (the sign flip cancels the two edge contributions into (h_left - h_right),
    matching -Grad*d_e term for term). It's an algebraic identity independent
    of d_e's numerical value -- d_e only sets the *magnitude* of the gradient,
    never breaks the adjointness.
    """

    def __init__(self, grid: Grid, radius: float):
        self.grid = grid
        self.radius = radius
        self.edges = et = build_edge_topology(grid.level)
        n = grid.n_cells
        e = et.n_edges

        v0 = grid.vertices[grid.faces[:, 0]] * radius
        v1 = grid.vertices[grid.faces[:, 1]] * radius
        v2 = grid.vertices[grid.faces[:, 2]] * radius
        self.cell_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)  # m^2
        self.rhat = grid.centers  # unit radial direction per cell
        self.center = grid.centers * radius

        va = grid.vertices[et.vertex_a] * radius
        vb = grid.vertices[et.vertex_b] * radius
        edge_vec = vb - va
        self.length = np.linalg.norm(edge_vec, axis=1)  # m

        rhat_e = grid.centers[et.cell_left] + grid.centers[et.cell_right]
        rhat_e /= np.linalg.norm(rhat_e, axis=1, keepdims=True)
        self.rhat_edge = rhat_e

        normal = np.cross(edge_vec, rhat_e)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        center_left = self.center[et.cell_left]
        center_right = self.center[et.cell_right]
        sign = np.sign(np.einsum("ij,ij->i", normal, center_right - center_left))
        sign[sign == 0] = 1.0
        self.normal = normal * sign[:, None]                      # (E,3) unit, left->right
        self.tangent = np.cross(self.normal, rhat_e)               # (E,3) unit
        self.tangent /= np.linalg.norm(self.tangent, axis=1, keepdims=True)

        self.d_center = np.linalg.norm(center_right - center_left, axis=1)  # m

        self.f_cell = 2.0 * OMEGA * grid.centers[:, 2]
        self.f_edge = 2.0 * OMEGA * rhat_e[:, 2]

        # local tangent-plane basis at cell centers, for vector reconstruction
        arbitrary = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        near_pole = np.abs(self.rhat[:, 2]) > 0.99
        arbitrary[near_pole] = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(self.rhat, arbitrary)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(self.rhat, e1)
        self.e1, self.e2 = e1, e2

        # Perot (2000) vector reconstruction at each cell from its 3 edges:
        #   v_i = (1/A_i) * sum_e u_e * n_e * d_{i,e} * L_e
        # where d_{i,e} is the perpendicular distance from cell i's center to
        # edge e's line, and n_e is the (global, left->right) edge normal.
        # This is exact for a truly linear velocity field on a triangle, and
        # -- checked empirically below -- unlike the least-squares fit it
        # first replaced, it makes the resulting Coriolis operator do zero
        # net work to machine precision (a real least-squares fit doesn't
        # have this property, which is exactly what was leaking energy and
        # causing runs to blow up).
        cell_edge = et.cell_edge  # (F,3)
        edge_mid = 0.5 * (va + vb)  # (E,3)
        mid_local = edge_mid[cell_edge]              # (F,3,3)
        normal_local = self.normal[cell_edge]         # (F,3,3)
        length_local = self.length[cell_edge]          # (F,3)
        d_local = np.abs(np.einsum(
            "fej,fej->fe", mid_local - self.center[:, None, :], normal_local
        ))  # (F,3) perpendicular distance, always >= 0
        self._perot_weight = (
            normal_local * (d_local * length_local / self.cell_area[:, None])[:, :, None]
        )  # (F,3,3): per-cell, per-slot vector weight
        self.cell_edge = cell_edge
        self.cell_edge_sign = et.cell_edge_sign
        self.cell_left = et.cell_left
        self.cell_right = et.cell_right

        # 4 neighbor edges of each edge (the other 2 edges of each of its
        # 2 flanking cells), for an isotropic edge-graph Laplacian. Used
        # instead of / alongside grad(div(u)): grad-div only damps the
        # *divergent* part of the flow, but a rotational (vorticity-mode)
        # grid-scale noise can grow completely undamped by it -- exactly
        # what happened when Coriolis+forcing were switched on and the
        # scheme blew up over ~6 days despite grad-div diffusion being on.
        left_edges = cell_edge[et.cell_left]    # (E,3)
        right_edges = cell_edge[et.cell_right]  # (E,3)
        all_edges = np.concatenate([left_edges, right_edges], axis=1)  # (E,6)
        self_idx = np.arange(e)[:, None]
        mask = all_edges != self_idx  # exactly 4 True per row (e appears twice)
        order = np.argsort(mask, axis=1)  # False(0) < True(1); last 4 cols = True
        self.edge_neighbors = np.take_along_axis(all_edges, order[:, -4:], axis=1)
        self._dx2 = self.cell_area.mean()  # characteristic length^2, m^2
        self.cell_neighbors = grid.neighbors  # (F,3), reused for a cell-centered Laplacian on h

    def divergence(self, flux_edge: np.ndarray) -> np.ndarray:
        """Cell-centered divergence of an edge-normal flux field."""
        contrib = self.cell_edge_sign * self.length[self.cell_edge] * flux_edge[self.cell_edge]
        return contrib.sum(axis=1) / self.cell_area

    def gradient_normal(self, h_cell: np.ndarray) -> np.ndarray:
        """Edge-normal component of grad(h), the exact adjoint partner of divergence()."""
        return (h_cell[self.cell_right] - h_cell[self.cell_left]) / self.d_center

    def reconstruct_cell_vector(self, u_edge: np.ndarray) -> np.ndarray:
        """Perot-reconstructed 3D tangent-plane velocity vector at each cell center."""
        u_local = u_edge[self.cell_edge]  # (F,3)
        return np.sum(self._perot_weight * u_local[:, :, None], axis=1)

    def edge_laplacian(self, u_edge: np.ndarray) -> np.ndarray:
        """Isotropic graph-Laplacian smoother on edge values: damps grid-scale
        noise regardless of whether it's divergent or rotational."""
        return (u_edge[self.edge_neighbors].mean(axis=1) - u_edge) / self._dx2

    def cell_laplacian(self, phi_cell: np.ndarray) -> np.ndarray:
        """Graph-Laplacian smoother on cell-centered values (e.g. h)."""
        return (phi_cell[self.cell_neighbors].mean(axis=1) - phi_cell) / self._dx2

    def tangential_at_edges(self, u_edge: np.ndarray) -> np.ndarray:
        """Reconstruct cell-center vectors, average to edges, project onto edge tangent."""
        v_cell = self.reconstruct_cell_vector(u_edge)
        v_edge = 0.5 * (v_cell[self.cell_left] + v_cell[self.cell_right])
        return np.einsum("ij,ij->i", v_edge, self.tangent)


@dataclass
class AtmosState:
    h: np.ndarray   # (n_cells,) m
    u: np.ndarray   # (n_edges,) m/s, edge-normal component
    q: np.ndarray   # (n_cells,) mm precipitable water (passive moisture tracer)


class ShallowWaterAtmosphere:
    def __init__(self, grid: Grid, params: AtmosParams | None = None, terrain=None):
        self.grid = grid
        self.p = params or AtmosParams()
        self.geo = CGridGeometry(grid, self.p.radius)
        self.t = 0.0
        self.terrain = terrain

        p = self.p
        if terrain is not None:
            self.is_land = terrain.is_land
            self.topo_b = np.where(
                terrain.is_land,
                np.clip(terrain.elevation - terrain.sea_level, 0.0, None) * p.topo_scale,
                0.0,
            )
        else:
            self.is_land = np.zeros(grid.n_cells, dtype=bool)
            self.topo_b = np.zeros(grid.n_cells)

        if terrain is not None:
            # smooth away single-cell slope spikes: an unresolved, near-vertical
            # ridge crest one triangle wide produces a huge local grad(b), which
            # blows the orographic rain term way past anything physical (checked:
            # unsmoothed, mean precip hit 150+ mm/day at a handful of cells).
            # Real terrain-following models smooth orography for the same reason.
            for _ in range(3):
                neighbor_mean = self.topo_b[grid.neighbors].mean(axis=1)
                self.topo_b = 0.5 * self.topo_b + 0.5 * neighbor_mean

        # blend ocean/land rates by averaging RATES (not timescales) so a
        # coast edge (one land cell, one ocean cell) gets a sensible middle
        # value rather than something skewed by which timescale is larger
        land_l = self.is_land[self.geo.cell_left].astype(float)
        land_r = self.is_land[self.geo.cell_right].astype(float)
        fric_rate_ocean, fric_rate_land = 1.0 / p.tau_fric, 1.0 / p.tau_fric_land
        self.fric_rate_edge = 0.5 * (
            (land_l * fric_rate_land + (1 - land_l) * fric_rate_ocean)
            + (land_r * fric_rate_land + (1 - land_r) * fric_rate_ocean)
        )
        rad_rate_ocean, rad_rate_land = 1.0 / p.tau_rad, 1.0 / p.tau_rad_land
        self.rad_rate_cell = np.where(self.is_land, rad_rate_land, rad_rate_ocean)

        # static terrain slope at cell centers, for the orographic rain term:
        # reuse reconstruct_cell_vector -- it's a generic "edge-normal-component
        # field -> cell-centered vector" operator, so feeding it grad(b)'s
        # edge-normal component (instead of a velocity) reconstructs grad(b)
        # itself as a proper vector.
        grad_b_edge = self.geo.gradient_normal(self.topo_b)
        self.grad_b_cell = self.geo.reconstruct_cell_vector(grad_b_edge)

        self.div_avg = np.zeros(grid.n_cells)  # running time-mean of div(u), see precipitation()

    def initial_state(self, perturb: float = 0.0, seed: int = 0) -> AtmosState:
        h = np.full(self.grid.n_cells, self.p.h0)
        if perturb:
            rng = np.random.default_rng(seed)
            h = h + perturb * rng.standard_normal(self.grid.n_cells)
        u = np.zeros(self.geo.edges.n_edges)
        q = np.zeros(self.grid.n_cells)  # start bone dry, let evaporation fill it in
        return AtmosState(h=h, u=u, q=q)

    def precipitation(self, s: AtmosState, u_cell: np.ndarray) -> np.ndarray:
        """Rain-out rate (mm/s): general convective (moisture above a threshold
        that convergence lowers and divergence raises -- real deserts are mostly
        subtropical subsidence zones, not just "far from mountains") plus
        orographic (forced ascent up a slope wrings out moisture directly,
        independent of the convective threshold -- this is what produces a
        windward-wet/leeward-dry rain shadow next to mountains)."""
        p = self.p
        # div_avg (updated in step(), not the instantaneous divergence): see
        # tau_div_avg docstring above -- <0 converging/rising, >0 diverging/sinking
        q_crit = p.q_crit_frac * p.q_sat + p.conv_sensitivity * self.div_avg
        convective = np.clip(s.q - q_crit, 0.0, None) / p.tau_precip
        ascent = np.einsum("ij,ij->i", u_cell, self.grad_b_cell)  # m/s "upslope wind"
        orographic = p.oro_coeff * np.clip(ascent, 0.0, None) * s.q
        return convective + orographic

    def h_equilibrium(self, t: float) -> np.ndarray:
        """Latitude+season dependent equilibrium thickness -- the insolation proxy.
        Land cells get an amplified deviation from h0 (land_amplify > 1): lower
        heat capacity means continental interiors run hotter in summer, colder
        in winter than the ocean at the same latitude."""
        lat = np.radians(self.grid.lat)
        year = 365.25 * 86400.0
        decl = self.p.obliquity * np.sin(2 * np.pi * t / year)
        coszen = np.clip(np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl), 0.0, None)
        insolation_norm = coszen - coszen.mean()
        amplify = np.where(self.is_land, self.p.land_amplify, 1.0)
        return self.p.h0 + self.p.delta_h_eq * amplify * insolation_norm / max(insolation_norm.max(), 1e-9)

    def cfl_dt(self, safety: float = 0.4) -> float:
        dx = np.sqrt(self.geo.cell_area.mean())
        wave_speed = np.sqrt(self.p.g * self.p.h0)
        return safety * dx / wave_speed

    def tendency(self, s: AtmosState, t: float) -> AtmosState:
        p, geo = self.p, self.geo

        h_eq = self.h_equilibrium(t)
        h_edge = 0.5 * (s.h[geo.cell_left] + s.h[geo.cell_right])
        flux = h_edge * s.u
        dhdt = (
            -geo.divergence(flux)
            - (s.h - h_eq) * self.rad_rate_cell
            + p.kappa_h * geo.cell_laplacian(s.h)
        )

        grad_eta = geo.gradient_normal(s.h + self.topo_b)  # free-surface = fluid + bottom
        u_tan = geo.tangential_at_edges(s.u)
        coriolis = geo.f_edge * u_tan

        diffusion = p.nu * geo.edge_laplacian(s.u)

        dudt = -p.g * grad_eta - coriolis + diffusion - self.fric_rate_edge * s.u

        u_cell = geo.reconstruct_cell_vector(s.u)
        q_edge = 0.5 * (s.q[geo.cell_left] + s.q[geo.cell_right])
        evap = np.where(self.is_land, 0.0, np.clip(p.q_sat - s.q, 0.0, None) / p.tau_evap)
        precip = self.precipitation(s, u_cell)
        dqdt = -geo.divergence(q_edge * s.u) + evap - precip + p.kappa_q * geo.cell_laplacian(s.q)

        return AtmosState(h=dhdt, u=dudt, q=dqdt)

    def step(self, s: AtmosState, dt: float) -> AtmosState:
        """
        Classic 4-stage RK4, not RK2. Gravity waves here are essentially
        undamped oscillators, and RK2's stability region only touches the
        imaginary axis at the origin -- any purely oscillatory mode
        amplifies every step regardless of step size (confirmed: at this
        grid's Nyquist wave frequency, RK2 amplifies energy ~1.6x/step,
        matching the blowup observed). RK4's stability region actually
        covers part of the imaginary axis, so it damps rather than
        amplifies oscillatory modes within range.
        """

        def add(a: AtmosState, b: AtmosState, scale: float) -> AtmosState:
            return AtmosState(h=a.h + scale * b.h, u=a.u + scale * b.u, q=a.q + scale * b.q)

        k1 = self.tendency(s, self.t)
        k2 = self.tendency(add(s, k1, 0.5 * dt), self.t + 0.5 * dt)
        k3 = self.tendency(add(s, k2, 0.5 * dt), self.t + 0.5 * dt)
        k4 = self.tendency(add(s, k3, dt), self.t + dt)

        h_new = s.h + (dt / 6.0) * (k1.h + 2 * k2.h + 2 * k3.h + k4.h)
        u_new = s.u + (dt / 6.0) * (k1.u + 2 * k2.u + 2 * k3.u + k4.u)
        q_new = np.clip(s.q + (dt / 6.0) * (k1.q + 2 * k2.q + 2 * k3.q + k4.q), 0.0, None)
        self.t += dt

        # exponential running mean of divergence, for precipitation()'s
        # desert-belt modulation (a simple low-pass filter, not part of the
        # RK4-integrated dynamics -- it's a smoothing diagnostic, not a
        # conserved physical quantity)
        div_now = self.geo.divergence(u_new)
        relax = min(1.0, dt / self.p.tau_div_avg)
        self.div_avg += relax * (div_now - self.div_avg)

        return AtmosState(h=h_new, u=u_new, q=q_new)

    def energy(self, s: AtmosState) -> float:
        """Diagnostic total energy (KE + available potential energy), for stability checks."""
        h_edge = 0.5 * (s.h[self.geo.cell_left] + s.h[self.geo.cell_right])
        ke = 0.5 * np.sum(h_edge * s.u ** 2 * self.geo.length * self.geo.d_center)
        ape = 0.5 * self.p.g * np.sum((s.h - self.p.h0) ** 2 * self.geo.cell_area)
        return ke + ape


if __name__ == "__main__":
    g = build_grid(4)

    print("=== Stage 0: pure gravity wave (no Coriolis/forcing/diffusion/friction) ===")
    quiet = ShallowWaterAtmosphere(
        g, AtmosParams(delta_h_eq=0.0, tau_fric=1e18, nu=0.0, omega=0.0)
    )
    dt = quiet.cfl_dt()
    print(f"cells={g.n_cells}  edges={quiet.geo.edges.n_edges}  dt={dt:.1f}s  "
          f"wave speed={np.sqrt(quiet.p.g*quiet.p.h0):.1f} m/s")
    s = quiet.initial_state(perturb=5.0, seed=0)
    e0 = quiet.energy(s)
    for i in range(300):
        s = quiet.step(s, dt)
        if i % 50 == 0 or i == 299:
            print(f"  step {i:4d}  h[min,max]=[{s.h.min():8.3f},{s.h.max():8.3f}]  "
                  f"|u|max={np.abs(s.u).max():8.4f}  E/E0={quiet.energy(s)/e0:8.4f}  "
                  f"finite={np.isfinite(s.h).all()}")

    print("\n=== Stage 1: no forcing, full model (Coriolis on) ===")
    quiet2 = ShallowWaterAtmosphere(g, AtmosParams(delta_h_eq=0.0))
    s = quiet2.initial_state()
    for i in range(200):
        s = quiet2.step(s, dt)
    print(f"  h range: [{s.h.min():.6f}, {s.h.max():.6f}] (started at {quiet2.p.h0})")
    print(f"  |u| max: {np.abs(s.u).max():.8f} m/s (started at rest)")

    print("\n=== Stage 2: full insolation forcing ===")
    model = ShallowWaterAtmosphere(g)
    s = model.initial_state()
    n_steps = int(20 * 86400 / dt)
    for i in range(n_steps):
        s = model.step(s, dt)
        if i % (n_steps // 10) == 0:
            print(f"  t={model.t/86400:5.1f}d  h[min,max]=[{s.h.min():8.2f},{s.h.max():8.2f}]  "
                  f"|u|mean/max={np.abs(s.u).mean():6.3f}/{np.abs(s.u).max():6.3f} m/s  "
                  f"finite={np.isfinite(s.h).all()}")
