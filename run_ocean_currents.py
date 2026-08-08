"""
Spin up the atmosphere over real procedural continents, take its
time-mean surface wind as a fixed forcing, and spin up the wind-driven
ocean layer under it -- then plot current speed + streamlines to see
whether gyres and western boundary intensification show up.

One-way coupling only (atmosphere -> ocean, no feedback yet): matches the
"wind-driven surface layer" scope in ocean.py. The atmosphere is spun up
first and its wind *time-averaged* over a second window, rather than
coupling live step-by-step -- the atmosphere's synoptic (day-scale)
variability isn't what drives the multi-year-mean gyre circulation, it's
the wind climatology, exactly the quantity Sverdrup/Munk theory itself is
forced by.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from grid import build_grid
from terrain import random_elevation, classify
from atmosphere import ShallowWaterAtmosphere
from ocean import WindDrivenOcean, OceanParams, wind_stress_forcing_edge, smooth_edge_field
from map_view import plot_scalar, plot_coastline, streamlines, geographic_components
from checkpoint import Checkpoint, save_checkpoint

if __name__ == "__main__":
    g = build_grid(4)
    elev = random_elevation(g, seed=1)
    terrain = classify(g, elev)
    print(f"land fraction: {g.area[terrain.is_land].sum() / g.area.sum():.3f}")

    # --- Atmosphere: spin up, then average its wind over a FULL YEAR ---
    # (not a partial window -- averaging less than a year biases the mean
    # wind toward whichever hemisphere's winter jet happened to fall in
    # that window, which is exactly what produced the lopsided,
    # one-hemisphere-only current pattern in the first pass)
    atmos = ShallowWaterAtmosphere(g, terrain=terrain)
    dt_a = atmos.cfl_dt()
    s_a = atmos.initial_state()

    spinup_days, average_days = 90, 365
    n_spinup = int(spinup_days * 86400 / dt_a)
    for i in range(n_spinup):
        s_a = atmos.step(s_a, dt_a)
        if i % max(1, n_spinup // 4) == 0:
            print(f"  atmosphere spin-up: t={atmos.t / 86400:.1f}d", flush=True)
    print(f"atmosphere spin-up done: t={atmos.t / 86400:.1f}d")

    ocean_p = OceanParams()
    n_avg_steps = int(average_days * 86400 / dt_a)
    forcing_accum = np.zeros(atmos.geo.edges.n_edges)
    for i in range(n_avg_steps):
        s_a = atmos.step(s_a, dt_a)
        forcing_accum += wind_stress_forcing_edge(atmos, s_a, ocean_p)
        if i % max(1, n_avg_steps // 8) == 0:
            print(f"  atmosphere wind-averaging: t={atmos.t / 86400:.1f}d", flush=True)
    mean_wind_forcing = forcing_accum / n_avg_steps
    # The atmosphere's time-mean wind still carries grid-scale spatial
    # structure that averaging over time doesn't remove (it's spatially
    # persistent, not transient noise) -- feeding it in raw drove the
    # ocean to NaN by ~day 180 in testing. Smooth it the same way
    # atmosphere.py smooths spiky single-cell terrain slopes. See
    # ocean.smooth_edge_field's docstring for the full story.
    mean_wind_forcing = smooth_edge_field(atmos.geo, mean_wind_forcing, passes=6)
    print(f"atmosphere wind-averaging done: t={atmos.t / 86400:.1f}d  "
          f"|forcing| mean/max = {np.abs(mean_wind_forcing).mean():.2e}/"
          f"{np.abs(mean_wind_forcing).max():.2e} m/s^2")

    # --- Ocean: same grid/edge geometry (literally shared), forced by that mean wind ---
    ocean = WindDrivenOcean(g, params=ocean_p, terrain=terrain, geo=atmos.geo)
    ocean.set_wind_forcing(mean_wind_forcing)
    dt_o = ocean.cfl_dt()
    s_o = ocean.initial_state()

    ocean_years = 4
    n_steps_o = int(ocean_years * 365 * 86400 / dt_o)
    print(f"ocean: dt={dt_o:.0f}s  steps={n_steps_o}  simulating {ocean_years} years")
    for i in range(n_steps_o):
        s_o = ocean.step(s_o, dt_o)
        if i % max(1, n_steps_o // 8) == 0 or i == n_steps_o - 1:
            speed = np.abs(s_o.u)
            print(f"  t={ocean.t / 365 / 86400:5.2f}y  h[min,max]=[{s_o.h.min():7.2f},{s_o.h.max():7.2f}]  "
                  f"|u| mean/max={speed.mean():.4f}/{speed.max():.4f} m/s  finite={np.isfinite(s_o.h).all()}", flush=True)

    cp = Checkpoint(
        grid=g, terrain=terrain, atmos=atmos, atmos_state=s_a,
        ocean=ocean, ocean_state=s_o, wind_forcing=mean_wind_forcing,
    )
    save_checkpoint("checkpoint_year1.npz", cp)
    print("wrote checkpoint_year1.npz "
          "(reload with checkpoint.load_checkpoint('checkpoint_year1.npz'))")

    # --- Plot ---
    vec = ocean.geo.reconstruct_cell_vector(s_o.u)
    speed_cell = np.linalg.norm(vec, axis=1)
    u_east, v_north = geographic_components(g, vec)
    # zero out current vectors on land for a clean streamline plot
    u_east_ocean = np.where(terrain.is_land, 0.0, u_east)
    v_north_ocean = np.where(terrain.is_land, 0.0, v_north)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    vmax_speed = np.percentile(speed_cell[~terrain.is_land], 99)
    plot_scalar(g, speed_cell, ax=axes[0], title=f"Ocean current speed after {ocean_years}y wind-driven spin-up",
                cmap="viridis", vmin=0, vmax=vmax_speed, cbar_label="m/s")
    plot_coastline(g, terrain.is_land, axes[0], color="white")

    plot_scalar(g, speed_cell, ax=axes[1], title="...with streamlines", cmap="viridis",
                vmin=0, vmax=vmax_speed, cbar_label="m/s")
    plot_coastline(g, terrain.is_land, axes[1], color="white")
    streamlines(g, u_east_ocean, v_north_ocean, axes[1], mask=terrain.is_land, color="white", density=2.5)

    plt.tight_layout()
    plt.savefig("map_ocean_currents.png", dpi=130)
    print("wrote map_ocean_currents.png")
