"""
Resume the ocean from a checkpoint, switch to a Munk-boundary-layer-
matched viscosity, and keep spinning up -- no atmosphere re-run needed.

Two changes from the checkpointed state, both discussed before running:
  1. nu -> munk_matched_nu(): the western boundary layer was structurally
     under-resolved before (~100km theoretical width against ~316km
     cells), which is the leading suspect for why the first spin-up
     produced turbulent, gyre-less currents instead of an organized loop.
  2. More simulated time: gyres set up via basin-crossing Rossby waves,
     whose transit time at this basin scale works out to years, not the
     4 already run.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from ocean import munk_matched_nu
from map_view import plot_scalar, plot_coastline, streamlines, geographic_components

if __name__ == "__main__":
    cp = load_checkpoint("checkpoint_year1.npz")
    ocean, s_o, g, terrain = cp.ocean, cp.ocean_state, cp.grid, cp.terrain

    old_nu, old_dt = ocean.p.nu, ocean.cfl_dt()
    ocean.p.nu = munk_matched_nu(ocean.geo, target_cells=3.0)
    new_dt = ocean.cfl_dt()
    print(f"nu: {old_nu:.2e} -> {ocean.p.nu:.2e} m^2/s")
    print(f"dt: {old_dt:.0f}s -> {new_dt:.0f}s")
    print(f"resuming from ocean t={ocean.t / 86400 / 365:.2f}y")

    more_years = 8
    n_steps = int(more_years * 365 * 86400 / new_dt)
    print(f"running {more_years} more years, {n_steps} steps")
    for i in range(n_steps):
        s_o = ocean.step(s_o, new_dt)
        if i % max(1, n_steps // 20) == 0 or i == n_steps - 1:
            speed = np.abs(s_o.u)
            print(f"  t={ocean.t / 365 / 86400:6.2f}y  h[min,max]=[{s_o.h.min():7.2f},{s_o.h.max():7.2f}]  "
                  f"|u| mean/max={speed.mean():.4f}/{speed.max():.4f} m/s  "
                  f"finite={np.isfinite(s_o.h).all()}", flush=True)
        if not np.isfinite(s_o.h).all():
            print("BLEW UP, stopping early")
            break

    cp2 = Checkpoint(
        grid=g, terrain=terrain, atmos=cp.atmos, atmos_state=cp.atmos_state,
        ocean=ocean, ocean_state=s_o, wind_forcing=cp.wind_forcing,
    )
    total_years = ocean.t / 86400 / 365
    ckpt_name = f"checkpoint_year{total_years:.0f}.npz"
    save_checkpoint(ckpt_name, cp2)
    print(f"wrote {ckpt_name}")

    # --- Plot ---
    vec = ocean.geo.reconstruct_cell_vector(s_o.u)
    speed_cell = np.linalg.norm(vec, axis=1)
    u_east, v_north = geographic_components(g, vec)
    u_east_ocean = np.where(terrain.is_land, 0.0, u_east)
    v_north_ocean = np.where(terrain.is_land, 0.0, v_north)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    vmax_speed = np.percentile(speed_cell[~terrain.is_land], 99)
    plot_scalar(g, speed_cell, ax=axes[0],
                title=f"Ocean current speed after {total_years:.1f}y (nu={ocean.p.nu:.1e} m^2/s, Munk-matched)",
                cmap="viridis", vmin=0, vmax=vmax_speed, cbar_label="m/s")
    plot_coastline(g, terrain.is_land, axes[0], color="white")

    plot_scalar(g, speed_cell, ax=axes[1], title="...with streamlines", cmap="viridis",
                vmin=0, vmax=vmax_speed, cbar_label="m/s")
    plot_coastline(g, terrain.is_land, axes[1], color="white")
    streamlines(g, u_east_ocean, v_north_ocean, axes[1], mask=terrain.is_land, color="white", density=2.5)

    plt.tight_layout()
    plt.savefig("map_ocean_currents_v2.png", dpi=130)
    print("wrote map_ocean_currents_v2.png")
