"""Run terrain-coupled atmosphere + moisture, accumulate precipitation, and map it
against elevation to look for windward-wet / leeward-dry (rain shadow) patterns."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from grid import build_grid
from terrain import random_elevation, classify
from atmosphere import ShallowWaterAtmosphere
from visualize import biome_colors
from map_view import plot_rgb, plot_scalar, plot_coastline, quiver_wind, geographic_components

if __name__ == "__main__":
    g = build_grid(4)
    elev = random_elevation(g, seed=1)
    terrain = classify(g, elev)

    model = ShallowWaterAtmosphere(g, terrain=terrain)
    dt = model.cfl_dt()
    s = model.initial_state()

    spinup_days = 60   # let winds + moisture reach a statistical equilibrium first
    accum_days = 60     # then accumulate precipitation over this window
    spinup_steps = int(spinup_days * 86400 / dt)
    accum_steps = int(accum_days * 86400 / dt)

    for i in range(spinup_steps):
        s = model.step(s, dt)
    print(f"spin-up done: t={model.t/86400:.1f}d")

    accum_precip = np.zeros(g.n_cells)  # mm
    for i in range(accum_steps):
        s = model.step(s, dt)
        u_cell = model.geo.reconstruct_cell_vector(s.u)
        precip_rate = model.precipitation(s, u_cell)  # mm/s
        accum_precip += precip_rate * dt

    mean_precip_mmday = accum_precip / accum_days
    u_cell = model.geo.reconstruct_cell_vector(s.u)
    u_east, v_north = geographic_components(g, u_cell)

    print(f"t={model.t/86400:.1f}d  mean precip range=[{mean_precip_mmday.min():.2f}, "
          f"{mean_precip_mmday.max():.2f}] mm/day")

    fig, axes = plt.subplots(3, 1, figsize=(14, 17))

    plot_rgb(g, biome_colors(terrain), ax=axes[0], title="Terrain (land/ocean/biome)")

    elev_land = np.where(terrain.is_land, terrain.elevation - terrain.sea_level, np.nan)
    plot_scalar(g, np.nan_to_num(elev_land, nan=0.0), ax=axes[1],
                title="Land elevation above sea level (unitless terrain field)",
                cmap="terrain", vmin=0, cbar_label="elevation units")
    plot_coastline(g, terrain.is_land, axes[1], color="black")
    quiver_wind(g, u_east, v_north, ax=axes[1], stride=8)

    vmax_precip = np.percentile(mean_precip_mmday, 98)
    plot_scalar(g, mean_precip_mmday, ax=axes[2],
                title=f"Mean precipitation, days {spinup_days}-{spinup_days + accum_days}",
                cmap="YlGnBu", vmin=0, vmax=vmax_precip, cbar_label="mm/day")
    plot_coastline(g, terrain.is_land, axes[2], color="red")

    plt.tight_layout()
    plt.savefig("map_precip.png", dpi=130)
    print("wrote map_precip.png")
