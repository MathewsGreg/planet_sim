"""Run terrain-coupled atmosphere + moisture, accumulate precipitation, and
re-classify biomes from *simulated* precipitation instead of a fixed latitude
band. Compares against the original latitude-only classification."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from grid import build_grid
from terrain import random_elevation, classify
from atmosphere import ShallowWaterAtmosphere
from visualize import biome_colors
from map_view import plot_rgb, plot_scalar, plot_coastline

if __name__ == "__main__":
    g = build_grid(4)
    elev = random_elevation(g, seed=1)
    terrain_latitude = classify(g, elev)  # original latitude-only biomes

    model = ShallowWaterAtmosphere(g, terrain=terrain_latitude)
    dt = model.cfl_dt()
    s = model.initial_state()

    spinup_days, accum_days = 60, 60
    for i in range(int(spinup_days * 86400 / dt)):
        s = model.step(s, dt)
    print(f"spin-up done: t={model.t/86400:.1f}d")

    accum_precip = np.zeros(g.n_cells)
    for i in range(int(accum_days * 86400 / dt)):
        s = model.step(s, dt)
        u_cell = model.geo.reconstruct_cell_vector(s.u)
        accum_precip += model.precipitation(s, u_cell) * dt
    mean_precip = accum_precip / accum_days
    print(f"mean precip range=[{mean_precip.min():.2f}, {mean_precip.max():.2f}] mm/day")

    terrain_weather = classify(g, elev, precip=mean_precip)  # weather-driven biomes

    for name, t in [("latitude-only", terrain_latitude), ("weather-driven", terrain_weather)]:
        for code, (biome_name, _) in [(2, ("desert", None)), (3, ("forest", None)), (4, ("tundra", None))]:
            frac = g.area[t.biome == code].sum() / g.area.sum()
            print(f"  {name:15s} {biome_name:8s} {frac:.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 17))

    plot_rgb(g, biome_colors(terrain_latitude), ax=axes[0], title="Biomes: latitude-only (original)")

    vmax_precip = np.percentile(mean_precip, 98)
    plot_scalar(g, mean_precip, ax=axes[1], title=f"Mean precipitation, days {spinup_days}-{spinup_days+accum_days}",
                cmap="YlGnBu", vmin=0, vmax=vmax_precip, cbar_label="mm/day")
    plot_coastline(g, terrain_latitude.is_land, axes[1], color="red")

    plot_rgb(g, biome_colors(terrain_weather), ax=axes[2], title="Biomes: weather-driven (from simulated precipitation)")

    plt.tight_layout()
    plt.savefig("map_biomes_weather.png", dpi=130)
    print("wrote map_biomes_weather.png")
