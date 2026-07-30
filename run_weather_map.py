"""Run the terrain-coupled shallow-water atmosphere and map the result."""
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

    n_days = 90
    n_steps = int(n_days * 86400 / dt)
    for i in range(n_steps):
        s = model.step(s, dt)

    vec = model.geo.reconstruct_cell_vector(s.u)
    speed = np.linalg.norm(vec, axis=1)
    h_anom = s.h - model.p.h0
    u_east, v_north = geographic_components(g, vec)

    print(f"t={model.t/86400:.1f}d  |u| mean/max={speed.mean():.2f}/{speed.max():.2f} m/s  "
          f"h anomaly range=[{h_anom.min():.1f}, {h_anom.max():.1f}] m")

    fig, axes = plt.subplots(3, 1, figsize=(14, 17))

    plot_rgb(g, biome_colors(terrain), ax=axes[0], title="Terrain (land/ocean/biome)")

    plot_scalar(g, speed, ax=axes[1], title=f"Surface wind speed, day {n_days}",
                cmap="magma", vmin=0, cbar_label="m/s")
    quiver_wind(g, u_east, v_north, ax=axes[1], stride=8)
    plot_coastline(g, terrain.is_land, axes[1], color="cyan")

    vmax_h = np.abs(h_anom).max()
    plot_scalar(g, h_anom, ax=axes[2], title=f"Height anomaly (pressure proxy), day {n_days}",
                cmap="RdBu_r", vmin=-vmax_h, vmax=vmax_h, cbar_label="m")
    plot_coastline(g, terrain.is_land, axes[2], color="black")

    plt.tight_layout()
    plt.savefig("map_weather.png", dpi=130)
    print("wrote map_weather.png")
