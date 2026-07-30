"""Run the shallow-water atmosphere and render 2D maps of wind speed + height anomaly."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from grid import build_grid
from atmosphere import ShallowWaterAtmosphere, AtmosParams
from map_view import plot_scalar, quiver_wind, geographic_components

if __name__ == "__main__":
    g = build_grid(4)
    model = ShallowWaterAtmosphere(g)
    dt = model.cfl_dt()
    s = model.initial_state()

    n_days = 60
    n_steps = int(n_days * 86400 / dt)
    for i in range(n_steps):
        s = model.step(s, dt)

    speed = np.linalg.norm(model.geo.reconstruct_cell_vector(s.u), axis=1)
    h_anom = s.h - model.p.h0
    u_east, v_north = geographic_components(g, model.geo.reconstruct_cell_vector(s.u))

    print(f"t={model.t/86400:.1f} days  |u| mean/max={speed.mean():.2f}/{speed.max():.2f} m/s  "
          f"h anomaly range=[{h_anom.min():.1f}, {h_anom.max():.1f}] m")

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    plot_scalar(g, speed, ax=axes[0], title=f"Surface wind speed, day {n_days}",
                cmap="magma", vmin=0, cbar_label="m/s")
    quiver_wind(g, u_east, v_north, ax=axes[0], stride=8)

    vmax_h = np.abs(h_anom).max()
    plot_scalar(g, h_anom, ax=axes[1], title=f"Height anomaly (pressure proxy), day {n_days}",
                cmap="RdBu_r", vmin=-vmax_h, vmax=vmax_h, cbar_label="m")

    plt.tight_layout()
    plt.savefig("map_atmosphere.png", dpi=130)
    print("wrote map_atmosphere.png")
