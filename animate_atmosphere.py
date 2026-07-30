"""Animate the shallow-water atmosphere as a 2D map GIF (wind speed + height anomaly)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection

from grid import build_grid
from atmosphere import ShallowWaterAtmosphere
from map_view import triangle_lonlat_coords, geographic_components

if __name__ == "__main__":
    g = build_grid(4)
    model = ShallowWaterAtmosphere(g)
    dt = model.cfl_dt()
    s = model.initial_state()

    n_days = 150
    frame_every_days = 1.5
    steps_per_frame = max(1, int(frame_every_days * 86400 / dt))
    n_frames = int(n_days / frame_every_days)
    stride = 8  # quiver arrow subsampling

    coords, wrap_mask = triangle_lonlat_coords(g)
    keep = ~wrap_mask
    coords_keep = coords[keep]
    idx = np.arange(g.n_cells)[::stride]

    speed_frames, h_anom_frames, u_frames, v_frames, t_frames = [], [], [], [], []

    for f in range(n_frames):
        for _ in range(steps_per_frame):
            s = model.step(s, dt)
        vec = model.geo.reconstruct_cell_vector(s.u)
        speed = np.linalg.norm(vec, axis=1)
        u_east, v_north = geographic_components(g, vec)
        speed_frames.append(speed[keep])
        h_anom_frames.append((s.h - model.p.h0)[keep])
        u_frames.append(u_east[idx])
        v_frames.append(v_north[idx])
        t_frames.append(model.t / 86400.0)
        print(f"frame {f + 1}/{n_frames}  t={model.t / 86400:.1f}d", end="\r")
    print()

    speed_all = np.array(speed_frames)
    h_all = np.array(h_anom_frames)
    vmax_speed = np.percentile(speed_all, 99.5)
    vmax_h = np.percentile(np.abs(h_all), 99.5)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    cmap_speed = plt.get_cmap("magma")
    norm_speed = plt.Normalize(0, vmax_speed)
    pc_speed = PolyCollection(coords_keep, facecolors=cmap_speed(norm_speed(speed_all[0])), edgecolors="none")
    axes[0].add_collection(pc_speed)
    axes[0].set_xlim(-180, 180)
    axes[0].set_ylim(-90, 90)
    axes[0].set_aspect("equal")
    axes[0].set_title("Surface wind speed")
    sm = plt.cm.ScalarMappable(norm=norm_speed, cmap=cmap_speed)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0], label="m/s", fraction=0.025, pad=0.02)
    quiv = axes[0].quiver(
        g.lon[idx], g.lat[idx], u_frames[0], v_frames[0],
        color="white", scale_units="inches", scale=120, width=0.0018, alpha=0.85,
    )

    cmap_h = plt.get_cmap("RdBu_r")
    norm_h = plt.Normalize(-vmax_h, vmax_h)
    pc_h = PolyCollection(coords_keep, facecolors=cmap_h(norm_h(h_all[0])), edgecolors="none")
    axes[1].add_collection(pc_h)
    axes[1].set_xlim(-180, 180)
    axes[1].set_ylim(-90, 90)
    axes[1].set_aspect("equal")
    axes[1].set_title("Height anomaly (pressure proxy)")
    sm2 = plt.cm.ScalarMappable(norm=norm_h, cmap=cmap_h)
    sm2.set_array([])
    plt.colorbar(sm2, ax=axes[1], label="m", fraction=0.025, pad=0.02)

    suptitle = fig.suptitle(f"day {t_frames[0]:.1f}")

    def update(i):
        pc_speed.set_facecolor(cmap_speed(norm_speed(speed_all[i])))
        pc_h.set_facecolor(cmap_h(norm_h(h_all[i])))
        quiv.set_UVC(u_frames[i], v_frames[i])
        suptitle.set_text(f"day {t_frames[i]:.1f}")
        return pc_speed, pc_h, quiv, suptitle

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=120)
    anim.save("atmosphere_animation.gif", writer=animation.PillowWriter(fps=8))
    print("wrote atmosphere_animation.gif")
