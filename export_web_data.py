"""
Export a fresh terrain + animated atmosphere + ocean-current dataset as a
single compact JSON, for the interactive web globe. Self-contained web
artifacts can't fetch external data at runtime (no server, strict CSP), so
this produces the exact payload that gets inlined directly into the page.

Uses the *new* l_min=2 terrain default (see terrain.py) -- the earlier
checkpoints predate that change and still have the old one-giant-landmass
geography, so this regenerates everything fresh for consistency across
all three layers (terrain/wind/currents must agree on where the coastlines
are).

Quantizes aggressively (speeds to 0-255, coordinates to 4 decimals) since
this all has to live inside one HTML file: still far more precision than
a rendered globe can show, at a fraction of the JSON size.
"""
from __future__ import annotations

import json

import numpy as np

from grid import build_grid
from terrain import random_elevation, classify, BIOMES
from atmosphere import ShallowWaterAtmosphere
from ocean import WindDrivenOcean, OceanParams, wind_stress_forcing_edge, smooth_edge_field
from visualize import biome_colors
from map_view import geographic_components

if __name__ == "__main__":
    g = build_grid(4)
    elev = random_elevation(g, seed=1)  # new l_min=2 default -> a couple of continents
    terrain = classify(g, elev)
    print(f"land fraction: {g.area[terrain.is_land].sum() / g.area.sum():.3f}")

    terrain_rgb = biome_colors(terrain, elevation_shading=True)
    terrain_hex = ["#%02x%02x%02x" % tuple(c) for c in terrain_rgb]

    # --- Atmosphere: animate the first 90 days, then 30 more (uncaptured)
    #     days of wind-averaging to force the ocean ---
    atmos = ShallowWaterAtmosphere(g, terrain=terrain)
    dt_a = atmos.cfl_dt()
    s_a = atmos.initial_state()

    animate_days, frame_every = 90, 1.5
    steps_per_frame = max(1, int(frame_every * 86400 / dt_a))
    n_frames = int(animate_days / frame_every)

    wind_frames_speed = []
    wind_days = []
    for f in range(n_frames):
        for _ in range(steps_per_frame):
            s_a = atmos.step(s_a, dt_a)
        vec = atmos.geo.reconstruct_cell_vector(s_a.u)
        speed = np.linalg.norm(vec, axis=1)
        wind_frames_speed.append(speed)
        wind_days.append(atmos.t / 86400.0)
        print(f"  wind frame {f + 1}/{n_frames}  t={atmos.t / 86400:.1f}d", end="\r", flush=True)
    print()

    wind_all = np.array(wind_frames_speed)
    wind_vmax = float(np.percentile(wind_all, 99.5))
    wind_q = np.clip(wind_all / wind_vmax * 255, 0, 255).astype(np.uint8)

    ocean_p = OceanParams()
    average_days = 30
    n_avg_steps = int(average_days * 86400 / dt_a)
    forcing_accum = np.zeros(atmos.geo.edges.n_edges)
    for i in range(n_avg_steps):
        s_a = atmos.step(s_a, dt_a)
        forcing_accum += wind_stress_forcing_edge(atmos, s_a, ocean_p)
        if i % max(1, n_avg_steps // 4) == 0:
            print(f"  wind-averaging: t={atmos.t / 86400:.1f}d", flush=True)
    mean_wind_forcing = smooth_edge_field(atmos.geo, forcing_accum / n_avg_steps, passes=6)

    # --- Ocean: a modest spin-up at the original (not yet Munk-matched)
    #     viscosity -- good enough for a first visualization pass; the
    #     boundary-layer/biharmonic work is tracked separately ---
    ocean = WindDrivenOcean(g, params=ocean_p, terrain=terrain, geo=atmos.geo)
    ocean.set_wind_forcing(mean_wind_forcing)
    dt_o = ocean.cfl_dt()
    s_o = ocean.initial_state()
    ocean_years = 3
    n_steps_o = int(ocean_years * 365 * 86400 / dt_o)
    print(f"ocean: dt={dt_o:.0f}s steps={n_steps_o}")
    for i in range(n_steps_o):
        s_o = ocean.step(s_o, dt_o)
        if i % max(1, n_steps_o // 6) == 0:
            print(f"  ocean: t={ocean.t / 86400 / 365:.2f}y", flush=True)

    vec_o = ocean.geo.reconstruct_cell_vector(s_o.u)
    current_speed = np.linalg.norm(vec_o, axis=1)
    current_speed = np.where(terrain.is_land, 0.0, current_speed)
    current_vmax = float(np.percentile(current_speed[~terrain.is_land], 99))
    current_q = np.clip(current_speed / max(current_vmax, 1e-9) * 255, 0, 255).astype(np.uint8)

    # --- Pack ---
    data = {
        "vertices": np.round(g.vertices, 4).tolist(),
        "faces": g.faces.tolist(),
        "isLand": terrain.is_land.astype(int).tolist(),
        "terrainColor": terrain_hex,
        "windDays": [round(d, 1) for d in wind_days],
        "windVmax": round(wind_vmax, 2),
        "windFrames": wind_q.tolist(),
        "currentVmax": round(current_vmax, 3),
        "currentSpeed": current_q.tolist(),
    }
    with open("web_data.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))

    import os
    size_kb = os.path.getsize("web_data.json") / 1024
    print(f"wrote web_data.json ({size_kb:.0f} KB, {len(wind_frames_speed)} wind frames, "
          f"{g.n_cells} cells, {len(g.vertices)} vertices)")
