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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grid import build_grid
from terrain import random_elevation, classify, BIOMES
from atmosphere import ShallowWaterAtmosphere
from ocean import WindDrivenOcean, OceanParams, wind_stress_forcing_edge, smooth_edge_field, munk_matched_nu4
from visualize import biome_colors, sea_ice_overlay
from map_view import geographic_components, streamlines


def lonlat_to_xyz(lon, lat):
    lonr, latr = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(latr) * np.cos(lonr), np.cos(latr) * np.sin(lonr), np.sin(latr)], axis=-1)

if __name__ == "__main__":
    g = build_grid(4)
    elev = random_elevation(g, seed=1)  # new l_min=2 default -> a couple of continents
    # Elevation-only pass first: is_land/sea_level only depend on elevation,
    # and the atmosphere needs a terrain (land contrast) to run at all --
    # biome labels get replaced below once real precipitation exists, but
    # is_land itself never changes, so the ocean/current code downstream
    # that depends on it is unaffected by the later reclassification.
    terrain = classify(g, elev)
    print(f"land fraction: {g.area[terrain.is_land].sum() / g.area.sum():.3f}")

    # --- Atmosphere: animate the first 90 days, then 30 more (uncaptured)
    #     days of wind-averaging to force the ocean ---
    atmos = ShallowWaterAtmosphere(g, terrain=terrain)
    dt_a = atmos.cfl_dt()
    s_a = atmos.initial_state()

    animate_days, frame_every = 90, 1.5
    steps_per_frame = max(1, int(frame_every * 86400 / dt_a))
    n_frames = int(animate_days / frame_every)

    # subsampled cells for direction arrows -- same "every Nth cell" idea
    # map_view.quiver_wind uses for the 2D plots, just applied on the globe
    arrow_stride = 8
    arrow_cells = np.arange(g.n_cells)[::arrow_stride]

    # Accumulated alongside the wind animation below (same run, no extra
    # atmosphere spin-up needed) -- same per-step mm/s * dt accumulation
    # run_weather_biomes.py validated, just spread over the animated window
    # instead of a separate dedicated spin-up+accumulate pair.
    accum_precip = np.zeros(g.n_cells)

    wind_frames_speed = []
    wind_arrow_dir = []  # (n_frames, n_arrows, 3) unit tangent vectors, quantized
    wind_days = []
    for f in range(n_frames):
        for _ in range(steps_per_frame):
            s_a = atmos.step(s_a, dt_a)
            u_step = atmos.geo.reconstruct_cell_vector(s_a.u)
            accum_precip += atmos.precipitation(s_a, u_step) * dt_a
        vec = atmos.geo.reconstruct_cell_vector(s_a.u)
        speed = np.linalg.norm(vec, axis=1)
        wind_frames_speed.append(speed)
        arrow_vec = vec[arrow_cells]
        arrow_speed = speed[arrow_cells]
        arrow_dir = arrow_vec / np.clip(arrow_speed, 1e-6, None)[:, None]
        wind_arrow_dir.append(np.round(arrow_dir * 127).astype(np.int8))
        wind_days.append(atmos.t / 86400.0)
        print(f"  wind frame {f + 1}/{n_frames}  t={atmos.t / 86400:.1f}d", end="\r", flush=True)
    print()

    # --- Reclassify biomes from real simulated precipitation (mm/day,
    # averaged over the animated window above) instead of the fixed
    # 12-35 degree latitude band. Ice/tundra stay latitude+elevation-driven
    # -- there's no simulated temperature field yet, see terrain.classify --
    # only the desert/forest split changes. is_land/sea_level are unchanged
    # (classify() recomputes them identically from the same elevation), so
    # nothing downstream (atmos, ocean) needs to be rebuilt.
    mean_precip = accum_precip / animate_days  # mm/day
    print(f"mean precip range=[{mean_precip.min():.2f}, {mean_precip.max():.2f}] mm/day")
    old_desert_frac = g.area[terrain.biome == 2].sum() / g.area.sum()
    terrain = classify(g, elev, precip=mean_precip)
    new_desert_frac = g.area[terrain.biome == 2].sum() / g.area.sum()
    print(f"desert land fraction: latitude-band={old_desert_frac:.3f} -> weather-driven={new_desert_frac:.3f}")

    # terrain_hex (the final packed color) isn't built yet -- the sea-ice
    # overlay below needs the ocean's spun-up SST first, so terrain_rgb
    # gets finalized after the ocean spin-up further down.
    terrain_rgb = biome_colors(terrain, elevation_shading=True)

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

    # --- Ocean: biharmonic (nabla^4) hyperviscosity for the western
    #     boundary layer, harmonic nu left at its original small value --
    #     see ocean.py's module docstring and README for the harmonic-vs-
    #     biharmonic story. target_cells=2.0 chosen empirically (swept
    #     1.0/1.5/2.0 against the cached wind forcing): 2.0 is where
    #     streamlines actually go from turbulent scribble to smooth,
    #     coherent flow; the cost is a real one -- dt drops ~5.5x, this
    #     phase is the slow part of this script now.
    ocean = WindDrivenOcean(g, params=ocean_p, terrain=terrain, geo=atmos.geo)
    ocean_p.nu4 = munk_matched_nu4(ocean.geo, target_cells=2.0)
    ocean.set_wind_forcing(mean_wind_forcing)
    dt_o = ocean.cfl_dt()
    s_o = ocean.initial_state()
    ocean_years = 4
    n_steps_o = int(ocean_years * 365 * 86400 / dt_o)
    print(f"ocean: nu4={ocean_p.nu4:.2e}  dt={dt_o:.0f}s steps={n_steps_o}")
    for i in range(n_steps_o):
        s_o = ocean.step(s_o, dt_o)
        if i % max(1, n_steps_o // 6) == 0:
            print(f"  ocean: t={ocean.t / 86400 / 365:.2f}y", flush=True)

    # --- Sea ice: real, ocean-state-dependent, wherever spun-up SST is at
    # or below freezing -- unlike land ice/tundra (still latitude+elevation
    # only, see terrain.classify), this can appear anywhere the simulated
    # ocean actually gets cold enough, including the land-sparse southern
    # high latitudes that never got land-based ice before (see README on
    # the hemisphere land-fraction asymmetry). Overlaid onto terrain_rgb
    # now that SST has had the full ocean spin-up to reach equilibrium.
    ice_frac = float((~terrain.is_land & (s_o.T <= ocean_p.T_freeze)).sum()) / g.n_cells
    print(f"SST range=[{s_o.T.min():.1f}, {s_o.T.max():.1f}]degC  "
          f"sea ice fraction of all cells={ice_frac:.3f}")
    terrain_rgb = sea_ice_overlay(terrain_rgb, s_o.T, terrain.is_land, ocean_p.T_freeze)
    terrain_hex = ["#%02x%02x%02x" % tuple(c) for c in terrain_rgb]

    vec_o = ocean.geo.reconstruct_cell_vector(s_o.u)
    current_speed = np.linalg.norm(vec_o, axis=1)
    current_speed = np.where(terrain.is_land, 0.0, current_speed)
    current_vmax = float(np.percentile(current_speed[~terrain.is_land], 99))
    current_q = np.clip(current_speed / max(current_vmax, 1e-9) * 255, 0, 255).astype(np.uint8)

    # --- Current streamlines: reuse map_view.streamlines()'s existing,
    # already-tested griddata-interpolation + matplotlib streamplot tracer
    # rather than writing a second integrator -- pull the traced segments
    # back out of the (never-saved) figure and convert lon/lat -> xyz for
    # the globe to rotate+project each frame like everything else. One-off
    # cost (not per-frame, currents are a static layer), negligible next
    # to the physics runtime above.
    u_east_o, v_north_o = geographic_components(g, vec_o)
    u_east_o = np.where(terrain.is_land, 0.0, u_east_o)
    v_north_o = np.where(terrain.is_land, 0.0, v_north_o)
    _fig, _ax = plt.subplots()
    _sp = streamlines(g, u_east_o, v_north_o, _ax, mask=terrain.is_land, density=2.6)
    segments_lonlat = _sp.lines.get_segments()  # list of (2,2) [lon,lat] arrays
    plt.close(_fig)

    stream_segments = []
    for seg in segments_lonlat:
        p0 = lonlat_to_xyz(seg[0, 0], seg[0, 1])
        p1 = lonlat_to_xyz(seg[1, 0], seg[1, 1])
        stream_segments.append(np.round(np.concatenate([p0, p1]), 4).tolist())
    print(f"current streamline segments: {len(stream_segments)}")

    # --- Pack ---
    data = {
        "vertices": np.round(g.vertices, 4).tolist(),
        "faces": g.faces.tolist(),
        "isLand": terrain.is_land.astype(int).tolist(),
        "terrainColor": terrain_hex,
        "windDays": [round(d, 1) for d in wind_days],
        "windVmax": round(wind_vmax, 2),
        "windFrames": wind_q.tolist(),
        "arrowCells": arrow_cells.tolist(),
        "windArrowDir": [f.tolist() for f in wind_arrow_dir],
        "currentVmax": round(current_vmax, 3),
        "currentSpeed": current_q.tolist(),
        "currentStreamlines": stream_segments,
    }
    with open("web_data.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))

    import os
    size_kb = os.path.getsize("web_data.json") / 1024
    print(f"wrote web_data.json ({size_kb:.0f} KB, {len(wind_frames_speed)} wind frames, "
          f"{g.n_cells} cells, {len(g.vertices)} vertices)")
