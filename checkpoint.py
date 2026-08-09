"""
Save/load a full simulation snapshot (terrain + atmosphere + ocean state)
to a single .npz file, so a multi-minute spin-up doesn't have to be re-run
from scratch every time we want to build on top of it -- add new physics,
re-plot, or keep spinning up further. Self-contained: also stores the
terrain fields directly (not just a seed), so a later change to
terrain.py's generation code can never silently invalidate an old
checkpoint.

Doesn't serialize AtmosParams/OceanParams -- if you've changed those
between saving and loading, the reloaded model objects use *current*
defaults, not the ones the checkpoint was actually produced under. Fine
for now (single-user, single-machine), but worth remembering if params
ever start drifting between sessions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grid import Grid, build_grid
from terrain import Terrain
from atmosphere import ShallowWaterAtmosphere, AtmosState
from ocean import WindDrivenOcean, OceanState


@dataclass
class Checkpoint:
    grid: Grid
    terrain: Terrain
    atmos: ShallowWaterAtmosphere
    atmos_state: AtmosState
    ocean: WindDrivenOcean
    ocean_state: OceanState
    wind_forcing: np.ndarray  # the (smoothed) mean forcing ocean was spun up under


def save_checkpoint(path: str, cp: Checkpoint) -> None:
    np.savez(
        path,
        grid_level=cp.grid.level,
        elevation=cp.terrain.elevation,
        is_land=cp.terrain.is_land,
        biome=cp.terrain.biome,
        sea_level=cp.terrain.sea_level,
        atmos_h=cp.atmos_state.h,
        atmos_u=cp.atmos_state.u,
        atmos_q=cp.atmos_state.q,
        atmos_t=cp.atmos.t,
        ocean_h=cp.ocean_state.h,
        ocean_u=cp.ocean_state.u,
        ocean_T=cp.ocean_state.T,
        ocean_t=cp.ocean.t,
        wind_forcing=cp.wind_forcing,
    )


def load_checkpoint(path: str) -> Checkpoint:
    d = np.load(path)
    grid = build_grid(int(d["grid_level"]))
    terrain = Terrain(
        elevation=d["elevation"], is_land=d["is_land"], biome=d["biome"],
        sea_level=float(d["sea_level"]),
    )

    atmos = ShallowWaterAtmosphere(grid, terrain=terrain)
    atmos.t = float(d["atmos_t"])
    atmos_state = AtmosState(h=d["atmos_h"], u=d["atmos_u"], q=d["atmos_q"])

    ocean = WindDrivenOcean(grid, terrain=terrain, geo=atmos.geo)
    ocean.t = float(d["ocean_t"])
    ocean.set_wind_forcing(d["wind_forcing"])
    # older checkpoints (pre-SST) don't have ocean_T -- fall back to a
    # uniform T_mean field, same as a fresh initial_state(), rather than
    # failing to load an otherwise-good multi-hour spin-up
    if "ocean_T" in d:
        ocean_T = d["ocean_T"]
    else:
        ocean_T = np.full(grid.n_cells, ocean.p.T_mean)
    ocean_state = OceanState(h=d["ocean_h"], u=d["ocean_u"], T=ocean_T)

    return Checkpoint(
        grid=grid, terrain=terrain, atmos=atmos, atmos_state=atmos_state,
        ocean=ocean, ocean_state=ocean_state, wind_forcing=d["wind_forcing"],
    )


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoint.npz"
    cp = load_checkpoint(path)
    speed = np.abs(cp.ocean_state.u)
    print(f"loaded {path}: grid level {cp.grid.level} ({cp.grid.n_cells} cells)")
    print(f"  atmosphere t={cp.atmos.t / 86400:.1f}d")
    print(f"  ocean t={cp.ocean.t / 86400 / 365:.2f}y  "
          f"h[min,max]=[{cp.ocean_state.h.min():.1f},{cp.ocean_state.h.max():.1f}]  "
          f"|u| mean/max={speed.mean():.4f}/{speed.max():.4f} m/s  "
          f"T[min,max]=[{cp.ocean_state.T.min():.1f},{cp.ocean_state.T.max():.1f}]degC")
