"""
Procedural terrain for a Grid: elevation, land/ocean mask, and a
latitude+elevation-driven biome classification (placeholder for real
climate output later).

Elevation is built from a random sum of real-valued spherical harmonics
with a "red" power spectrum (amplitude ~ 1/l^p): low-degree harmonics
dominate, which is what produces smooth continent-scale blobs instead of
salt-and-pepper noise. This is the same basis functions spectral GCMs use
to represent atmospheric fields, so it's worth getting familiar with them
now.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import sph_harm_y

from grid import Grid


def random_elevation(
    grid: Grid,
    seed: int,
    l_max: int = 10,
    spectral_slope: float = 1.8,
) -> np.ndarray:
    """
    Real-valued elevation field on grid cell centers, built from random
    spherical harmonic coefficients with amplitude falling off as 1/l^slope.
    Returned field has zero mean and unit standard deviation (unitless;
    terrain.classify() below turns it into physical-ish elevation).
    """
    rng = np.random.default_rng(seed)

    # colatitude theta in [0, pi], longitude phi in [0, 2pi) for sph_harm_y
    theta = np.radians(90.0 - grid.lat)
    phi = np.radians(grid.lon % 360.0)

    field = np.zeros(grid.n_cells, dtype=np.float64)
    for l in range(1, l_max + 1):
        sigma = 1.0 / (l ** spectral_slope)
        # m = 0 term is already real
        c0 = rng.normal(0.0, sigma)
        field += c0 * np.real(sph_harm_y(l, 0, theta, phi))
        # m > 0: complex coefficient, add 2*Re(c * Y_l^m) which is the
        # standard trick to keep the overall field real (equivalent to
        # pairing each +m term with its -m conjugate partner).
        for m in range(1, l + 1):
            c = rng.normal(0.0, sigma) + 1j * rng.normal(0.0, sigma)
            field += 2.0 * np.real(c * sph_harm_y(l, m, theta, phi))

    field -= field.mean()
    field /= field.std()
    return field


@dataclass
class Terrain:
    elevation: np.ndarray      # unitless, roughly [-3, 3], 0 = sea level datum before shifting
    is_land: np.ndarray        # bool
    biome: np.ndarray          # int codes, see BIOMES
    sea_level: float           # elevation threshold used to split land/ocean


BIOMES = {
    0: ("ocean", "#1c5f8a"),
    1: ("ocean_deep", "#0b2f4a"),
    2: ("desert", "#d9c064"),
    3: ("forest", "#2f6b3a"),
    4: ("tundra", "#8a9a8a"),
    5: ("ice", "#f2f6f8"),
}


def classify(
    grid: Grid,
    elevation: np.ndarray,
    land_fraction: float = 0.30,
    seed: int = 0,
) -> Terrain:
    """
    Turn a raw elevation field into land/ocean + biome, targeting a given
    area-weighted land fraction (Earth is ~0.29-0.30).
    """
    # Area-weighted percentile: pick the elevation threshold such that the
    # cells above it cover `land_fraction` of total sphere area.
    order = np.argsort(elevation)[::-1]  # highest elevation first
    cum_area = np.cumsum(grid.area[order])
    target_area = land_fraction * grid.area.sum()
    cutoff_pos = np.searchsorted(cum_area, target_area)
    cutoff_pos = min(cutoff_pos, len(order) - 1)
    sea_level = elevation[order[cutoff_pos]]

    is_land = elevation > sea_level

    biome = np.zeros(grid.n_cells, dtype=np.int64)
    abs_lat = np.abs(grid.lat)

    # Ocean: shallow near coasts/shelves, deep in basins.
    ocean_depth = sea_level - elevation  # >0 for ocean cells
    biome[~is_land] = np.where(ocean_depth[~is_land] > 0.6, 1, 0)

    # Land biomes driven by latitude (proxy for the real climate we'll
    # eventually compute) plus elevation-dependent snow line: snow line
    # sits high in the tropics and drops to sea level near the poles.
    land_elev_above_sea = elevation - sea_level
    snow_line = 2.2 * np.cos(np.radians(np.clip(abs_lat, 0, 90))) ** 1.3

    ice = is_land & ((abs_lat > 66.5) | (land_elev_above_sea > snow_line))
    desert = is_land & ~ice & (abs_lat < 35.0) & (abs_lat >= 12.0)
    tundra = is_land & ~ice & (abs_lat > 55.0)
    forest = is_land & ~ice & ~desert & ~tundra  # equatorial belt + temperate belt

    biome[ice] = 5
    biome[desert] = 2
    biome[tundra] = 4
    biome[forest] = 3

    return Terrain(elevation=elevation, is_land=is_land, biome=biome, sea_level=sea_level)


if __name__ == "__main__":
    from grid import build_grid

    g = build_grid(5)
    elev = random_elevation(g, seed=1)
    t = classify(g, elev)

    print(f"cells: {g.n_cells}")
    print(f"land fraction (area-weighted): {g.area[t.is_land].sum() / g.area.sum():.3f}")
    for code, (name, _) in BIOMES.items():
        mask = t.biome == code
        frac = g.area[mask].sum() / g.area.sum()
        print(f"  {name:12s} {frac:.3f}")
