"""
2D equirectangular (Plate Carree) map rendering: project the grid's
triangles to (lon, lat) and draw them directly as a matplotlib
PolyCollection. Works for both flat RGB coloring (terrain/biomes) and
colormapped scalar fields (wind speed, height anomaly, ...).

Triangles that straddle the antimeridian (lon jumps from +180 to -180)
would otherwise get drawn as a distorted band stretching across the whole
map, so they're masked out -- leaves a thin, harmless gap at the seam.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from grid import Grid


def triangle_lonlat_coords(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(np.arctan2(grid.vertices[:, 1], grid.vertices[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(grid.vertices[:, 2], -1.0, 1.0)))
    tri_lon = lon[grid.faces]  # (F,3)
    tri_lat = lat[grid.faces]  # (F,3)
    wrap_mask = (tri_lon.max(axis=1) - tri_lon.min(axis=1)) > 180.0
    coords = np.stack([tri_lon, tri_lat], axis=-1)  # (F,3,2)
    return coords, wrap_mask


def _setup_axes(ax, title: str):
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_xticks(range(-180, 181, 30))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(alpha=0.15, linewidth=0.5)


def plot_rgb(grid: Grid, rgb: np.ndarray, ax=None, title: str = "", figsize=(14, 7)):
    """rgb: (n_cells,3) uint8, one flat color per cell (e.g. from visualize.biome_colors)."""
    coords, wrap_mask = triangle_lonlat_coords(grid)
    keep = ~wrap_mask
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    colors = rgb[keep].astype(float) / 255.0
    ax.add_collection(PolyCollection(coords[keep], facecolors=colors, edgecolors="none"))
    _setup_axes(ax, title)
    return ax


def plot_scalar(
    grid: Grid,
    values: np.ndarray,
    ax=None,
    title: str = "",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    figsize=(14, 7),
):
    """values: (n_cells,), one scalar per cell, colormapped."""
    coords, wrap_mask = triangle_lonlat_coords(grid)
    keep = ~wrap_mask
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(norm(values[keep]))
    ax.add_collection(PolyCollection(coords[keep], facecolors=colors, edgecolors="none"))
    _setup_axes(ax, title)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=cbar_label, fraction=0.025, pad=0.02)
    return ax


def quiver_wind(grid: Grid, u_east: np.ndarray, v_north: np.ndarray, ax, stride: int = 6, **kwargs):
    """Overlay a subsampled wind-vector field. u_east/v_north: (n_cells,) m/s, geographic components."""
    lon, lat = grid.lon, grid.lat
    idx = np.arange(grid.n_cells)[::stride]
    ax.quiver(
        lon[idx], lat[idx], u_east[idx], v_north[idx],
        color="white", scale_units="inches", scale=kwargs.pop("scale", 120),
        width=0.0018, alpha=0.85, **kwargs,
    )


def plot_coastline(grid: Grid, is_land: np.ndarray, ax, color: str = "black", size: float = 6.0):
    """Scatter-outline the land/ocean boundary: land cells with >=1 ocean neighbor."""
    neighbor_is_ocean = ~is_land[grid.neighbors]  # (F,3)
    has_ocean_neighbor = neighbor_is_ocean.any(axis=1)
    coastal = is_land & has_ocean_neighbor
    ax.scatter(grid.lon[coastal], grid.lat[coastal], s=size, c=color, marker=".",
               linewidths=0.5, edgecolors="white", zorder=5)


def geographic_components(grid: Grid, vec3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project cell-centered 3D tangent vectors onto local (east, north) unit vectors."""
    lon = np.radians(grid.lon)
    lat = np.radians(grid.lat)
    east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], axis=1)
    north = np.stack([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ], axis=1)
    u_east = np.einsum("ij,ij->i", vec3, east)
    v_north = np.einsum("ij,ij->i", vec3, north)
    return u_east, v_north


if __name__ == "__main__":
    from grid import build_grid
    from terrain import random_elevation, classify
    from visualize import biome_colors

    g = build_grid(5)
    elev = random_elevation(g, seed=1)
    t = classify(g, elev)
    rgb = biome_colors(t)

    ax = plot_rgb(g, rgb, title="Procedural terrain (seed=1, level=5)")
    plt.tight_layout()
    plt.savefig("map_terrain.png", dpi=130)
    print("wrote map_terrain.png")
