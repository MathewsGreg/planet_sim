"""Render a Grid + Terrain as a 3D globe with pyvista."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from grid import Grid, build_grid
from terrain import BIOMES, Terrain, classify, random_elevation


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def biome_colors(terrain: Terrain, elevation_shading: bool = True) -> np.ndarray:
    """Per-cell RGB uint8 array, biome base color lightly shaded by elevation."""
    palette = {code: _hex_to_rgb(hexcode) for code, (_, hexcode) in BIOMES.items()}
    n = len(terrain.biome)
    rgb = np.zeros((n, 3), dtype=np.float64)
    for code, color in palette.items():
        rgb[terrain.biome == code] = color

    if elevation_shading:
        e = terrain.elevation.copy()
        e = (e - e.min()) / (e.max() - e.min())  # 0..1
        shade = 0.75 + 0.5 * e  # darker low, brighter high
        rgb *= shade[:, None]
        rgb = np.clip(rgb, 0, 255)

    return rgb.astype(np.uint8)


def build_mesh(grid: Grid) -> pv.PolyData:
    faces = np.hstack([np.full((grid.n_cells, 1), 3), grid.faces]).astype(np.int64)
    return pv.PolyData(grid.vertices, faces)


def render_globe(
    grid: Grid,
    terrain: Terrain,
    screenshot: str | None = None,
    interactive: bool = False,
    window_size=(1000, 1000),
):
    mesh = build_mesh(grid)
    mesh.cell_data["rgb"] = biome_colors(terrain)

    plotter = pv.Plotter(off_screen=not interactive, window_size=window_size)
    plotter.add_mesh(mesh, scalars="rgb", rgb=True, show_edges=False, lighting=False)
    plotter.set_background("black")
    plotter.camera_position = [(0, -3.4, 1.4), (0, 0, 0), (0, 0, 1)]
    plotter.reset_camera()

    if screenshot:
        plotter.show(screenshot=screenshot, auto_close=not interactive)
    elif interactive:
        plotter.show()
    else:
        plotter.screenshot("render.png")

    return plotter


if __name__ == "__main__":
    g = build_grid(5)
    elev = random_elevation(g, seed=1)
    t = classify(g, elev)
    render_globe(g, t, screenshot="render.png", interactive=False)
    print("wrote render.png")
