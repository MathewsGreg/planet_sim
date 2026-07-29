"""
Geodesic (icosahedral) sphere grid.

Start from a regular icosahedron (20 triangular faces, 12 vertices) and
recursively subdivide each triangle into 4, projecting new vertices back
onto the unit sphere each time. This gives a mesh of near-equal-area,
near-equilateral triangles covering the sphere with no pole singularity --
the same base construction used by MPAS/ICON-style climate models (they
then usually take the Voronoi *dual* of this mesh to get hexagonal cells;
we work directly with the triangles for simplicity).

Each triangle is treated as one grid cell. `Grid` exposes, per cell:
  - center unit vector + (lat, lon)
  - spherical area (steradians)
  - indices of its (up to) 3 edge-neighbor cells

Subdivision level -> face count = 20 * 4^level:
  level 0:    20
  level 3:  1,280
  level 4:  5,120
  level 5: 20,480
  level 6: 81,920
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

_PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Base 12 vertices / 20 faces of a unit-radius icosahedron."""
    verts = np.array([
        [-1, _PHI, 0], [1, _PHI, 0], [-1, -_PHI, 0], [1, -_PHI, 0],
        [0, -1, _PHI], [0, 1, _PHI], [0, -1, -_PHI], [0, 1, -_PHI],
        [_PHI, 0, -1], [_PHI, 0, 1], [-_PHI, 0, -1], [-_PHI, 0, 1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return verts, faces


def _subdivide(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split every triangle into 4 by bisecting edges, re-projected to the unit sphere."""
    verts = list(verts)
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(i: int, j: int) -> int:
        key = (i, j) if i < j else (j, i)
        if key in midpoint_cache:
            return midpoint_cache[key]
        v = (verts[i] + verts[j])
        v /= np.linalg.norm(v)
        verts.append(v)
        idx = len(verts) - 1
        midpoint_cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.append([a, ab, ca])
        new_faces.append([b, bc, ab])
        new_faces.append([c, ca, bc])
        new_faces.append([ab, bc, ca])

    return np.array(verts, dtype=np.float64), np.array(new_faces, dtype=np.int64)


def _spherical_triangle_area(v0, v1, v2) -> np.ndarray:
    """
    Area of spherical triangles on the unit sphere via L'Huilier's theorem,
    vectorized over arrays of vertex triples. v0/v1/v2: (N,3) unit vectors.
    """
    a = np.arccos(np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0))
    b = np.arccos(np.clip(np.sum(v0 * v2, axis=1), -1.0, 1.0))
    c = np.arccos(np.clip(np.sum(v0 * v1, axis=1), -1.0, 1.0))
    s = (a + b + c) / 2.0
    tan_e4 = np.sqrt(
        np.clip(
            np.tan(s / 2) * np.tan((s - a) / 2) * np.tan((s - b) / 2) * np.tan((s - c) / 2),
            0.0, None,
        )
    )
    excess = 4.0 * np.arctan(tan_e4)
    return excess  # steradians; total over closed sphere sums to 4*pi


@dataclass
class Grid:
    level: int
    vertices: np.ndarray       # (V, 3) unit vectors
    faces: np.ndarray          # (F, 3) vertex indices per triangular cell
    centers: np.ndarray        # (F, 3) unit vector cell centers (normalized centroid)
    lat: np.ndarray            # (F,) degrees, -90..90
    lon: np.ndarray            # (F,) degrees, -180..180
    area: np.ndarray           # (F,) steradians, sums to 4*pi
    neighbors: np.ndarray      # (F, 3) int, neighbor cell index across each edge (a-b, b-c, c-a)

    @property
    def n_cells(self) -> int:
        return len(self.faces)


@lru_cache(maxsize=8)
def build_grid(level: int = 4) -> Grid:
    """Build (and cache) a geodesic grid at the given subdivision level."""
    verts, faces = _icosahedron()
    for _ in range(level):
        verts, faces = _subdivide(verts, faces)

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    centers = v0 + v1 + v2
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    lat = np.degrees(np.arcsin(np.clip(centers[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))

    area = _spherical_triangle_area(v0, v1, v2)

    neighbors = _build_neighbors(faces)

    return Grid(level, verts, faces, centers, lat, lon, area, neighbors)


def _build_neighbors(faces: np.ndarray) -> np.ndarray:
    """For each face, find the neighbor sharing each of its 3 edges."""
    n_faces = len(faces)
    edge_to_faces: dict[tuple[int, int], list[int]] = {}

    def edge_key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    for f, (a, b, c) in enumerate(faces):
        for i, j in ((a, b), (b, c), (c, a)):
            edge_to_faces.setdefault(edge_key(i, j), []).append(f)

    neighbors = np.full((n_faces, 3), -1, dtype=np.int64)
    for f, (a, b, c) in enumerate(faces):
        for slot, (i, j) in enumerate(((a, b), (b, c), (c, a))):
            pair = edge_to_faces[edge_key(i, j)]
            other = [x for x in pair if x != f]
            if other:
                neighbors[f, slot] = other[0]
    return neighbors


@dataclass
class EdgeTopology:
    """
    Global edge list for a Grid, with a fixed left/right cell convention
    per edge (used to build C-grid divergence/gradient operators that are
    exact discrete adjoints of each other -- see atmosphere.py).
    """
    n_edges: int
    vertex_a: np.ndarray        # (E,) vertex index
    vertex_b: np.ndarray        # (E,) vertex index
    cell_left: np.ndarray       # (E,) face index that "owns" the +normal direction
    cell_right: np.ndarray      # (E,) face index on the -normal side
    cell_edge: np.ndarray       # (F,3) global edge index per local slot (a-b, b-c, c-a)
    cell_edge_sign: np.ndarray  # (F,3) +1 if this cell is cell_left for that edge, else -1


@lru_cache(maxsize=8)
def build_edge_topology(level: int = 4) -> EdgeTopology:
    g = build_grid(level)
    faces = g.faces
    n_faces = len(faces)

    edge_index: dict[tuple[int, int], int] = {}
    va, vb, cl, cr = [], [], [], []
    cell_edge = np.full((n_faces, 3), -1, dtype=np.int64)
    cell_edge_sign = np.zeros((n_faces, 3), dtype=np.float64)

    def key(i: int, j: int) -> tuple[int, int]:
        return (i, j) if i < j else (j, i)

    for f, (a, b, c) in enumerate(faces):
        for slot, (i, j) in enumerate(((a, b), (b, c), (c, a))):
            k = key(i, j)
            if k not in edge_index:
                edge_index[k] = len(va)
                va.append(i)
                vb.append(j)
                cl.append(f)
                cr.append(-1)
                cell_edge[f, slot] = edge_index[k]
                cell_edge_sign[f, slot] = 1.0
            else:
                e = edge_index[k]
                cr[e] = f
                cell_edge[f, slot] = e
                cell_edge_sign[f, slot] = -1.0

    return EdgeTopology(
        n_edges=len(va),
        vertex_a=np.array(va, dtype=np.int64),
        vertex_b=np.array(vb, dtype=np.int64),
        cell_left=np.array(cl, dtype=np.int64),
        cell_right=np.array(cr, dtype=np.int64),
        cell_edge=cell_edge,
        cell_edge_sign=cell_edge_sign,
    )


if __name__ == "__main__":
    for lvl in range(0, 7):
        g = build_grid(lvl)
        et = build_edge_topology(lvl)
        print(f"level {lvl}: {g.n_cells:6d} cells, {et.n_edges:6d} edges "
              f"(expect 3F/2 = {3 * g.n_cells // 2}), "
              f"area sum / 4pi = {g.area.sum() / (4 * np.pi):.6f}, "
              f"neighbors == -1: {(g.neighbors == -1).sum()}, "
              f"cell_right == -1: {(et.cell_right == -1).sum()}")
