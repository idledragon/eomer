from __future__ import annotations

from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

OUT = Path(__file__).resolve().parent / "files"
OUT.mkdir(parents=True, exist_ok=True)

RAIL_CENTER_X = 24.0
HOST_RAIL_BASE_W = 8.0
HOST_RAIL_HEAD_W = 12.0
HOST_RAIL_HEIGHT = 7.0
BODY_W = 72.0
BODY_L = 34.0
BODY_T = 9.8

VARIANTS = [
    ("A", 0.30, 0.35),
    ("B", 0.40, 0.45),
    ("C", 0.50, 0.45),
    ("D", 0.40, 0.55),
    ("E", 0.50, 0.55),
]


def clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.fix_normals()
    return mesh


def make_coupon(side: float, depth: float) -> trimesh.Trimesh:
    section = Polygon([
        (-BODY_W/2, 0.0), (BODY_W/2, 0.0),
        (BODY_W/2, BODY_T), (-BODY_W/2, BODY_T),
    ])
    voids = []
    z_deep = BODY_T - (HOST_RAIL_HEIGHT + depth)
    half_open = HOST_RAIL_BASE_W/2.0 + side
    half_deep = HOST_RAIL_HEAD_W/2.0 + side
    for x in (-RAIL_CENTER_X, RAIL_CENTER_X):
        voids.append(Polygon([
            (x-half_deep, z_deep),
            (x+half_deep, z_deep),
            (x+half_open, BODY_T+1.0),
            (x-half_open, BODY_T+1.0),
        ]))
    section = section.difference(unary_union(voids))
    if not isinstance(section, Polygon) or section.is_empty:
        raise RuntimeError("Invalid coupon cross-section")

    local = trimesh.creation.extrude_polygon(section, height=BODY_L, engine="earcut")
    vertices = local.vertices.copy()
    local.vertices = np.column_stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]])
    local.fix_normals()
    return clean(local)


def validate(mesh: trimesh.Trimesh, name: str) -> None:
    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    checks = {
        "watertight": mesh.is_watertight,
        "winding": mesh.is_winding_consistent,
        "single_component": len(mesh.split(only_watertight=False)) == 1,
        "manifold_edges": int(np.count_nonzero(counts != 2)) == 0,
        "footprint": mesh.extents[0] <= 72.001 and mesh.extents[1] <= 34.001,
        "thickness": mesh.extents[2] <= 9.801,
    }
    if not all(checks.values()):
        raise RuntimeError(f"{name} validation failed: {checks}")


def number(value: float) -> str:
    value = float(value)
    if abs(value) < 0.0000005:
        value = 0.0
    return format(value, ".9g")


def compact_ascii(mesh: trimesh.Trimesh, solid_name: str) -> str:
    lines = [f"solid {solid_name}"]
    for normal, triangle in zip(mesh.face_normals, mesh.triangles):
        n = " ".join(number(v) for v in normal)
        vertices = " ".join(
            "vertex " + " ".join(number(v) for v in vertex)
            for vertex in triangle
        )
        lines.append(f"facet normal {n} outer loop {vertices} endloop endfacet")
    lines.append(f"endsolid {solid_name}")
    return "\n".join(lines) + "\n"


for letter, side, depth in VARIANTS:
    mesh = make_coupon(side, depth)
    stem = f"UH-RCV-WAL-001-R5-CAL-{letter}-SIDE-{side:.2f}-DEPTH-{depth:.2f}"
    validate(mesh, stem)
    text = compact_ascii(mesh, stem.replace("-", "_"))
    path = OUT / f"{stem}.stl"
    path.write_text(text, encoding="ascii")
    (OUT / f"{stem}.csv").write_text(text, encoding="ascii")
    reloaded = trimesh.load_mesh(path, process=True)
    validate(reloaded, stem + " reloaded")
    print(path.name, path.stat().st_size, len(reloaded.faces), reloaded.extents.tolist())
