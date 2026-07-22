from __future__ import annotations

from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Polygon

OUT = Path(__file__).resolve().parent / "files"
OUT.mkdir(parents=True, exist_ok=True)

RAIL_CENTER_X = 24.0
HOST_RAIL_BASE_W = 8.0
HOST_RAIL_HEAD_W = 12.0
HOST_RAIL_HEIGHT = 7.0
BODY_W = 72.0
BODY_L = 34.0
BODY_T = 9.8
BODY_CHAMFER = 4.0
ENTRY_FLARE_L = 2.5
ENTRY_SIDE_EXTRA = 0.20
ENTRY_DEPTH_EXTRA = 0.15

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


def difference(base: trimesh.Trimesh, cutters: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    result = trimesh.boolean.difference([base] + cutters, engine="manifold")
    if result is None:
        raise RuntimeError("Boolean difference failed")
    return clean(result)


def extrude(poly: Polygon, z0: float, z1: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.extrude_polygon(poly, height=z1-z0, engine="earcut")
    mesh.apply_translation([0.0, 0.0, z0])
    return clean(mesh)


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=[x1-x0, y1-y0, z1-z0])
    mesh.apply_translation([(x0+x1)/2.0, (y0+y1)/2.0, (z0+z1)/2.0])
    return mesh


def chamfered_rect(width: float, length: float, chamfer: float) -> Polygon:
    x0, x1 = -width/2.0, width/2.0
    y0, y1 = 0.0, length
    return Polygon([
        (x0+chamfer, y0), (x1-chamfer, y0),
        (x1, y0+chamfer), (x1, y1-chamfer),
        (x1-chamfer, y1), (x0+chamfer, y1),
        (x0, y1-chamfer), (x0, y0+chamfer),
    ])


def dovetail_void_segment(x_center: float, y0: float, y1: float,
                           side_clearance: float, depth_clearance: float) -> trimesh.Trimesh:
    z_open = BODY_T + 0.5
    z_deep = BODY_T - (HOST_RAIL_HEIGHT + depth_clearance)
    half_open = HOST_RAIL_BASE_W/2.0 + side_clearance
    half_deep = HOST_RAIL_HEAD_W/2.0 + side_clearance
    vertices = np.array([
        [x_center-half_deep, y0, z_deep],
        [x_center+half_deep, y0, z_deep],
        [x_center+half_open, y0, z_open],
        [x_center-half_open, y0, z_open],
        [x_center-half_deep, y1, z_deep],
        [x_center+half_deep, y1, z_deep],
        [x_center+half_open, y1, z_open],
        [x_center-half_open, y1, z_open],
    ], dtype=float)
    faces = np.array([
        [0,2,1],[0,3,2],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[1,2,6],[1,6,5],
        [2,3,7],[2,7,6],[3,0,4],[3,4,7],
    ], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    return clean(mesh)


def channel_with_leadins(x_center: float, side: float, depth: float) -> list[trimesh.Trimesh]:
    return [
        dovetail_void_segment(x_center, -0.6, ENTRY_FLARE_L,
                              side+ENTRY_SIDE_EXTRA, depth+ENTRY_DEPTH_EXTRA),
        dovetail_void_segment(x_center, ENTRY_FLARE_L-0.05,
                              BODY_L-ENTRY_FLARE_L+0.05, side, depth),
        dovetail_void_segment(x_center, BODY_L-ENTRY_FLARE_L, BODY_L+0.6,
                              side+ENTRY_SIDE_EXTRA, depth+ENTRY_DEPTH_EXTRA),
    ]


def make_coupon(letter: str, side: float, depth: float) -> trimesh.Trimesh:
    body = extrude(chamfered_rect(BODY_W, BODY_L, BODY_CHAMFER), 0.0, BODY_T)
    cutters: list[trimesh.Trimesh] = []
    for x in (-RAIL_CENTER_X, RAIL_CENTER_X):
        cutters.extend(channel_with_leadins(x, side, depth))

    # A–E are identified by one through five simple rectangular edge notches.
    # This avoids the heavy triangulation caused by circular holes while keeping
    # identification clear and leaving the rail test geometry untouched.
    count = ord(letter)-ord("A")+1
    pitch = 4.0
    start = -(count-1)*pitch/2.0
    for index in range(count):
        x = start + index*pitch
        cutters.append(box(x-0.9, x+0.9, BODY_L-3.0, BODY_L+0.5, -0.2, BODY_T+0.2))
    return difference(body, cutters)


def validate(mesh: trimesh.Trimesh, name: str) -> None:
    edges = np.sort(mesh.edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    checks = {
        "watertight": mesh.is_watertight,
        "winding": mesh.is_winding_consistent,
        "single_component": len(mesh.split(only_watertight=False)) == 1,
        "manifold_edges": int(np.count_nonzero(counts != 2)) == 0,
        "footprint": mesh.extents[0] <= 72.001 and mesh.extents[1] <= 34.001,
    }
    if not all(checks.values()):
        raise RuntimeError(f"{name} validation failed: {checks}")


for letter, side, depth in VARIANTS:
    mesh = make_coupon(letter, side, depth)
    stem = f"UH-RCV-WAL-001-R5-CAL-{letter}-SIDE-{side:.2f}-DEPTH-{depth:.2f}"
    validate(mesh, stem)
    text = trimesh.exchange.stl.export_stl_ascii(mesh)
    if isinstance(text, bytes):
        text = text.decode("ascii")
    path = OUT / f"{stem}.stl"
    path.write_text(text, encoding="ascii")
    (OUT / f"{stem}.csv").write_text(text, encoding="ascii")
    reloaded = trimesh.load_mesh(path, process=True)
    validate(reloaded, stem + " reloaded")
    print(path.name, path.stat().st_size, reloaded.extents.tolist())
