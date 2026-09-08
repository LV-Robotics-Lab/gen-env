"""A render-only stand-in for a support surface physics simulates as an analytic plane.

When SimFoundry classifies the surface a scene rests on as ground rather than an object, it
becomes `gs.morphs.Plane`: infinite, analytic, no geometry. Physics is right to use it -- the
bodies really do rest on that plane -- but a render of it shows Genesis' default checkerboard
where the photograph shows a desk.

This builds a slab from the observed support points purely so the render matches the scene it
came from. It carries `collision=False`, so nothing it does can change a physics result, and
its top face is pinned to the plane physics actually uses. That pinning is the point: a
visual surface at any other height would draw the body floating above or sunk into a desk it
is not touching, which is precisely the kind of picture-that-outruns-the-evidence this
pipeline exists to prevent. `verify_top_face` re-checks it from the written mesh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "genenv.visual_support_surface.v1"
# The slab is drawn downwards from the plane, so its thickness never reaches the bodies.
THICKNESS_M = 0.02
# The written top face must sit on the plane this tightly. Well under the 1 mm penetration
# budget, so a viewer cannot mistake a modelling offset for real contact behaviour.
TOP_FACE_TOLERANCE_M = 1e-5


def footprint(observation):
    """Axis-aligned visible extent of the observed surface, in world XY."""
    corners = np.asarray(observation["visible_footprint_world_xy_m"], dtype=float)
    if corners.shape != (2, 2) or not np.isfinite(corners).all():
        raise ValueError("support observation has no usable world footprint")
    low, high = corners.min(0), corners.max(0)
    if not (high - low > 1e-3).all():
        raise ValueError("observed support footprint is degenerate")
    return low, high


def slab(low, high, z_m, thickness=THICKNESS_M):
    """Box mesh whose top face lies exactly on z_m, extending downwards."""
    xs, ys = (low[0], high[0]), (low[1], high[1])
    zs = (z_m - thickness, z_m)
    vertices = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    # Vertex order above is (z, y, x) major, so 0-3 are the bottom face and 4-7 the top.
    faces = np.array([
        [0, 2, 3], [0, 3, 1],   # bottom
        [4, 5, 7], [4, 7, 6],   # top
        [0, 1, 5], [0, 5, 4],   # -y
        [2, 6, 7], [2, 7, 3],   # +y
        [0, 4, 6], [0, 6, 2],   # -x
        [1, 3, 7], [1, 7, 5],   # +x
    ], dtype=int)
    return vertices, faces


def write_obj(path, vertices, faces):
    lines = [f"# {SCHEMA}: render-only support surface, never simulated"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    Path(path).write_text("\n".join(lines) + "\n")


def build(observation, z_m, out_path):
    """Write the slab and return the record describing it, or raise if unusable."""
    low, high = footprint(observation)
    vertices, faces = slab(low, high, float(z_m))
    write_obj(out_path, vertices, faces)
    return dict(
        schema_version=SCHEMA,
        mesh=Path(out_path).name,
        z_m=float(z_m),
        thickness_m=THICKNESS_M,
        footprint_xy_m=[low.tolist(), high.tolist()],
        area_m2=float((high[0] - low[0]) * (high[1] - low[1])),
        source="simfoundry support observation, visible extent only",
        # Repeated from the observation so a reader of the scene package alone still sees
        # that the drawn edges are where the image stopped, not where the desk stops.
        boundary_censored=observation.get("boundary_censored"),
        collision=False,
        meaning="rendering only; physics uses the analytic environment plane",
    )


def verify_top_face(root, record, z_m):
    """Re-measure the written mesh: its highest vertex must be the plane physics uses."""
    path = Path(root) / record["mesh"]
    if not path.is_file():
        raise ValueError("visual support mesh is missing")
    if record.get("collision") is not False:
        raise ValueError("visual support surface must declare collision=False")
    heights = [
        float(line.split()[3])
        for line in path.read_text().splitlines()
        if line.startswith("v ")
    ]
    if not heights:
        raise ValueError("visual support mesh has no vertices")
    top = max(heights)
    if abs(top - float(z_m)) > TOP_FACE_TOLERANCE_M:
        raise ValueError(
            f"visual support top face {top:.6f} m is not the environment plane {z_m:.6f} m; "
            "a render would show bodies floating above or sunk into it"
        )
    return dict(measured_top_z_m=top, vertex_count=len(heights))


def load(root):
    """The record for a scene package that has one, or None."""
    path = Path(root) / "visual_support.json"
    return json.loads(path.read_text()) if path.is_file() else None
