"""Bounded pose-preserving placement proposals; never a physical acceptance verdict.

Input surfaces are explicit convex, hole-free patches in each target's local XY
frame. Geometry contains all local visual AND collision vertices. Fixed roots and
orientations never move. Only horizontal world support planes are supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis.physics_math import rotation

VERSION = "genenv.position_solver.v2"
DEFAULTS = dict(margin_m=0.01, clearance_m=0.001, obstacle_gap_m=0.002,
                max_translation_m=0.5, candidate_limit=96, attempt_limit=4096)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array(value, shape=None):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all() or (shape and result.shape != shape):
        raise ValueError("invalid geometry dimensions or nonfinite values")
    return result


def polygon(value):
    p = array(value)
    if p.ndim != 2 or p.shape[1] != 2 or not 3 <= len(p) <= 128:
        raise ValueError("support patch needs 3..128 ordered convex vertices")
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-10):
        raise ValueError("degenerate support polygon")
    normals = np.c_[-edges[:, 1], edges[:, 0]] / lengths[:, None]
    signed = np.einsum('ijk,ik->ij', p[None, :, :] - p[:, None, :], normals)
    if np.max(np.abs(signed)) < 1e-9:
        raise ValueError("degenerate support polygon")
    if np.all(signed <= 1e-9):
        return polygon(p[::-1])
    if not np.all(signed >= -1e-9) or np.max(signed) < 1e-9:
        raise ValueError("support polygon must be convex and nondegenerate")
    return p, normals


def clip_halfplane(points, normal, bound):
    out = []
    for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
        da, db = float(a @ normal - bound), float(b @ normal - bound)
        if da >= -1e-10:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(a + (b-a) * da/(da-db))
    return np.asarray(out).reshape(-1, 2)


def closest(points, target):
    candidates = []
    for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
        edge = b-a
        t = np.clip((target-a) @ edge / max(edge @ edge, 1e-20), 0, 1)
        candidates.append(a+t*edge)
    return min(candidates, key=lambda p: np.linalg.norm(p-target))


def solve(document, **overrides):
    """Return bounded geometric proposal, deltas and rejection trace, bound to input.

    A failed search means no candidate was found within the budget, not a proof
    of infeasibility. A feasible proposal still needs collision and dynamics checks.
    """
    if set(overrides)-set(DEFAULTS):
        raise ValueError("unknown solver setting")
    cfg = DEFAULTS | overrides
    for k, v in cfg.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError("invalid solver setting")
        if k.endswith('_limit'):
            if not isinstance(v, int) or not 1 <= v <= (256 if k == 'candidate_limit' else 10000):
                raise ValueError("invalid search budget")
        elif v < 0 or v > 10:
            raise ValueError("invalid metric setting")
    rows = document['objects']
    if not 1 <= len(rows) <= 12:
        raise ValueError("requires 1..12 objects")
    objects = {o['object_id']: o for o in rows}
    if len(objects) != len(rows) or any(not isinstance(n, str) or not n for n in objects):
        raise ValueError("invalid or duplicate object IDs")
    poses, rotations, vertices, fixed = {}, {}, {}, set()
    for n, obj in objects.items():
        if not isinstance(obj['fixed'], bool):
            raise ValueError("fixed must be boolean")
        poses[n] = array(obj['translation_m'], (3,))
        rotations[n] = rotation(obj['orientation_wxyz'])
        v = array(obj['geometry_vertices_m'])
        if v.ndim != 2 or v.shape[1] != 3 or not 4 <= len(v) <= 1000000:
            raise ValueError("invalid full geometry vertex set")
        if np.any(np.ptp(v, axis=0) <= 1e-10):
            raise ValueError("degenerate object geometry")
        vertices[n] = v @ rotations[n].T
        if obj['fixed']:
            fixed.add(n)
    if not fixed:
        raise ValueError("finite fixed support root required")
    surfaces = {}
    for s in document['surfaces']:
        sid, target = s['surface_id'], s['object_id']
        if sid in surfaces or target not in objects:
            raise ValueError("duplicate patch or unknown target")
        if s.get('holes') or s.get('obstacles'):
            raise ValueError("split holes/obstacles into explicit free convex patches")
        # Geometry above the plane only blocks the patch where it actually covers it. A
        # measured top face records its own asset's perimeter bevel, which lies outside
        # the convex hull and obstructs nothing placeable.
        patch = array(s['polygon_xy_m'])
        if any(builder.polygon_area(builder.clip_polygon(array(t), patch)) > 1e-9
               for t in s.get('above_triangles_xy_m', ())):
            raise ValueError("support patch is covered by geometry above its plane")
        if not s.get('source'):
            raise ValueError("support patch provenance required")
        p, normals = polygon(s['polygon_xy_m'])
        z = float(s['z_m'])
        if not np.isfinite(z):
            raise ValueError("invalid support height")
        if np.linalg.norm(rotations[target][:, 2] - [0, 0, 1]) > 1e-6:
            raise ValueError("tilted support requires a separate orientation solver")
        surfaces[sid] = (target, p, normals, z)
    spatial.relations(document)
    parents, patches = {}, {}
    for rel in document['relations']:
        a, b = rel['source'], rel['target']
        if rel['relation'] != 'on':
            continue
        if a not in objects or b not in objects or a == b or a in parents or a in fixed:
            raise ValueError("invalid or multiple support parent")
        sid = rel['surface_id']
        if sid not in surfaces or surfaces[sid][0] != b:
            raise ValueError("missing or mismatched measured support patch")
        parents[a], patches[a] = b, sid
    if set(parents) != set(objects)-fixed:
        raise ValueError("every dynamic object requires exactly one explicit support")
    order, active, done = [], set(), set(fixed)

    def visit(n):
        if n in active:
            raise ValueError("support cycle")
        if n in done:
            return
        active.add(n)
        visit(parents[n])
        active.remove(n)
        done.add(n)
        order.append(n)

    for n in objects:
        visit(n)
    placed = {n: poses[n].copy() for n in fixed}
    trace = []
    stopped = False

    def reasons(n, pos):
        errors = []
        if np.linalg.norm(pos-poses[n]) > cfg['max_translation_m']+1e-8:
            errors.append('translation_budget')
        target, poly, normals, z = surfaces[patches[n]]
        local = (vertices[n] + pos-placed[target]) @ rotations[target]
        if any(np.min((local[:, :2]-p) @ normal) < cfg['margin_m']-1e-8
               for p, normal in zip(poly, normals, strict=True)):
            errors.append('full_footprint_outside_patch')
        if abs(local[:, 2].min()-z-cfg['clearance_m']) > 1e-8:
            errors.append('support_height')
        world = vertices[n]+pos
        low, high = world.min(0), world.max(0)
        for other, other_pos in placed.items():
            if other == n or other == target or parents.get(other) == n:
                continue
            v = vertices[other]+other_pos
            # Conservative broad phase: may reject valid close arrangements. Never
            # treats AABB overlap as physical support or collision acceptance.
            if np.all(np.minimum(high, v.max(0))-np.maximum(low, v.min(0))
                      > -cfg['obstacle_gap_m']+1e-9):
                errors.append('conservative_overlap:'+other)
        known = dict(placed, **{n: pos})
        for rel in document['relations']:
            a, b = rel['source'], rel['target']
            if rel['relation'] != 'on' and a in known and b in known:
                va, vb = vertices[a]+known[a], vertices[b]+known[b]
                if not spatial.relation_ok(rel['relation'],
                                           np.array([va.min(0), va.max(0)]),
                                           np.array([vb.min(0), vb.max(0)])):
                    errors.append('relation:'+rel['relation']+':'+a+':'+b)
        return errors

    def candidates(n):
        target, poly, normals, z = surfaces[patches[n]]
        rt = rotations[target]
        local = vertices[n] @ rt
        low = poly.min(0)-local[:, :2].max(0)
        high = poly.max(0)-local[:, :2].min(0)
        domain = np.array([[low[0], low[1]], [high[0], low[1]],
                           [high[0], high[1]], [low[0], high[1]]])
        bounds = [float(p @ normal + cfg['margin_m']-np.min(local[:, :2] @ normal))
                  for p, normal in zip(poly, normals, strict=True)]
        for normal, bound in zip(normals, bounds, strict=True):
            domain = clip_halfplane(domain, normal, bound)
            if not len(domain):
                return []
        original = (poses[n]-placed[target]) @ rt
        xy = original[:2]
        inside = all(xy @ normal >= bound-1e-10
                     for normal, bound in zip(normals, bounds, strict=True))
        points = [xy if inside else closest(domain, xy), domain.mean(0), *domain]
        points.extend(np.array([x, y]) for x in np.linspace(*[domain[:, 0].min(),
                                                           domain[:, 0].max()], 7)
                      for y in np.linspace(domain[:, 1].min(), domain[:, 1].max(), 7))
        # Anchors beside already placed objects reduce reliance on a coarse grid.
        for other, pos in placed.items():
            if other == target:
                continue
            other_local = (vertices[other]+pos-placed[target]) @ rt
            for axis in (0, 1):
                for sign in (-1, 1):
                    point = xy.copy()
                    point[axis] = ((other_local[:, axis].min()-local[:, axis].max()
                                    - cfg['obstacle_gap_m']) if sign < 0 else
                                   (other_local[:, axis].max()-local[:, axis].min()
                                    + cfg['obstacle_gap_m']))
                    points.append(point)
        unique = {tuple(np.round(p, 12)): p for p in points}
        choices = [np.r_[p, z+cfg['clearance_m']-local[:, 2].min()] @ rt.T+placed[target]
                   for p in unique.values()]
        return sorted(choices, key=lambda p: float(np.linalg.norm(p-poses[n])))[:
            cfg['candidate_limit']]

    def search(i):
        nonlocal stopped
        if i == len(order):
            return True
        n = order[i]
        choices = candidates(n)
        if not choices and len(trace) < cfg['attempt_limit']:
            trace.append(dict(object_id=n, reasons=['empty_feasible_patch']))
        for pos in choices:
            if len(trace) >= cfg['attempt_limit']:
                stopped = True
                return False
            errors = reasons(n, pos)
            trace.append(dict(object_id=n, translation_m=pos.tolist(), reasons=errors))
            if errors:
                continue
            placed[n] = pos
            if search(i+1):
                return True
            del placed[n]
            if stopped:
                return False
        return False

    ok = search(0)
    if ok:
        checks = {n: reasons(n, placed[n]) for n in order}
        if any(checks.values()):
            raise RuntimeError('final placement consistency check failed')
    result = dict(schema_version=VERSION, input_sha256=digest(document), settings=cfg,
                  status='proposal_ready' if ok else 'search_exhausted',
                  budget_exhausted=stopped, physics_status='not_run', attempts=trace,
                  meaning=('geometric proposal only; contact, collision fidelity '
                           'and stability unverified'),
                  poses={n: dict(translation_m=placed[n].tolist(),
                                 orientation_wxyz=objects[n]['orientation_wxyz'],
                                 delta_m=(placed[n]-poses[n]).tolist(), fixed=n in fixed)
                         for n in objects} if ok else {})
    return result


def from_scene(directory):
    """Read a verified imported package; do not mutate or reseal the source."""
    from self_improving.sim_adapters.genesis import import_simfoundry_scene as imported
    from self_improving.sim_adapters.genesis import standard_urdf as standard

    root = Path(directory).resolve()
    layout = imported.verify(root)
    objects = []
    for obj in layout['objects']:
        _, entry, _ = standard.verify_package(root/obj['standard_package'])
        measured = standard.inspect(entry)
        objects.append({k: obj[k] for k in ('object_id', 'fixed', 'translation_m',
                                           'orientation_wxyz')} | dict(
            geometry_vertices_m=np.concatenate(
                [measured['visual'], measured['collision']]).tolist()))
    surfaces = [dict(s, surface_id=n, object_id=s.get("object_id", n))
                for n, s in layout['support_surfaces'].items()]
    relations = [dict(r, surface_id=r.get('surface_id', r['target']))
                 if r['relation'] == 'on' else dict(r) for r in layout['relations']]
    return dict(objects=objects, surfaces=surfaces, relations=relations,
                source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path)
    inputs.add_argument('--scene-package', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.scene_package:
        root, out = args.scene_package.resolve(), args.output.resolve()
        if out.is_relative_to(root) or root.is_relative_to(out):
            raise ValueError("source scene and output must be separate")
    document = from_scene(args.scene_package) if args.scene_package else json.loads(
        args.input.read_text())
    result = solve(document, **document.get('settings', {}))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'input.json').write_text(json.dumps(document, ensure_ascii=False, indent=2)+'\n')
    (args.output/'proposal.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], physics_status=result['physics_status'],
                          output=str(args.output)), ensure_ascii=False))
    return 0 if result['status'] == 'proposal_ready' else 2


if __name__ == '__main__':
    raise SystemExit(main())
