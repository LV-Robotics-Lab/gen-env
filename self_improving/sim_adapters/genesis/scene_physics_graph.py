"""Finite-support graph constraints shared by preparation and free replay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import position_solver as positions
from self_improving.sim_adapters.genesis import scene_layout as spatial
from self_improving.sim_adapters.genesis import standard_urdf as standard
from self_improving.sim_adapters.genesis.physics_math import angle, rotation

VERSION = 'genenv.scene_physics_graph.v1'


def topology(layout):
    objects = layout['objects']
    if any(type(o['fixed']) is not bool for o in objects):
        raise ValueError('fixed must be boolean')
    parents = spatial.relations(dict(objects=objects, relations=layout['relations']))
    fixed = {o['object_id'] for o in objects if o['fixed']}
    dynamic = {o['object_id'] for o in objects if not o['fixed']}
    # The environment plane is a fixed root when the scene has one. A body resting on the
    # floor is an ordinary scene, and refusing it forced every scene to contain a table;
    # what the plane cannot do is stand in for an undeclared support, so a body still needs
    # an explicit relation naming it.
    grounded = layout['environment']['ground'] is not None
    if grounded:
        fixed = fixed | {spatial.GROUND}
    elif any(r['target'] == spatial.GROUND for r in layout['relations']):
        raise ValueError('support declared on a ground plane this scene does not have')
    if not fixed or not dynamic or set(parents) != dynamic:
        raise ValueError('every dynamic body needs one support chain reaching a fixed root')
    for r in layout['relations']:
        if r['relation'] != 'on':
            continue
        if r['target'] == spatial.GROUND:
            # Analytic and unbounded: there is no polygon to measure and no edge to fall
            # off. What remains checkable -- that the body stays above the plane -- is
            # checked per step in geometric_checks().
            continue
        sid = r.get('surface_id', r['target'])
        surface = layout['support_surfaces'].get(sid)
        if surface is None or surface.get('object_id', r['target']) != r['target']:
            raise ValueError('missing or mismatched support surface')
        positions.polygon(surface['polygon_xy_m'])
        if not surface.get('source') or not np.isfinite(surface['z_m']):
            raise ValueError('support surface requires finite height and provenance')
        if any(surface.get(k) for k in ('holes', 'obstacles')):
            raise ValueError('support requires an explicit free convex patch')
        # Geometry recorded above the plane disqualifies the patch only where it actually
        # covers it. A measured top face carries the perimeter bevel of its own asset;
        # rejecting on the field's presence would refuse every honestly measured surface.
        patch = np.asarray(surface['polygon_xy_m'])
        if any(builder.polygon_area(builder.clip_polygon(np.asarray(t), patch)) > 1e-9
               for t in surface.get('above_triangles_xy_m', ())):
            raise ValueError('support patch is covered by geometry above its plane')
    return parents, fixed


def inside(points, polygon, tolerance=1e-6):
    """Signed distance of the worst point into a convex polygon; negative means outside."""
    corners, normals = positions.polygon(polygon)
    return min(float(np.min((np.asarray(points)-p) @ normal))
               for p, normal in zip(corners, normals, strict=True)) >= -tolerance


def verify_support(root, layout):
    """Prove every declared support patch is backed by the asset's real flat top.

    topology() asserts the patch is a free convex surface but never re-derives it from
    geometry. A patch spanning the asset's whole bounding rectangle therefore passed
    unchallenged, and bodies placed over the parts that are open air were released above
    nothing -- which reads downstream as an unexplained tumble, not as a bad surface.
    """
    root = Path(root)
    measured = {}
    for relation in layout['relations']:
        if relation['relation'] != 'on':
            continue
        target = relation['target']
        if target == spatial.GROUND:
            continue
        surface_id = relation.get('surface_id', target)
        if surface_id in measured:
            continue
        declared = layout['support_surfaces'][surface_id]
        obj = next(o for o in layout['objects'] if o['object_id'] == target)
        _, entry, _ = standard.verify_package(root/obj['standard_package'])
        parsed = standard.inspect(entry)
        actual = builder.support_surface(parsed['collision'], parsed['collision_faces'])
        if abs(actual['z_m']-declared['z_m']) > 1e-4:
            raise ValueError(
                f"{surface_id}: declared support height {declared['z_m']:.5f} m is not the "
                f"measured collision top face {actual['z_m']:.5f} m")
        if not inside(declared['polygon_xy_m'], actual['polygon_xy_m']):
            raise ValueError(
                f"{surface_id}: declared support patch is not contained in the measured "
                f"top face ({declared.get('area_m2', float('nan')):.4f} m2 declared vs "
                f"{actual['area_m2']:.4f} m2 measured); it claims surface the asset does "
                'not have')
        measured[surface_id] = dict(measured_area_m2=actual['area_m2'],
                                    measured_z_m=actual['z_m'])
    return measured


def geometry(root, layout):
    result = {}
    for obj in layout['objects']:
        _, entry, _ = standard.verify_package(root / obj['standard_package'])
        measured = standard.inspect(entry)
        vertices = np.concatenate([measured['visual'], measured['collision']])
        result[obj['object_id']] = vertices[ConvexHull(vertices).vertices].tolist()
    return result


def geometric_checks(row, layout, vertices, cfg, *, lateral=True):
    world = {n: np.asarray(v) @ rotation(row['objects'][n]['orientation_wxyz']).T
             + row['objects'][n]['position'] for n, v in vertices.items()}
    boxes = {n: np.array([v.min(0), v.max(0)]) for n, v in world.items()}
    checks = []
    for r in layout['relations']:
        a, b = r['source'], r['target']
        if r['relation'] == 'on' and b == spatial.GROUND:
            # An unbounded surface cannot be left laterally, so the containment margin is
            # not a question here. Sinking below it is, and that is a real reading.
            floor = layout['environment']['z_m']
            lowest = float(world[a][:, 2].min())
            checks.append(dict(source=a, target=b, relation='on',
                               lowest_z_m=lowest, plane_z_m=floor,
                               passed=lowest >= floor - cfg['penetration_m']))
        elif r['relation'] == 'on':
            target = row['objects'][b]
            rt = rotation(target['orientation_wxyz'])
            surface = layout['support_surfaces'][r.get('surface_id', b)]
            poly, normals = positions.polygon(surface['polygon_xy_m'])
            local = (world[a] - target['position']) @ rt
            margin = min(float(np.min((local[:, :2]-p) @ normal))
                         for p, normal in zip(poly, normals, strict=True))
            tilt = float(np.degrees(np.arccos(np.clip(rt[2, 2], -1, 1))))
            checks.append(dict(source=a, target=b, relation='on', margin_m=margin,
                               tilt_deg=tilt, passed=margin >= cfg['margin_m']-1e-8
                               and tilt <= cfg['surface_tilt_deg']))
        elif lateral:
            checks.append(dict(source=a, target=b, relation=r['relation'],
                               passed=bool(spatial.relation_ok(r['relation'], boxes[a], boxes[b]))))
    return checks


def evaluate(rows, layout, vertices, cfg):
    parents, fixed = topology(layout)
    failures = []
    minimum_margin = float('inf')
    allowed = {frozenset((a, b)) for a, b in parents.items()}
    for row in rows:
        for n in fixed:
            if n == spatial.GROUND:
                continue  # analytic and immovable; there is no state row to compare
            obj = next(o for o in layout['objects'] if o['object_id'] == n)
            state = row['objects'][n]
            if (np.linalg.norm(np.array(state['position'])-obj['translation_m']) > 1e-6
                    or angle(state['orientation_wxyz'], obj['orientation_wxyz']) > 1e-4):
                failures.append(dict(step=row['step'], reason='fixed_root_moved', object_id=n))
        checks = geometric_checks(row, layout, vertices, cfg,
                                   lateral=row['time_s'] >= cfg['dt']*cfg['steps']-cfg['window_s']
                                   or row['step'] == 0)
        for c in checks:
            minimum_margin = min(minimum_margin, c.get('margin_m', float('inf')))
            if not c['passed']:
                failures.append(dict(step=row['step'], reason='graph_geometry', check=c))
        for c in row['contacts']:
            if c['penetration'] > cfg['penetration_m']:
                failures.append(dict(step=row['step'], reason='penetration',
                                     penetration_m=c['penetration'],
                                     limit_m=cfg['penetration_m']))
            if (frozenset((c['a'], c['b'])) not in allowed
                    and not {c['a'], c['b']} <= fixed):
                failures.append(dict(step=row['step'], reason='undeclared_contact',
                                     a=c['a'], b=c['b']))
    return dict(version=VERSION, passed=not failures,
                # None, not infinity: an unbounded plane has no containment margin to
                # report, and "inf" is both unserialisable and reads like a huge margin
                # rather than an inapplicable one.
                minimum_margin_m=None if minimum_margin == float('inf') else minimum_margin,
                failure_count=len(failures), failures=failures[:100])
