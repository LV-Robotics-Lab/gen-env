"""Bounded, translation-only layout of verified native assets (visual evidence only)."""
from __future__ import annotations

import re

import numpy as np

from self_improving.sim_adapters.genesis import clip_select as clip

SCHEMA = 'genenv.asset_scene.v2'
VERSION = 'genenv.visual_layout.v1'
GAP, MARGIN, NEAR, FAR = .06, .02, .18, .30
# MARGIN is the acceptance red line. Planning aims further in, so that normal settling
# drift cannot carry a legally placed object back across it.
SETTLING_SLACK = .015
PLANNING_MARGIN = MARGIN+SETTLING_SLACK
LATERAL = {'left_of', 'right_of', 'in_front_of', 'behind', 'near', 'far_from'}
REGIONS = {'center', 'left', 'right', 'front', 'back'}


class UnsupportedScene(ValueError):
    """A capability failure cannot be repaired by changing model preferences."""


class LayoutError(ValueError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


def _acyclic(ids, edges, label):
    active, done = set(), set()
    def visit(name):
        if name in active:
            raise ValueError(f'{label} cycle')
        if name in done:
            return
        active.add(name)
        for a, b in edges:
            if a == name:
                visit(b)
        active.remove(name)
        done.add(name)
    for name in ids:
        visit(name)


# Reserved id for the Genesis built-in plane. It is a legitimate fixed root -- a body on
# the floor is an ordinary scene -- but it is analytic rather than a mesh, so the surface
# checks that re-measure an asset's top face do not apply to it.
GROUND = 'ground'


def relations(document):
    ids = [o['object_id'] for o in document['objects']]
    if not 1 <= len(ids) <= 12 or len(ids) != len(set(ids)):
        raise ValueError('invalid or duplicate asset object ID')
    if GROUND in ids:
        raise ValueError(f'{GROUND!r} is reserved for the environment plane')
    parents, seen, axes, distances = {}, set(), [[], []], {}
    rows = document['relations']
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('invalid relation list')
    for row in rows:
        kind, a, b = row['relation'], row['source'], row['target']
        # The environment plane may be named as what a body rests on, and only that: it is
        # not an object, so it has no extent to be left of and nothing can rest under it.
        # Whether the scene actually has a plane is checked where the layout is known.
        if b == GROUND and kind == 'on' and a in ids:
            pass
        elif a not in ids or b not in ids or a == b:
            raise ValueError('invalid relation object reference')
        if kind not in LATERAL | {'on'}:
            raise UnsupportedScene(f'unsupported relation: {kind}; inside is deferred')
        if re.search(r'\d+(?:\.\d+)?\s*(?:cm|mm|meters?|metres?|厘米|毫米|米)',
                     row.get('evidence', ''), re.I):
            raise UnsupportedScene('explicit numeric distances are not supported')
        key = kind, a, b
        if key in seen:
            raise ValueError('duplicate relation')
        seen.add(key)
        if kind == 'on':
            if a in parents:
                raise ValueError('multiple support targets')
            parents[a] = b
        elif kind in {'left_of', 'right_of'}:
            axes[0].append((a, b) if kind == 'left_of' else (b, a))
        elif kind in {'in_front_of', 'behind'}:
            axes[1].append((a, b) if kind == 'in_front_of' else (b, a))
        else:
            pair = tuple(sorted((a, b)))
            if pair in distances and distances[pair] != kind:
                raise ValueError('contradictory near/far constraints')
            distances[pair] = kind
    _acyclic([*ids, GROUND], list(parents.items()), 'support')
    for edges in axes:
        _acyclic(ids, edges, 'direction')
    return parents


def validate_proposal(proposal, document):
    fields = {'object_ids', 'relations', 'preferences'}
    if not isinstance(proposal, dict) or set(proposal) != fields:
        raise ValueError('unexpected scene planning fields')
    ids = [o['object_id'] for o in document['objects']]
    if proposal['object_ids'] != ids:
        raise ValueError('model changed or omitted the ordered object set')
    if proposal['relations'] != document['relations']:
        raise ValueError('model changed explicit relations')
    relations(document)
    prefs = proposal['preferences']
    if not isinstance(prefs, list) or len(prefs) > 48:
        raise ValueError('invalid preference list')
    for pref in prefs:
        if not isinstance(pref, dict):
            raise ValueError('invalid preference')
        if set(pref) == {'object_id', 'region'}:
            if pref['object_id'] not in ids or pref['region'] not in REGIONS:
                raise ValueError('invalid region preference')
        elif set(pref) == {'relation', 'source', 'target'}:
            if (pref['relation'] not in LATERAL or pref['source'] not in ids
                    or pref['target'] not in ids or pref['source'] == pref['target']):
                raise ValueError('invalid lateral preference; no inferred support allowed')
        else:
            raise ValueError('unexpected preference fields')
    return proposal


def graph_for(document, bindings, proposal):
    validate_proposal(proposal, document)
    return dict(schema_version=SCHEMA,
                nodes=[dict(object_id=o['object_id'], category=o['category'],
                            asset_id=bindings[o['object_id']]['asset_id'])
                       for o in document['objects']],
                edges=document['relations'], preferences=proposal['preferences'],
                preference_source='model_suggestion',
                environment=dict(ground='genesis_builtin_plane'),
                frame=dict(up='+Z', right='+X', front='-Y', units='m'))


def rectangle(box):
    return np.array([[box[0, 0], box[0, 1]], [box[1, 0], box[0, 1]],
                     [box[1, 0], box[1, 1]], [box[0, 0], box[1, 1]]])


def xy_distance(a, b):
    return float(np.linalg.norm(np.maximum(0, np.maximum(a[0, :2]-b[1, :2],
                                                       b[0, :2]-a[1, :2]))))


def relation_ok(kind, a, b):
    if kind == 'left_of':
        return a[1, 0]+GAP <= b[0, 0]+1e-8
    if kind == 'right_of':
        return b[1, 0]+GAP <= a[0, 0]+1e-8
    if kind == 'in_front_of':
        return a[1, 1]+GAP <= b[0, 1]+1e-8
    if kind == 'behind':
        return b[1, 1]+GAP <= a[0, 1]+1e-8
    distance = xy_distance(a, b)
    return distance <= NEAR+1e-8 if kind == 'near' else distance >= FAR-1e-8


def solve(document, bindings, geometry, graph, seed=42):
    # Local import keeps the geometry implementation shared with legacy rule mode.
    from self_improving.sim_adapters.genesis.build_scene import fits_surface

    parents = relations(document)
    ids = [o['object_id'] for o in document['objects']]
    if set(ids) != set(bindings) or set(ids) != set(geometry):
        raise ValueError('geometry/binding object set mismatch')
    boxes = {n: np.asarray(geometry[n]['bounds'], float) for n in ids}
    if any(b.shape != (2, 3) or not np.isfinite(b).all() or not (b[1] > b[0]).all()
           for b in boxes.values()):
        raise ValueError('invalid native visual bounds')
    for name in set(parents.values()):
        if 'surface' not in geometry[name]:
            raise UnsupportedScene(f'{name}: no measured support surface')
    def depth(n):
        return 0 if n not in parents else 1+depth(parents[n])
    depths = {n: depth(n) for n in ids}
    order = sorted(ids, key=lambda n: (depths[n], ids.index(n)))
    rng = np.random.default_rng(seed)
    poses, world = {}, {}
    trace = dict(version=VERSION, seed=seed, candidate_limit=96, backtrack_limit=48,
                 backtracks=0, attempts=[], status='running')
    extent = sum(float(np.max(b[1, :2]-b[0, :2]))+FAR for b in boxes.values())

    def failures(n, pos):
        b = boxes[n]+pos
        reasons = []
        if n in parents:
            target = parents[n]
            local = b-poses[target]
            if not fits_surface(geometry[target]['surface'], rectangle(local)):
                reasons.append('full footprint outside measured support or under an obstacle')
        for other, a in world.items():
            if parents.get(n) == other or parents.get(other) == n:
                continue
            if np.all(np.minimum(b[1], a[1])-np.maximum(b[0], a[0]) > 1e-7):
                reasons.append(f'conservative geometry overlap: {other}')
            if depths[n] == depths[other] and xy_distance(a, b) < GAP-1e-8:
                reasons.append(f'same-level gap: {other}')
        for row in document['relations']:
            if row['relation'] == 'on':
                continue
            a, c = row['source'], row['target']
            known = dict(world, **{n: b})
            if a in known and c in known and not relation_ok(row['relation'], known[a], known[c]):
                reasons.append(f"{row['relation']}: {a} -> {c}")
        return reasons

    def candidates(n):
        box = boxes[n]
        half = (box[1, :2]-box[0, :2])/2
        if n in parents:
            target = parents[n]
            surface = geometry[target]['surface']
            polygon = np.asarray(surface['polygon_xy_m'])+poses[target][:2]
            low = polygon.min(axis=0)+half+PLANNING_MARGIN
            high = polygon.max(axis=0)-half-PLANNING_MARGIN
            z = poses[target][2]+surface['z_m']-box[0, 2]
        else:
            low, high = np.full(2, -extent/2), np.full(2, extent/2)
            z = -box[0, 2]
        if np.any(low > high):
            return []
        center = (low+high)/2
        points = [center]
        # Exact gap anchors make near/directional constraints practical for small assets.
        for b in world.values():
            for axis in (0, 1):
                for sign in (-1, 1):
                    for gap in (GAP, FAR):
                        p = b[:, :2].mean(axis=0)
                        p[axis] = b[0 if sign < 0 else 1, axis]+sign*(half[axis]+gap)
                        points.append(np.clip(p, low, high))
        points.extend(np.array([x, y]) for x in np.linspace(low[0], high[0], 7)
                      for y in np.linspace(low[1], high[1], 7))
        points.extend(rng.uniform(low, high) for _ in range(32))
        def score(p):
            score = float(np.linalg.norm(p-center))*.01
            candidate = box+np.r_[p-box[:, :2].mean(axis=0), z]
            for pref in graph['preferences']:
                if pref.get('object_id') == n:
                    region = pref['region']
                    desired = center.copy()
                    if region != 'center':
                        axis = 0 if region in {'left', 'right'} else 1
                        desired[axis] = (low if region in {'left', 'front'} else high)[axis]
                    score += float(np.linalg.norm(p-desired))
                elif 'relation' in pref:
                    a, b = pref['source'], pref['target']
                    known = dict(world, **{n: candidate})
                    if a in known and b in known:
                        score += 1.0*(not relation_ok(pref['relation'], known[a], known[b]))
            return score
        unique = {tuple(np.round(p, 10)): p for p in points}
        return [np.r_[p-box[:, :2].mean(axis=0), z]
                for p in sorted(unique.values(), key=score)[:96]]

    def place(i):
        if i == len(order):
            return True
        n = order[i]
        choices = candidates(n)
        if not choices:
            trace['attempts'].append(dict(object_id=n,
                                          reasons=['footprint exceeds support bounds']))
        for pos in choices:
            reasons = failures(n, pos)
            trace['attempts'].append(dict(object_id=n, translation_m=pos.tolist(), reasons=reasons))
            if reasons:
                continue
            poses[n], world[n] = pos, boxes[n]+pos
            if place(i+1):
                return True
            del poses[n], world[n]
            if trace['backtracks'] >= 48:
                return False
            trace['backtracks'] += 1
            if trace['backtracks'] >= 48:
                return False
        return False

    if not place(0):
        trace['status'] = 'failed'
        raise LayoutError('bounded layout search exhausted; hard constraints unsatisfied', trace)
    # Independent final pass also covers constraints whose other endpoint was placed later.
    checks = []
    for n in order:
        b = world.pop(n)
        reasons = failures(n, poses[n])
        world[n] = b
        checks.append(dict(object_id=n, status='passed' if not reasons else 'failed',
                           reasons=reasons))
    if any(c['reasons'] for c in checks):
        raise LayoutError('final geometry validation failed', dict(trace, checks=checks))
    trace.update(status='passed', checks=checks, physics_status='not_run')
    surfaces = {}
    for target in set(parents.values()):
        surfaces[target] = dict(geometry[target]['surface'],
                                world_z_m=poses[target][2]+geometry[target]['surface']['z_m'],
                                child_placements=[dict(object_id=n, margin_m=PLANNING_MARGIN,
                                footprint_target_xy_m=rectangle(world[n]-poses[target]).tolist(),
                                coverage='passed') for n in ids if parents.get(n) == target])
    layout = dict(schema_version=SCHEMA, scene_graph_sha256=clip.digest(graph),
                  native_geometry_sha256=clip.digest(geometry), solver_version=VERSION,
                  objects=[dict(bindings[n], object_id=n, translation_m=poses[n].tolist(),
                           scale=1.0, orientation_policy='preserve_native_orientation',
                           local_visual_bounds_m=boxes[n].tolist(),
                           world_visual_bounds_m=world[n].tolist(),
                           support=parents.get(n, 'ground')) for n in ids],
                  support_surfaces=surfaces, relations=document['relations'],
                  environment=dict(ground='genesis_builtin_plane', z_m=0),
                  minimum_visual_gap_m=GAP, physics_steps=0, physics_status='not_run',
                  meaning='initial visual geometry only; contact and stability unverified')
    return layout, trace
