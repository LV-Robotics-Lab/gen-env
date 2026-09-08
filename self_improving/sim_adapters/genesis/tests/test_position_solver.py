"""Geometry regressions and adversarial placement cases; no synthetic physics verdicts."""
import copy
import itertools

import numpy as np
import pytest

from self_improving.sim_adapters.genesis import position_solver as solver


def box(name, size, pos, *, fixed=False, yaw=0):
    half = np.asarray(size)/2
    return dict(object_id=name, fixed=fixed, translation_m=pos,
                orientation_wxyz=[float(np.cos(yaw/2)), 0, 0, float(np.sin(yaw/2))],
                geometry_vertices_m=list(map(list, itertools.product(*zip(-half, half)))))


def patch(name, size=(1, 1), z=.05, sid=None):
    x, y = np.asarray(size)/2
    return dict(surface_id=sid or name, object_id=name, z_m=z,
                polygon_xy_m=[[-x, -y], [x, -y], [x, y], [-x, y]], source='measured_fixture')


def on(a, b, sid=None):
    return dict(relation='on', source=a, target=b, surface_id=sid or b)


def scene():
    return dict(objects=[box('table', [1, 1, .1], [0, 0, .5], fixed=True),
                         box('a', [.1, .1, .1], [.1, .1, .8])],
                surfaces=[patch('table')], relations=[on('a', 'table')])


def test_preserves_xy_orientation_and_fixed_root():
    s = scene()
    original = copy.deepcopy(s)
    r = solver.solve(s)
    assert s == original
    assert r['status'] == 'proposal_ready' and r['physics_status'] == 'not_run'
    assert r['poses']['table']['delta_m'] == [0, 0, 0]
    assert r['poses']['a']['translation_m'] == pytest.approx([.1, .1, .601])
    assert r == solver.solve(s)


def test_rotated_translated_target_local_frame():
    s = scene()
    s['objects'][0] = box('table', [1, 1, .1], [2, 3, .5], fixed=True, yaw=np.pi/2)
    s['objects'][1]['translation_m'] = [2.1, 3.1, .8]
    r = solver.solve(s)
    assert r['poses']['a']['translation_m'] == pytest.approx([2.1, 3.1, .601])


def test_overhanging_object_is_projected_inside():
    s = scene()
    s['objects'][1]['translation_m'] = [.49, 0, .8]
    r = solver.solve(s)
    assert r['poses']['a']['translation_m'][0] == pytest.approx(.44)


def test_overlapping_objects_separated():
    s = scene()
    s['objects'].append(box('b', [.1, .1, .1], [.1, .1, .8]))
    s['relations'].append(on('b', 'table'))
    r = solver.solve(s)
    assert r['status'] == 'proposal_ready'
    delta = np.abs(np.array(r['poses']['a']['translation_m']) -
                   r['poses']['b']['translation_m'])
    assert delta[:2].max() >= .102-1e-8


def test_multiple_tables_and_stacking():
    s = scene()
    s['objects'] += [box('table2', [1, 1, .1], [2, 0, .7], fixed=True),
                     box('b', [.1, .1, .1], [2, 0, .9]),
                     box('c', [.04, .04, .04], [.1, .1, .95])]
    s['surfaces'] += [patch('table2'), patch('a', (.1, .1))]
    s['relations'] += [on('b', 'table2'), on('c', 'a')]
    r = solver.solve(s)
    assert r['status'] == 'proposal_ready'
    assert r['poses']['b']['translation_m'][2] == pytest.approx(.801)
    assert r['poses']['c']['translation_m'][2] == pytest.approx(.672)


def test_no_infinite_ground_or_silent_missing_support():
    s = scene()
    s['surfaces'] = []
    with pytest.raises(ValueError, match='missing'):
        solver.solve(s)
    s = scene()
    s['relations'] = []
    with pytest.raises(ValueError, match='every dynamic'):
        solver.solve(s)


@pytest.mark.parametrize('kind', ['cycle', 'inside', 'tilt', 'nan', 'hole', 'concave', 'line'])
def test_rejects_unsupported_or_invalid_input(kind):
    s = scene()
    if kind == 'cycle':
        s['objects'].append(box('b', [.1, .1, .1], [0, 0, .8]))
        s['surfaces'] += [patch('a'), patch('b')]
        s['relations'] = [on('a', 'b'), on('b', 'a')]
    elif kind == 'inside':
        s['relations'][0]['relation'] = 'inside'
    elif kind == 'tilt':
        s['objects'][0]['orientation_wxyz'] = [np.cos(.1), np.sin(.1), 0, 0]
    elif kind == 'nan':
        s['objects'][1]['translation_m'][0] = float('nan')
    elif kind == 'hole':
        s['surfaces'][0]['holes'] = [[[0, 0], [.1, 0], [0, .1]]]
    elif kind == 'concave':
        s['surfaces'][0]['polygon_xy_m'] = [[0, 0], [1, 0], [.2, .2], [0, 1]]
    else:
        s['surfaces'][0]['polygon_xy_m'] = [[0, 0], [.1, 0], [.2, 0]]
    with pytest.raises(ValueError):
        solver.solve(s)


def test_full_footprint_not_center_only():
    s = scene()
    s['objects'][1] = box('a', [2, 2, .1], [0, 0, .8])
    r = solver.solve(s)
    assert r['status'] == 'search_exhausted' and not r['poses']
    assert r['physics_status'] == 'not_run'


def test_input_hash_and_budget_fail_closed():
    s = scene()
    r = solver.solve(s, max_translation_m=.001, attempt_limit=2)
    assert r['status'] == 'search_exhausted'
    assert len(r['attempts']) <= 2
    s['objects'][1]['translation_m'][0] += .01
    assert solver.solve(s)['input_sha256'] != r['input_sha256']


def test_support_polygon_winding_and_collinear_vertices():
    s = scene()
    s['surfaces'][0]['polygon_xy_m'].reverse()
    assert solver.solve(s)['status'] == 'proposal_ready'


def test_two_disjoint_patches_on_one_table():
    s = scene()
    s['surfaces'] = [patch('table', (.3, .3), sid='left'),
                     patch('table', (.3, .3), sid='right')]
    for p, x in zip(s['surfaces'], [-.3, .3], strict=True):
        p['polygon_xy_m'] = (np.array(p['polygon_xy_m'])+[x, 0]).tolist()
    s['relations'][0]['surface_id'] = 'left'
    s['objects'].append(box('b', [.1, .1, .1], [0, 0, .8]))
    s['relations'].append(on('b', 'table', 'right'))
    r = solver.solve(s)
    assert r['status'] == 'proposal_ready'
    assert r['poses']['a']['translation_m'][0] < 0
    assert r['poses']['b']['translation_m'][0] > 0


def test_backtracks_earlier_object_when_later_patch_is_reserved():
    s = scene()
    s['objects'][1] = box('a', [.3, .3, .1], [0, 0, .8])
    s['objects'].append(box('b', [.1, .1, .1], [0, 0, .8]))
    s['surfaces'].append(patch('table', (.14, .14), sid='reserved'))
    s['relations'].append(on('b', 'table', 'reserved'))
    r = solver.solve(s)
    assert r['status'] == 'proposal_ready'
    assert sum(a['object_id'] == 'a' for a in r['attempts']) > 1
    assert np.max(np.abs(r['poses']['a']['translation_m'][:2])) > .19
