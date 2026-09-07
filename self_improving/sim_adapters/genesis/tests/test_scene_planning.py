"""Scene planning contracts, bounded geometry and lifecycle without a live model."""
import copy
import json

import numpy as np
import pytest
from test_asset_extraction import config
from test_build_scene import prepared
from test_clip_select import setup  # noqa: F401

from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import scene_layout as layout
from self_improving.sim_adapters.genesis.scene_planning import ScenePlanner


def example():
    doc = dict(request='桌上有两个方块。', objects=[dict(object_id=n, category=n,
               description=n) for n in ['table', 'a', 'b']], relations=[dict(
               relation='on', source=n, target='table', evidence='桌上有两个方块。')
               for n in ['a', 'b']])
    bindings = {o['object_id']: dict(asset_id=o['object_id']) for o in doc['objects']}
    geometry = dict(table=dict(bounds=[[-.6, -.4, 0], [.6, .4, .7]], surface=surface(.6, .4, .7)),
                    a=dict(bounds=[[-.05, -.05, 0], [.05, .05, .1]]),
                    b=dict(bounds=[[-.05, -.05, 0], [.05, .05, .1]]))
    return doc, bindings, geometry


def surface(x, y, z):
    vertices = [[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z], [0, 0, 0]]
    return builder.support_surface(vertices, [[0, 1, 2], [0, 2, 3]])


def proposal(doc, preferences=None):
    return dict(object_ids=[o['object_id'] for o in doc['objects']],
                relations=copy.deepcopy(doc['relations']), preferences=preferences or [])


def solve(doc, bindings, geometry, preferences=None):
    graph = layout.graph_for(doc, bindings, proposal(doc, preferences))
    return layout.solve(doc, bindings, geometry, graph)


def test_reproducible_layout_and_complete_relation_checks():
    doc, bindings, geometry = example()
    doc['relations'] += [dict(relation='left_of', source='a', target='b'),
                         dict(relation='near', source='a', target='b')]
    result, report = solve(doc, bindings, geometry)
    assert report['status'] == 'passed'
    assert result == solve(doc, bindings, geometry)[0]
    boxes = {o['object_id']: np.array(o['world_visual_bounds_m']) for o in result['objects']}
    assert layout.relation_ok('left_of', boxes['a'], boxes['b'])
    assert layout.xy_distance(boxes['a'], boxes['b']) <= .18
    assert result['physics_steps'] == 0 and all(o['scale'] == 1 for o in result['objects'])


@pytest.mark.parametrize('kind', sorted(layout.LATERAL))
def test_all_lateral_relations(kind):
    doc, bindings, geometry = example()
    doc['relations'].append(dict(relation=kind, source='a', target='b'))
    result, _ = solve(doc, bindings, geometry)
    boxes = {o['object_id']: np.array(o['world_visual_bounds_m']) for o in result['objects']}
    assert layout.relation_ok(kind, boxes['a'], boxes['b'])


def test_three_levels_and_no_invented_table():
    doc, bindings, geometry = example()
    geometry['a'] = dict(bounds=[[-.2, -.2, 0], [.2, .2, .1]], surface=surface(.2, .2, .1))
    doc['relations'][1]['target'] = 'a'
    result, _ = solve(doc, bindings, geometry)
    assert [o['world_visual_bounds_m'][0][2] for o in result['objects']] == pytest.approx([0,.7,.8])
    assert result['objects'][-1]['support'] == 'a'
    doc['objects'] = doc['objects'][1:]
    doc['relations'] = []
    del bindings['table'], geometry['table']
    result, _ = solve(doc, bindings, geometry)
    assert all(o['support'] == 'ground' for o in result['objects'])


@pytest.mark.parametrize('change',
                         ['inside', 'unknown', 'cycle', 'multiple', 'direction', 'distance'])
def test_bad_hard_constraints_fail(change):
    doc, bindings, geometry = example()
    if change in {'inside', 'unknown'}:
        doc['relations'][0]['relation'] = change
    elif change == 'cycle':
        doc['relations'].append(dict(relation='on', source='table', target='a'))
    elif change == 'multiple':
        doc['relations'].append(dict(relation='on', source='a', target='b'))
    elif change == 'direction':
        doc['relations'] += [dict(relation='left_of', source='a', target='b'),
                             dict(relation='right_of', source='a', target='b')]
    else:
        doc['relations'] += [dict(relation='near', source='a', target='b'),
                             dict(relation='far_from', source='a', target='b')]
    with pytest.raises(ValueError):
        solve(doc, bindings, geometry)


@pytest.mark.parametrize('change', ['extra_object', 'reverse', 'support', 'pose', 'unknown_id'])
def test_model_cannot_change_assets_hard_edges_or_emit_poses(change):
    doc, _, _ = example()
    plan = proposal(doc)
    if change == 'extra_object':
        plan['object_ids'].append('invented')
    elif change == 'reverse':
        plan['relations'][0].update(source='table', target='a')
    elif change == 'pose':
        plan['world_xyz'] = [0, 0, 0]
    elif change == 'unknown_id':
        plan['preferences'] = [dict(object_id='invented', region='left')]
    else:
        plan['preferences'] = [dict(relation='on', source='a', target='b')]
    with pytest.raises(ValueError):
        layout.validate_proposal(plan, doc)


def test_oversize_obstacle_and_overlap_fail_with_trace():
    doc, bindings, geometry = example()
    geometry['a']['bounds'] = [[-1, -1, 0], [1, 1, .1]]
    with pytest.raises(layout.LayoutError) as error:
        solve(doc, bindings, geometry)
    assert error.value.trace['backtracks'] <= 48
    doc, bindings, geometry = example()
    geometry['table']['surface']['above_triangles_xy_m'] = [
        [[-.6, -.4], [.6, -.4], [.6, .4]], [[-.6, -.4], [.6, .4], [-.6, .4]]]
    with pytest.raises(layout.LayoutError):
        solve(doc, bindings, geometry)
    # Full-area boxes cannot coexist on the same support even if both individually fit.
    doc, bindings, geometry = example()
    for n in ['a', 'b']:
        geometry[n]['bounds'] = [[-.5, -.3, 0], [.5, .3, .1]]
    with pytest.raises(layout.LayoutError):
        solve(doc, bindings, geometry)


def planner(tmp_path, send):
    return ScenePlanner(config(), transport_fn=send, cache_dir=tmp_path/'cache')


def test_repair_cache_revalidation_and_asset_invalidation(tmp_path):
    doc, bindings, geometry = example()
    calls = []
    def send(system, user):
        calls.append(json.loads(user))
        return '{}' if len(calls) == 1 else json.dumps(proposal(doc))
    p = planner(tmp_path, send)
    out = tmp_path/'out'
    out.mkdir()
    result = p.plan(doc, bindings, geometry, 42, lambda: None, out)
    assert len(calls) == 2 and 'feedback' in calls[1]
    assert p.evidence['status'] == 'passed'
    cached = planner(tmp_path, lambda *_: pytest.fail('cache must avoid model'))
    assert cached.plan(doc, bindings, geometry, 42, lambda: None, out) == result
    assert cached.evidence['calls'] == 0 and cached.evidence['cache']['hit']
    bindings['a']['source_files'] = ['new hash']
    p = planner(tmp_path, lambda *_: json.dumps(proposal(doc)))
    p.plan(doc, bindings, geometry, 42, lambda: None, out)
    assert p.evidence['calls'] == 1 and not p.evidence['cache']['hit']


def test_cache_corruption_is_terminal(tmp_path):
    doc, bindings, geometry = example()
    out = tmp_path/'out'
    out.mkdir()
    p = planner(tmp_path, lambda *_: json.dumps(proposal(doc)))
    p.plan(doc, bindings, geometry, 42, lambda: None, out)
    path = next((tmp_path/'cache').glob('*.json'))
    entry = json.loads(path.read_text())
    entry['proposal']['preferences'] = [dict(relation='on', source='a', target='b')]
    entry['sha256'] = clip.digest(entry['proposal'])
    path.write_text(json.dumps(entry))
    p = planner(tmp_path, lambda *_: pytest.fail('do not repair bad cache'))
    with pytest.raises(ValueError, match='cached scene plan'):
        p.plan(doc, bindings, geometry, 42, lambda: None, out)
    assert p.evidence['calls'] == 0


@pytest.mark.parametrize('mode', ['malformed', 'timeout', 'secret', 'unsupported', 'infeasible'])
def test_terminal_failures_and_call_budget(tmp_path, mode):
    doc, bindings, geometry = example()
    out = tmp_path/'out'
    out.mkdir()
    def send(*_):
        if mode == 'timeout':
            raise TimeoutError(config().api_key)
        if mode == 'secret':
            return config().api_key
        return '[]' if mode == 'malformed' else json.dumps(proposal(doc))
    if mode == 'unsupported':
        doc['relations'][0]['relation'] = 'inside'
    if mode == 'infeasible':
        geometry['a']['bounds'] = [[-2, -2, 0], [2, 2, .1]]
    p = planner(tmp_path, send)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        p.plan(doc, bindings, geometry, 42, lambda: None, out)
    expected = 0 if mode == 'unsupported' else 2 if mode in {'malformed', 'infeasible'} else 1
    assert p.evidence['calls'] == expected
    assert config().api_key not in json.dumps(p.evidence)
    assert not list((tmp_path/'cache').glob('*.json'))


def test_scene_llm_entry_preserves_assets_and_reports_calls(setup, monkeypatch):  # noqa: F811
    s = setup
    task = prepared(s, monkeypatch)
    before = {p: p.read_bytes() for p in task.stage('objects').rglob('*') if p.is_file()}
    def send(system, user):
        payload = json.loads(user)
        return json.dumps(dict(object_ids=payload['object_ids'],
                               relations=payload['explicit_relations'], preferences=[]))
    p = planner(s.root, send)
    def render(doc, bindings, bounds, output, check, *, plan_layout):
        from test_build_scene import geometry
        result = plan_layout(geometry())
        clip.write_json(output/'scene_layout.json', result)
        check()
        return dict(status='passed')
    result = builder.run(task.root, s.index/'index.json', provider=p, renderer=render)
    assert result['status'] == 'scene_built', result
    assert result['model_calls'] == 1
    task.verify()
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(task.stage('physics').iterdir())
    failed = planner(s.root/'bad', lambda *_: '{}')
    result = builder.run(task.root, s.index/'index.json', provider=failed, renderer=render)
    assert result['status'] == 'error' and result['model_calls'] == 2
    assert not (task.stage('scene')/'scene_layout.json').exists()
    task.verify()
    assert task.report['stages']['objects'] == 'passed'


def test_automatic_entry_and_independent_build_are_equivalent(setup, monkeypatch):  # noqa: F811
    from test_asset_extraction import QUERY, four_objects, provider
    from test_build_scene import document, geometry

    from self_improving.sim_adapters.genesis import extract_assets as entry
    from self_improving.sim_adapters.genesis.task_output import TaskOutput

    s = setup
    monkeypatch.setattr(entry, 'load_llm_provider_config', lambda _: s.config)
    extracted, _ = provider(s.root, four_objects(), document()['relations'], cfg=s.config)
    def selector(index, query, output, **kwargs):
        return clip.select(index, query, output, **kwargs, encoder=s.encoder,
                           weights_dir=s.kwargs['weights_dir'], cache_dir=s.root/'selection',
                           vlm=s.kwargs['vlm'])
    def send(system, user):
        data = json.loads(user)
        return json.dumps(dict(object_ids=data['object_ids'],
                               relations=data['explicit_relations'], preferences=[]))
    def renderer(doc, bindings, bounds, output, check, *, plan_layout):
        clip.write_json(output/'scene_layout.json', plan_layout(geometry()))
        check()
        return dict(status='passed')
    p = planner(s.root, send)
    report = entry.run(QUERY, s.root/'automatic', clip_index=s.index/'index.json',
                       vlm_config=None, provider=extracted, selector=selector,
                       scene_provider=p, scene_renderer=renderer)
    assert report['status'] == 'scene_built', report
    task = TaskOutput(report['output_dir'])
    task.verify()
    before = (task.stage('scene')/'scene_layout.json').read_bytes()
    again = builder.run(task.root, s.index/'index.json', provider=planner(s.root, send),
                         renderer=renderer)
    assert again['status'] == 'scene_built' and again['model_calls'] == 0
    assert (task.stage('scene')/'scene_layout.json').read_bytes() == before
    # Exact handoff manifest mismatch must not clear a completed stage.
    with pytest.raises(ValueError, match='manifest changed'):
        builder.run(task.root, s.index/'index.json', expected_manifest_sha256='stale')
    assert (task.stage('scene')/'scene_layout.json').read_bytes() == before
    task.verify()


def test_input_change_during_model_call_is_terminal(tmp_path):
    doc, bindings, geometry = example()
    out = tmp_path/'out'
    out.mkdir()
    changed = False
    def send(*_):
        nonlocal changed
        changed = True
        return json.dumps(proposal(doc))
    def check():
        if changed:
            raise ValueError('input hash changed')
    p = planner(tmp_path, send)
    with pytest.raises(ValueError, match='input hash changed'):
        p.plan(doc, bindings, geometry, 42, check, out)
    assert p.evidence['calls'] == 1
    assert not list((tmp_path/'cache').glob('*.json'))


def test_config_default_is_llm_and_stop_after_is_explicit(monkeypatch, capsys):
    from self_improving.sim_adapters.genesis import extract_assets as entry

    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return dict(status='scene_built', output=0, total_s=0)
    monkeypatch.setattr(builder, 'run', run)
    assert builder.main(['--scene-dir', 'task', '--clip-index', 'index']) == 0
    assert calls[-1]['planner'] == 'llm'
    def extract(**kwargs):
        calls.append(kwargs)
        return dict(status='assets_selected', output_dir='task')
    monkeypatch.setattr(entry, 'run', extract)
    assert entry.main(['--request', '桌上有苹果', '--clip-index', 'index',
                       '--vlm-config', 'config', '--stop-after', 'assets']) == 0
    assert calls[-1]['stop_after'] == 'assets'


def test_deep_failure_never_exceeds_global_backtrack_budget():
    doc, bindings, geometry = example()
    geometry['a'] = dict(bounds=[[-.2, -.2, 0], [.2, .2, .1]], surface=surface(.2, .2, .1))
    geometry['b']['bounds'] = [[-.5, -.5, 0], [.5, .5, .1]]
    doc['relations'][1]['target'] = 'a'
    with pytest.raises(layout.LayoutError) as error:
        solve(doc, bindings, geometry)
    assert error.value.trace['backtracks'] <= 48


def test_reused_provider_counts_only_this_run(tmp_path):
    doc, bindings, geometry = example()
    out = tmp_path/'out'
    out.mkdir()
    p = planner(tmp_path, lambda *_: json.dumps(proposal(doc)))
    first = p.plan(doc, bindings, geometry, 42, lambda: None, out)
    assert p.evidence['calls'] == 1
    assert p.plan(doc, bindings, geometry, 42, lambda: None, out) == first
    assert p.evidence['calls'] == 0 and len(p.evidence['attempts']) == 1
    assert p.evidence['transport'] == 'injected'


def test_missing_measured_support_is_terminal_before_model(tmp_path):
    doc, bindings, geometry = example()
    del geometry['table']['surface']
    with pytest.raises(layout.UnsupportedScene, match='measured support'):
        solve(doc, bindings, geometry)


def test_explicit_numeric_distance_is_not_replaced_with_default():
    doc, bindings, geometry = example()
    doc['relations'].append(dict(relation='near', source='a', target='b',
                                 evidence='a距离b 5厘米'))
    with pytest.raises(layout.UnsupportedScene, match='numeric distances'):
        solve(doc, bindings, geometry)
