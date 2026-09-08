"""Graph rejection, intervention evidence and independent free-dynamics guards."""
import copy

import numpy as np
import pytest

from self_improving.sim_adapters.genesis import asset_physics as evidence
from self_improving.sim_adapters.genesis import position_solver as solver
from self_improving.sim_adapters.genesis import scene_physics_graph as graph
from self_improving.sim_adapters.genesis import scene_stabilization as settling
from self_improving.sim_adapters.genesis import validate_imported_scene as free
from self_improving.sim_adapters.genesis.tests.test_position_solver import box, on, scene


def graph_case():
    doc = scene()
    proposal = solver.solve(doc, margin_m=.02)
    objects = []
    for obj in doc['objects']:
        objects.append(dict(obj, **{k: v for k, v in proposal['poses'][obj['object_id']].items()
                                    if k in ('translation_m', 'orientation_wxyz')},
                            category='box', source_velocity_mps=[0, 0, 0],
                            source_angular_velocity_radps=[0, 0, 0]))
    layout = dict(objects=objects, relations=doc['relations'],
                  support_surfaces={s['surface_id']: s for s in doc['surfaces']},
                  environment=dict(ground=None))
    vertices = {o['object_id']: o['geometry_vertices_m'] for o in objects}
    return layout, vertices


def trajectory(layout, count):
    rows = []
    for step in range(count):
        states = {o['object_id']: dict(position=o['translation_m'],
                  orientation_wxyz=o['orientation_wxyz'], velocity=[0, 0, 0],
                  angular_velocity=[0, 0, 0]) for o in layout['objects']}
        c = dict(a='a', b='table', position=[.1, .1, .55], normal=[0, 0, 1],
                 penetration=0, force_a=[0, 0, 1] if step else None,
                 force_b=[0, 0, -1] if step else None,
                 geom_a=1, geom_b=0, link_a=1, link_b=0)
        rows.append(dict(step=step, time_s=.004*step, objects=copy.deepcopy(states),
                         contacts=[c], contact_phase='solved_step' if step else 'initial_detection',
                         zeroed_after_sample=['a'] if step else []))
    return rows


def test_lateral_relation_moves_object_and_rejects_conflict():
    doc = scene()
    doc['objects'].append(box('b', [.1, .1, .1], [.1, .1, .8]))
    doc['relations'] += [on('b', 'table'), dict(relation='left_of', source='a', target='b')]
    result = solver.solve(doc)
    assert result['status'] == 'proposal_ready'
    assert (result['poses']['b']['translation_m'][0]
            - result['poses']['a']['translation_m'][0]) >= .16
    doc['relations'].append(dict(relation='left_of', source='b', target='a'))
    with pytest.raises(ValueError, match='cycle'):
        solver.solve(doc)


def test_multi_root_graph_is_valid_but_fixed_child_is_not():
    layout, _ = graph_case()
    layout['objects'].append(box('another_table', [1, 1, .1], [2, 0, .5], fixed=True))
    assert graph.topology(layout)[1] == {'table', 'another_table'}
    layout['objects'][1]['fixed'] = True
    with pytest.raises(ValueError, match='dynamic'):
        graph.topology(layout)


def test_missing_patch_and_holes_rejected():
    layout, _ = graph_case()
    layout['support_surfaces']['table']['holes'] = [[[0, 0], [.1, 0], [0, .1]]]
    with pytest.raises(ValueError, match='free convex'):
        graph.topology(layout)
    layout['support_surfaces'].clear()
    with pytest.raises(ValueError, match='missing'):
        graph.topology(layout)


def test_stabilization_requires_fifty_consecutive_pre_reset_pose_samples():
    layout, vertices = graph_case()
    rows = trajectory(layout, 51)
    # Nonzero recorded velocities are allowed here: these are PRE-intervention values.
    for row in rows[1:]:
        row['objects']['a']['velocity'] = [.02, 0, 0]
    assert settling.assess(rows[:50], layout, vertices)['status'] == 'running'
    assert settling.assess(rows, layout, vertices)['status'] == 'candidate_ready'
    rows[-1]['zeroed_after_sample'] = []
    with pytest.raises(ValueError, match='intervention'):
        settling.assess(rows, layout, vertices)


def test_intervened_stability_does_not_imply_free_acceptance():
    layout, vertices = graph_case()
    # Stabilization judges its own intervened run, where contact holds the body still.
    assert settling.assess(trajectory(layout, 51), layout, vertices)[
        'status'] == 'candidate_ready'
    # Free replay is a separate rollout released from that candidate pose. Carrying a
    # velocity there means travelling, and nothing zeroes it away any more.
    rows = trajectory(layout, 1001)
    cfg = evidence.settings('baseline')
    speed = 2*cfg['effective_speed_mps']
    for row in rows[1:]:
        row['objects']['a']['velocity'] = [speed, 0, 0]
        row['objects']['a']['position'][0] += speed*row['step']*cfg['dt']
    result = free.evaluate(rows, layout, cfg, vertices)
    assert result['physics_status'] == 'failed'
    assert not next(c for c in result['objects']['a']['checks']
                    if c['name'] == 'drift_rate_mps')['passed']


def test_free_fall_creep_is_not_convergence():
    """Zeroing velocity turns free fall into a creep of 0.5*g*dt^2 per step.

    A convergence delta above that creep certifies a body falling through empty air,
    which is how an airborne pose reached free replay and tumbled.
    """
    layout, vertices = graph_case()
    creep = .5*abs(evidence.settings('baseline')['gravity'][2])*settling.SETTINGS['dt']**2
    rows = trajectory(layout, 201)
    for row in rows[1:]:
        row['contacts'] = []
        row['objects']['a']['position'][2] -= creep*row['step']
    assert settling.assess(rows, layout, vertices)['status'] != 'candidate_ready'
    # Holding perfectly still is not support either, without something holding it up.
    still = trajectory(layout, 201)
    for row in still:
        row['contacts'] = []
    assert settling.assess(still, layout, vertices)['status'] != 'candidate_ready'


def test_stabilization_delta_must_beat_free_fall_creep(monkeypatch):
    monkeypatch.setitem(settling.SETTINGS, 'position_delta_m', .0001)
    with pytest.raises(ValueError, match='free fall'):
        settling.discriminating(evidence.settings('baseline'))


@pytest.mark.parametrize('mode', ['drift', 'penetration', 'oscillate'])
def test_stabilization_failure_modes(mode):
    layout, vertices = graph_case()
    rows = trajectory(layout, 51)
    if mode == 'drift':
        rows[-1]['objects']['a']['position'][0] += .021
    elif mode == 'penetration':
        rows[-1]['contacts'][0]['penetration'] = .0011
    else:
        for row in rows[1:]:
            row['objects']['a']['position'][0] += .001*(row['step'] % 2)
    assert settling.assess(rows, layout, vertices)['status'] != 'candidate_ready'


def test_full_footprint_and_intermediate_fixed_root_motion_fail():
    layout, vertices = graph_case()
    rows = trajectory(layout, 1001)
    rows[20]['objects']['a']['position'][0] = .49
    rows[30]['objects']['table']['position'][2] += .01
    report = graph.evaluate(rows, layout, vertices, evidence.settings('baseline'))
    assert not report['passed']
    assert {'graph_geometry', 'fixed_root_moved'} <= {f['reason'] for f in report['failures']}


def test_zero_contacts_fail_even_when_geometrically_supported():
    layout, vertices = graph_case()
    rows = trajectory(layout, 1001)
    for row in rows:
        row['contacts'] = []
    assert free.evaluate(rows, layout, evidence.settings('baseline'), vertices)[
        'physics_status'] == 'failed'


def test_above_geometry_rejects_only_where_it_covers_the_patch():
    """A measured top face carries its own asset's perimeter bevel.

    Rejecting on the field's presence refuses every honest measurement; what matters is
    whether anything above the plane actually sits over the declared patch.
    """
    layout, _ = graph_case()
    surface = layout['support_surfaces']['table']
    poly = np.asarray(surface['polygon_xy_m'])
    lo, hi = poly.min(0), poly.max(0)
    outside = [[[hi[0]+.05, lo[1]], [hi[0]+.15, lo[1]], [hi[0]+.10, hi[1]]]]
    surface['above_triangles_xy_m'] = outside
    assert graph.topology(layout)
    middle = (lo+hi)/2
    surface['above_triangles_xy_m'] = outside + [
        [middle.tolist(), (middle+[.05, 0]).tolist(), (middle+[0, .05]).tolist()]]
    with pytest.raises(ValueError, match='covered by geometry above'):
        graph.topology(layout)


def test_declared_support_patch_must_be_backed_by_geometry(tmp_path):
    """The 'free convex patch' contract was asserted but never re-derived from the mesh.

    A patch spanning more than the asset's real top face releases bodies over open air.
    """
    from self_improving.sim_adapters.genesis import import_simfoundry_scene as imported
    from self_improving.sim_adapters.genesis.tests.workflow_fixtures import package

    root = package(tmp_path/'scene', scene())
    layout = imported.verify(root)
    assert graph.verify_support(root, layout)
    surface = layout['support_surfaces']['table']
    surface['polygon_xy_m'] = (np.asarray(surface['polygon_xy_m'])*3).tolist()
    with pytest.raises(ValueError, match='not contained in the measured'):
        graph.verify_support(root, layout)
    surface['polygon_xy_m'] = (np.asarray(surface['polygon_xy_m'])/3).tolist()
    surface['z_m'] += .05
    with pytest.raises(ValueError, match='not the measured collision top face'):
        graph.verify_support(root, layout)


def fake_workflow(tmp_path, monkeypatch):
    import json
    from collections import Counter

    from self_improving.sim_adapters.genesis import build_official_index as official
    from self_improving.sim_adapters.genesis import scene_physics_workflow as workflow
    from self_improving.sim_adapters.genesis.tests.workflow_fixtures import package

    root = package(tmp_path/'scene', scene())
    calls = Counter()
    fail = {'half_dt': False}

    def stabilize(source, out, sdf):
        calls['stabilization'] += 1
        out.mkdir()
        layout = workflow.imported.verify(source)
        states = {o['object_id']: dict(position=o['translation_m'],
                  orientation_wxyz=o['orientation_wxyz']) for o in layout['objects']}
        official.write_json(out/'final_state.json', dict(objects=states))
        official.write_json(out/'report.json', dict(status='candidate_ready'))
        return {'status': 'candidate_ready'}

    def replay(source, out, *, profile, **kwargs):
        calls[profile] += 1
        out.mkdir()
        if fail.get(profile, False):
            return dict(status='error', error='simulated interruption')
        result = dict(status='complete', physics_status='passed')
        official.write_json(out/'physics_result.json', result)
        official.write_json(out/'final_state.json', dict(step=0, objects={
            'a': dict(position=[0, 0, .5], orientation_wxyz=[1, 0, 0, 0])}))
        return result

    monkeypatch.setattr(workflow.settle, 'run', stabilize)
    monkeypatch.setattr(workflow.settle, 'verify',
                        lambda p: json.loads((p/'report.json').read_text()))
    monkeypatch.setattr(workflow.free, 'run', replay)
    monkeypatch.setattr(workflow.free, 'verify_evidence',
                        lambda p: json.loads((p/'physics_result.json').read_text()))
    return workflow, root, calls, fail


def test_resume_reuses_only_complete_matching_stages(tmp_path, monkeypatch):
    workflow, source, calls, fail = fake_workflow(tmp_path, monkeypatch)
    out = tmp_path/'physics'
    fail['half_dt'] = True
    assert workflow.run(source, out)['status'] == 'error'
    fail['half_dt'] = False
    result = workflow.run(source, out, resume=True)
    assert result['physics_status'] == 'passed'
    assert calls == {'stabilization': 1, 'baseline': 1, 'half_dt': 2}
    assert result['stages']['position']['reused']
    (out/'baseline/extra').write_text('cache corruption')
    workflow.run(source, out, resume=True)
    assert calls == {'stabilization': 1, 'baseline': 2, 'half_dt': 2}
    result = workflow.run(source, out, resume=True, support_sdf_cell_size=.0015,
                          support_sdf_max_res=384)
    assert result['stages']['position']['reused']
    assert calls == {'stabilization': 2, 'baseline': 3, 'half_dt': 3}


def test_support_scene_changes_invalidate_position_and_downstream(tmp_path, monkeypatch):
    import json

    from self_improving.sim_adapters.genesis import build_official_index as official

    workflow, source, calls, _ = fake_workflow(tmp_path, monkeypatch)
    out = tmp_path/'physics'
    workflow.run(source, out)
    # Add a bound source note: asset dependency identity changes even if poses match.
    (source/'note.json').write_text('{}')
    manifest = json.loads((source/'manifest.json').read_text())
    manifest['files'].append(official.fingerprint(source/'note.json', source))
    official.write_json(source/'manifest.json', manifest)
    result = workflow.run(source, out, resume=True)
    assert not result['stages']['position']['reused']
    assert calls == {'stabilization': 2, 'baseline': 2, 'half_dt': 2}


def test_graph_and_layout_mismatch_rejected_before_simulation(tmp_path, monkeypatch):
    from self_improving.sim_adapters.genesis import build_official_index as official
    from self_improving.sim_adapters.genesis import clip_select as clip

    workflow, source, calls, _ = fake_workflow(tmp_path, monkeypatch)
    document = workflow.lib.read_json(source/'scene_graph.json')
    document['edges'] = []
    official.write_json(source/'scene_graph.json', document)
    layout = workflow.lib.read_json(source/'scene_layout.json')
    layout['scene_graph_sha256'] = clip.digest(document)
    official.write_json(source/'scene_layout.json', layout)
    manifest = workflow.lib.read_json(source/'manifest.json')
    manifest['files'] = [official.fingerprint(source/f['path'], source) for f in manifest['files']]
    official.write_json(source/'manifest.json', manifest)
    result = workflow.run(source, tmp_path/'physics')
    assert result['failed_stage'] == 'graph'
    assert not calls


def test_workflow_render_gate_precedes_any_rendering(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from self_improving.sim_adapters.genesis import build_official_index as official
    from self_improving.sim_adapters.genesis import reconstruct_media as media

    physical = tmp_path/'03_physics'
    physical.mkdir()
    report = physical/'workflow_report.json'
    official.write_json(report, dict(schema_version='genenv.scene_physics_workflow.v1',
                                     physics_status='failed'))
    official.write_json(physical/'workflow_input.json', {})
    official.write_json(physical/'validated_scene.json', dict(
        workflow_report=official.fingerprint(report, tmp_path)))
    task = SimpleNamespace(root=tmp_path, stage=lambda _: physical)
    monkeypatch.setattr(media.scene_import, 'preview',
                        lambda *a, **kw: pytest.fail('rendering must not run'))
    with pytest.raises(ValueError, match='both free replays'):
        media.render_validated(task)


def test_physics_code_change_keeps_media_upstream_checkpoints(tmp_path):
    from types import SimpleNamespace

    from self_improving.sim_adapters.genesis import media_checkpoints as checkpoints

    task = SimpleNamespace(root=tmp_path)
    directory = tmp_path/'objects'
    directory.mkdir()
    (directory/'asset').write_text('unchanged')
    checkpoints.save(task, 'objects', directory,
                     {'model': 'same', 'physics_implementation_sha256': 'old'})
    assert checkpoints.read(task, 'objects', directory,
                            {'model': 'same', 'physics_implementation_sha256': 'new'})
    with pytest.raises(ValueError):
        checkpoints.read(task, 'objects', directory,
                         {'model': 'different', 'physics_implementation_sha256': 'new'})
