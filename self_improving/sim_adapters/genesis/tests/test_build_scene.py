"""Measured visual support and stage routing, without a simulator."""

import numpy as np
import pytest
import test_clip_select
from test_asset_extraction import QUERY, four_objects, provider

from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import extract_assets as entry
from self_improving.sim_adapters.genesis.task_output import TaskOutput

setup = test_clip_select.setup


def table_mesh():
    vertices = np.array([[-.6, -.35, .75], [.6, -.35, .75], [.6, .35, .75], [-.6, .35, .75],
                         [-.6, -.35, 0]])
    return vertices, np.array([[0, 1, 2], [0, 2, 3]])


def document():
    return dict(objects=four_objects(), relations=[dict(relation='on', source=o['object_id'],
                target='table_1', evidence=QUERY) for o in four_objects()[1:]])


def geometry():
    return {o['object_id']: (dict(bounds=[[-.6, -.35, 0], [.6, .35, .75]],
                                  surface=builder.support_surface(*table_mesh()))
                            if o['category'] == 'table' else
                            dict(bounds=[[-.03, -.03, -.04], [.03, .03, .04]]))
            for o in four_objects()}


def test_real_plane_triangles_define_surface_and_complete_footprint():
    surface = builder.support_surface(*table_mesh())
    assert surface['z_m'] == .75
    assert surface['area_m2'] == pytest.approx(1.2*.7)
    assert builder.fits_surface(surface, [[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])
    assert not builder.fits_surface(surface, [[-.7, -.1], [.1, -.1], [.1, .1], [-.7, .1]])


def test_holes_disconnected_slanted_and_overlapping_surfaces_rejected():
    vertices, faces = table_mesh()
    with pytest.raises(ValueError, match='overlapping'):
        builder.support_surface(vertices, faces[:1].tolist()+[[1, 2, 3]])
    slanted = vertices.copy()
    slanted[0, 2] += .02
    with pytest.raises(ValueError, match='horizontal'):
        builder.support_surface(slanted, faces)
    # Disconnected triangles leave a gap in the apparent AABB; cannot accept that as tabletop.
    triangles = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1],
                          [2, 0, 1], [3, 0, 1], [3, 1, 1]])
    with pytest.raises(ValueError, match='solid convex'):
        builder.support_surface(triangles, [[0, 1, 2], [3, 4, 5]])


def test_on_actual_table_root_on_ground_no_proxy_or_scaling():
    doc = document()
    bindings = {o['object_id']: dict(asset_id=o['object_id']) for o in doc['objects']}
    layout = builder.layout_objects(doc, bindings, geometry())
    by_id = {o['object_id']: o for o in layout['objects']}
    assert len(by_id) == 4
    assert by_id['table_1']['support'] == 'ground'
    assert by_id['table_1']['world_visual_bounds_m'][0][2] == 0
    for name in ('apple_1', 'cup_1', 'bowl_1'):
        assert by_id[name]['support'] == 'table_1'
        assert by_id[name]['world_visual_bounds_m'][0][2] == pytest.approx(.75)
        assert by_id[name]['scale'] == 1
    assert 'table' not in layout and layout['physics_steps'] == 0


def test_ground_only_does_not_invent_a_table_and_oversize_fails():
    doc = document()
    doc['objects'] = doc['objects'][1:]
    doc['relations'] = []
    bindings = {o['object_id']: {} for o in doc['objects']}
    geom = geometry()
    geom.pop('table_1')
    layout = builder.layout_objects(doc, bindings, geom)
    assert all(o['support'] == 'ground' for o in layout['objects'])
    doc = document()
    geom = geometry()
    geom['bowl_1']['bounds'] = [[-1, -1, 0], [1, 1, 1]]
    with pytest.raises(ValueError, match='footprint'):
        builder.layout_objects(doc, {o['object_id']: {} for o in doc['objects']}, geom)


@pytest.mark.parametrize('change', ['inside', 'left_of', 'cycle', 'unbound'])
def test_unsupported_relations_not_silently_dropped(change):
    doc = document()
    if change in {'inside', 'left_of'}:
        doc['relations'][0]['relation'] = change
    elif change == 'cycle':
        doc['relations'].append(dict(relation='on', source='table_1', target='apple_1'))
    else:
        doc['relations'][0]['target'] = 'missing'
    with pytest.raises(ValueError):
        builder.support_graph(doc)


def prepared(s, monkeypatch):
    monkeypatch.setattr(entry, 'load_llm_provider_config', lambda _: s.config)
    p, _ = provider(s.root, four_objects(), document()['relations'])
    def select(index, query, output, **kwargs):
        return entry.clip.select(index, query, output, **kwargs, encoder=s.encoder,
                                 weights_dir=s.kwargs['weights_dir'], cache_dir=s.root/'cache',
                                 vlm=s.kwargs['vlm'])
    result = entry.run(QUERY, s.root/'task', clip_index=s.index/'index.json', vlm_config=None,
                       provider=p, selector=select, stop_after='assets')
    assert result['status'] == 'assets_selected'
    return TaskOutput(result['output_dir'])


def test_scene_routing_preserves_selection_and_clears_downstream(setup, monkeypatch):
    s = setup
    task = prepared(s, monkeypatch)
    before = {p: p.read_bytes() for p in task.stage('objects').rglob('*') if p.is_file()}
    def renderer(doc, bindings, bounds, output, check, **kwargs):
        check()
        assert len(doc['objects']) == len(bindings) == 4
        entry.clip.write_json(output/'scene_layout.json', {'test': True})
        (output/'overview.png').write_bytes(b'offline test fixture')
        return dict(status='passed')
    result = builder.run(task.root, s.index/'index.json', renderer=renderer, planner='rule')
    assert result['status'] == 'scene_built'
    task.verify()
    assert task.report['stages']['objects'] == task.report['stages']['scene'] == 'passed'
    for name in ('physics', 'final_render'):
        (task.stage(name)/'old_success.json').write_text('{}')
    task.report['stages'].update(physics='passed', final_render='passed')
    task.seal()
    def failure(*args, **kwargs):
        raise ValueError('offline scene failure')
    failed = builder.run(task.root, s.index/'index.json', renderer=failure, planner='rule')
    assert failed['status'] == 'error'
    task.verify()
    assert task.report['stages']['objects'] == 'passed'
    assert task.report['stages']['scene'] == 'failed'
    assert not (task.stage('scene')/'overview.png').exists()
    assert not list(task.stage('physics').iterdir())
    assert not list(task.stage('final_render').iterdir())
    assert all(p.read_bytes() == data for p, data in before.items())


def test_upstream_tamper_during_render_cannot_report_success(setup, monkeypatch):
    s = setup
    task = prepared(s, monkeypatch)
    def bad_renderer(doc, bindings, bounds, output, check, **kwargs):
        (task.stage('objects')/'apple_1.json').write_text('{}')
        return dict(status='passed')
    failed = builder.run(task.root, s.index/'index.json', renderer=bad_renderer, planner='rule')
    assert failed['status'] == 'error'
    task.verify()
    assert task.report['stages']['scene'] == 'failed'


def test_shallow_edge_trim_does_not_replace_the_real_tabletop():
    vertices, faces = table_mesh()
    trim = np.array([[-.6, -.35, .751], [-.59, -.35, .751], [-.59, .35, .751],
                     [-.6, .35, .751]])
    all_vertices = np.concatenate([vertices, trim])
    all_faces = np.concatenate([faces, np.array([[5, 6, 7], [5, 7, 8]])])
    # Two small disjoint strips make the highest plane unsuitable, as on the real official table.
    trim2 = trim.copy()
    trim2[:, 0] += 1.19
    all_vertices = np.concatenate([all_vertices, trim2])
    all_faces = np.concatenate([all_faces, np.array([[9, 10, 11], [9, 11, 12]])])
    surface = builder.support_surface(all_vertices, all_faces)
    assert surface['z_m'] == .75 and surface['recess_m'] == pytest.approx(.001)
    assert builder.fits_surface(surface, [[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])


def test_above_surface_obstacle_and_deep_container_are_not_support():
    surface = builder.support_surface(*table_mesh())
    surface['above_triangles_xy_m'] = [[[-.1, -.1], [.1, -.1], [0, .1]]]
    assert not builder.fits_surface(surface, [[-.2, -.2], [.2, -.2], [.2, .2], [-.2, .2]])
    vertices, faces = table_mesh()
    vertices = np.concatenate([vertices, [[0, 0, .8]]])
    with pytest.raises(ValueError, match='horizontal'):
        builder.support_surface(vertices, faces)
