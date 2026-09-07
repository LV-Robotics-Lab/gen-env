"""Offline contracts for the bounded multi-object initial-scene preview."""
import json

import numpy as np
import pytest
import test_clip_select

from scene_gen.schema import RelationSpec, SceneObjectSpec, SceneSpec, SceneSpecError
from self_improving.sim_adapters.genesis import scene_preview as preview

setup = test_clip_select.setup


def spec():
    return SceneSpec(scene_id='preview_test', request='桌上放着一个苹果和一个黄色杯子。',
                     language='zh', objects=(SceneObjectSpec(object_id='apple_1', category='apple'),
                     SceneObjectSpec(object_id='cup_1', category='cup', color='yellow')),
                     relations=(RelationSpec(relation='on_table', source='apple_1', target='table'),
                     RelationSpec(relation='on_table', source='cup_1', target='table')))


def bindings(scene):
    return [dict(object_id=o.object_id, local_visual_bounds_m=[[-.03, -.04, -.06],
                                                              [.05, .04, .06]])
            for o in scene.objects]


def test_queries_preserve_typed_attributes():
    obj = SceneObjectSpec(object_id='bowl_1', category='bowl', color='orange', material='plastic')
    assert preview.object_query(obj) == '橙色塑料碗'
    assert preview.object_query(spec().objects[1]) == '黄色杯子'


def test_layout_original_scale_full_footprints_and_no_physics_claim():
    scene = spec()
    inputs = bindings(scene)
    layout = preview.row_layout(scene, inputs)
    assert layout['physics_status'] == 'not_evaluated' and layout['physics_steps'] == 0
    boxes = [np.array(o['world_visual_bounds_m']) for o in layout['objects']]
    for obj, original, box in zip(layout['objects'], inputs, boxes, strict=True):
        assert obj['scale'] == 1 and obj['intended_dynamic']
        assert np.allclose(box, np.array(original['local_visual_bounds_m'])+obj['translation_m'])
        assert box[0, 2] == pytest.approx(scene.workspace.table_height_m)
        assert box[0, 0] >= scene.workspace.x_bounds_m[0]
        assert box[1, 0] <= scene.workspace.x_bounds_m[1]
        assert box[0, 1] > scene.workspace.robot_keepout_y_m[1]
    assert boxes[1][0, 0] - boxes[0][1, 0] >= .06-1e-12
    assert preview.row_layout(scene, inputs) == layout


@pytest.mark.parametrize('bad', [
    [[0, 0, 0], [2, 2, 2]], [[0, 0, 0], [0, 1, 1]], [[0, 0, 0], [float('nan'), 1, 1]],
])
def test_unfit_or_invalid_geometry_is_not_rescaled(bad):
    scene = spec()
    inputs = bindings(scene)
    inputs[0]['local_visual_bounds_m'] = bad
    with pytest.raises(ValueError):
        preview.row_layout(scene, inputs)


def test_binding_order_mismatch_rejected():
    scene = spec()
    with pytest.raises(ValueError, match='binding set/order'):
        preview.row_layout(scene, list(reversed(bindings(scene))))


@pytest.mark.parametrize('change', ['nested', 'lateral', 'region', 'articulation'])
def test_unsupported_constraints_not_silently_ignored(change):
    scene = spec()
    data = scene.model_dump(mode='json')
    if change == 'nested':
        data['relations'][0] = dict(relation='on_top_of', source='apple_1', target='cup_1')
    elif change == 'lateral':
        data['relations'].append(dict(relation='left_of', source='apple_1', target='cup_1'))
    elif change == 'region':
        data['objects'][0]['region'] = 'left'
    else:
        data['objects'][1]['articulation'] = dict(state='open', open_fraction=1)
        # Cup articulation is also rejected by the unchanged SceneSpec contract.
    with pytest.raises((ValueError, SceneSpecError)):
        changed = SceneSpec.model_validate(data)
        preview.check_scope(changed)


def test_verified_selected_binding_uses_index_geometry(setup):
    s = setup
    obj = spec().objects[1]
    directory = s.root/'query'
    report = preview.clip.select(output_dir=directory, **(s.kwargs | {'query': '黄色杯子'}))
    assert report['status'] == 'selected'
    value = preview.trusted_binding(spec(), obj, directory, s.index/'index.json')
    assert value['local_visual_bounds_m'] == [[0]*3, [1]*3]
    assert value['model_entrypoint'].endswith('/model.xml')
    assert value['visible_differences'] == ['未确认米老鼠印花']
    source = s.assets/value['source_files'][0]['path']
    source.write_bytes(source.read_bytes()+b' ')
    with pytest.raises(ValueError):
        preview.trusted_binding(spec(), obj, directory, s.index/'index.json')


def test_rehashed_selection_cannot_invent_model_path(setup):
    s = setup
    directory = s.root/'query'
    preview.clip.select(output_dir=directory, **(s.kwargs | {'query': '黄色杯子'}))
    binding = json.loads((directory/'selected_asset.json').read_text())
    binding['model_entrypoint'] = '/tmp/invented.obj'
    preview.clip.write_json(directory/'selected_asset.json', binding)
    report = json.loads((directory/'run_report.json').read_text())
    report['files'] = [preview.official.fingerprint(directory/r['path'], directory)
                       for r in report['files']]
    preview.clip.write_json(directory/'run_report.json', report)
    with pytest.raises(ValueError, match='source binding'):
        preview.trusted_binding(spec(), spec().objects[1], directory, s.index/'index.json')


def test_output_directory_protection(tmp_path):
    existing = tmp_path/'old'
    existing.mkdir()
    (existing/'keep').write_text('keep')
    with pytest.raises(FileExistsError):
        preview.run('桌上有苹果', existing, clip_index=tmp_path/'clip/index.json', vlm_config=None)
    assert (existing/'keep').read_text() == 'keep'
    (existing/'asset_index.json').write_text('{}')
    with pytest.raises(ValueError, match='official_package'):
        preview.run('桌上有苹果', existing/'nested', clip_index=tmp_path/'clip/index.json',
                    vlm_config=None)
    assert not (existing/'nested').exists()

