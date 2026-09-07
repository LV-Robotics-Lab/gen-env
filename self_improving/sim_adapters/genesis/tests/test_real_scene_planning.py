"""Opt-in real Genesis geometry/render checks; fixed offline proposal, NO network calls."""
import json
import os
from pathlib import Path

import pytest

from scene_gen.llm_provider import LLMProviderConfig
from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import build_scene as builder
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis.scene_planning import ScenePlanner

pytestmark = pytest.mark.skipif(os.environ.get('GENESIS_SCENE_REAL') != '1',
                                reason='opt-in real native asset geometry and rendering')
INDEX = Path('assets/genesis/clip_non_robot_v1/index.json')


@pytest.mark.parametrize('stack', [False, True], ids=['four_assets', 'three_levels'])
def test_native_geometry_planned_layout_and_views(tmp_path, stack):
    root = Path(os.environ.get('GENESIS_SCENE_OUTPUT', tmp_path))
    output = root/('three_levels' if stack else 'four_assets')
    output.mkdir(parents=True, exist_ok=False)
    index, _ = clip.load_index(INDEX)
    index_hash = library.sha256(INDEX)
    chosen = [('table_1', 'dex_table_d3996872'), ('apple_1', 'apple_15')]
    if stack:
        chosen.insert(1, ('microwave_1', 'microwave_microwave_59704527'))
    else:
        chosen += [('cup_1', 'cup_2'), ('bowl_1', 'glb_orange_plastic_bowl_97f9d352')]
    document = dict(request='桌上有微波炉，微波炉上有苹果。' if stack else
                    '桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。',
                    objects=[dict(object_id=n, category=n.split('_')[0], description=n)
                             for n, _ in chosen], relations=[])
    for n, _ in chosen[1:]:
        document['relations'].append(dict(relation='on', source=n,
                target='microwave_1' if stack and n == 'apple_1' else 'table_1',
                evidence=document['request']))
    bindings, bounds = {}, {}
    preview_root = Path(index['official_index']['path']).parent
    for name, asset_id in chosen:
        asset = next(a for a in index['assets'] if a['asset_id'] == asset_id)
        source_root = Path(asset['source_root'])
        official.verify_files(source_root, asset['source_files'])
        bindings[name] = dict(asset_id=asset_id, source_root=str(source_root),
                              model_entrypoint=str(official.safe_file(source_root,
                                                                     asset['entrypoint'])),
                              source_files=asset['source_files'])
        record = library.read_json(preview_root/asset['record'])
        preview = library.read_json(preview_root/record['preview_result'])
        bounds[name] = preview.get('loaded_bounds',
                                   preview.get('geometry', {}).get('loaded_bounds_m'))
    def check():
        assert library.sha256(INDEX) == index_hash
        for binding in bindings.values():
            official.verify_files(Path(binding['source_root']), binding['source_files'])
    def send(system, user):
        payload = json.loads(user)
        return json.dumps(dict(object_ids=payload['object_ids'],
                               relations=payload['explicit_relations'], preferences=[]))
    planner = ScenePlanner(LLMProviderConfig(endpoint='https://example.invalid/v1', model='offline',
                           api_key='OFFLINE_TEST_SECRET'), transport_fn=send,
                           cache_dir=output/'planning_cache')
    def plan(geometry):
        return planner.plan(document, bindings, geometry, 42, check, output)
    report = builder.render_scene(document, bindings, bounds, output, check, plan_layout=plan)
    assert report['status'] == 'passed' and report['physics_steps'] == 0
    assert len(report['views']) == 3
    for name, _ in chosen:
        assert max(v['visible_pixels'][name] for v in report['views']) >= 64
    scene = library.read_json(output/'scene_layout.json')
    assert scene['physics_steps'] == 0 and all(o['scale'] == 1 for o in scene['objects'])
    if stack:
        apple = next(o for o in scene['objects'] if o['object_id'] == 'apple_1')
        assert apple['support'] == 'microwave_1'
    clip.write_json(output/'acceptance.json', dict(status='passed', real_genesis=True,
                    live_model_calls=0, planning_transport='fixed_offline_proposal',
                    source='verified existing official asset index; no new asset selection',
                    physics_steps=0, files=[official.fingerprint(p, output)
                    for p in sorted(output.rglob('*')) if p.is_file()]))
