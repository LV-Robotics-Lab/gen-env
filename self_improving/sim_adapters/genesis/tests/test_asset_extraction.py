"""Complete asset extraction and selection, without live models or Genesis."""
import copy
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_clip_select

from self_improving.sim_adapters.genesis import asset_extraction as parsing
from self_improving.sim_adapters.genesis import extract_assets as entry
from self_improving.sim_adapters.genesis import scene_preview
from self_improving.sim_adapters.genesis.task_output import TaskOutput

setup = test_clip_select.setup
QUERY = '桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。'


def obj(category, mention, description=None, *, number=1, attributes=()):
    return dict(object_id=f'{category}_{number}', category=category,
                description=description or parsing.quantity(mention)[1],
                attributes=list(attributes), mentions=[mention])


def four_objects():
    return [obj('table', '桌', '桌子'), obj('apple', '一个苹果'),
            obj('cup', '一个黄色杯子', attributes=['黄色']),
            obj('bowl', '一个橙色塑料碗', attributes=['橙色', '塑料'])]


def config():
    return entry.clip.LLMProviderConfig(endpoint='https://example.invalid/v1', model='gpt-4o',
                                        api_key='TEST_SECRET_NEVER_LOG')


def provider(tmp_path, objects, relations=(), *, ambiguity_stage=None, cfg=None):
    calls = []

    def send(system, user):
        stage = 'objects' if 'stage 1' in system else 'relations'
        calls.append((stage, json.loads(user)))
        return json.dumps({stage: objects if stage == 'objects' else list(relations),
                           'ambiguities': ['指代不明确'] if stage == ambiguity_stage else []},
                          ensure_ascii=False)

    p = parsing.AssetProvider(cfg or config(), transport_fn=send, cache_dir=tmp_path/'parse')
    return p, calls


def test_four_objects_with_table_and_no_default_relations(tmp_path):
    rows = [dict(relation='on', source=o['object_id'], target='table_1', evidence=QUERY)
            for o in four_objects()[1:]]
    p, calls = provider(tmp_path, four_objects(), rows)
    result = p.extract(QUERY)
    assert len(result['objects']) == 4 and result['objects'][0]['category'] == 'table'
    assert result['relations'] == rows and len(calls) == 2
    assert 'workspace' not in result and result['environment']['ground'] == 'genesis_builtin'


def test_no_table_is_invented(tmp_path):
    objects = [obj('apple', '一个苹果'), obj('cup', '一个黄色杯子')]
    p, _ = provider(tmp_path, objects)
    data = p.extract('一个苹果和一个黄色杯子。')
    assert len(data['objects']) == 2 and data['relations'] == []
    bad = dict(objects=objects+[obj('table', '桌子')], ambiguities=[])
    with pytest.raises(ValueError, match='substrings'):
        parsing.clean_objects(bad, data['request'])


def test_missing_support_and_narrowed_mentions_are_rejected():
    with pytest.raises(ValueError, match='omitted'):
        parsing.clean_objects(dict(objects=four_objects()[1:], ambiguities=[]), QUERY)
    objects = four_objects()
    objects[2] = obj('cup', '杯子')
    with pytest.raises(ValueError, match='attributes'):
        parsing.clean_objects(dict(objects=objects, ambiguities=[]), QUERY)


def test_cabinet_container_and_resolved_reference(tmp_path):
    query = '一个柜子里有一个杯子，它是黄色的。'
    objects = [obj('cabinet', '一个柜子'),
               obj('cup', '一个杯子', '黄色的杯子', attributes=['黄色'])]
    objects[1]['mentions'].append('它是黄色的')
    p, _ = provider(tmp_path, objects, [dict(relation='inside', source='cup_1',
                                           target='cabinet_1', evidence='一个柜子里有一个杯子')])
    data = p.extract(query)
    assert data['relations'][0]['target'] == 'cabinet_1'
    assert data['objects'][1]['description'] == '黄色的杯子'


def test_repeated_quantity_and_handle_print_not_separate_assets(tmp_path):
    phrase = '两个印有米老鼠图案且带把手的黄色杯子'
    objects = [obj('cup', phrase, number=i,
                   attributes=['米老鼠图案', '把手', '黄色']) for i in (1, 2)]
    p, _ = provider(tmp_path, objects)
    data = p.extract(phrase+'。')
    assert len(data['objects']) == 2
    assert all(o['description'] == '印有米老鼠图案且带把手的黄色杯子' for o in data['objects'])
    with pytest.raises(ValueError, match='quantity'):
        parsing.clean_objects(dict(objects=objects[:1], ambiguities=[]), phrase+'。')
    stripped = copy.deepcopy(objects)
    for o in stripped:
        o.update(description='杯子', attributes=[], mentions=['杯子'])
    with pytest.raises(ValueError):
        parsing.clean_objects(dict(objects=stripped, ambiguities=[]), phrase+'。')


def test_apple_print_is_not_an_independent_apple(tmp_path):
    p, _ = provider(tmp_path, [obj('cup', '一个印有苹果图案的杯子')])
    assert len(p.extract('一个印有苹果图案的杯子。')['objects']) == 1


@pytest.mark.parametrize('stage', ['objects', 'relations'])
def test_ambiguity_stops_selection_keeps_evidence(tmp_path, monkeypatch, stage):
    p, calls = provider(tmp_path, four_objects(), ambiguity_stage=stage)
    monkeypatch.setattr(entry, 'load_llm_provider_config', lambda _: config())
    report = entry.run(QUERY, tmp_path/'task', clip_index=tmp_path/'clip/index.json',
                       vlm_config=None, provider=p,
                       selector=lambda *a, **kw: pytest.fail('selected ambiguous request'))
    assert report['status'] == 'error' and len(calls) == (1 if stage == 'objects' else 2)
    evidence_file = tmp_path/'task/01_obj/parse_records/extraction_evidence.json'
    evidence = json.loads(evidence_file.read_text())
    assert evidence['ambiguities'] == ['指代不明确']
    assert not list((tmp_path/'parse').glob('*.json'))
    TaskOutput(tmp_path/'task').verify()


@pytest.mark.parametrize('change', ['extra', 'path', 'unknown_id', 'self', 'implicit_relation'])
def test_malformed_model_outputs(tmp_path, change):
    objects = four_objects()
    if change == 'extra':
        objects[0]['model_entrypoint'] = '/tmp/evil.xml'
    if change == 'path':
        objects[0]['description'] = '/tmp/evil.xml'
    if change in {'extra', 'path'}:
        with pytest.raises(ValueError):
            parsing.clean_objects(dict(objects=objects, ambiguities=[]), QUERY)
        return
    row = dict(relation='on', source='apple_1', target='table_1', evidence=QUERY)
    if change == 'unknown_id':
        row['target'] = 'table'
    if change == 'self':
        row['target'] = 'apple_1'
    if change == 'implicit_relation':
        row['evidence'] = '一个苹果、一个黄色杯子'
    with pytest.raises(ValueError):
        parsing.clean_relations(dict(relations=[row], ambiguities=[]), QUERY, objects)


def test_cache_hit_version_config_and_prompt_invalidation(tmp_path, monkeypatch):
    p, calls = provider(tmp_path, four_objects())
    original = p.extract(QUERY)
    assert p.extract(QUERY) == original and p.evidence()['calls'] == 0 and len(calls) == 2
    p.prompts['objects'] += '\nPrompt version changed.'
    p.extract(QUERY)
    assert len(calls) == 4
    p.config = replace(p.config, temperature=0.2)
    p.extract(QUERY)
    assert len(calls) == 6
    monkeypatch.setattr(parsing, 'VERSION', 'new_schema_version')
    p.extract(QUERY)
    assert len(calls) == 8
    assert len(list((tmp_path/'parse').glob('*.json'))) == 4
    assert 'scene_gen.llm_extractor' not in json.dumps(p.evidence())


def test_cache_tampering_and_invalid_json_do_not_get_cached(tmp_path):
    p, _ = provider(tmp_path, four_objects())
    p.extract(QUERY)
    cache = next((tmp_path/'parse').glob('*.json'))
    data = json.loads(cache.read_text())
    data['payload']['objects']['objects'].pop()
    cache.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='integrity'):
        p.extract(QUERY)
    broken = parsing.AssetProvider(config(), cache_dir=tmp_path/'broken',
                                    transport_fn=lambda *_: '{invalid')
    with pytest.raises(ValueError):
        broken.extract(QUERY)
    assert not list((tmp_path/'broken').glob('*.json'))


def test_secret_transport_error_and_timeout_leave_safe_evidence(tmp_path):
    p = parsing.AssetProvider(config(), cache_dir=tmp_path/'parse',
                              transport_fn=lambda *_: config().api_key)
    with pytest.raises(ValueError):
        p.extract(QUERY)
    assert config().api_key not in json.dumps(p.evidence())
    def fail(*args):
        raise TimeoutError(config().api_key)
    p.send = fail
    with pytest.raises(TimeoutError):
        p.extract(QUERY)
    assert config().api_key not in json.dumps(p.evidence())


def test_selection_all_objects_cache_and_no_scene_calls(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(entry, 'load_llm_provider_config', lambda _: s.config)
    p, _ = provider(s.root, four_objects(), cfg=s.config)
    monkeypatch.setitem(sys.modules, 'genesis', SimpleNamespace(
        Scene=lambda *a, **kw: pytest.fail('must not create a scene')))
    monkeypatch.setattr(scene_preview, 'render', lambda *a: pytest.fail('must not render'))
    queries = []
    def selector(index, query, output, **kwargs):
        queries.append(query)
        return entry.clip.select(index, query, output, **kwargs, encoder=s.encoder,
                                 weights_dir=s.kwargs['weights_dir'],
                                 cache_dir=s.root/'selection_cache',
                                 vlm=s.kwargs['vlm'])
    opts = dict(stop_after='assets', output_root=s.root/'output',
                clip_index=s.index/'index.json', vlm_config=None,
                provider=p, selector=selector)
    report = scene_preview.run(QUERY, **opts)
    assert report['status'] == 'assets_selected', report
    assert queries == [o['description'] for o in four_objects()]
    task = TaskOutput(report['output_dir'])
    task.verify()
    assert len(list(task.stage('objects').glob('*_1.json'))) == 4
    assert (task.stage('objects')/'asset_request.json').is_file()
    assert not (task.stage('objects')/'scene_spec.json').exists()
    for name in ('scene', 'physics', 'final_render'):
        assert task.report['stages'][name] == 'not_run' and not list(task.stage(name).iterdir())
    again = entry.run(QUERY, **opts)
    assert again['status'] == 'assets_selected' and again['parse_calls'] == 0
    assert all(i['vlm_calls'] == 0 for i in again['selections'])
    # A failed same-query rerun clears the previous success and any legacy images.
    (task.stage('scene')/'old.png').write_bytes(b'old image')
    task.seal()
    ambiguous, _ = provider(s.root/'new', four_objects(), ambiguity_stage='objects')
    failed = entry.run(QUERY, **(opts | {'provider': ambiguous}))
    assert failed['status'] == 'error'
    task.verify()
    assert not list(task.stage('scene').iterdir())
    assert not list(task.stage('objects').glob('*_1.json'))


def test_rejection_and_exception_preserve_all_objects_and_continue(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(entry, 'load_llm_provider_config', lambda _: s.config)
    p, _ = provider(s.root, four_objects())
    calls = []
    def selector(index, query, output, **kwargs):
        calls.append(query)
        if len(calls) == 2:
            raise TimeoutError('offline timeout')
        return entry.clip.select(index, query, output, **kwargs, encoder=s.encoder,
                                 weights_dir=s.kwargs['weights_dir'],
                                 cache_dir=s.root/'selection_cache',
                                 vlm=lambda _: test_clip_select.response(status='rejected',
                                                                         number=None))
    report = entry.run(QUERY, s.root/'task', clip_index=s.index/'index.json', vlm_config=None,
                       provider=p, selector=selector)
    assert report['status'] == 'partial' and len(calls) == 4
    assert [i['status'] for i in report['selections']] == [
        'rejected', 'error', 'rejected', 'rejected']
    task = TaskOutput(report['output_dir'])
    task.verify()
    assert len(list(task.stage('objects').glob('*_1.json'))) == 4
    assert not list(task.root.rglob('selected_asset.json'))


def test_coordinated_relation_cannot_rewrite_its_quote():
    row = dict(relation='on', source='cup_1', target='table_1', evidence='桌上放着一个黄色杯子')
    with pytest.raises(ValueError, match='exact request substring'):
        parsing.clean_relations(dict(relations=[row], ambiguities=[]), QUERY, four_objects())
    row['evidence'] = QUERY
    assert parsing.clean_relations(dict(relations=[row], ambiguities=[]), QUERY, four_objects())


def test_rehashed_fake_binding_path_never_survives_as_success(setup, monkeypatch):
    s = setup
    monkeypatch.setattr(entry, 'load_llm_provider_config', lambda _: s.config)
    p, _ = provider(s.root, [obj('apple', '一个苹果')])
    def selector(index, query, output, **kwargs):
        report = entry.clip.select(index, query, output, **kwargs, encoder=s.encoder,
                                   weights_dir=s.kwargs['weights_dir'], cache_dir=s.root/'cache',
                                   vlm=s.kwargs['vlm'])
        selected = output/'selected_asset.json'
        data = entry.library.read_json(selected)
        data['model_entrypoint'] = '/tmp/invented.xml'
        entry.clip.write_json(selected, data)
        report['files'] = [entry.official.fingerprint(output/r['path'], output)
                           for r in report['files']]
        entry.clip.write_json(output/'run_report.json', report)
        return report
    report = entry.run('一个苹果。', s.root/'task', clip_index=s.index/'index.json',
                       vlm_config=None, provider=p, selector=selector)
    assert report['status'] == 'partial' and report['selections'][0]['status'] == 'error'
    task = TaskOutput(report['output_dir'])
    task.verify()
    directory = task.stage('objects')/'asset_selection/apple_1'
    assert not (directory/'selected_asset.json').exists()
    saved = entry.library.read_json(directory/'run_report.json')
    assert saved['status'] == 'error'
    entry.official.verify_files(directory, saved['files'])
