"""Failure recovery and native response contracts; no model or GPU calls."""
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from self_improving.sim_adapters.genesis import media_checkpoints as cp
from self_improving.sim_adapters.genesis import reconstruct_media as media
from self_improving.sim_adapters.genesis.task_output import TaskOutput
from self_improving.sim_adapters.simfoundry import native_service as native


def test_checkpoint_rejects_changed_added_missing_files(tmp_path):
    task = SimpleNamespace(root=tmp_path)
    stage = tmp_path / 'stage'
    stage.mkdir()
    (stage / 'mesh').write_bytes(b'original')
    cp.save(task, 'objects', stage, {'model': 'a'}, {'ok': True})
    assert cp.read(task, 'objects', stage, {'model': 'a'})['result'] == {'ok': True}
    with pytest.raises(ValueError):
        cp.read(task, 'objects', stage, {'model': 'b'})
    (stage / 'mesh').write_bytes(b'changed')
    with pytest.raises(ValueError):
        cp.read(task, 'objects', stage, {'model': 'a'})
    (stage / 'mesh').write_bytes(b'original')
    (stage / 'extra').write_bytes(b'x')
    with pytest.raises(ValueError):
        cp.read(task, 'objects', stage, {'model': 'a'})
    (stage / 'extra').unlink()
    (stage / 'mesh').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        cp.read(task, 'objects', stage, {'model': 'a'})


def test_config_rotation_does_not_invalidate_but_model_endpoint_do(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('model: a\nendpoint: https://one\napi_key: private1\n')
    before = media.configuration_hash(path)
    path.write_text('model: a\nendpoint: https://one\napi_key: private2\n')
    assert media.configuration_hash(path) == before
    path.write_text('model: b\nendpoint: https://one\napi_key: private2\n')
    assert media.configuration_hash(path) != before
    path.write_text('model: a\nendpoint: https://two\napi_key: private2\n')
    assert media.configuration_hash(path) != before


def test_render_retry_does_not_invoke_reconstruction_or_physics(tmp_path, monkeypatch):
    source = tmp_path / 'mouse.jpg'
    Image.new('RGB', (4, 4)).save(source)
    index = tmp_path / 'index.json'
    index.write_text('{}')
    config = tmp_path / 'config.yaml'
    config.write_text('profiles: {}')
    task = TaskOutput(tmp_path / 'task')
    task.start('image:mouse.jpg')
    folder = task.stage('objects') / 'input'
    folder.mkdir()
    (folder / 'media_manifest.json').write_text(json.dumps({'effective_config':
        media.task_config('image', source, index, config, ())}))
    task.report.update(status='execution_failed', physics={
        'status': 'execution_failed', 'physics_status': 'passed', 'render_status': 'failed',
        'exit_code': 1})
    task.report['stages'].update(objects='passed', scene='passed', physics='passed',
                                 final_render='failed')
    effective = media.task_config('image', source, index, config, ())
    cp.save(task, 'objects', task.stage('objects'), effective)
    cp.save(task, 'scene', task.stage('scene'), effective)
    task.seal()
    render = Mock(side_effect=RuntimeError('renderer failed'))
    runner = Mock(side_effect=AssertionError('reconstruction must not run'))
    monkeypatch.setattr(media, 'render_validated', render)
    failed = media.run(source, 'image', task.root, index, config, resume=True,
                       process_runner=runner)
    assert failed['physics']['physics_status'] == 'passed'
    assert failed['physics']['render_status'] == 'failed'
    render.side_effect = None
    passed = media.run(source, 'image', task.root, index, config, resume=True,
                       process_runner=runner)
    assert passed['physics']['render_status'] == 'passed'
    assert passed['exit_code'] == 0
    runner.assert_not_called()
    assert render.call_count == 2
    task.verify()


def test_native_multimodal_mapping_and_no_external_image_urls():
    client = object.__new__(native.NativeClient)
    client.generate = Mock(return_value=[{'text': '{"status":"rejected"}'}])
    assert 'rejected' in client([{'role': 'system', 'content': 'policy'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'match'},
         {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}]}])
    parts = client.generate.call_args.args[0]
    assert parts[1]['inlineData'] == {'mimeType': 'image/png', 'data': 'eA=='}
    with pytest.raises(ValueError):
        client([{'role': 'user', 'content': [{'type': 'image_url',
                'image_url': {'url': 'https://untrusted/image.png'}}]}])


def test_probe_requires_decodable_image(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (8, 8)).save(source)
    fake = Mock()
    fake.generate.side_effect = [[{'text': 'OK'}], [{'text': 'mouse'}], [{'text': 'no image'}]]
    monkeypatch.setattr(native, 'NativeClient', lambda _: fake)
    report = native.probe('unused', source, tmp_path / 'failed')
    assert report['status'] == 'failed'
    assert report['probes'][-1]['error'] == 'gemini_missing_image'
    data = io.BytesIO()
    Image.new('RGB', (8, 8)).save(data, 'PNG')
    fake.generate.side_effect = [[{'text': 'OK'}], [{'text': 'mouse'}],
        [{'inlineData': {'data': base64.b64encode(data.getvalue()).decode()}}]]
    assert native.probe('unused', source, tmp_path / 'passed')['status'] == 'passed'


def test_actual_sampled_frames_reject_wrong_copy(tmp_path):
    all_frames = tmp_path / 's1_video/frames_all'
    sample = tmp_path / 's1_video/frames_subsampled_15'
    all_frames.mkdir(parents=True)
    sample.mkdir()
    for i in range(4):
        (all_frames / f'frame_{i:04d}.png').write_bytes(bytes([i]))
    for i in (0, 2):
        (sample / f'frame_{i:04d}.png').write_bytes(bytes([i]))
    manifest = dict(decoded_frame_count=4, sampled_frame_indices=[0, 2])
    assert media.verify_sampled_frames(tmp_path, manifest, 'video')['decoded_frame_count'] == 4
    (sample / 'frame_0002.png').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='bytes differ'):
        media.verify_sampled_frames(tmp_path, manifest, 'video')


def test_native_refusal_preserves_reason(monkeypatch):
    client = object.__new__(native.NativeClient)
    client.config = SimpleNamespace(api_key='private-fixture')
    envelope = {'candidates': [{'finishReason': 'SAFETY'}]}
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = json.dumps(envelope).encode()
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr(native.urllib.request, 'build_opener', lambda _: opener)
    with pytest.raises(ValueError, match='gemini_incomplete_or_blocked'):
        client.generate([{'text': 'fixture'}])


def test_corrupt_stage_archives_and_invalidates_downstream(tmp_path):
    task = TaskOutput(tmp_path / 'task')
    task.start('image:mouse.jpg')
    for name in ('objects', 'scene', 'physics', 'final_render'):
        (task.stage(name) / 'evidence').write_text(name)
        task.report['stages'][name] = 'passed'
        cp.save(task, name, task.stage(name), {'model': 'a'})
    (task.stage('scene') / 'evidence').write_text('damaged')
    assert cp.reusable(task, 'scene', task.stage('scene'), {'model': 'a'}) is None
    assert (task.stage('objects') / 'evidence').read_text() == 'objects'
    assert task.report['stages']['objects'] == 'passed'
    for name in ('scene', 'physics', 'final_render'):
        assert task.report['stages'][name] == 'not_run'
        assert not list(task.stage(name).iterdir())
    assert len(list((task.root / 'attempt_history').glob('*/02_scene/evidence'))) == 1
    task.verify()


def test_failed_upstream_resume_reuses_only_bound_success(tmp_path):
    source = tmp_path / 'mouse.jpg'
    Image.new('RGB', (4, 4)).save(source)
    index, config = tmp_path / 'index.json', tmp_path / 'config.yaml'
    index.write_text('{}')
    config.write_text('profiles: {}')
    output = tmp_path / 'task'
    commands = []

    def interrupted_runner(command, **kwargs):
        commands.append(command)
        stage = output / '01_obj/reconstruction/s2_da'
        stage.mkdir(parents=True, exist_ok=True)
        (stage / 'depth.bin').write_bytes(b'completed-depth-fixture')
        (stage / 'stage_info.json').write_text('{"success": true}')
        (output / '01_obj/input/backend_failure.txt').write_text(
            'injected failure: CUDA out of memory')
        return SimpleNamespace(returncode=137)

    first = media.run(source, 'image', output, index, config,
                      process_runner=interrupted_runner)
    assert first['status'] == 'execution_failed'
    assert (output / 'checkpoints/sf_s2_da.json').is_file()
    second = media.run(source, 'image', output, index, config, resume=True,
                       process_runner=interrupted_runner)
    assert second['status'] == 'execution_failed'
    assert '--skip-successful' not in commands[0]
    assert '--skip-successful' in commands[1]
    (output / '01_obj/reconstruction/s2_da/depth.bin').write_bytes(b'corrupt')
    media.run(source, 'image', output, index, config, resume=True,
              process_runner=interrupted_runner)
    assert '--skip-successful' not in commands[2]
    assert list((output / 'attempt_history').glob('*/reconstruction/s2_da/depth.bin'))
    TaskOutput(output).verify()


def test_asset_bytes_invalidate_without_index_change(tmp_path):
    index = tmp_path / 'index.json'
    mesh = tmp_path / 'mouse.obj'
    mesh.write_bytes(b'original')
    index.write_text(json.dumps({'official_index': {'path': str(tmp_path / 'asset_index.json')},
        'assets': [{'source_root': str(tmp_path), 'source_files': [{'path': 'mouse.obj'}]}]}))
    before = media.asset_dependency_hash(index)
    mesh.write_bytes(b'changed')
    assert media.asset_dependency_hash(index) != before
