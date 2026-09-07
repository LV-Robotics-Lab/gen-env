"""Task ownership, stage replacement and physical/render routing without a simulator."""
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from test_physics import example, physics, trace

from self_improving.sim_adapters.genesis import task_output as storage

REQUEST = '桌上放着一个苹果、一个黄色杯子和一个橙色塑料碗。'


def make_task(tmp_path):
    task = storage.TaskOutput(storage.destination(REQUEST, output_root=tmp_path/'output'))
    with task.lock():
        task.start(REQUEST)
        (task.stage('objects')/'apple_1.json').write_text('{"object_id":"apple_1"}')
        (task.stage('scene')/'overview.png').write_bytes(b'offline image fixture')
        task.finish_preview(dict(status='preview_passed', stage='complete'))
    return task


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_chinese_name_and_default_root():
    assert storage.task_name(REQUEST) == REQUEST
    assert storage.destination(REQUEST) == storage.DEFAULT_ROOT/REQUEST


@pytest.mark.parametrize('query', ['../杯子/苹果', '桌上\n苹果', '杯子\\碗', 'CON', '杯子?苹果',
                                     '苹果'*100, ' ', '.', '盘子.'])
def test_safe_names_and_utf8_length(query):
    if not query.strip():
        with pytest.raises(ValueError):
            storage.task_name(query)
        return
    name = storage.task_name(query)
    assert len(name.encode()) <= 180
    assert not any(c in name for c in '/\\?\n') and name not in {'.', '..', 'CON'}
    assert name.endswith(storage.request_hash(query)[:8])
    assert name == storage.task_name(query)


def test_sanitization_collisions_and_api_conflict():
    assert storage.task_name('苹果/碗') != storage.task_name('苹果\\碗')
    with pytest.raises(ValueError, match='mutually exclusive'):
        storage.destination(REQUEST, output_dir='a', output_root='b')


def test_full_rerun_replaces_all_stages_and_failure_drops_stale_success(tmp_path):
    task = make_task(tmp_path)
    with task.lock():
        for stage in ('physics', 'final_render'):
            (task.stage(stage)/'old_success.json').write_text('{}')
        task.report['stages'].update(physics='passed', final_render='passed')
        task.seal()
        task.start(REQUEST)
        task.finish_preview(dict(status='error', stage='parse', error='offline failure'))
    task.verify()
    assert all(not list(task.stage(s).iterdir()) for s in storage.STAGES)
    assert task.report['stages'] == dict(objects='failed', scene='not_run',
                                         physics='not_run', final_render='not_run')
    assert len([p for p in task.root.parent.iterdir() if not p.name.startswith('.')]) == 1
    assert (task.root/'request.txt').read_text() == REQUEST
    assert 'old_success' not in (task.root/'README.md').read_text()


def test_physics_reset_preserves_upstream_bytes(tmp_path):
    task = make_task(tmp_path)
    before = {s: snapshot(task.stage(s)) for s in ('objects', 'scene')}
    with task.lock():
        for stage in ('physics', 'final_render'):
            (task.stage(stage)/'old_success.json').write_text('{}')
        task.report['stages'].update(physics='passed', final_render='passed')
        task.seal()
        task.start_physics()
        assert [p.name for p in task.stage('physics').iterdir()] == ['scene_input_manifest.json']
        assert not list(task.stage('final_render').iterdir())
        task.finish_physics(dict(status='failed', physics_status='failed', render_status='not_run'))
    task.verify()
    assert all(snapshot(task.stage(s)) == before[s] for s in before)
    assert task.report['stages']['scene'] == 'passed'


def test_unowned_wrong_request_and_copied_owner_are_not_overwritten(tmp_path):
    other = tmp_path/'unrelated'
    other.mkdir()
    (other/'keep').write_text('user file')
    with pytest.raises(FileExistsError):
        storage.TaskOutput(other).start(REQUEST)
    task = make_task(tmp_path)
    before = snapshot(task.root)
    with pytest.raises(ValueError, match='request mismatch'):
        task.start('另一个请求')
    assert snapshot(task.root) == before
    for name in (storage.OWNER, 'request.txt'):
        (other/name).write_bytes((task.root/name).read_bytes())
    with pytest.raises(ValueError, match='owner'):
        storage.TaskOutput(other).start(REQUEST)
    assert (other/'keep').read_text() == 'user file'


@pytest.mark.parametrize('kind', ['root_symlink', 'child_symlink', 'official', 'nested_task'])
def test_protected_paths(tmp_path, kind):
    task = make_task(tmp_path)
    if kind == 'root_symlink':
        alias = tmp_path/'alias'
        alias.symlink_to(task.root, target_is_directory=True)
        with pytest.raises(ValueError, match='symlink'):
            storage.TaskOutput(alias)
    elif kind == 'child_symlink':
        (task.root/'link').symlink_to(tmp_path/'outside')
        with pytest.raises(ValueError, match='symlink'):
            task.start(REQUEST)
    elif kind == 'official':
        (task.stage('objects')/'asset_index.json').write_text('{}')
        with pytest.raises(ValueError, match='protected package'):
            task.start(REQUEST)
    else:
        with pytest.raises(ValueError, match='nested'):
            storage.TaskOutput(task.root/'child')
    assert (task.stage('objects')/'apple_1.json').exists()


def test_exclusive_lock_and_release(tmp_path):
    task = make_task(tmp_path)
    with task.lock():
        with pytest.raises(ValueError, match='already running'):
            with storage.TaskOutput(task.root).lock():
                pytest.fail('concurrent task entered')
        different = storage.TaskOutput(tmp_path/'different')
        with different.lock():
            pass
    with task.lock():
        pass


def test_tampered_manifest_stops_physics_and_does_not_clear(tmp_path):
    task = make_task(tmp_path)
    image = task.stage('scene')/'overview.png'
    image.write_bytes(b'tampered')
    with pytest.raises(ValueError):
        task.start_physics()
    assert image.read_bytes() == b'tampered'


def install_fake_runtime(monkeypatch, package, *, failed=False, render_code=0):
    rows = trace(package)
    if failed:
        for row in rows:
            row['contacts'] = []

    def simulate(package, spec, out, raw):
        (out/'trace.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        return rows, {'offline_fixture': True}

    calls = []

    def render(command, **kwargs):
        calls.append(command)
        assert not failed, 'physics failure must never call rendering'
        scene_file = Path(command[command.index('--scene')+1])
        data = json.loads(scene_file.read_text())
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'preview.png').write_bytes(b'offline renderer fixture')
        (destination/physics.render.EVIDENCE_OUTPUT).write_text(json.dumps(
            dict(status='success', package_digest=data['package_digest'])))
        return SimpleNamespace(returncode=render_code)

    monkeypatch.setattr(physics, 'simulate', simulate)
    monkeypatch.setattr(physics.subprocess, 'run', render)
    return calls


def test_physics_routing_original_command_then_failed_rerun(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    package, compiled, _ = example(tmp_path)
    calls = install_fake_runtime(monkeypatch, package)
    before = {s: snapshot(task.stage(s)) for s in ('objects', 'scene')}
    assert physics.run(compiled.manifest_path, scene_dir=task.root) == 0
    task.verify()
    report = json.loads((task.stage('physics')/'physics_result.json').read_text())
    terminal = physics.CompileResult.read(report['settled_compile_manifest'])
    assert calls == [terminal.runtime_command]
    assert Path(terminal.artifact_path).is_relative_to(task.stage('final_render'))
    assert Path(report['render_evidence']).is_relative_to(task.stage('final_render'))
    assert (task.stage('final_render')/'render.log').exists()
    assert (task.stage('physics')/'trace.jsonl').exists()
    assert not (task.stage('physics')/'settled').exists()
    assert all(snapshot(task.stage(s)) == before[s] for s in before)
    calls = install_fake_runtime(monkeypatch, package, failed=True)
    assert physics.run(compiled.manifest_path, scene_dir=task.root) == 1
    task.verify()
    assert not calls and not list(task.stage('final_render').iterdir())
    assert task.report['stages']['physics'] == 'failed'
    assert task.report['stages']['final_render'] == 'not_run'
    assert all(snapshot(task.stage(s)) == before[s] for s in before)


def test_invalid_input_rerun_clears_old_final_output(tmp_path):
    task = make_task(tmp_path)
    (task.stage('final_render')/'old.png').write_bytes(b'old')
    task.report['stages'].update(physics='passed', final_render='passed')
    task.seal()
    assert physics.run(tmp_path/'missing.json', scene_dir=task.root) == 1
    task.verify()
    assert not list(task.stage('final_render').iterdir())
    assert task.report['stages']['physics'] == 'failed'


def test_inputs_in_replaced_stages_are_not_deleted(tmp_path):
    task = make_task(tmp_path)
    _, compiled, _ = example(task.stage('physics'))
    task.seal()
    before = snapshot(task.root)
    with pytest.raises(ValueError, match='inputs cannot'):
        physics.run(compiled.manifest_path, scene_dir=task.root)
    assert snapshot(task.root) == before


def test_readme_links_and_manifest_cover_all_outputs(tmp_path):
    import re
    task = make_task(tmp_path)
    task.verify()
    for target in re.findall(r'\]\(([^)]+)\)', (task.root/'README.md').read_text()):
        assert (task.root/unquote(target)).exists()


def test_upstream_changed_during_physics_never_renders(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    package, compiled, _ = example(tmp_path)
    rows = trace(package)

    def simulate(package, spec, out, raw):
        (task.stage('objects')/'apple_1.json').write_text('changed during physics')
        return rows, {}

    monkeypatch.setattr(physics, 'simulate', simulate)
    monkeypatch.setattr(physics.subprocess, 'run', lambda *a, **k: pytest.fail('render called'))
    assert physics.run(compiled.manifest_path, scene_dir=task.root) == 1
    task.verify()
    assert task.report['stages']['physics'] == 'failed'
    assert not list(task.stage('final_render').iterdir())


@pytest.mark.parametrize('physics_status', ['failed', 'not_run', 'running'])
@pytest.mark.parametrize('render_status', ['passed', 'running', 'failed'])
def test_final_render_status_requires_physics_pass(tmp_path, physics_status, render_status):
    task = make_task(tmp_path)
    before = snapshot(task.root)
    with pytest.raises(ValueError, match='final_render requires passed physics'):
        task.finish_physics(dict(status='physics_failed', physics_status=physics_status,
                                 render_status=render_status))
    assert snapshot(task.root) == before


def test_failure_diagnostics_belong_to_physics_and_rerun_clears_them(tmp_path):
    task = make_task(tmp_path)
    task.start_physics()
    diagnostic = task.stage('physics')/'video'
    diagnostic.mkdir()
    (diagnostic/'replay.mp4').write_bytes(b'offline video fixture')
    task.report['physics_diagnostics'] = {'video': {'status': 'passed'}}
    task.finish_physics(dict(status='physics_failed', physics_status='failed',
                             render_status='not_run'))
    task.verify()
    assert task.report['stages']['final_render'] == 'not_run'
    assert not list(task.stage('final_render').iterdir())
    task.start_physics()
    assert not diagnostic.exists() and 'physics_diagnostics' not in task.report


def test_failed_physics_cannot_seal_files_in_final_render(tmp_path):
    task = make_task(tmp_path)
    task.finish_physics(dict(status='physics_failed', physics_status='failed',
                             render_status='not_run'))
    before = (task.root/'manifest.json').read_bytes()
    (task.stage('final_render')/'diagnostic.png').write_bytes(b'wrong stage')
    with pytest.raises(ValueError, match='final_render requires passed physics'):
        task.seal()
    assert (task.root/'manifest.json').read_bytes() == before


def test_verify_rejects_hash_consistent_failed_physics_render(tmp_path):
    task = make_task(tmp_path)
    task.report['stages'].update(physics='passed', final_render='passed')
    (task.stage('final_render')/'image.png').write_bytes(b'offline image fixture')
    task.seal()
    # Rehashing an invalid lifecycle must not make it acceptable.
    task.report['stages']['physics'] = 'failed'
    storage.clip.write_json(task.root/'run_report.json', task.report)
    manifest = storage.library.read_json(task.root/'manifest.json')
    manifest['files'] = [storage.official.fingerprint(task.root/r['path'], task.root)
                         for r in manifest['files']]
    storage.clip.write_json(task.root/'manifest.json', manifest)
    with pytest.raises(ValueError, match='final_render requires passed physics'):
        task.verify()


def test_physics_pass_without_render_remains_not_run(tmp_path):
    task = make_task(tmp_path)
    task.finish_physics(dict(status='physics_passed', physics_status='passed',
                             render_status='not_run'))
    task.verify()
    assert task.report['status'] == 'physics_passed'
    assert task.report['stages']['final_render'] == 'not_run'
