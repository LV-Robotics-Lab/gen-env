"""Owned, locked natural-language task directories and stage-bound evidence manifests."""
from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis.storage_paths import CACHE_ROOT

SCHEMA = 'genenv.task_output.v1'
DEFAULT_ROOT = Path(__file__).resolve().parents[3] / 'output'
STAGES = {'objects': '01_obj', 'scene': '02_scene',
          'physics': '03_physics', 'final_render': '04_final_render'}
OWNER = '.task.json'


def request_hash(request):
    return hashlib.sha256(request.encode('utf-8')).hexdigest()


def task_name(request):
    if not isinstance(request, str) or not request.strip():
        raise ValueError('empty task request')
    name = re.sub(r'[\x00-\x1f\x7f/\\:*?"<>|]', '_', request).strip(' .')
    if re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?', name):
        name = '_' + name
    changed = name != request or not name or len(name.encode('utf-8')) > 180
    if changed:
        # Reserve ten bytes for the separator and an eight-character source hash.
        name = name.encode('utf-8')[:170].decode('utf-8', errors='ignore').rstrip(' .')
        name = (name or '任务') + '__' + request_hash(request)[:8]
    return name


def no_symlinks(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('task path contains symlink')
    return path.resolve()


def destination(request, *, output_dir=None, output_root=None):
    if output_dir is not None and output_root is not None:
        raise ValueError('output_dir and output_root are mutually exclusive')
    name = task_name(request)
    return no_symlinks(output_dir if output_dir is not None
                       else Path(output_root or DEFAULT_ROOT) / name)


class TaskOutput:
    def __init__(self, root):
        self.root = no_symlinks(root)
        clip.writable_storage(self.root)
        for parent in self.root.parents:
            if (parent / OWNER).exists():
                raise ValueError('task directories cannot be nested')

    def stage(self, name):
        return self.root / STAGES[name]

    def _safe_tree(self):
        no_symlinks(self.root)
        for path in self.root.rglob('*'):
            if path.is_symlink():
                raise ValueError('task contents contain symlink')
            if path.name in {'asset_index.json', 'download_manifest.json', OWNER}:
                if path != self.root / OWNER:
                    raise ValueError('task contains another protected package')

    def owner(self, request=None):
        self._safe_tree()
        try:
            owner = library.read_json(self.root / OWNER)
            text = (self.root / 'request.txt').read_text(encoding='utf-8')
        except (OSError, ValueError) as exc:
            raise FileExistsError('not an owned task directory') from exc
        if (owner != dict(schema_version=SCHEMA, request=text, request_sha256=request_hash(text),
                          directory=str(self.root)) or (request is not None and text != request)):
            raise ValueError('task owner or original request mismatch')
        return owner

    @contextmanager
    def lock(self):
        # Keep the lock inode outside the directory that full reruns clear. Do not unlink it.
        locks = no_symlinks(CACHE_ROOT / 'task_locks')
        locks.mkdir(parents=True, exist_ok=True)
        path = locks / (request_hash(str(self.root)) + '.lock')
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('task is already running') from None
            yield self
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def start(self, request):
        task_name(request)
        if self.root.exists():
            self.owner(request)
            for child in self.root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        clip.write_json(self.root / OWNER, dict(schema_version=SCHEMA, request=request,
                        request_sha256=request_hash(request), directory=str(self.root)))
        (self.root / 'request.txt').write_text(request, encoding='utf-8')
        for name in STAGES:
            self.stage(name).mkdir()
        self.report = dict(schema_version=SCHEMA, status='running',
                           request_sha256=request_hash(request),
                           stages={name: 'not_run' for name in STAGES})
        self.report['stages']['objects'] = 'running'
        self.save()

    def verify(self):
        self.owner()
        manifest = library.read_json(self.root / 'manifest.json')
        if manifest['schema_version'] != SCHEMA:
            raise ValueError('invalid task manifest')
        official.verify_files(self.root, manifest['files'])
        actual = {p.relative_to(self.root).as_posix() for p in self.root.rglob('*')
                  if p.is_file() and p != self.root / 'manifest.json'}
        if actual != {r['path'] for r in manifest['files']}:
            raise ValueError('task manifest file set mismatch')
        if any(not self.stage(name).is_dir() for name in STAGES):
            raise ValueError('task stage directory missing')
        self.report = library.read_json(self.root / 'run_report.json')
        if (manifest['status'] != self.report['status']
                or manifest['request_sha256'] != self.owner()['request_sha256']):
            raise ValueError('task manifest status or request mismatch')
        self._check_final_render_gate(**self._stage_gate_status())
        return manifest

    def start_scene(self):
        manifest = self.verify()
        if self.report['stages']['objects'] != 'passed':
            raise ValueError('scene construction requires all assets selected')
        upstream = [r for r in manifest['files']
                    if r['path'].split('/')[0] == STAGES['objects']
                    or r['path'] in {OWNER, 'request.txt'}]
        input_hash = library.sha256(self.root/'manifest.json')
        for name in ('scene', 'physics', 'final_render'):
            shutil.rmtree(self.stage(name))
            self.stage(name).mkdir()
            self.report['stages'][name] = 'not_run'
        for key in ('preview', 'scene_build', 'physics', 'physics_input_manifest_sha256',
                    'physics_diagnostics', 'diagnostic_final_render', 'diagnostic_video'):
            self.report.pop(key, None)
        clip.write_json(self.stage('scene')/'input_manifest.json',
                        dict(source_manifest_sha256=input_hash, files=upstream))
        self.report.update(status='running')
        self.report['stages']['scene'] = 'running'
        (self.root/'manifest.json').unlink()
        self.save()

    def verify_scene_inputs(self):
        snapshot = library.read_json(self.stage('scene')/'input_manifest.json')
        for row in snapshot['files']:
            if (row['path'] == OWNER and (self.root/'.scene_source_owner.json').exists()
                    and library.sha256(self.root/OWNER) != row['sha256']):
                # Physics copies preserve every 01/02 byte, including the original owner hash.
                source = self.root/'.scene_source_owner.json'
                archived = library.read_json(source)
                owner = self.owner()
                if (archived['request'] != owner['request']
                        or archived['request_sha256'] != owner['request_sha256']):
                    raise ValueError('copied scene owner request mismatch')
                official.verify_files(self.root, [dict(row, path=source.name)])
            else:
                official.verify_files(self.root, [row])

    def copy_for_physics(self, destination):
        """Copy a sealed task for independent trials; never alter its 01/02 bytes."""
        target = TaskOutput(destination)
        clip.separate(self.root, target.root)
        with self.lock(), target.lock():
            self.verify()
            self.verify_scene_inputs()
            if self.report['stages']['scene'] != 'passed':
                raise ValueError('physics copy requires completed scene')
            if target.root.exists():
                raise FileExistsError('physics copy destination already exists')
            shutil.copytree(self.root, target.root)
            archived = target.root/'.scene_source_owner.json'
            if not archived.exists():
                archived.write_bytes((self.root/OWNER).read_bytes())
            owner = self.owner()
            clip.write_json(target.root/OWNER, dict(owner, directory=str(target.root)))
            target.report = library.read_json(target.root/'run_report.json')
            target.seal()
            target.verify_scene_inputs()
        return target

    def finish_scene(self, report):
        self.report.update(status=report['status'], scene_build=report)
        self.report['stages']['scene'] = ('passed' if report['status'] == 'scene_built'
                                           else 'failed')
        self.seal()

    def start_physics(self, protected_inputs=()):
        manifest = self.verify()
        upstream = [r for r in manifest['files']
                    if r['path'].split('/')[0] in (STAGES['objects'], STAGES['scene'])
                    or r['path'] in (OWNER, 'request.txt', '.scene_source_owner.json')]
        input_hash = library.sha256(self.root / 'manifest.json')
        if (self.report['stages']['scene'] != 'passed'
                or self.report['stages']['objects'] != 'passed'):
            raise ValueError('physics requires a completed initial scene')
        for path in protected_inputs:
            resolved = Path(path).resolve()
            if any(resolved.is_relative_to(self.stage(s)) for s in ('physics', 'final_render')):
                raise ValueError('compile inputs cannot be inside stages being replaced')
        for name in ('physics', 'final_render'):
            shutil.rmtree(self.stage(name))
            self.stage(name).mkdir()
            self.report['stages'][name] = 'not_run'
        clip.write_json(self.stage('physics') / 'scene_input_manifest.json',
                        dict(source_manifest_sha256=input_hash, files=upstream,
                             request_sha256=self.report['request_sha256']))
        for key in ('physics', 'physics_diagnostics',
                    'diagnostic_final_render', 'diagnostic_video'):
            self.report.pop(key, None)
        self.report['physics_input_manifest_sha256'] = input_hash
        self.report['status'] = 'running'
        self.report['stages']['physics'] = 'running'
        (self.root / 'manifest.json').unlink()
        self.save()

    def verify_physics_inputs(self):
        snapshot = library.read_json(self.stage('physics') / 'scene_input_manifest.json')
        official.verify_files(self.root, snapshot['files'])

    def finish_preview(self, report):
        self.report.update(status=report['status'], preview=report)
        objects_ok = report['stage'] in {'layout', 'render', 'complete'}
        self.report['stages'].update(objects='passed' if objects_ok else 'failed',
                                     scene=('passed' if report['status'] == 'preview_passed' else
                                            'failed' if objects_ok else 'not_run'))
        self.seal()

    def finish_assets(self, report):
        self.report.update(status=report['status'], asset_selection=report)
        self.report['stages'].update(objects=('passed' if report['status'] == 'assets_selected'
                                             else 'partial' if report['status'] == 'partial'
                                             else 'failed'),
                                     scene='not_run', physics='not_run', final_render='not_run')
        self.seal()

    def _stage_gate_status(self):
        stages = self.report['stages']
        return dict(physics_status=stages['physics'], render_status=stages['final_render'])

    def _check_final_render_gate(self, *, physics_status, render_status):
        """04 is reserved for results that passed physics; diagnostics belong to 03."""
        if physics_status != 'passed' and (
                render_status != 'not_run' or any(self.stage('final_render').iterdir())):
            raise ValueError('final_render requires passed physics; '
                             'store diagnostics in 03_physics')

    def finish_physics(self, report):
        self._check_final_render_gate(physics_status=report['physics_status'],
                                      render_status=report['render_status'])
        self.report.update(status=report['status'], physics=report)
        self.report['stages'].update(physics=report['physics_status'],
                                     final_render=report['render_status'])
        self.seal()

    def save(self):
        clip.write_json(self.root / 'run_report.json', self.report)
        lines = ['# 任务结果', '', (self.root / 'request.txt').read_text(), '',
                 f"当前状态：`{self.report['status']}`", '',
                 '| 阶段 | 状态 |', '| --- | --- |']
        for name, folder in STAGES.items():
            lines.append(f"| [{folder}]({quote(folder)}/) | {self.report['stages'][name]} |")
        lines.extend(['', '未执行的阶段不生成图片或成功报告。初始场景图不代表物理验证通过。', '',
                      '## 已有产物', ''])
        for path in sorted(self.root.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.png', '.mp4', '.jsonl'}:
                relative = path.relative_to(self.root).as_posix()
                if relative not in {OWNER, 'manifest.json', 'run_report.json'}:
                    lines.append(f'- [{relative}]({quote(relative)})')
        (self.root / 'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    def seal(self):
        self._check_final_render_gate(**self._stage_gate_status())
        self.save()
        self._safe_tree()
        files = [official.fingerprint(p, self.root) for p in sorted(self.root.rglob('*'))
                 if p.is_file() and p != self.root / 'manifest.json']
        clip.write_json(self.root / 'manifest.json', dict(schema_version=SCHEMA, files=files,
                        status=self.report['status'], request_sha256=self.report['request_sha256']))
