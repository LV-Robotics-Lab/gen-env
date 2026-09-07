"""Offline multi-format source provenance and CLIP binding regression tests."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from test_clip_select import FakeEncoder, response

from self_improving.sim_adapters.genesis import asset_library as lib
from self_improving.sim_adapters.genesis import build_library_previews as previews
from self_improving.sim_adapters.genesis import clip_select as clip


def inventory(root, files):
    rows = []
    for name, content in files.items():
        path = root / 'sources' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        digest = lib.sha256(path)
        rows.append(dict(path=name, size_bytes=path.stat().st_size, sha256=digest,
                         lfs_sha256=digest, git_blob_sha1=None, status='verified'))
    path = root / 'download_manifest.json'
    clip.write_json(path, dict(repository=clip.official.REPOSITORY, revision=lib.REVISION,
                               failures=[], files=rows))
    return path


def reseal(root):
    index = lib.read_json(root / 'asset_index.json')
    index['files'] = [clip.official.fingerprint(p, root) for p in sorted(root.rglob('*'))
                      if p.is_file() and p.name != 'asset_index.json']
    clip.write_json(root / 'asset_index.json', index)


@pytest.fixture
def library(tmp_path):
    source = tmp_path / 'download'
    manifest = inventory(source, {
        'object.gltf': '{"asset":{"version":"2.0"}}',
        'bowl.urdf': '<robot name="passive"><link name="base"/></robot>',
        'broken.obj': 'mtllib absent.mtl\nv 0 0 0\n',
    })
    source_root, files = lib.verify_download(manifest)
    candidates, excluded = lib.discover(source_root, files)
    root = tmp_path / 'previews'
    reference = dict(path=str(manifest), sha256=lib.sha256(manifest))
    assets = []
    for c in candidates:
        passed = c['discovery_error'] is None
        record_path = f"assets/{c['asset_id']}.json"
        preview_path = f"previews/{c['asset_id']}/preview_result.json"
        source_files = [files[p] for p in c['dependency_paths']]
        views = []
        if passed:
            for v in lib.VIEWS:
                path = root / 'previews' / c['asset_id'] / f'view_{v}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (512, 512), (100, 80, 20)).save(path)
                views.append(dict(camera=dict(name=v),
                                  image=clip.official.fingerprint(path, root)))
        clip.write_json(root / preview_path, dict(status='passed' if passed else 'failed',
                        physics_steps=0, physics_status='not_evaluated', resolution=[512, 512],
                        genesis_commit=clip.official.GENESIS_COMMIT,
                        source_files=source_files, source_root=str(source_root),
                        entrypoint=c['entrypoint'], views=views,
                        material_status='passed' if passed else 'not_evaluated'))
        record = dict(c, source_root=str(source_root), source_inventory=reference,
                      source_files=source_files, preview_result=preview_path,
                      status='preview_passed' if passed else 'failed')
        clip.write_json(root / record_path, record)
        assets.append(dict(asset_id=c['asset_id'], record=record_path, status=record['status']))
    clip.write_json(root / 'asset_index.json', dict(schema_version=lib.SCHEMA,
                    source_inventory=reference, assets=assets, excluded=excluded, files=[],
                    status='partial'))
    reseal(root)
    encoder = FakeEncoder()
    index = tmp_path / 'clip'
    clip.build_index(root / 'asset_index.json', index, encoder=encoder,
                     weights_dir=tmp_path / 'weights')
    calls = []

    def vlm(messages):
        calls.append(messages)
        return response(number=2)

    config = clip.LLMProviderConfig(endpoint='https://example.invalid/v1', model='gpt-4o',
                                    api_key='TEST_SECRET_NEVER_LOG')
    kwargs = dict(clip_index=index / 'index.json', query='一个碗', encoder=encoder,
                  vlm_config=config, vlm=vlm, weights_dir=tmp_path / 'weights',
                  cache_dir=tmp_path / 'cache')
    return SimpleNamespace(root=root, source=source, manifest=manifest, kwargs=kwargs,
                           calls=calls, index=index, tmp=tmp_path, encoder=encoder)


def test_multi_format_binding_and_cache(library):
    s = library
    index = lib.read_json(s.index / 'index.json')
    assert len(index['assets']) == 2 and len(index['rows']) == 12
    assert {a['model_format'] for a in index['assets']} == {'gltf', 'urdf'}
    assert not any('broken' in a['asset_id'] for a in index['assets'])
    first = clip.select(output_dir=s.tmp / 'first', **s.kwargs)
    second = clip.select(output_dir=s.tmp / 'cached', **s.kwargs)
    assert first['status'] == second['status'] == 'selected'
    assert first['vlm_calls'] == 1 and second['vlm_calls'] == 0 and len(s.calls) == 1
    assert s.encoder.image_calls == 2
    binding = lib.read_json(s.tmp / 'first/selected_asset.json')
    assert Path(binding['model_entrypoint']) == s.source / 'sources/object.gltf'
    assert binding['source_inventory']['sha256'] == lib.sha256(s.manifest)
    assert binding['model_format'] == 'gltf'
    assert binding['visible_differences'] == ['未确认米老鼠印花']
    wire = json.dumps(s.calls)
    assert all(a['asset_id'] not in wire for a in index['assets'])
    assert 'score' not in wire and 'TEST_SECRET_NEVER_LOG' not in wire


@pytest.mark.parametrize('target', ['image', 'vector', 'source', 'inventory'])
def test_extended_integrity_before_cached_binding(library, target):
    s = library
    clip.select(output_dir=s.tmp / 'first', **s.kwargs)
    path = {'image': next(s.root.rglob('view_top.png')), 'vector': s.index / 'vectors.npy',
            'source': s.source / 'sources/object.gltf', 'inventory': s.manifest}[target]
    path.write_bytes(path.read_bytes() + b' ')
    result = clip.select(output_dir=s.tmp / 'corrupt', **s.kwargs)
    assert result['status'] == 'error' and result['vlm_calls'] == 0
    assert not (s.tmp / 'corrupt/selected_asset.json').exists()


@pytest.mark.parametrize('field,value', [
    ('format', 'xml'), ('entrypoint', 'sources/bowl.urdf'), ('source_root', '/tmp/forged'),
    ('source_files', []), ('source_inventory', {'path': '/tmp/forged', 'sha256': '0'}),
])
def test_rehashed_record_cannot_change_binding(library, field, value):
    s = library
    path = next(p for p in (s.root / 'assets').glob('*.json') if 'object' in p.name)
    record = lib.read_json(path)
    record[field] = value
    clip.write_json(path, record)
    reseal(s.root)
    with pytest.raises(ValueError):
        lib.verify_preview_index(s.root / 'asset_index.json')


@pytest.mark.parametrize('field,value', [('material_status', 'failed'), ('physics_steps', 1),
                                        ('views', []), ('resolution', [256, 256]),
                                        ('genesis_commit', 'wrong')])
def test_failed_or_incomplete_previews_not_accepted(library, field, value):
    s = library
    path = next(p for p in s.root.rglob('preview_result.json') if 'object' in str(p))
    record = lib.read_json(path)
    record[field] = value
    clip.write_json(path, record)
    reseal(s.root)
    with pytest.raises(ValueError, match='preview evidence'):
        lib.verify_preview_index(s.root / 'asset_index.json')


def test_discovery_dependency_dedup_and_scene_filters(tmp_path):
    manifest = inventory(tmp_path, {
        'thing.urdf': '<robot><link><visual><mesh filename="mesh.obj"/></visual></link></robot>',
        'mesh.obj': 'mtllib mesh.mtl\nv 0 0 0\n', 'mesh.mtl': 'newmtl surface\nmap_Kd color.png',
        'color.png': 'texture bytes', 'duplicate.obj': 'v 1 2 3\n', 'same.obj': 'v 1 2 3\n',
        'shape/collision/coll.obj': 'v 4 5 6\n', 'connect.xml': '<mujoco/>',
        'broken.obj': 'mtllib missing.mtl\n',
    })
    root, files = lib.verify_download(manifest)
    candidates, excluded = lib.discover(root, files)
    assert {c['entrypoint'] for c in candidates} == {
        'sources/thing.urdf', 'sources/duplicate.obj', 'sources/broken.obj'}
    thing = next(c for c in candidates if c['format'] == 'urdf')
    assert len(thing['dependency_paths']) == 4
    assert next(c for c in candidates if 'broken' in c['asset_id'])['discovery_error']
    assert any('byte_identical' in e['reason'] for e in excluded)


@pytest.mark.parametrize('ref', ['/etc/passwd', '../../../escape', 'https://example.invalid/a',
                                  'file://local', 'a\\b'])
def test_external_reference_rejected(tmp_path, ref):
    path = tmp_path / 'sources/model.obj'
    path.parent.mkdir()
    path.write_text('')
    with pytest.raises(ValueError):
        lib.local_reference(path, ref, tmp_path)


def test_parent_relative_allowed_but_symlink_directory_rejected(tmp_path):
    (tmp_path / 'real').mkdir()
    (tmp_path / 'real/texture').write_text('ok')
    (tmp_path / 'alias').symlink_to(tmp_path / 'real', target_is_directory=True)
    assert lib.local_reference(tmp_path / 'real/model', '../real/texture', tmp_path).is_file()
    with pytest.raises(ValueError, match='symlink'):
        lib.local_reference(tmp_path / 'model', 'alias/texture', tmp_path)


def test_output_and_download_package_protection(library):
    s = library
    for output in (s.source / 'accidental', s.root / 'accidental'):
        with pytest.raises(clip.SelectionError, match='official_package|overlap'):
            clip.select(output_dir=output, **s.kwargs)
        assert not output.exists()
    with pytest.raises(ValueError, match='separate'):
        previews.build(s.manifest, s.source / 'accidental')
    with pytest.raises(FileExistsError):
        previews.build(s.manifest, s.root)


def test_preview_builder_protects_other_sealed_packages(library):
    other = library.tmp / 'another_official'
    other.mkdir()
    (other / 'asset_index.json').write_text('{}')
    with pytest.raises(ValueError, match='sealed asset package'):
        previews.build(library.manifest, other / 'new_previews')
    assert not (other / 'new_previews').exists()


def test_missing_dependency_cannot_be_promoted_by_rehashed_status(library):
    s = library
    path = next(p for p in (s.root / 'assets').glob('*.json') if 'broken' in p.name)
    record = lib.read_json(path)
    record['status'] = 'preview_passed'
    record['discovery_error'] = None
    clip.write_json(path, record)
    index = lib.read_json(s.root / 'asset_index.json')
    item = next(a for a in index['assets'] if a['asset_id'] == record['asset_id'])
    item['status'] = 'preview_passed'
    clip.write_json(s.root / 'asset_index.json', index)
    reseal(s.root)
    with pytest.raises(ValueError, match='discovery failure'):
        lib.verify_preview_index(s.root / 'asset_index.json')
