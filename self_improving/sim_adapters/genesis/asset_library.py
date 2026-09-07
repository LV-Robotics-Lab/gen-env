"""Verified non-robot download inventory and multi-format six-view preview contract."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import struct
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis.storage_paths import local_path, same_evidence_path
from self_improving.sim_adapters.genesis.vision_request import strict_json

SCHEMA = 'genenv.nonrobot_preview_index.v1'
REVISION = '5b01555c225977d2d3973710884adc5add269fe8'
FORMATS = {'.xml', '.urdf', '.glb', '.gltf', '.obj', '.stl', '.usd', '.usda', '.usdz'}
USD_FORMATS = {'.usd', '.usda', '.usdz'}
VIEWS = ('az000', 'az090', 'az180', 'az270', 'top', 'bottom')
# These are explicit multi-object scenes or parser fixtures, not single-object candidates.
SCENES = {'Lightwheel_Kitchen/KitchenRoom.usd',
          'usd/Lightwheel_Kitchen001/Kitchen001/Kitchen001.usd',
          'usd/Lightwheel_Kitchen001/Kitchen001/Assets/LayoutKitchen001/LayoutKitchen001.usd',
          'usd/franka_mocap_teleop/table_scene.usd', 'usd/chair_array.usd',
          'usd/nodegraph.usda', 'connect.xml', 'weld.xml'}
# Format/orientation variants represent one asset; choose the visual format, preserve exclusions.
ALIASES = {'usd/RoughnessTest.usdz': 'usd/RoughnessTest.glb',
           'usd/sneaker_airforce.usdz': 'usd/sneaker_airforce.glb',
           'glb/tycoon_draco_no_normal.glb': 'glb/tycoon_with_normal_draco.glb'}


def sha256(path):
    result = hashlib.sha256()
    with local_path(path).open('rb') as stream:
        while data := stream.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def read_json(path):
    return strict_json(local_path(path).read_text())


def verify_download(manifest_path):
    manifest_path = local_path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if (manifest['repository'] != official.REPOSITORY or manifest['revision'] != REVISION
            or manifest['failures']):
        raise ValueError('unsupported or incomplete nonrobot source inventory')
    root = manifest_path.parent
    records = {}
    for row in manifest['files']:
        relative = 'sources/' + row['path']
        if relative in records or row['status'] != 'verified':
            raise ValueError('duplicate or unverified source record')
        path = official.safe_file(root, relative)
        actual = sha256(path)
        if path.stat().st_size != row['size_bytes'] or actual != row['sha256']:
            raise ValueError('downloaded source integrity mismatch')
        if row['lfs_sha256']:
            if actual != row['lfs_sha256']:
                raise ValueError('source differs from official LFS hash')
        else:
            blob = hashlib.sha1(f"blob {row['size_bytes']}\0".encode())
            with path.open('rb') as stream:
                while data := stream.read(1024 * 1024):
                    blob.update(data)
            if blob.hexdigest() != row['git_blob_sha1']:
                raise ValueError('source differs from official Git hash')
        records[relative] = dict(path=relative, size_bytes=row['size_bytes'], sha256=actual)
    actual_paths = {p.relative_to(root).as_posix() for p in (root / 'sources').rglob('*')
                    if p.is_file()}
    if actual_paths != set(records):
        raise ValueError('download source file set mismatch')
    return root, records


def local_reference(path, reference, root):
    if not isinstance(reference, str) or not reference or '\\' in reference:
        raise ValueError('invalid local reference')
    if reference.startswith('data:'):
        return None
    if PurePosixPath(reference).is_absolute() or ':' in reference:
        raise ValueError('external reference is not accepted')
    target = Path(path).parent / reference
    resolved = target.resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError('reference escapes source inventory')
    # Permit ordinary parent-relative references inside the authenticated source tree.
    relative = resolved.relative_to(Path(root).resolve()).as_posix()
    checked = official.safe_file(root, relative)
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('symlink reference')
    return checked


def direct_references(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {'.xml', '.urdf'}:
        content = path.read_text()
        if '<!DOCTYPE' in content or '<!ENTITY' in content:
            raise ValueError('XML entities are not accepted')
        tree = ET.fromstring(content)
        if suffix == '.xml' and any(n.get(k) for n in tree.findall('compiler')
                                    for k in ('assetdir', 'meshdir', 'texturedir')):
            raise ValueError('unsupported MJCF directory overrides')
        return [n.get(k) for n in tree.iter() for k in ('file', 'filename') if n.get(k)]
    if suffix in {'.obj', '.mtl'}:
        refs = []
        for line in path.read_text(errors='strict').splitlines():
            # Geometry rows dominate large OBJ files; only tokenize dependency directives.
            head = line.split(None, 1)
            if not head:
                continue
            directive = head[0].lower()
            if directive != 'mtllib' and not directive.startswith('map_') and directive not in {
                    'bump', 'disp', 'decal', 'norm', 'refl'}:
                continue
            words = shlex.split(line, comments=True)
            if not words:
                continue
            if words[0] == 'mtllib':
                refs.extend(words[1:])
            elif words[0].lower().startswith('map_') or words[0].lower() in {
                    'bump', 'disp', 'decal', 'norm', 'refl'}:
                if len(words) < 2 or words[1].startswith('-'):
                    raise ValueError('unsupported MTL texture options')
                refs.append(' '.join(words[1:]))
        return refs
    if suffix in {'.glb', '.gltf'}:
        if suffix == '.gltf':
            content = read_json(path)
        else:
            with path.open('rb') as stream:
                header = stream.read(20)
                if len(header) != 20:
                    raise ValueError('invalid GLB header')
                magic, version, total, size, kind = struct.unpack('<5I', header)
                if (magic != 0x46546C67 or version != 2 or kind != 0x4E4F534A
                        or total != path.stat().st_size or size > total - 20):
                    raise ValueError('invalid GLB header')
                content = json.loads(stream.read(size))
        return [v['uri'] for kind in ('buffers', 'images') for v in content.get(kind, [])
                if 'uri' in v]
    return []


def dependencies(path, root, records):
    """Complete ordinary file closure; USD is resolved by its native resolver in the worker."""
    pending, visited = [Path(path)], set()
    while pending:
        current = pending.pop()
        relative = current.relative_to(root).as_posix()
        if relative in visited:
            continue
        if relative not in records:
            raise ValueError('dependency absent from authenticated inventory')
        visited.add(relative)
        for ref in direct_references(current):
            target = local_reference(current, ref, root)
            if target is not None:
                pending.append(target)
    return sorted(visited)


def asset_id(relative):
    p = PurePosixPath(relative)
    if p.name == 'model.xml' and p.parent.name in official.ASSETS:
        return p.parent.name
    stem = re.sub('[^a-zA-Z0-9_]+', '_', str(p.with_suffix(''))).strip('_').lower()
    return stem[:90] + '_' + hashlib.sha256(relative.encode()).hexdigest()[:8]


def discover(root, records):
    candidates, excluded, closures, errors = [], [], {}, {}
    model_paths = sorted(p for p in records if Path(p).suffix.lower() in FORMATS)
    referred = set()
    for relative in model_paths:
        suffix = Path(relative).suffix.lower()
        if suffix in USD_FORMATS:
            continue
        try:
            closure = dependencies(root / relative, root, records)
            closures[relative] = closure
            if suffix in {'.urdf', '.xml'} and Path(relative).name != 'output.xml':
                referred.update(set(closure) - {relative})
        except Exception as exc:
            errors[relative] = str(exc)
    seen = {}
    for relative in model_paths:
        path = relative.removeprefix('sources/')
        suffix = Path(path).suffix.lower()
        reason = None
        if path in SCENES:
            reason = 'multi_object_scene_or_scene_fixture'
        elif path in ALIASES:
            reason = 'alternate_representation:' + ALIASES[path]
        elif path.startswith('yup_zup_coverage/') and path != 'yup_zup_coverage/cannon_z.glb':
            reason = 'coordinate_or_format_variant:yup_zup_coverage/cannon_z.glb'
        elif Path(path).name == 'output.xml':
            reason = 'alternate_MJCF_use_model.xml'
        elif relative in referred or any(t in path.lower() for t in (
                '/collision/', '/body_collision_components/', '/props/', '_coll.',
                '_receptacle_mesh.', '/configuration/')):
            reason = 'dependency_or_collision_component'
        closure = closures.get(relative, [relative])
        # Do not merge distinct textures just because geometry bytes match.
        key = tuple(sorted((records[p]['sha256'], records[p]['size_bytes']) for p in closure))
        if reason is None and key in seen:
            reason = 'byte_identical_dependency_closure:' + seen[key]
        if reason:
            excluded.append(dict(path=relative, reason=reason))
            continue
        seen[key] = relative
        candidates.append(dict(asset_id=asset_id(path), entrypoint=relative,
                               format=suffix.lstrip('.'), dependency_paths=closure,
                               discovery_error=errors.get(relative)))
    return candidates, excluded


def verify_preview_index(path):
    path = local_path(path).resolve()
    index = read_json(path)
    if index['schema_version'] != SCHEMA:
        raise ValueError('invalid multi-format preview index schema')
    reference = index['source_inventory']
    manifest_path = Path(reference['path'])
    if sha256(manifest_path) != reference['sha256']:
        raise ValueError('source inventory changed')
    source_root, sources = verify_download(manifest_path)
    official.verify_files(path.parent, index['files'])
    expected_files = {r['path'] for r in index['files']} | {'asset_index.json'}
    actual_files = {p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*')
                    if p.is_file()}
    if expected_files != actual_files:
        raise ValueError('preview file set mismatch')
    candidates, excluded = discover(source_root, sources)
    expected = {c['asset_id']: c for c in candidates}
    if index['excluded'] != excluded or {a['asset_id'] for a in index['assets']} != set(expected):
        raise ValueError('preview discovery binding mismatch')
    if len(index['assets']) != len(expected):
        raise ValueError('duplicate preview asset')
    for item in index['assets']:
        candidate = expected[item['asset_id']]
        record = read_json(official.safe_file(path.parent, item['record']))
        if (record['asset_id'] != item['asset_id'] or record['status'] != item['status']
                or record['entrypoint'] != candidate['entrypoint']
                or not same_evidence_path(record['source_root'], str(source_root))
                or record['source_inventory'] != reference
                or record['format'] != candidate['format']):
            raise ValueError('preview asset binding mismatch')
        if item['status'] != 'preview_passed':
            continue
        if candidate['discovery_error'] is not None:
            raise ValueError('discovery failure cannot become preview success')
        if not record['source_files'] or candidate['entrypoint'] not in {
                r['path'] for r in record['source_files']}:
            raise ValueError('entrypoint absent from source hashes')
        for row in record['source_files']:
            if sources.get(row['path']) != row:
                raise ValueError('preview source file differs from inventory')
        if not set(candidate['dependency_paths']).issubset(
                r['path'] for r in record['source_files']):
            raise ValueError('preview dependency closure incomplete')
        preview = read_json(official.safe_file(path.parent, record['preview_result']))
        if (preview['status'] != 'passed' or preview['physics_steps'] != 0
                or preview['source_files'] != record['source_files']
                or not same_evidence_path(preview['source_root'], str(source_root))
                or preview['entrypoint'] != candidate['entrypoint']
                or len(preview['views']) != 6
                or {v['camera']['name'] for v in preview['views']} != set(VIEWS)
                or preview['material_status'] != 'passed'
                or preview['physics_status'] != 'not_evaluated'
                or preview['resolution'] != [512, 512]
                or preview['genesis_commit'] != official.GENESIS_COMMIT):
            raise ValueError('preview evidence mismatch')
        for view in preview['views']:
            expected_image = f"previews/{item['asset_id']}/view_{view['camera']['name']}.png"
            if view['image']['path'] != expected_image:
                raise ValueError('preview view mapping mismatch')
            official.verify_files(path.parent, [view['image']])
    return index
