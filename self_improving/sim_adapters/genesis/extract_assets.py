"""Extract every explicit asset and select bindings; then build an initial scene by default."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scene_gen.llm_provider import load_llm_provider_config
from self_improving.sim_adapters.genesis import asset_extraction as extraction
from self_improving.sim_adapters.genesis import asset_library as library
from self_improving.sim_adapters.genesis import build_official_index as official
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis.storage_paths import same_evidence_path
from self_improving.sim_adapters.genesis.task_output import TaskOutput, destination


def verified_binding(obj, directory, index, index_hash):
    """Resolve asset identity/paths from the trusted index, regardless of model format."""
    result = library.read_json(directory/'run_report.json')
    official.verify_files(directory, result['files'])
    selected = library.read_json(directory/'selected_asset.json')
    if (result['status'] != 'selected' or selected['query'] != obj['description']
            or selected['query_sha256'] != extraction.transport._sha256_text(obj['description'])):
        raise ValueError('selected query mismatch')
    if (selected['clip_index_sha256'] != index_hash
            or selected['official_index'] != index['official_index']):
        raise ValueError('selected index mismatch')
    asset = next(a for a in index['assets'] if a['asset_id'] == selected['asset_id'])
    if index.get('physics_after_selection') or 'standard_package' in asset:
        from self_improving.sim_adapters.genesis.validate_single_asset import verify_evidence
        verify_evidence(directory, selected)
    source_root = Path(asset.get('source_root', Path(index['official_index']['path']).parent))
    source = official.safe_file(source_root, asset['entrypoint'])
    if (selected['source_files'] != asset['source_files']
            or selected['official_record'] != asset['record']
            or not same_evidence_path(selected['model_entrypoint'], str(source))):
        raise ValueError('selected source binding mismatch')
    official.verify_files(source_root, asset['source_files'])
    return dict(asset_id=asset['asset_id'], selection_file=str(directory/'selected_asset.json'),
                selection_sha256=library.sha256(directory/'selected_asset.json'),
                model_entrypoint=selected['model_entrypoint'], source_root=str(source_root),
                source_files=asset['source_files'], official_index=index['official_index'],
                visible_differences=selected['visible_differences'])


def selection_error(directory, reason):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'selected_asset.json').unlink(missing_ok=True)
    clip.write_json(directory/'binding_validation.json', dict(status='error', error=reason))
    report_path = directory/'run_report.json'
    report = library.read_json(report_path) if report_path.exists() else dict(vlm_calls=0)
    report.update(status='error', error=reason)
    report['files'] = [official.fingerprint(p, directory) for p in sorted(directory.iterdir())
                       if p.is_file() and p != report_path]
    clip.write_json(report_path, report)


def _extract(request, output_dir=None, *, output_root=None, clip_index, vlm_config, seed=42,
        provider=None, selector=None):
    output = destination(request, output_dir=output_dir, output_root=output_root)
    clip.separate(output, Path(clip_index).resolve().parent, clip.WEIGHTS_DIR, clip.CACHE_DIR,
                  extraction.CACHE_DIR)
    task = TaskOutput(output)
    with task.lock():
        task.start(request)
        started = time.perf_counter()
        report = dict(schema_version='genenv.asset_selection_run.v1', status='error', stage='parse',
                      extraction_status='error', output_dir=str(output), selections=[],
                      timings_s={}, physics_steps=0, physics_status='not_run', parse_calls=0)
        objects_dir = task.stage('objects')
        records = objects_dir/'parse_records'
        records.mkdir()
        config = None
        try:
            extraction.boundary(request, request=True)
            config = load_llm_provider_config(vlm_config)
            provider = provider or extraction.AssetProvider(config)
            clip.separate(output, getattr(provider, 'cache_dir', extraction.CACHE_DIR))

            def save_objects(objects):
                for obj in objects:
                    clip.write_json(objects_dir/f"{obj['object_id']}.json", obj)

            tick = time.perf_counter()
            document = provider.extract(request, seed, on_objects=save_objects)
            report['timings_s']['parse'] = time.perf_counter()-tick
            # Revalidate injected providers as well as online/cached output.
            objects = extraction.clean_objects(dict(objects=document['objects'], ambiguities=[]),
                                                request)
            extraction.clean_relations(dict(relations=document['relations'], ambiguities=[]),
                                       request, objects)
            save_objects(objects)
            clip.write_json(objects_dir/'asset_request.json', document)
            clip.write_json(objects_dir/'relations.json', document['relations'])
            clip.write_json(objects_dir/'object_queries.json', [dict(object_id=o['object_id'],
                            query=o['description']) for o in objects])
            report.update(extraction_status='passed', stage='selection', object_count=len(objects))
            index, _ = clip.load_index(clip_index)
            index_hash = library.sha256(clip_index)
            tick = time.perf_counter()
            for obj in objects:
                directory = objects_dir/'asset_selection'/obj['object_id']
                item = dict(object_id=obj['object_id'], query=obj['description'], status='error',
                            vlm_calls=0)
                try:
                    result = (selector or clip.select)(clip_index, obj['description'], directory,
                                                       vlm_config=config)
                    if result['status'] not in {'selected', 'rejected', 'error'}:
                        raise ValueError('invalid selection status')
                    item.update(status=result['status'], vlm_calls=result['vlm_calls'],
                                cache=result.get('cache', {}))
                    if result.get('physics_status') != 'not_evaluated':
                        item['asset_physics_status'] = result.get('physics_status')
                        item['asset_physics_exit_code'] = result.get('exit_code')
                        item['asset_physics_evidence'] = str(
                            directory/'asset_physics/physics_result.json')
                    if result['status'] == 'selected':
                        item.update(verified_binding(obj, directory, index, index_hash))
                    elif result['status'] == 'error':
                        item['error'] = result.get('error', 'selection_failed')
                except Exception as exc:
                    item.update(status='error', error=type(exc).__name__)
                    # A selector failure must never leave a success binding behind.
                    selection_error(directory, type(exc).__name__)
                report['selections'].append(item)
                clip.write_json(objects_dir/'asset_resolution.json', dict(
                    schema_version='genenv.asset_resolution.v1', objects=report['selections'],
                    status='running', physics_status='not_run'))
            report['timings_s']['selection'] = time.perf_counter()-tick
            clip.load_index(clip_index)
            if library.sha256(clip_index) != index_hash:
                raise ValueError('index changed during selection')
            for obj, item in zip(objects, report['selections'], strict=True):
                if item['status'] == 'selected':
                    verified_binding(obj, objects_dir/'asset_selection'/obj['object_id'],
                                     index, index_hash)
            status = ('assets_selected' if all(i['status'] == 'selected'
                                               for i in report['selections']) else 'partial')
            report.update(status=status, stage='complete')
        except Exception as exc:
            message = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            if config is not None:
                message = message.replace(config.api_key, '[REDACTED]')
            report['error'] = message
            if report['stage'] == 'selection':
                for item in report['selections']:
                    if item['status'] == 'selected':
                        item.update(status='error', error='run_integrity_failed')
                        item.pop('selection_file', None)
                        selection_error(objects_dir/'asset_selection'/item['object_id'],
                                        'run_integrity_failed')
        finally:
            report['timings_s']['total'] = time.perf_counter()-started
            if provider is not None:
                evidence = provider.evidence()
                report['parse_calls'] = evidence.get('calls', 0)
                report['parse_cache'] = evidence.get('cache', {})
                clip.write_json(records/'extraction_evidence.json', evidence)
            clip.write_json(objects_dir/'asset_resolution.json', dict(
                schema_version='genenv.asset_resolution.v1', status=report['status'],
                objects=report['selections'], physics_status='not_run'))
            task.finish_assets(report)
        return dict(report, input_manifest_sha256=library.sha256(task.root/'manifest.json'))


def run(request, output_dir=None, *, output_root=None, clip_index, vlm_config, seed=42,
        provider=None, selector=None, stop_after='scene', planner='llm', llm_config=None,
        scene_provider=None, scene_renderer=None):
    if stop_after not in {'assets', 'scene'} or planner not in {'llm', 'rule'}:
        raise ValueError('invalid stop-after or planner')
    report = _extract(request, output_dir, output_root=output_root, clip_index=clip_index,
                      vlm_config=vlm_config, seed=seed, provider=provider, selector=selector)
    manifest_hash = report.pop('input_manifest_sha256')
    if report['status'] == 'assets_selected' and stop_after == 'scene':
        from self_improving.sim_adapters.genesis.build_scene import run as build
        # The asset lock has been released. Recheck its exact manifest under the scene lock.
        scene = build(report['output_dir'], clip_index, planner=planner,
                      llm_config=llm_config if llm_config is not None else vlm_config, seed=seed,
                      provider=scene_provider, renderer=scene_renderer,
                      expected_manifest_sha256=manifest_hash)
        report = dict(report, status=scene['status'], stage='scene', scene_build=scene)
        if 'error' in scene:
            report['error'] = scene['error']
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument('--output-root', type=Path, help='默认 output/，按原始请求命名')
    outputs.add_argument('--output-dir', type=Path, help='显式任务目录，同请求覆盖')
    parser.add_argument('--clip-index', required=True, type=Path)
    parser.add_argument('--vlm-config', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stop-after', choices=['assets', 'scene'], default='scene')
    parser.add_argument('--planner', choices=['llm', 'rule'], default='llm')
    parser.add_argument('--llm-config', type=Path, help='场景规划配置；默认沿用 --vlm-config')
    report = run(**vars(parser.parse_args(argv)))
    print(dict(output=report['output_dir'], status=report['status'], error=report.get('error')))
    return 0 if report['status'] in {'assets_selected', 'scene_built'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
