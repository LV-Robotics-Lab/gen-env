"""Opt-in real non-robot retrieval with trace artifacts; never changes expected answers."""
import base64
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis.vision_request import ChatVisionClient

pytestmark = pytest.mark.skipif(os.environ.get('GENESIS_LIBRARY_CLIP_REAL') != '1',
                                reason='explicit real CLIP and online VLM opt-in required')
CASES = [
    ('mango', '我要一个黄绿色的芒果。', 'glb_mango_b6d2f230'),
    ('yellow_bowl', '一个黄色塑料碗',
     'table_bussing_yellow_plastic_bowl_yellow_plastic_bowl_c981d094'),
    ('green_bin', '绿色垃圾桶', 'table_bussing_green_trash_bin_green_trash_bin_bb0c682f'),
]


@pytest.fixture(scope='module')
def library_runs():
    destination = Path(os.environ['GENESIS_LIBRARY_CLIP_OUTPUT']).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    index_path = Path(os.environ.get('GENESIS_LIBRARY_CLIP_INDEX',
                                    'assets/genesis/clip_non_robot_v1/index.json'))
    index, vectors = clip.load_index(index_path)
    config = replace(clip.load_llm_provider_config('configs/llm.yaml'),
                     timeout_s=60, max_attempts=1)
    client = ChatVisionClient(config)
    original = clip.ChineseClipEncoder.encode_text
    runs, summary = {}, []
    active = destination

    def encode_and_capture(self, query):
        raw = original(self, query)
        text = clip.normalize(raw)
        np.save(active / 'text_vector.npy', text, allow_pickle=False)
        scores = vectors @ text[0]
        clip.write_json(active / 'all_view_scores.json', [dict(row, score=float(score))
                        for row, score in zip(index['rows'], scores, strict=True)])
        return raw

    def request_and_capture(messages):
        payload = dict(model=config.model, messages=messages,
                       response_format={'type': 'json_object'}, max_tokens=800)
        if config.temperature is not None:
            payload['temperature'] = config.temperature
        clip.write_json(active / 'vlm_request.json', payload)  # body only, no auth headers
        number, view = None, 0
        for item in messages[1]['content'][1:]:
            if item['type'] == 'text':
                number, view = int(item['text'].split()[-1]), 0
            elif item['type'] == 'image_url':
                view += 1
                data = base64.b64decode(item['image_url']['url'].split(',', 1)[1])
                (active / f'candidate_{number}_view_{view}.png').write_bytes(data)
        return client(messages)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(clip.ChineseClipEncoder, 'encode_text', encode_and_capture)
        for name, query, expected in CASES:
            active = destination / name
            report = clip.select(index_path, query, active, vlm_config=config,
                                 cache_dir=destination / 'cache', vlm=request_and_capture)
            retrieval = json.loads((active / 'retrieval_result.json').read_text())
            selected = active / 'selected_asset.json'
            binding = json.loads(selected.read_text()) if selected.exists() else None
            candidates = [c['asset_id'] for c in retrieval['candidates']]
            row = dict(name=name, query=query, expected_asset=expected, candidates=candidates,
                       top3_hit=expected in candidates, status=report['status'],
                       selected_asset=binding['asset_id'] if binding else None,
                       timings_s=report['timings_s'], vlm_calls=report['vlm_calls'],
                       cache=report['cache'], error=report.get('error'),
                       visible_differences=binding['visible_differences'] if binding else [])
            summary.append(row)
            clip.write_json(destination / 'acceptance_summary.json', summary)
            runs[name] = row
        active = destination / 'mango_cached'
        runs['cache'] = clip.select(index_path, CASES[0][1], active, vlm_config=config,
                                    cache_dir=destination / 'cache', vlm=request_and_capture)
    # Check all trace artifacts without printing credentials on failure.
    for path in destination.rglob('*.json'):
        assert config.api_key not in path.read_text(), 'credential found in trace'
    return runs


@pytest.mark.parametrize('name,query,expected', CASES)
def test_new_library_assets(library_runs, name, query, expected):
    row = library_runs[name]
    assert row['top3_hit'], row
    assert row['status'] == 'selected' and row['selected_asset'] == expected, row
    assert row['vlm_calls'] == 1


def test_library_cache(library_runs):
    report = library_runs['cache']
    assert report['status'] == 'selected' and report['cache']['hit']
    assert report['vlm_calls'] == 0 and report['image_encoding_calls'] == 0
