"""Generation extensions cannot change the qualification or cache trust boundary."""

import copy
import json
import time

import pytest
import trimesh

from self_improving.sim_adapters.genesis import repair_assets as prep
from self_improving.sim_adapters.genesis import repair_collision_v3 as v3
from self_improving.sim_adapters.genesis import repair_apple_fast as apple
from self_improving.sim_adapters.genesis.tests.test_repair_collision_v2 import successful_worker


def fake_success(request, directory, visual):
    result = successful_worker(request, directory, visual)
    result['parts_sha256'] = {n: prep.lib.sha256(directory / n) for n in result['parts']}
    return result


def test_candidate_fallback_and_cache_requalification(tmp_path, monkeypatch):
    visual = trimesh.creation.box([.1, .1, .1])
    calls = []

    def run(path, directory, timeout):
        request = json.loads(path.read_text())
        calls.append(request)
        assert 300 < timeout <= 1200
        if request['candidate']['max_convex_hull'] == 256:
            return dict(passed=False, status='quality_rejected')
        if request.get('reuse_dir'):
            return v3.worker(request)
        return fake_success(request, directory, visual)

    monkeypatch.setattr(v3, 'run_worker', run)
    cache = tmp_path / 'cache'
    for index in range(2):
        parts, report = v3.decompose(visual, None, tmp_path / str(index),
                                    category='apple', cache_root=cache)
        assert len(parts) == 1
        assert report['policy'] == 'collision_repair_v3'
        assert report['cache_hit'] == bool(index)
        assert report['quality']['passed']
    assert [r['candidate']['max_convex_hull'] for r in calls] == [256, 512, 256, 512]
    assert calls[-1]['reuse_parts_sha256']
    assert report['coacd_executed'] is False
    assert report['operation'] == 'cached_quality_recheck'


@pytest.mark.parametrize('attack', ['part', 'missing_hash', 'identity', 'quality', 'input'])
def test_worker_output_tampering_rejected(tmp_path, monkeypatch, attack):
    visual = trimesh.creation.box([.1, .1, .1])

    def run(path, directory, timeout):
        request = json.loads(path.read_text())
        result = fake_success(request, directory, visual)
        if attack == 'part':
            (directory / result['parts'][0]).write_text('changed')
        elif attack == 'missing_hash':
            result.pop('parts_sha256')
        elif attack == 'identity':
            result['request'] = {}
        elif attack == 'quality':
            result['quality']['passed'] = False
        else:
            (directory / next(iter(result['processed_input_sha256']))).write_text('changed')
        return result

    monkeypatch.setattr(v3, 'run_worker', run)
    with pytest.raises(ValueError):
        v3.decompose(visual, None, tmp_path / 'asset', category='apple',
                     cache_root=tmp_path / 'cache')
    assert not (tmp_path / 'asset/collision_success.json').exists()


@pytest.mark.parametrize('field', ['candidate', 'surface', 'coacd_options', 'coacd_timeout_s'])
def test_request_options_bound_to_identity(tmp_path, field):
    visual = trimesh.creation.box([.1, .1, .1])
    candidate = apple.candidates(len(visual.faces))[0]
    request = dict(strategy='apple_fast_v1', candidate=candidate, surface=None,
                   coacd_options=apple.options(candidate), coacd_timeout_s=300,
                   identity=v3.identity(visual, None, candidate, 'apple_fast_v1'))
    request = copy.deepcopy(request)
    request[field] = 'tampered'
    with pytest.raises(ValueError):
        v3.worker(request)


def test_publication_cannot_reseal_changed_parts(tmp_path, monkeypatch):
    visual = trimesh.creation.box([.1, .1, .1])
    monkeypatch.setattr(v3, 'run_worker', lambda p, d, t: fake_success(
        json.loads(p.read_text()), d, visual))
    publish = v3.gate._publish_cache

    def changed(cache, frozen, directory, result, **kwargs):
        path = directory / result['parts'][0]
        path.write_text(path.read_text() + '\n# changed after qualification\n')
        return publish(cache, frozen, directory, result, **kwargs)

    monkeypatch.setattr(v3.gate, '_publish_cache', changed)
    with pytest.raises(ValueError, match='publication differs'):
        v3.decompose(visual, None, tmp_path / 'asset', category='apple',
                     cache_root=tmp_path / 'cache')


def test_expired_budget_does_not_start_generator(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        pytest.fail('generator started after asset deadline')
    monkeypatch.setattr(v3, 'run_worker', forbidden)
    with pytest.raises(ValueError, match='ASSET_PREPARATION_FAILED'):
        v3.decompose(trimesh.creation.box(), None, tmp_path, category='apple',
                     deadline=time.monotonic() - 1, cache_root=tmp_path / 'cache')
    failure = json.loads((tmp_path / 'collision_failure.json').read_text())
    assert failure['budget_exhausted']
    assert all(a['elapsed_s'] == 0 for a in failure['attempts'])


def test_apple_strategy_finite_and_keeps_gate_parameters():
    candidates = apple.candidates(50000)
    assert [c['max_convex_hull'] for c in candidates] == [256, 512]
    for c in candidates:
        options = apple.options(c)
        assert options['seed'] == 0
        assert options['preprocess_mode'] == 'off'
        assert options['mcts_iterations'] == 5
    with pytest.raises(ValueError):
        apple.options(dict(candidates[0], threshold=.5))
