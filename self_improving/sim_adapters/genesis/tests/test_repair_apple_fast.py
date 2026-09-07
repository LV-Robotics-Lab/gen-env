"""Bounded native calls and serialized qualification for authorized apple generation."""

import copy
import json
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from self_improving.sim_adapters.genesis import repair_apple_fast as apple
from self_improving.sim_adapters.genesis import repair_collision_v3 as v3


@pytest.mark.parametrize('remaining', [1200, 75])
def test_native_timeout_kills_child_without_new_process_group(tmp_path, monkeypatch, remaining):
    seen = {}

    class Child:
        def wait(self, timeout=None):
            if timeout is not None:
                seen['timeout'] = timeout
                raise subprocess.TimeoutExpired('coacd', timeout)
            seen['joined'] = True
            return -9

        def kill(self):
            seen['killed'] = True

    def popen(*args, **kwargs):
        assert not kwargs.get('start_new_session')
        return Child()

    monkeypatch.setattr(apple.subprocess, 'Popen', popen)
    mesh = SimpleNamespace(vertices=np.zeros((3, 3)), indices=np.array([[0, 1, 2]]))
    with pytest.raises(TimeoutError, match='native CoACD'):
        apple.bounded_coacd(mesh, {}, dict(output_dir=str(tmp_path), identity={}), 0,
                            time.monotonic() + remaining)
    assert 0 < seen['timeout'] <= min(remaining, 300)
    assert seen['joined'] and seen['killed']
    timing = json.loads((tmp_path / 'coacd_calls/000/timing.json').read_text())
    assert timing['status'] == 'coacd_timeout'


def test_serialized_parts_are_qualified_and_mutations_rejected(tmp_path, monkeypatch):
    import coacd
    visual = trimesh.creation.box([.1, .1, .1])
    candidate = apple.candidates(len(visual.faces))[0]
    data = dict(strategy='apple_fast_v1', candidate=candidate, surface=None,
                coacd_options=apple.options(candidate), output_dir=str(tmp_path),
                identity=v3.identity(visual, None, candidate, 'apple_fast_v1'))
    for key, value in dict(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
                           MKL_NUM_THREADS='1').items():
        monkeypatch.setenv(key, value)
    original_run, original_options = coacd.run_coacd, copy.deepcopy(apple.prep.COACD)
    part = tmp_path / 'parts/part_000.obj'
    part.parent.mkdir()
    visual.export(part)
    called = []

    def quality(reference, parts, surface=None, **kwargs):
        called.append(True)
        assert np.allclose(parts[0].vertices, visual.vertices)
        part.write_text(part.read_text() + '\n# changed during gate\n')
        return dict(passed=True)

    monkeypatch.setattr(apple.collision, 'quality_gate', quality)
    monkeypatch.setattr(apple.collision, 'worker', lambda request:
                        apple.collision.quality_gate(visual, [trimesh.creation.box()]))
    with pytest.raises(ValueError, match='changed during qualification'):
        apple.worker(data)
    assert called
    assert apple.prep.COACD == original_options
    assert coacd.run_coacd is original_run
    assert apple.collision.quality_gate is quality
