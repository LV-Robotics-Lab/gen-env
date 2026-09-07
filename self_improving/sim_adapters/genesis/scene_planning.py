"""Auditable two-attempt LLM planning, successful-plan cache and geometric validation."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from scene_gen import llm_provider as transport
from self_improving.sim_adapters.genesis import clip_select as clip
from self_improving.sim_adapters.genesis import scene_layout as spatial

CACHE_DIR = Path(__file__).resolve().parents[3]/'.cache/genesis/scene_plan_cache'
PROMPT = Path(__file__).parent/'prompts/scene_plan.md'


class ScenePlanner:
    def __init__(self, config, *, transport_fn=None, cache_dir=CACHE_DIR):
        self.config = replace(config, timeout_s=60, max_attempts=1)
        self.send = transport_fn or (lambda system, user:
                                    transport._http_transport(self.config, system, user))
        self.cache_dir = Path(cache_dir)
        self.prompt = PROMPT.read_text()
        self.transport_mode = 'injected' if transport_fn else 'http'
        self.reset_evidence()

    def reset_evidence(self):
        self.evidence = dict(status='not_run', calls=0, attempts=[], cache=dict(hit=False),
                             transport=self.transport_mode,
                             config=self.config.safe_dict(), prompt_sha256=clip.digest(self.prompt))

    def plan(self, document, bindings, geometry, seed, check_inputs, output):
        self.reset_evidence()
        spatial.relations(document)
        payload = dict(request=document['request'],
                       object_ids=[o['object_id'] for o in document['objects']],
                       objects=[dict(object_id=o['object_id'], description=o['description'],
                       bounds_m=geometry[o['object_id']]['bounds'],
                       support_surface=geometry[o['object_id']].get('surface'))
                       for o in document['objects']], explicit_relations=document['relations'])
        key = clip.digest(dict(input=payload, bindings=bindings, geometry=geometry, seed=seed,
                               config=self.config.fingerprint(), prompt=self.prompt,
                               schema=spatial.SCHEMA, solver=spatial.VERSION))
        self.evidence.update(status='running', cache=dict(key=key, hit=False))
        clip.writable_storage(self.cache_dir)
        path = self.cache_dir/f'{key}.json'
        cached = None
        try:
            check_inputs()
            if path.exists():
                entry = clip.strict_json(path.read_text())
                if (entry['key'] != key or clip.digest(entry['proposal']) != entry['sha256']
                        or self.config.api_key in json.dumps(entry, ensure_ascii=False)):
                    raise ValueError('scene planning cache integrity mismatch')
                cached = entry['proposal']
                self.evidence['cache']['hit'] = True
            feedback = None
            for attempt in range(1 if cached is not None else 2):
                check_inputs()
                user = dict(payload)
                if feedback is not None:
                    user['feedback'] = feedback
                record = dict(attempt=attempt+1, request=user)
                self.evidence['attempts'].append(record)
                if cached is not None:
                    raw = json.dumps(cached, ensure_ascii=False)
                else:
                    self.evidence['calls'] += 1
                    # Transport errors are terminal, never disguised as repairable layout errors.
                    raw = self.send(self.prompt, json.dumps(user, ensure_ascii=False))
                check_inputs()
                if not isinstance(raw, str) or len(raw.encode('utf-8')) > 262144:
                    raise RuntimeError('invalid or oversized model response')
                safe = raw.replace(self.config.api_key, '[REDACTED]')
                record['response'] = safe
                if raw != safe:
                    raise RuntimeError('model response contained secret material')
                try:
                    proposal = transport._strict_json_loads(raw)
                    spatial.validate_proposal(proposal, document)
                    graph = spatial.graph_for(document, bindings, proposal)
                    clip.write_json(output/'scene_graph.json', graph)
                    layout, validation = spatial.solve(document, bindings, geometry, graph, seed)
                except spatial.UnsupportedScene:
                    raise
                except (ValueError, TypeError, KeyError) as exc:
                    reason = str(exc).replace(self.config.api_key, '[REDACTED]')
                    record.update(status='failed', error=reason)
                    validation = getattr(exc, 'trace', dict(status='failed', error=reason))
                    clip.write_json(output/f'layout_attempt_{attempt+1}.json', validation)
                    record['validation_file'] = f'layout_attempt_{attempt+1}.json'
                    clip.write_json(output/'layout_validation_report.json', validation)
                    if cached is not None:
                        raise ValueError('cached scene plan failed revalidation') from exc
                    if attempt == 1:
                        raise ValueError(
                            f'scene planning exhausted two attempts: {reason}') from exc
                    feedback = dict(error=reason, previous_proposal=safe,
                                    recent_layout_failures=validation.get('attempts', [])[-8:])
                    continue
                check_inputs()
                record['status'] = 'passed'
                self.evidence['status'] = 'passed'
                clip.write_json(output/'layout_validation_report.json', validation)
                if cached is None:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(prefix='.plan-', dir=self.cache_dir)
                    try:
                        with os.fdopen(fd, 'w') as handle:
                            json.dump(dict(key=key, proposal=proposal,
                                           sha256=clip.digest(proposal)),
                                      handle, ensure_ascii=False)
                        os.replace(temporary, path)
                    finally:
                        Path(temporary).unlink(missing_ok=True)
                return layout
        except Exception as exc:
            self.evidence.update(status='error', error=type(exc).__name__)
            raise
        finally:
            clip.write_json(output/'planning_evidence.json', self.evidence)
