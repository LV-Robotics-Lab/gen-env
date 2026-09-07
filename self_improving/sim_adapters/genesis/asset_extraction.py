"""Platform-only semantic assets: no tabletop SceneSpec, layout, or simulation."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from scene_gen import llm_provider as transport
from scene_gen.parser import FORBIDDEN_PROMPT_PATTERNS, OBJECT_TERMS
from self_improving.sim_adapters.genesis import clip_select as clip

SCHEMA = 'genenv.asset_request.v1'
VERSION = 'genenv.asset_extractor.v1'
CACHE_DIR = Path(__file__).resolve().parents[3] / '.cache/genesis/asset_parse_cache'
PROMPTS = Path(__file__).parent / 'prompts'
ENVIRONMENT = {'floor', 'ground', 'wall', 'ceiling', 'world', 'workspace', 'tabletop'}
TERMS = dict(OBJECT_TERMS, table=('table', 'desk', '桌子', '桌', '工作台'),
             chair=('chair', '椅子', '椅'), cabinet=('cabinet', '柜子', '柜', '橱柜'),
             shelf=('shelf', '架子', '书架'), sofa=('sofa', '沙发'), bed=('bed', '床'))
RELATIONS = {'on', 'inside', 'left_of', 'right_of', 'in_front_of', 'behind', 'near',
             'far_from', 'other'}


def boundary(text, *, request=False):
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError('text must contain 1 to 2000 characters')
    if request and len(text.strip()) < 3:
        raise ValueError('request must contain at least 3 characters')
    if any((ord(c) < 32 and c not in '\n\r\t') or ord(c) == 127
           or c in '\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069'
           for c in text):
        raise ValueError('forbidden control character')
    text.encode('utf-8')
    for pattern, label in FORBIDDEN_PROMPT_PATTERNS:
        if re.search(pattern, text, re.I):
            raise ValueError(f'forbidden {label}')


def exact_keys(data, keys):
    if not isinstance(data, dict) or set(data) != set(keys):
        raise ValueError('unexpected extraction fields')


def strings(value):
    if not isinstance(value, list) or len(value) > 64:
        raise ValueError('expected a bounded text list')
    for text in value:
        boundary(text)
    return value


def occurrences(text, phrase):
    return [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), text)]


def noun_matches(text):
    matches = []
    for category, terms in TERMS.items():
        for term in terms:
            pattern = (rf'\b{re.escape(term)}(?:s|es)?\b' if term.isascii()
                       else re.escape(term))
            for m in re.finditer(pattern, text, re.I):
                matches.append((m.start(), m.end(), category))
    selected = []
    for item in sorted(matches, key=lambda m: (m[0], m[0]-m[1])):
        if not any(item[0] < b and item[1] > a for a, b, _ in selected):
            selected.append(item)
    return selected


def quantity(phrase):
    match = re.match(r'\s*(\d+|[一二两三四五六七八九十]+)\s*[个只张把台件本]?\s*', phrase)
    if match:
        token = match[1]
        number = int(token) if token.isdigit() else {
            '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12,
        }.get(token, 99)
        return number, phrase[match.end():]
    match = re.match(r'\s*(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+',
                     phrase, re.I)
    if match:
        number = dict(a=1, an=1, one=1, two=2, three=3, four=4, five=5, six=6,
                      seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12)[match[1].lower()]
        return number, phrase[match.end():]
    return None, phrase


def clean_objects(document, request):
    exact_keys(document, ('objects', 'ambiguities'))
    strings(document['ambiguities'])
    objects = document['objects']
    if not isinstance(objects, list) or not 1 <= len(objects) <= 12:
        raise ValueError('expected 1 to 12 explicit objects')
    counts, covered, groups = Counter(), [], {}
    first_positions = []
    for obj in objects:
        exact_keys(obj, ('object_id', 'category', 'description', 'attributes', 'mentions'))
        category = obj['category']
        if not isinstance(category, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,47}', category):
            raise ValueError('invalid semantic category')
        if category in ENVIRONMENT:
            raise ValueError('environment surfaces are not retrieved assets')
        counts[category] += 1
        if obj['object_id'] != f'{category}_{counts[category]}':
            raise ValueError('object IDs must follow category introduction order')
        boundary(obj['description'])
        attrs, mentions = strings(obj['attributes']), strings(obj['mentions'])
        if not mentions or any(m not in request for m in mentions):
            raise ValueError('object mentions must be exact request substrings')
        spans = [span for mention in mentions for span in occurrences(request, mention)]
        covered.extend(spans)
        first_positions.append(min(a for a, _ in occurrences(request, mentions[0])))
        if category in TERMS and not any(c == category for m in mentions
                                        for _, _, c in noun_matches(m)):
            raise ValueError('category contradicts its source mentions')
        for attr in attrs:
            if not any(attr in mention for mention in mentions) or attr not in obj['description']:
                raise ValueError('attribute missing from source or retrieval description')
        # Full noun phrase coverage, including prints/handles, instead of a color-only query.
        n, phrase = quantity(mentions[0])
        if phrase not in obj['description']:
            # Short support words can be expanded, e.g. 桌 -> 桌子. No attribute may disappear.
            raise ValueError('retrieval description dropped part of the source noun phrase')
        groups.setdefault((category, mentions[0]), []).append(obj)
        if n is not None and not 1 <= n <= 12:
            raise ValueError('unsupported object quantity')
    if first_positions != sorted(first_positions):
        raise ValueError('objects must follow source introduction order')
    for (_, mention), members in groups.items():
        n, _ = quantity(mention)
        if ((n is not None and len(members) != n * len(occurrences(request, mention)))
                or (n is None and len(members) > len(occurrences(request, mention)))):
            raise ValueError('explicit quantity contradicts extracted instances')
    for a, b, _ in noun_matches(request):
        if not any(start <= a and b <= end for start, end in covered):
            raise ValueError('an explicitly mentioned object was omitted')
    # A narrowed model mention must not hide an explicit quantity or descriptive prefix.
    for a, b, category in noun_matches(request):
        prefix = re.split(r'[，。；,;.!?和与]', request[:a])[-1]
        matches = list(re.finditer(r'(?:\d+|[一二两三四五六七八九十]+)[个只张把台件本]'
                                  r'|\b(?:a|an|one|two|three|four|five)\s+', prefix, re.I))
        if not matches:
            continue
        q = matches[-1]
        between = prefix[q.end():]
        nondecorative = re.sub(r'(?:印有|印着).*?(?:图案|印花)', '', between)
        decorative_noun = re.match(r'(?:图案|印花)', request[b:])
        if noun_matches(nondecorative) or decorative_noun:
            continue  # Decorative nouns are not independent instances.
        n, _ = quantity(prefix[q.start():])
        members = [o for o in objects if o['category'] == category and any(
            start <= a and b <= end for m in o['mentions']
            for start, end in occurrences(request, m))]
        if len(members) != n:
            raise ValueError('explicit source quantity was lost')
        phrase = between + request[a:b]
        if any(phrase not in o['description'] for o in members):
            raise ValueError('source attributes were dropped from retrieval description')
    return objects


def clean_relations(document, request, objects):
    exact_keys(document, ('relations', 'ambiguities'))
    strings(document['ambiguities'])
    rows = document['relations']
    if not isinstance(rows, list) or len(rows) > 64:
        raise ValueError('expected at most 64 explicit relations')
    ids, seen = {o['object_id'] for o in objects}, set()
    for row in rows:
        exact_keys(row, ('relation', 'source', 'target', 'evidence'))
        if (row['relation'] not in RELATIONS or row['source'] not in ids
                or row['target'] not in ids or row['source'] == row['target']):
            raise ValueError('invalid or unbound explicit relation')
        boundary(row['evidence'])
        if row['evidence'] not in request:
            raise ValueError('relation evidence must be an exact request substring')
        key = (row['relation'], row['source'], row['target'])
        if key in seen:
            raise ValueError('duplicate relation')
        cues = {'on': r'上|\bon\b', 'inside': r'里|内|中|inside|into|contains|\bin\b',
                'left_of': r'左|left', 'right_of': r'右|right',
                'in_front_of': r'前|front', 'behind': r'后|behind', 'near': r'旁|近|near|next',
                'far_from': r'远|far|away'}
        if row['relation'] in cues and not re.search(cues[row['relation']], row['evidence'], re.I):
            raise ValueError('relation not expressed by its evidence')
        seen.add(key)
    return rows


class AssetProvider:
    def __init__(self, config, *, transport_fn=None, cache_dir=CACHE_DIR):
        self.config = replace(config, timeout_s=60, max_attempts=1)
        self.send = transport_fn or (lambda system, user:
                                    transport._http_transport(self.config, system, user))
        self.cache_dir = Path(cache_dir)
        self.prompts = {stage: (PROMPTS/f'asset_{stage}.md').read_text()
                        for stage in ('objects', 'relations')}
        self.last_evidence = {}

    def evidence(self):
        return self.last_evidence

    def extract(self, request, seed=42, *, on_objects=None):
        boundary(request, request=True)
        hashes = {k: clip.digest(v) for k, v in self.prompts.items()}
        key = clip.digest(dict(schema=SCHEMA, version=VERSION, request=request, seed=seed,
                               config=self.config.fingerprint(), prompts=hashes))
        self.last_evidence = dict(schema_version=VERSION, status='running', calls=0,
                                  cache=dict(key=key, hit=False), prompt_hashes=hashes,
                                  config=self.config.safe_dict(), stages={})
        clip.writable_storage(self.cache_dir)
        path = self.cache_dir/f'{key}.json'
        try:
            cached = None
            if path.exists():
                entry = clip.strict_json(path.read_text())
                if entry['key'] != key or clip.digest(entry['payload']) != entry['payload_sha256']:
                    raise ValueError('extraction cache integrity mismatch')
                cached = entry['payload']
                self.last_evidence['cache']['hit'] = True
            payload = {}
            objects, relations = [], []
            for stage in ('objects', 'relations'):
                if cached is not None:
                    document = cached[stage]
                    if self.config.api_key in json.dumps(document, ensure_ascii=False):
                        raise ValueError('cache contained secret material')
                    self.last_evidence['stages'][stage] = dict(response=document)
                else:
                    user = dict(request=request)
                    if stage == 'relations':
                        user['objects'] = objects
                    self.last_evidence['calls'] += 1
                    raw = self.send(self.prompts[stage], json.dumps(user, ensure_ascii=False))
                    if not isinstance(raw, str):
                        raise ValueError('model response must be text')
                    safe = raw.replace(self.config.api_key, '[REDACTED]')
                    self.last_evidence['stages'][stage] = dict(response=safe)
                    if raw != safe:
                        raise ValueError('model response contained secret material')
                    document = transport._strict_json_loads(raw)
                # Ambiguity wins over validation: retain the original diagnostic, never select.
                if isinstance(document, dict) and document.get('ambiguities'):
                    strings(document['ambiguities'])
                    self.last_evidence['ambiguities'] = document['ambiguities']
                    raise ValueError('extraction is ambiguous')
                if stage == 'objects':
                    objects = clean_objects(document, request)
                    if on_objects:
                        on_objects(objects)
                else:
                    relations = clean_relations(document, request, objects)
                payload[stage] = document
                self.last_evidence['stages'][stage]['validation'] = 'passed'
            if cached is None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                clip.write_json(path, dict(key=key, payload=payload,
                                           payload_sha256=clip.digest(payload)))
            self.last_evidence['status'] = 'passed'
            return dict(schema_version=SCHEMA, request=request, objects=objects,
                        relations=relations,
                        environment=dict(ground='genesis_builtin', physics_status='not_run'))
        except Exception as exc:
            self.last_evidence.update(status='error', error=type(exc).__name__)
            raise
