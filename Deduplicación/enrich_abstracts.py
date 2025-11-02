import os
import json
import time
import argparse
import csv
import sys
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from langdetect import detect

# Load env (cargar desde la raíz del repo para soportar ejecución desde subcarpetas)
REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / '.env', encoding='utf-8-sig')
load_dotenv(override=False, encoding='utf-8-sig')  # también desde CWD para permitir overrides locales
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
HTTP_REFERER = os.getenv('OPENROUTER_SITE_URL', '')
X_TITLE = os.getenv('OPENROUTER_APP_NAME', 'SLR Abstract Enricher')
CROSSREF_MAILTO = os.getenv('CROSSREF_MAILTO', '')
ENABLE_SCRAPING = os.getenv('ENABLE_SCRAPING', 'false').lower() == 'true'
OVERWRITE_IF_BETTER = os.getenv('OVERWRITE_IF_BETTER', 'true').lower() != 'false'

USER_AGENT = (
    f"abstract-enricher/1.0 (+mailto:{CROSSREF_MAILTO})"
    if CROSSREF_MAILTO
    else 'abstract-enricher/1.0 (+mailto:unknown@example.com)'
)
HTTP_TIMEOUT = 30

# Policies
LANG_POLICY = 'preserve'
TARGET_WORDS = (120, 200)
MIN_CHARS = 400
MIN_KEYWORDS = 2
MIN_ABSTRACT_VALID_CHARS = 400
BANNED_ABSTRACT_PHRASES = ['graphical abstract', 'no abstract available', 'no abstract provided']
KEYWORDS = [
    'rag', 'retrieval-augmented', 'llm', 'large language model',
    'hallucination', 'faithfulness', 'factuality', 'trust', 'confidence', 'credibility',
    'citation', 'grounding', 'provenance', 'explainability', 'xai',
    'user perception', 'perceived reliability', 'overtrust', 'trustworthiness',
    'hallucinations', 'accuracy', 'grounded', 'correctness', 'consistency',
    'explainable ai', 'interpretable', 'interpretability',
    'transparency', 'attribution', 'saliency', 'attention',
    'citations', 'reference', 'source attribution',
    'evidence', 'evidence highlighting', 'highlighting',
    'calibration', 'brier', 'ece',
    'evaluation', 'evaluation metric', 'evaluation metrics', 'metric', 'metrics',
    'methodology', 'methodologies', 'method', 'methods', 'protocol',
    'instrument', 'questionnaire', 'scale', 'survey', 'user study',
    'benchmark', 'dataset', 'guideline', 'framework', 'pipeline'
]

sys.setrecursionlimit(max(5000, sys.getrecursionlimit()))

# OpenRouter client
client = None
if OPENROUTER_API_KEY:
    client = OpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=OPENROUTER_API_KEY,
        default_headers={'HTTP-Referer': HTTP_REFERER, 'X-Title': X_TITLE},
    )


class DummyHTTPResponse:
    def __init__(self, *, json_data=None, text='', status_code=200):
        self._json = json_data
        self.text = text
        self.status_code = status_code

    def json(self):
        return self._json if self._json is not None else {}

    def raise_for_status(self):
        if 400 <= self.status_code:
            raise httpx.HTTPStatusError('error', request=None, response=None)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def chat_complete(model, system, user):
    if client is None:
        raise RuntimeError('OpenRouter API key not configured')
    return client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        temperature=0.2,
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text or '').strip()


def strip_jats(text: str) -> str:
    if not text:
        return ''
    soup = BeautifulSoup(text, 'lxml')
    return normalize_whitespace(soup.get_text(separator=' ', strip=True))


def is_valid_abs(text: str) -> bool:
    cleaned = normalize_whitespace(text)
    if len(cleaned) < MIN_ABSTRACT_VALID_CHARS:
        return False
    lowered = cleaned.lower()
    if any(phrase in lowered for phrase in BANNED_ABSTRACT_PHRASES):
        return False
    return True


def score_abstract(text: str) -> int:
    return len(normalize_whitespace(text))


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def _build_headers():
    return {'User-Agent': USER_AGENT}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def fetch_crossref_abstract(doi: str) -> tuple[str, str]:
    doi = (doi or '').strip()
    if not doi:
        return '', 'none'
    encoded = quote(doi)
    url = f'https://api.crossref.org/works/{encoded}'
    try:
        response = httpx.get(url, headers=_build_headers(), timeout=HTTP_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # 4xx responses (e.g. missing DOI) are not retriable; surface as empty result.
        if exc.response is not None and 400 <= exc.response.status_code < 500:
            return '', 'none'
        raise
    data = {}
    try:
        data = response.json()
    except ValueError:
        return '', 'none'
    message = data.get('message', {}) if isinstance(data, dict) else {}
    raw = message.get('abstract') or ''
    cleaned = strip_jats(raw)
    if is_valid_abs(cleaned):
        return cleaned, 'crossref'
    return '', 'none'


def fetch_url_abstract(url: str) -> tuple[str, str]:
    if not ENABLE_SCRAPING or not url:
        return '', 'none'
    try:
        response = httpx.get(url, headers=_build_headers(), timeout=HTTP_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except Exception:
        return '', 'none'
    soup = BeautifulSoup(response.text or '', 'lxml')

    meta_selectors = [
        'meta[name="dc.Description"]',
        'meta[name="dc.description"]',
        'meta[name="DC.Description"]',
        'meta[name="citation_abstract"]',
        'meta[property="og:description"]',
        'meta[property="article:abstract"]',
        'meta[name="description"]',
    ]
    for selector in meta_selectors:
        tag = soup.select_one(selector)
        if not tag:
            continue
        content = tag.get('content') or tag.get('value') or ''
        cleaned = strip_jats(content)
        if is_valid_abs(cleaned):
            return cleaned, 'meta'

    html_selectors = [
        'div.abstract',
        'div.abstract-text',
        'div[class*="abstract"]',
        'section.abstract',
        'section#abstract',
        'section#Abs1 p',
        'div#Abs1-content p',
        'blockquote.abstract',
        'div.html-abstract p',
        'section.article-section__content p',
        '[itemprop="description"]',
    ]
    for selector in html_selectors:
        nodes = soup.select(selector)
        if not nodes:
            continue
        pieces = []
        for node in nodes:
            text_chunk = node.get_text(separator=' ', strip=True)
            cleaned_chunk = strip_jats(text_chunk)
            if cleaned_chunk:
                pieces.append(cleaned_chunk)
        candidate = normalize_whitespace(' '.join(pieces))
        if is_valid_abs(candidate):
            return candidate, 'html'

    return '', 'none'


def get_abstract_via_crossref_then_url(doi: str | None, url: str | None) -> tuple[str, str]:
    doi = (doi or '').strip()
    url = (url or '').strip()
    if doi:
        try:
            abstract, source = fetch_crossref_abstract(doi)
        except Exception:
            abstract, source = '', 'none'
        if abstract:
            return abstract, source
    if url:
        abstract, source = fetch_url_abstract(url)
        if abstract:
            return abstract, source
    return '', 'none'


def pick_language(text):
    try:
        return detect(text or '')
    except Exception:
        return 'en'


def needs_enrichment(abstract, min_chars: int, min_keywords: int) -> bool:
    if not abstract:
        return True
    t = abstract.strip()
    if len(t) < min_chars:
        return True
    lower = t.lower()
    found = sum(1 for k in KEYWORDS if k in lower)
    return found < min_keywords


def build_prompt(title, abstract, lang):
    goal = f"Expand and refine the abstract to be concise, specific, and faithful. Target {TARGET_WORDS[0]}-{TARGET_WORDS[1]} words."
    policy = 'Preserve the original language.' if LANG_POLICY == 'preserve' else f'Write in {lang}.'
    kws = ', '.join(KEYWORDS[:20]) + ', ...'
    hints = 'Include concrete contributions, methods, datasets, and evaluation metrics when available. Avoid speculation.'
    user = f"Title: {title or ''}\nOriginal abstract:\n{abstract or ''}\n\nKeywords of interest: {kws}"
    system = f"You are an expert research editor. {policy} {goal} {hints}"
    return system, user


def call_with_fallbacks(model, fallbacks, system, user):
    models = [model] + [m for m in fallbacks if m]
    last_err = None
    for m in models:
        try:
            resp = chat_complete(m, system, user)
            content = resp.choices[0].message.content.strip()
            if content:
                return content, m
        except Exception as e:
            last_err = e
            time.sleep(1.5)
    raise last_err or RuntimeError('All models failed')


def load_state(state_path):
    done = set()
    p = Path(state_path)
    if p.exists():
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get('status') == 'enriched':
                        done.add(obj.get('key'))
                except Exception:
                    pass
    return done


def append_state(state_path, obj):
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


ENTRY_SPLIT_RE = re.compile(r'(?=@[A-Za-z])')


def load_bib_database(in_path):
    text = Path(in_path).read_text(encoding='utf-8', errors='ignore')
    raw_chunks = ENTRY_SPLIT_RE.split(text)
    entries = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.startswith('@'):
            chunk = '@' + chunk
        entry = _parse_chunk(chunk)
        if entry:
            entries.append(entry)
    db = BibDatabase()
    db.entries = entries
    return db


def _parse_chunk(chunk: str):
    pos_brace = chunk.find('{')
    pos_paren = chunk.find('(')
    if pos_brace == -1 and pos_paren == -1:
        return None
    if pos_brace == -1 or (pos_paren != -1 and pos_paren < pos_brace):
        open_pos = pos_paren
        close_char = ')'
    else:
        open_pos = pos_brace
        close_char = '}'

    entry_type = chunk[1:open_pos].strip().lower()
    content = chunk[open_pos + 1 :].strip()
    if content.endswith(close_char):
        content = content[:-1]
    content = content.strip()

    entry = _parse_entry(entry_type, content)
    return entry


def _parse_entry(entry_type, content):
    key, body = _split_key_and_body(content)
    if not key:
        return None
    fields = _parse_fields(body)
    entry = {'ENTRYTYPE': entry_type, 'ID': key}
    entry.update(fields)
    return entry


def _split_key_and_body(content):
    depth = 0
    for idx, ch in enumerate(content):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth = max(0, depth - 1)
        elif ch == ',' and depth == 0:
            key = content[:idx].strip()
            body = content[idx + 1 :]
            return key, body
    return content.strip(), ''


def _parse_fields(body):
    fields = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in ' \t\r\n,':
            i += 1
        if i >= n:
            break
        name_start = i
        while i < n and body[i] not in '=\r\n':
            if body[i] == ',':
                break
            i += 1
        name = body[name_start:i].strip()
        while i < n and body[i] != '=':
            if body[i] == ',':
                break
            i += 1
        if i >= n or body[i] != '=':
            while i < n and body[i] != ',':
                i += 1
            continue
        i += 1
        while i < n and body[i] in ' \t\r\n':
            i += 1
        value, i = _parse_field_value(body, i)
        if name:
            fields[name.lower()] = value.strip()
        if i < n and body[i] == ',':
            i += 1
    return fields



def _parse_field_value(body, i):
    n = len(body)
    if i >= n:
        return '', n
    if body[i] == '{':
        start = i + 1
        depth = 1
        i += 1
        while i < n:
            ch = body[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    value = body[start:i]
                    i += 1
                    break
            elif ch == ',' and depth == 1:
                value = body[start:i]
                break
            i += 1
        else:
            value = body[start:i]
        value = _balance_braces(value)
        return value, i
    if body[i] == '"':
        start = i + 1
        i += 1
        while i < n and body[i] != '"':
            i += 1
        value = body[start:i]
        i = min(i + 1, n)
        return value, i
    start = i
    while i < n and body[i] != ',':
        i += 1
    value = body[start:i].strip()
    return value, i


def _balance_braces(value):
    opens = value.count('{')
    closes = value.count('}')
    if opens > closes:
        value = value + '}' * (opens - closes)
    return value


def enrich_bib(
    in_path,
    out_path,
    audit_path,
    state_path,
    model,
    fallbacks,
    dry_run=False,
    resume=False,
    force_retry=False,
    min_chars=MIN_CHARS,
    min_keywords=MIN_KEYWORDS,
    enrich_all=False,
):
    ensure_parent(out_path)
    ensure_parent(audit_path)
    ensure_parent(state_path)

    done_keys = load_state(state_path) if resume and not force_retry else set()
    db = load_bib_database(in_path)

    writer = BibTexWriter()
    writer.order_entries_by = ('ID',)

    audit_rows = []
    changed = 0
    total = 0
    for entry in db.entries:
        total += 1
        key = entry.get('ID') or entry.get('id') or entry.get('citekey') or f'entry_{total}'
        title = entry.get('title', '')
        abstract = entry.get('abstract', '') or entry.get('annotation', '') or ''

        if resume and key in done_keys:
            append_state(state_path, {'key': key, 'status': 'skipped', 'reason': 'resume', 'ts': time.time()})
            continue

        if not enrich_all and not needs_enrichment(abstract, min_chars, min_keywords):
            append_state(state_path, {'key': key, 'status': 'skipped', 'reason': 'sufficient', 'ts': time.time()})
            continue

        original_abstract = abstract or ''
        original_length = len(original_abstract)
        current_score = score_abstract(original_abstract)

        doi = (entry.get('doi') or '').strip()
        url_field = entry.get('url') or entry.get('link') or ''
        if isinstance(url_field, list):
            url_field = url_field[0] if url_field else ''
        url_field = str(url_field).strip(' {}')
        url = url_field if url_field.lower().startswith('http') else ''

        if (enrich_all or not original_abstract or OVERWRITE_IF_BETTER) and (doi or url):
            fetched_abs, fetched_source = get_abstract_via_crossref_then_url(doi, url)
            if fetched_abs and score_abstract(fetched_abs) > current_score:
                entry['abstract'] = fetched_abs
                lang = pick_language(fetched_abs)
                entry['abstract_lang'] = lang
                entry['abstract_source'] = fetched_source
                entry['x_enrich_status'] = 'fetched'
                entry['x_enrich_source'] = fetched_source
                entry['x_enrich_ts'] = now_iso()
                audit_rows.append([key, 'fetched', original_length, len(fetched_abs), lang, fetched_source])
                append_state(state_path, {'key': key, 'status': 'fetched', 'source': fetched_source, 'ts': time.time()})
                changed += 1 if fetched_abs != original_abstract else 0
                continue

        lang = pick_language((abstract or title) or '')
        system, user = build_prompt(title, abstract, lang)

        if dry_run or client is None:
            enriched = abstract or ''
            used_model = None
            status = 'dry-run'
        else:
            try:
                enriched, used_model = call_with_fallbacks(model, fallbacks, system, user)
                status = 'enriched'
            except Exception as e:
                append_state(state_path, {'key': key, 'status': 'error', 'error': str(e), 'ts': time.time()})
                continue

        if enriched and enriched != abstract:
            entry['abstract'] = enriched
            lang = pick_language(enriched)
            entry['abstract_lang'] = lang
            entry['abstract_source'] = used_model or 'llm'
            entry['x_enrich_status'] = 'ok'
            entry['x_enrich_source'] = used_model or 'llm'
            entry['x_enrich_ts'] = now_iso()
            changed += 1

        audit_rows.append([
            key,
            status,
            original_length,
            len(entry.get('abstract', '') or ''),
            lang,
            used_model or '',
        ])
        append_state(state_path, {'key': key, 'status': status, 'model': used_model, 'ts': time.time()})

    with open(out_path, 'w', encoding='utf-8') as f:
        bibtexparser.dump(db, f)

    with open(audit_path, 'w', encoding='utf-8', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['key', 'status', 'old_chars', 'new_chars', 'lang', 'model'])
        writer_csv.writerows(audit_rows)

    return {'total': total, 'changed': changed}


def run_fetch_tests():
    from unittest import mock

    global ENABLE_SCRAPING

    print('Running abstract fetch smoke tests...')

    crossref_payload = {'message': {'abstract': '<jats:p>This is a synthetic abstract ' + 'x' * 500 + '</jats:p>'}}
    with mock.patch('httpx.get', return_value=DummyHTTPResponse(json_data=crossref_payload)):
        text, source = fetch_crossref_abstract('10.1234/test')
        print('Crossref test:', source, len(text))

    html_snippet = '<html><head><meta name="dc.Description" content="This is an extracted abstract ' + 'y' * 500 + '"></head></html>'

    def side_effect(url, *args, **kwargs):
        if 'crossref' in url:
            return DummyHTTPResponse(json_data={'message': {}})
        return DummyHTTPResponse(text=html_snippet)

    with mock.patch('httpx.get', side_effect=side_effect):
        previous = ENABLE_SCRAPING
        ENABLE_SCRAPING = True
        try:
            text, source = get_abstract_via_crossref_then_url('10.0000/none', 'https://example.org/paper')
            print('URL fallback test:', source, len(text))
        finally:
            ENABLE_SCRAPING = previous

    html_short = '<html><head><meta name="description" content="Too short"></head></html>'

    def side_effect_none(url, *args, **kwargs):
        if 'crossref' in url:
            return DummyHTTPResponse(json_data={'message': {}})
        return DummyHTTPResponse(text=html_short)

    with mock.patch('httpx.get', side_effect=side_effect_none):
        previous = ENABLE_SCRAPING
        ENABLE_SCRAPING = True
        try:
            text, source = get_abstract_via_crossref_then_url('10.0000/none2', 'https://example.org/none')
            print('None test:', source, len(text))
        finally:
            ENABLE_SCRAPING = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--audit', default='logs/enrich_abstracts_audit.csv')
    parser.add_argument('--state', default='logs/enrich_state.jsonl')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force-retry', action='store_true')
    parser.add_argument('--model', default='xai/grok-2-mini')
    parser.add_argument('--fallbacks', default='openai/gpt-4o-mini,anthropic/claude-3.5-sonnet')
    parser.add_argument('--min-chars', type=int, default=MIN_CHARS,
                        help='Caracteres mínimos del abstract para considerarlo suficiente (por defecto 400).')
    parser.add_argument('--min-keywords', type=int, default=MIN_KEYWORDS,
                        help='Número mínimo de keywords detectadas para evitar enriquecimiento (por defecto 2).')
    parser.add_argument('--enrich-all', action='store_true',
                        help='Forzar enriquecimiento de todas las entradas, sin aplicar heurísticas.')
    parser.add_argument('--run-fetch-tests', action='store_true',
                        help='Ejecuta pruebas de la ruta Crossref/URL y termina.')
    args = parser.parse_args()

    if args.run_fetch_tests:
        run_fetch_tests()
        return

    fallbacks = [x.strip() for x in args.fallbacks.split(',') if x.strip()]
    enrich_bib(
        args.input,
        args.out,
        args.audit,
        args.state,
        args.model,
        fallbacks,
        dry_run=args.dry_run,
        resume=args.resume,
        force_retry=args.force_retry,
        min_chars=max(0, args.min_chars),
        min_keywords=max(0, args.min_keywords),
        enrich_all=args.enrich_all,
    )


if __name__ == '__main__':
    main()
