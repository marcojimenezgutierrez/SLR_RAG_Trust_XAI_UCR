import os, json, time, argparse, csv, sys
from pathlib import Path
from dotenv import load_dotenv
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from langdetect import detect

# Load env
load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
HTTP_REFERER = os.getenv('OPENROUTER_SITE_URL','')
X_TITLE = os.getenv('OPENROUTER_APP_NAME','SLR Abstract Enricher')

# Policies
LANG_POLICY = 'preserve'
TARGET_WORDS = (120, 200)
MIN_CHARS = 400
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
    client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=OPENROUTER_API_KEY, default_headers={'HTTP-Referer': HTTP_REFERER, 'X-Title': X_TITLE})

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def chat_complete(model, system, user):
    if client is None:
        raise RuntimeError('OpenRouter API key not configured')
    return client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        temperature=0.2,
    )

def pick_language(text):
    try:
        return detect(text or '')
    except Exception:
        return 'en'

def needs_enrichment(abstract):
    if not abstract:
        return True
    t = abstract.strip()
    if len(t) < MIN_CHARS:
        return True
    lower = t.lower()
    found = sum(1 for k in KEYWORDS if k in lower)
    return found < 2

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

def load_bib_database(in_path):
    text = Path(in_path).read_text(encoding='utf-8', errors='ignore')
    entries = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != '@':
            i += 1
            continue
        start_entry = i
        i += 1
        while i < n and text[i] not in '{(':
            i += 1
        entry_type = text[start_entry + 1:i].strip().lower()
        if i >= n:
            break
        open_char = text[i]
        close_char = '}' if open_char == '{' else ')'
        i += 1
        content_start = i
        depth = 1
        while i < n and depth > 0:
            ch = text[i]
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
            i += 1
        content = text[content_start:i - 1]
        entry = _parse_entry(entry_type, content)
        if entry:
            entries.append(entry)
    db = BibDatabase()
    db.entries = entries
    return db


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
            body = content[idx + 1:]
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


def enrich_bib(in_path, out_path, audit_path, state_path, model, fallbacks, dry_run=False, resume=False, force_retry=False):
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

        if not needs_enrichment(abstract):
            append_state(state_path, {'key': key, 'status': 'skipped', 'reason': 'sufficient', 'ts': time.time()})
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
            changed += 1

        audit_rows.append([key, status, len(abstract or ''), len(entry.get('abstract', '') or ''), lang, used_model or ''])
        append_state(state_path, {'key': key, 'status': status, 'model': used_model, 'ts': time.time()})

    with open(out_path, 'w', encoding='utf-8') as f:
        bibtexparser.dump(db, f)

    with open(audit_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['key', 'status', 'old_chars', 'new_chars', 'lang', 'model'])
        w.writerows(audit_rows)

    return {'total': total, 'changed': changed}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--audit', default='logs/enrich_abstracts_audit.csv')
    p.add_argument('--state', default='logs/enrich_state.jsonl')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--force-retry', action='store_true')
    p.add_argument('--model', default='xai/grok-2-mini')
    p.add_argument('--fallbacks', default='openai/gpt-4o-mini,anthropic/claude-3.5-sonnet')
    args = p.parse_args()
    fallbacks = [x.strip() for x in args.fallbacks.split(',') if x.strip()]
    enrich_bib(args.input, args.out, args.audit, args.state, args.model, fallbacks, dry_run=args.dry_run, resume=args.resume, force_retry=args.force_retry)
