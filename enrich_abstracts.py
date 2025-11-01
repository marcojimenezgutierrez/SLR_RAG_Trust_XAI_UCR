import os, re, json, time, argparse, csv
from pathlib import Path
from dotenv import load_dotenv
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
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

# OpenRouter client
client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=OPENROUTER_API_KEY, default_headers={'HTTP-Referer': HTTP_REFERER, 'X-Title': X_TITLE})

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def chat_complete(model, system, user):
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

def enrich_bib(in_path, out_path, audit_path, state_path, model, fallbacks, dry_run=False, resume=False, force_retry=False):
    ensure_parent(out_path)
    ensure_parent(audit_path)
    ensure_parent(state_path)

    done_keys = load_state(state_path) if resume and not force_retry else set()
    with open(in_path, 'r', encoding='utf-8') as f:
        db = bibtexparser.load(f)

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

        if dry_run or not OPENROUTER_API_KEY:
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

    import csv as _csv
    with open(audit_path, 'w', encoding='utf-8', newline='') as f:
        w = _csv.writer(f)
        w.writerow(['key', 'status', 'old_chars', 'new_chars', 'lang', 'model'])
        w.writerows(audit_rows)

    return {'total': total, 'changed': changed}

if __name__ == '__main__':
    import argparse as _argparse
    p = _argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--audit', default='logs/enrich_abstracts_audit.csv')
    p.add_argument('--state', default='logs/enrich_state.jsonl')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--force-retry', action='store_true')
    p.add_argument('--model', default='xai/grok-2-mini')
    p.add_argument('--fallbacks', default='openai/gpt-4o-mini,anthropic/claude-3.5-sonnet')
    a = p.parse_args()
    fallbacks = [x.strip() for x in a.fallbacks.split(',') if x.strip()]
    enrich_bib(a.input, a.out, a.audit, a.state, a.model, fallbacks, dry_run=a.dry_run, resume=a.resume, force_retry=a.force_retry)
