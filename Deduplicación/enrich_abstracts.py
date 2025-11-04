import os
import json
import time
import argparse
import csv
import sys
import re
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from langdetect import detect
import tldextract

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

# Logger global para bitácora
LOGGER = logging.getLogger('enrich_abstracts')

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

# Denylist de hosts donde evitamos scraping HTML directo por políticas/TOS
DENYLIST_HOSTS = {
    'dl.acm.org',
}

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


def _build_doi_json_headers():
    h = _build_headers()
    h['Accept'] = 'application/vnd.citationstyles.csl+json'
    return h


def _ensure_dir(p: Union[Path, str]):
    Path(p).mkdir(parents=True, exist_ok=True)


def _next_log_file(log_dir: Path, prefix: str = 'enrich_abstracts_log') -> Path:
    date = datetime.now().strftime('%Y%m%d')
    seq = 1
    while True:
        fname = f"{prefix}_{date}_{seq:03d}.log"
        candidate = log_dir / fname
        if not candidate.exists():
            return candidate
        seq += 1


def setup_logging(log_dir: str = 'logs', level: str = 'INFO') -> Path:
    """Configura logging a consola y archivo diario incremental.

    - Consola: nivel INFO por defecto.
    - Archivo: nivel DEBUG, nombre enrich_abstracts_log_YYYYMMDD_XXX.log en `log_dir`.
    Devuelve la ruta del archivo de log.
    """
    # Limpiar handlers previos si se re-invoca
    if LOGGER.handlers:
        for h in list(LOGGER.handlers):
            LOGGER.removeHandler(h)

    LOGGER.setLevel(logging.DEBUG)

    _ensure_dir(log_dir)
    log_dir_path = Path(log_dir)
    logfile = _next_log_file(log_dir_path)

    # Console handler
    ch = logging.StreamHandler()
    level_map = {
        'CRITICAL': logging.CRITICAL,
        'ERROR': logging.ERROR,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    ch.setLevel(level_map.get(level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))

    # File handler (rotating; 10MB x 3)
    fh = RotatingFileHandler(logfile, maxBytes=10 * 1024 * 1024, backupCount=3, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d - %(message)s'))

    LOGGER.addHandler(ch)
    LOGGER.addHandler(fh)
    LOGGER.propagate = False

    LOGGER.info('Bitácora inicializada. Archivo: %s', logfile)
    return logfile


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def fetch_crossref_abstract(doi: str) -> Tuple[str, str]:
    doi = (doi or '').strip()
    if not doi:
        return '', 'none'
    encoded = quote(doi)
    url = f'https://api.crossref.org/works/{encoded}'
    try:
        LOGGER.debug('Consultando Crossref para DOI=%s', doi)
        response = httpx.get(url, headers=_build_headers(), timeout=HTTP_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # 4xx responses (e.g. missing DOI) are not retriable; surface as empty result.
        if exc.response is not None and 400 <= exc.response.status_code < 500:
            LOGGER.info('Crossref sin abstract (HTTP %s) para DOI=%s', exc.response.status_code, doi)
            return '', 'none'
        LOGGER.warning('Error HTTP en Crossref para DOI=%s; reintentando...', doi)
        raise
    data = {}
    try:
        data = response.json()
    except ValueError:
        LOGGER.warning('Respuesta Crossref no-JSON para DOI=%s', doi)
        return '', 'none'
    message = data.get('message', {}) if isinstance(data, dict) else {}
    raw = message.get('abstract') or ''
    cleaned = strip_jats(raw)
    if is_valid_abs(cleaned):
        LOGGER.info('Abstract válido obtenido desde Crossref (len=%d) para DOI=%s', len(cleaned), doi)
        return cleaned, 'crossref'
    return '', 'none'


def fetch_url_abstract(url: str) -> Tuple[str, str]:
    if not ENABLE_SCRAPING or not url:
        return '', 'none'
    # Evita scraping para hosts en denylist
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        host = ''
    if host in DENYLIST_HOSTS:
        LOGGER.info('Scraping HTML deshabilitado para host en denylist: %s', host)
        return '', 'none'
    try:
        LOGGER.debug('Extrayendo abstract desde URL=%s', url)
        response = httpx.get(url, headers=_build_headers(), timeout=HTTP_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except Exception:
        LOGGER.info('No fue posible extraer abstract desde URL=%s', url)
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
            LOGGER.info('Abstract válido obtenido desde metadatos (len=%d) URL=%s', len(cleaned), url)
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
            LOGGER.info('Abstract válido obtenido desde HTML (len=%d) URL=%s', len(candidate), url)
            return candidate, 'html'

    return '', 'none'


def fetch_doi_json_abstract(doi: str) -> Tuple[str, str]:
    doi = (doi or '').strip()
    if not doi:
        return '', 'none'
    encoded = quote(doi)
    url = f'https://doi.org/{encoded}'
    try:
        LOGGER.debug('Consultando DOI JSON para DOI=%s', doi)
        response = httpx.get(url, headers=_build_doi_json_headers(), timeout=HTTP_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except Exception as e:
        LOGGER.info('DOI JSON no disponible para DOI=%s (%s)', doi, e)
        return '', 'none'
    try:
        data = response.json()
    except Exception:
        return '', 'none'
    abstract = ''
    if isinstance(data, dict):
        abstract = data.get('abstract') or ''
    cleaned = strip_jats(abstract)
    if is_valid_abs(cleaned):
        LOGGER.info('Abstract válido obtenido desde DOI JSON (len=%d) para DOI=%s', len(cleaned), doi)
        return cleaned, 'doi-json'
    return '', 'none'


def get_abstract_via_crossref_then_url(doi: Optional[str], url: Optional[str]) -> Tuple[str, str]:
    doi = (doi or '').strip()
    url = (url or '').strip()

    # 1) Intento por Crossref usando DOI
    if doi:
        try:
            abstract, source = fetch_crossref_abstract(doi)
        except Exception:
            abstract, source = '', 'none'
        if abstract:
            return abstract, source

    # 2) Intento por DOI JSON (content negotiation)
    if doi:
        abs_doi, src_doi = fetch_doi_json_abstract(doi)
        if abs_doi:
            return abs_doi, src_doi

    # 3) Intento por URL proporcionada
    if url:
        abstract, source = fetch_url_abstract(url)
        if abstract:
            return abstract, source

    # 4) Intento por URL construida del DOI (https://doi.org/...) si no hubo suerte y hay DOI
    if doi:
        doi_url = f"https://doi.org/{doi}"
        LOGGER.info('Intentando extracción desde URL de DOI: %s', doi_url)
        abstract, source = fetch_url_abstract(doi_url)
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


def build_prompt(doi: Optional[str], url: Optional[str]):
    """Crea un prompt para pedir al LLM solamente el abstract extraído del DOI o URL.

    Nota: Los LLM de OpenRouter no pueden navegar; este prompt solo sirve si
    el modelo tiene contexto de texto (p. ej., si se le adjuntara contenido).
    """
    doi = (doi or '').strip()
    url = (url or '').strip()
    system = (
        "You are a precise assistant. Extract only the publication abstract. "
        "Do not add commentary, headings, quotes, or extra text. "
        "Preserve the original language of the abstract."
    )
    user = (
        f"Extract the abstract from the DOI: {doi} or the URL: {url}. "
        "Return only the abstract text in its original language."
    )
    return system, user


def call_with_fallbacks(model, fallbacks, system, user):
    models = [model] + [m for m in fallbacks if m]
    last_err = None
    for m in models:
        try:
            LOGGER.info('Invocando LLM: %s', m)
            resp = chat_complete(m, system, user)
            content = resp.choices[0].message.content.strip()
            if content:
                LOGGER.info('LLM OK: %s (len=%d)', m, len(content))
                return content, m
        except Exception as e:
            last_err = e
            LOGGER.warning('LLM error con %s: %s', m, e)
            time.sleep(1.5)
    raise last_err or RuntimeError('All models failed')


def load_state(state_path):
    """Carga claves ya procesadas desde un archivo de estado JSONL.

    Considera como procesadas las entradas con status en {'enriched','fetched','skipped'}.
    Ignora 'error'.
    """
    done = set()
    p = Path(state_path)
    if p.exists():
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    status = obj.get('status')
                    if status in {'enriched', 'fetched', 'skipped'}:
                        k = obj.get('key')
                        if k:
                            done.add(k)
                except Exception:
                    pass
    return done


def _resolve_resume_state_path(state_path: str) -> str:
    """Si el archivo de estado no existe, intenta resolver el más reciente
    que coincida con el patrón base: <stem>_YYYYMMDD_XXX.jsonl en el mismo directorio.
    """
    p = Path(state_path)
    if p.exists():
        return str(p)
    parent = p.parent
    stem = p.stem
    # Elimina sufijo _YYYYMMDD_XXX si viene incluido
    base_stem = re.sub(r'_\d{8}_\d{3}$', '', stem)
    candidates = sorted(parent.glob(f"{base_stem}_*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.exists() and c.is_file():
            return str(c)
    return str(p)


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
    checkpoint_every: int = 0,
):
    ensure_parent(out_path)
    ensure_parent(audit_path)
    ensure_parent(state_path)
    partial_out_path = f"{out_path}.partial" if checkpoint_every and checkpoint_every > 0 else None
    audit_streaming = bool(checkpoint_every and checkpoint_every > 0)
    audit_writer = None
    audit_fh = None
    if audit_streaming:
        # Abrimos auditoría en append y escribimos encabezado si el archivo no existe o está vacío
        p = Path(audit_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        file_exists = p.exists() and p.stat().st_size > 0
        audit_fh = p.open('a', encoding='utf-8', newline='')
        audit_writer = csv.writer(audit_fh)
        if not file_exists:
            audit_writer.writerow(['key', 'status', 'old_chars', 'new_chars', 'lang', 'model'])

    if resume and not force_retry:
        resume_src = _resolve_resume_state_path(state_path)
        if resume_src != state_path:
            LOGGER.info('Reanudación: usando estado previo %s', resume_src)
        done_keys = load_state(resume_src)
    else:
        done_keys = set()
    LOGGER.info('Cargando base BibTeX desde %s', in_path)
    db = load_bib_database(in_path)
    LOGGER.info('Entradas cargadas: %d', len(db.entries))

    writer = BibTexWriter()
    writer.order_entries_by = ('ID',)

    audit_rows = []
    changed = 0
    total = 0
    processed_run = 0  # entradas efectivamente procesadas en esta corrida (excluye saltos por resume)
    for entry in db.entries:
        total += 1
        key = entry.get('ID') or entry.get('id') or entry.get('citekey') or f'entry_{total}'
        title = entry.get('title', '')
        abstract = entry.get('abstract', '') or entry.get('annotation', '') or ''

        if resume and key in done_keys:
            LOGGER.debug('Saltando (resume) %s', key)
            append_state(state_path, {'key': key, 'status': 'skipped', 'reason': 'resume', 'ts': time.time()})
            continue

        if not enrich_all and not needs_enrichment(abstract, min_chars, min_keywords):
            LOGGER.debug('Suficiente, no enriquecer: %s (len=%d)', key, len(abstract or ''))
            append_state(state_path, {'key': key, 'status': 'skipped', 'reason': 'sufficient', 'ts': time.time()})
            processed_run += 1
            # Checkpoint oportunista en resume tras la primera entrada procesada
            if partial_out_path and (processed_run % checkpoint_every == 0 or (resume and processed_run == 1)):
                try:
                    with open(partial_out_path, 'w', encoding='utf-8') as f:
                        bibtexparser.dump(db, f)
                    LOGGER.info('Checkpoint guardado en %s (procesadas=%d)', partial_out_path, processed_run)
                except Exception as e:
                    LOGGER.warning('Error al escribir checkpoint %s: %s', partial_out_path, e)
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
            LOGGER.info('Intentando obtener abstract vía Crossref/URL para %s (doi=%s url=%s)', key, doi, url)
            fetched_abs, fetched_source = get_abstract_via_crossref_then_url(doi, url)
            if fetched_abs and score_abstract(fetched_abs) > current_score:
                entry['abstract'] = fetched_abs
                lang = pick_language(fetched_abs)
                entry['abstract_lang'] = lang
                entry['abstract_source'] = fetched_source
                entry['x_enrich_status'] = 'fetched'
                entry['x_enrich_source'] = fetched_source
                entry['x_enrich_ts'] = now_iso()
                LOGGER.info('Abstract mejorado por %s para %s (old=%d new=%d)', fetched_source, key, original_length, len(fetched_abs))
                audit_rows.append([key, 'fetched', original_length, len(fetched_abs), lang, fetched_source])
                append_state(state_path, {'key': key, 'status': 'fetched', 'source': fetched_source, 'ts': time.time()})
                changed += 1 if fetched_abs != original_abstract else 0
                processed_run += 1
                # Audit streaming
                if audit_writer:
                    try:
                        audit_writer.writerow([key, 'fetched', original_length, len(fetched_abs), lang, fetched_source])
                    except Exception:
                        pass
                # Checkpoint
                if partial_out_path and (processed_run % checkpoint_every == 0 or (resume and processed_run == 1)):
                    try:
                        with open(partial_out_path, 'w', encoding='utf-8') as f:
                            bibtexparser.dump(db, f)
                        LOGGER.info('Checkpoint guardado en %s (procesadas=%d)', partial_out_path, processed_run)
                    except Exception as e:
                        LOGGER.warning('Error al escribir checkpoint %s: %s', partial_out_path, e)
                continue

        lang = pick_language((abstract or title) or '')
        system, user = build_prompt(doi, url)

        if dry_run:
            enriched = abstract or ''
            used_model = None
            status = 'dry-run'
        elif client is None:
            # Sin cliente LLM disponible: si no hay abstract original, coloca marcador explícito
            if not original_abstract:
                enriched = 'NO_ABSTRACT_FOUND'
                used_model = None
                status = 'no-abstract'
            else:
                enriched = abstract or ''
                used_model = None
                status = 'skipped-llm'
        else:
            try:
                enriched, used_model = call_with_fallbacks(model, fallbacks, system, user)
                status = 'enriched'
            except Exception as e:
                LOGGER.error('Fallo enriqueciendo %s: %s', key, e)
                if not original_abstract:
                    # Si no había abstract y el LLM falla, marca explícitamente no encontrado
                    enriched = 'NO_ABSTRACT_FOUND'
                    used_model = None
                    status = 'no-abstract'
                else:
                    append_state(state_path, {'key': key, 'status': 'error', 'error': str(e), 'ts': time.time()})
                    continue

        if enriched and enriched != abstract:
            entry['abstract'] = enriched
            # Para el marcador explícito, no intentamos detectar idioma
            lang = 'und' if enriched == 'NO_ABSTRACT_FOUND' else pick_language(enriched)
            entry['abstract_lang'] = lang
            entry['abstract_source'] = (used_model or 'none') if enriched == 'NO_ABSTRACT_FOUND' else (used_model or 'llm')
            entry['x_enrich_status'] = 'not_found' if enriched == 'NO_ABSTRACT_FOUND' else 'ok'
            entry['x_enrich_source'] = (used_model or 'none') if enriched == 'NO_ABSTRACT_FOUND' else (used_model or 'llm')
            entry['x_enrich_ts'] = now_iso()
            changed += 1
            if enriched == 'NO_ABSTRACT_FOUND':
                LOGGER.info('Sin abstract tras intentos para %s; marcado NO_ABSTRACT_FOUND', key)
            else:
                LOGGER.info('Enriquecido por LLM %s para %s (old=%d new=%d)', used_model, key, original_length, len(enriched))

        audit_rows.append([
            key,
            status,
            original_length,
            len(entry.get('abstract', '') or ''),
            lang,
            used_model or '',
        ])
        append_state(state_path, {'key': key, 'status': status, 'model': used_model, 'ts': time.time()})
        processed_run += 1
        # Audit streaming
        if audit_writer:
            try:
                audit_writer.writerow([key, status, original_length, len(entry.get('abstract', '') or ''), lang, used_model or ''])
            except Exception:
                pass

        # Checkpoint: escribe progreso parcial del BibTex cada N entradas procesadas en esta corrida
        if partial_out_path and (processed_run % checkpoint_every == 0 or (resume and processed_run == 1)):
            try:
                with open(partial_out_path, 'w', encoding='utf-8') as f:
                    bibtexparser.dump(db, f)
                LOGGER.info('Checkpoint guardado en %s (procesadas=%d)', partial_out_path, processed_run)
            except Exception as e:
                LOGGER.warning('Error al escribir checkpoint %s: %s', partial_out_path, e)

    with open(out_path, 'w', encoding='utf-8') as f:
        bibtexparser.dump(db, f)
    LOGGER.info('Archivo BibTeX guardado en %s', out_path)
    # Al finalizar, si hubo archivo parcial, lo actualizamos también con el estado final
    if partial_out_path:
        try:
            with open(partial_out_path, 'w', encoding='utf-8') as f:
                bibtexparser.dump(db, f)
            LOGGER.info('Archivo parcial actualizado con estado final: %s', partial_out_path)
        except Exception as e:
            LOGGER.warning('No se pudo actualizar archivo parcial final %s: %s', partial_out_path, e)

    if audit_writer:
        try:
            audit_fh.flush()
            audit_fh.close()
        except Exception:
            pass
        LOGGER.info('Auditoría actualizada en modo streaming: %s', audit_path)
    else:
        with open(audit_path, 'w', encoding='utf-8', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(['key', 'status', 'old_chars', 'new_chars', 'lang', 'model'])
            writer_csv.writerows(audit_rows)
        LOGGER.info('Auditoría guardada en %s', audit_path)

    LOGGER.info('Resumen: total=%d, cambiados=%d', total, changed)
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
    parser.add_argument('--audit', default='logs/enrich_abstracts_audit.csv',
                        help='Ruta base para auditoría; se le añade _YYYYMMDD_XXX automáticamente.')
    parser.add_argument('--state', default='logs/enrich_state.jsonl',
                        help='Ruta base para estado; se le añade _YYYYMMDD_XXX automáticamente.')
    parser.add_argument('--log-dir', default='logs', help='Directorio para archivos de log.')
    parser.add_argument('--log-level', default='INFO', help='Nivel de log para consola (DEBUG, INFO, WARNING, ERROR).')
    parser.add_argument('--checkpoint-every', type=int, default=30,
                        help='Cada cuántas entradas escribir un checkpoint en <out>.partial (0 desactiva).')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force-retry', action='store_true')
    parser.add_argument('--model', default='x-ai/grok-4-fast')
    parser.add_argument('--fallbacks', default='anthropic/claude-sonnet-4.5,x-ai/grok-4')
    parser.add_argument('--min-chars', type=int, default=MIN_CHARS,
                        help='Caracteres mínimos del abstract para considerarlo suficiente (por defecto 400).')
    parser.add_argument('--min-keywords', type=int, default=MIN_KEYWORDS,
                        help='Número mínimo de keywords detectadas para evitar enriquecimiento (por defecto 2).')
    parser.add_argument('--enrich-all', action='store_true',
                        help='Forzar enriquecimiento de todas las entradas, sin aplicar heurísticas.')
    parser.add_argument('--run-fetch-tests', action='store_true',
                        help='Ejecuta pruebas de la ruta Crossref/URL y termina.')
    args = parser.parse_args()

    logfile = setup_logging(args.log_dir, args.log_level)
    # Construye sufijo _YYYYMMDD_XXX a partir del logfile para mantener consistencia
    lf_stem = Path(logfile).stem  # enrich_abstracts_log_YYYYMMDD_XXX
    parts = lf_stem.split('_')
    log_suffix = ''
    if len(parts) >= 2:
        log_suffix = '_' + '_'.join(parts[-2:])  # _YYYYMMDD_XXX

    # Aplica sufijo a audit y state
    audit_path = Path(args.audit)
    state_path = Path(args.state)
    audit_path = audit_path.with_name(f"{audit_path.stem}{log_suffix}{audit_path.suffix}") if log_suffix else audit_path
    state_path = state_path.with_name(f"{state_path.stem}{log_suffix}{state_path.suffix}") if log_suffix else state_path

    LOGGER.info('Parámetros: input=%s out=%s audit=%s state=%s model=%s fallbacks=%s dry_run=%s resume=%s force_retry=%s min_chars=%d min_keywords=%d enrich_all=%s',
                args.input, args.out, str(audit_path), str(state_path), args.model, args.fallbacks, args.dry_run, args.resume, args.force_retry, args.min_chars, args.min_keywords, args.enrich_all)

    if args.run_fetch_tests:
        run_fetch_tests()
        return

    fallbacks = [x.strip() for x in args.fallbacks.split(',') if x.strip()]
    result = enrich_bib(
        args.input,
        args.out,
        str(audit_path),
        str(state_path),
        args.model,
        fallbacks,
        dry_run=args.dry_run,
        resume=args.resume,
        force_retry=args.force_retry,
        min_chars=max(0, args.min_chars),
        min_keywords=max(0, args.min_keywords),
        enrich_all=args.enrich_all,
        checkpoint_every=max(0, args.checkpoint_every),
    )
    LOGGER.info('Proceso finalizado. Archivo log: %s', logfile)


if __name__ == '__main__':
    main()
