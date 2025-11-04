from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import bibtexparser
import httpx
from openpyxl import Workbook


STATUS_OK = "ok"
STATUS_ID_NO_MATCH = "IdNoMatch"
STATUS_NO_ABSTRACT = "NoAbstract"
STATUS_LLM_ERROR = "LLMError"
STATUS_PARSE_ERROR = "ParseError"
STATUS_TIMEOUT = "Timeout"

NEW_COLUMNS = ["rqs_llm", "llm_explanation", "llm_response", "llm_reasoning", "llm_status"]
VALID_PROVIDERS = {"ollama_native", "openai_compat"}
DEFAULT_PROVIDER = "ollama_native"
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CONCURRENCY = 1
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 120.0
DEFAULT_SHEET_NAME = "results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "Resultado"
DEFAULT_OPENAI_MAX_TOKENS = 50_000
PROMPT_SEPARATOR_LINE = "-" * 56

REASONING_PATTERN = re.compile(r"Thinking\.\.\.(.*?)(?:\.\.\.done thinking\.)", re.DOTALL | re.IGNORECASE)

PROMPT_TEMPLATE = """You are a senior reviewer for a systematic literature review \
focused on Retrieval-Augmented Generation (RAG) systems, trust, hallucinations, \
explainability, calibration, and evaluation methodologies.

Use only the provided bibliographic metadata.

Title: {title}
Abstract: {abstract}
Kitchenham heuristic tags: {heuristic_tags}
Screener rationale: {reason}
Additional context keywords (not RQs): {extra_keywords}

Research questions to evaluate
------------------------------
{rq_section}

Instructions
------------
1. Decide which research questions are explicitly addressed by this work.
2. Base your judgement solely on the supplied title and abstract.
3. If the evidence is unclear, treat the research question as NOT covered.
4. Reply ONLY with a JSON object exactly like this example:
{{"rqs":"RQ1,RQ3","explanation":"breve justificación en español"}}

JSON requirements:
- Valid RQ codes: {allowed_codes}.
- The `rqs` value must be a comma-separated list without spaces (use an empty string "" if none apply).
- `explanation` must be a concise justification in Spanish (fewer than 60 words).
- Return only the JSON, without code fences or additional commentary.
"""


@dataclass(frozen=True)
class BibEntry:
    entry_id: str
    title: str
    abstract: str


@dataclass(frozen=True)
class RQDefinition:
    code: str
    question: str
    keywords: Sequence[str]

    def format_block(self) -> str:
        keywords = ", ".join(self.keywords) if self.keywords else "No keywords provided"
        return f"{self.code}: {self.question}\nKeywords: {keywords}"


@dataclass(frozen=True)
class WorkItem:
    row_index: int
    row_id: str
    prompt: str


@dataclass
class WorkResult:
    row_index: int
    status: str
    rqs: str = ""
    explanation: str = ""
    reasoning: str = ""
    raw_response: str = ""


class LLMClientError(Exception):
    """Generic error raised when the LLM backend fails."""


class LLMTimeoutError(LLMClientError):
    """Raised when the LLM backend does not answer before the timeout."""


class LLMClient:
    """Simple HTTP client that supports Ollama native and OpenAI-compatible APIs."""

    def __init__(
        self,
        *,
        provider: str,
        host: str,
        model: str,
        temperature: float,
        max_tokens: int,
        api_key: Optional[str],
        request_timeout: float,
    ) -> None:
        if provider not in VALID_PROVIDERS:
            raise ValueError(f"Unsupported provider '{provider}'. Options: {sorted(VALID_PROVIDERS)}")

        self.provider = provider
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        timeout = httpx.Timeout(request_timeout, connect=request_timeout, read=request_timeout, write=request_timeout)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def generate(self, prompt: str, *, capture_reasoning: bool) -> Tuple[str, str]:
        try:
            if self.provider == "ollama_native":
                return self._generate_ollama(prompt, capture_reasoning=capture_reasoning)
            return self._generate_openai(prompt, capture_reasoning=capture_reasoning)
        except httpx.TimeoutException as exc:  # pragma: no cover - network errors are runtime-specific
            raise LLMTimeoutError("LLM request timed out") from exc
        except httpx.RequestError as exc:  # pragma: no cover
            raise LLMClientError(f"LLM request failed: {exc}") from exc

    def _generate_ollama(self, prompt: str, *, capture_reasoning: bool) -> Tuple[str, str]:
        url = f"{self.host}/api/generate"
        options: Dict[str, object] = {"temperature": self.temperature}
        options["num_predict"] = self.max_tokens if self.max_tokens >= 0 else -1
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": options,
            "stream": bool(capture_reasoning),
        }
        if capture_reasoning:
            response_buffer: List[str] = []
            reasoning_buffer: List[str] = []
            done_reason: Optional[str] = None
            with self._client.stream("POST", url, json=payload) as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    response_part = event.get("response")
                    if isinstance(response_part, str):
                        response_buffer.append(response_part)
                    thinking_part = event.get("thinking")
                    if isinstance(thinking_part, str):
                        reasoning_buffer.append(thinking_part)
                    if event.get("done"):
                        done_reason = event.get("done_reason")
                        break
            if done_reason and done_reason not in {"stop", ""}:
                raise LLMClientError(f"Ollama generation stopped early: {done_reason}")
            response_text = "".join(response_buffer).strip()
            reasoning_text = "".join(reasoning_buffer).strip()
            if not response_text:
                raise LLMClientError("Ollama stream returned empty response")
            return response_text, reasoning_text

        payload["stream"] = False
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("response")
        if not isinstance(content, str):
            raise LLMClientError("Unexpected Ollama response payload")
        return content.strip(), ""

    def _generate_openai(self, prompt: str, *, capture_reasoning: bool) -> Tuple[str, str]:
        url = f"{self.host}/v1/chat/completions"
        payload: Dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = DEFAULT_OPENAI_MAX_TOKENS
        response = self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMClientError("OpenAI-compatible backend returned no choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMClientError("OpenAI-compatible backend returned invalid content")
        output = content.strip()
        if capture_reasoning:
            reasoning, _ = extract_reasoning_block(output)
            return output, reasoning
        return output, ""


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based second reader for Kitchenham screening results.")
    parser.add_argument("--input-tsv", required=True, help="Path to the TSV produced by rag_screening.py.")
    parser.add_argument("--bib-file", required=True, help="BibTeX file that contains titles and abstracts.")
    parser.add_argument("--rqconfig", required=True, help="JSON file with research question definitions.")
    parser.add_argument("--output-dir", help="Base directory for Resultado/YYYYMMDD.", default=None)
    parser.add_argument("--output-tsv", help="Explicit path for the working TSV copy (enables resume).")
    parser.add_argument("--provider", choices=sorted(VALID_PROVIDERS), default=DEFAULT_PROVIDER)
    parser.add_argument("--host", default="http://localhost:11434", help="LLM host or base URL.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model identifier.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=-1, help="Max tokens (<=0 means backend default).")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_TIMEOUT, help="Timeout per LLM call in seconds.")
    parser.add_argument("--bib-id", help="Process a single BibTeX id (useful for testing).")
    parser.add_argument("--api-key", help="API key for OpenAI-compatible providers.", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--run-date", help="Override YYYYMMDD folder name.")
    parser.add_argument("--sheet-name", default=DEFAULT_SHEET_NAME, help="Worksheet name for the Excel export.")
    parser.add_argument("--force", action="store_true", help="Recompute even if llm_response is already populated.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ...).")
    parser.add_argument("--summary-only", action="store_true", help="Skip LLM calls and only report configuration.")
    parser.add_argument("--dump-prompts", action="store_true", help="Save generated prompts under the run folder for debugging.")
    parser.add_argument("--capture-reasoning", dest="capture_reasoning", action="store_true", help="Capture reasoning blocks emitted by the LLM (default: enabled).")
    parser.add_argument("--no-capture-reasoning", dest="capture_reasoning", action="store_false", help="Disable reasoning capture.")
    parser.set_defaults(capture_reasoning=True)
    return parser.parse_args(argv)


def setup_logging(level_name: str) -> None:
    numeric_level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_bib_text(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[{}\n]+", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_bib_entries(path: Path) -> Dict[str, BibEntry]:
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    with path.open("r", encoding="utf-8") as stream:
        bib_database = bibtexparser.load(stream, parser=parser)
    records: Dict[str, BibEntry] = {}
    for entry in bib_database.entries:
        entry_id = entry.get("ID")
        if not entry_id:
            continue
        title = normalize_bib_text(entry.get("title"))
        abstract = normalize_bib_text(entry.get("abstract"))
        records[entry_id] = BibEntry(entry_id=entry_id, title=title, abstract=abstract)
    return records


def load_rq_config(path: Path) -> Tuple[List[RQDefinition], List[str]]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("RQ config must be a JSON object keyed by RQ code.")
    rq_definitions: List[RQDefinition] = []
    context_keywords: List[str] = []
    rq_code_pattern = re.compile(r"^RQ\d+$", re.IGNORECASE)
    for code, raw in payload.items():
        code_str = str(code).strip()
        if rq_code_pattern.match(code_str):
            question = str(raw.get("RQ", "")).strip()
            keywords_raw = raw.get("KW", [])
            if isinstance(keywords_raw, str):
                keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()]
            elif isinstance(keywords_raw, list):
                keywords = [str(kw).strip() for kw in keywords_raw if str(kw).strip()]
            else:
                keywords = []
            rq_definitions.append(RQDefinition(code=code_str, question=question, keywords=keywords))
        else:
            # Treat non-RQ sections (e.g., "RAG", "LLM") as extra context keywords
            kw_raw = raw.get("KW", []) if isinstance(raw, dict) else []
            if isinstance(kw_raw, str):
                kws = [kw.strip() for kw in kw_raw.split(",") if kw.strip()]
            elif isinstance(kw_raw, list):
                kws = [str(kw).strip() for kw in kw_raw if str(kw).strip()]
            else:
                kws = []
            context_keywords.extend(kws)
    # Deduplicate while preserving order
    seen: set = set()
    deduped_context = [kw for kw in context_keywords if not (kw in seen or seen.add(kw))]
    return rq_definitions, deduped_context


def read_tsv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV file '{path}' has no header.")
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames)


def write_tsv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_to_excel(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]], sheet_name: str) -> None:
    workbook = Workbook()
    worksheet = workbook.active
    safe_sheet = (sheet_name or DEFAULT_SHEET_NAME)[:31]
    worksheet.title = safe_sheet if safe_sheet else DEFAULT_SHEET_NAME
    worksheet.append(list(fieldnames))
    for row in rows:
        worksheet.append([row.get(column, "") for column in fieldnames])
    workbook.save(path)


def ensure_columns(fieldnames: List[str], rows: List[Dict[str, str]]) -> List[str]:
    updated_fieldnames = list(fieldnames)
    for column in NEW_COLUMNS:
        if column not in updated_fieldnames:
            updated_fieldnames.append(column)
    for row in rows:
        for column in NEW_COLUMNS:
            row.setdefault(column, "")
    return updated_fieldnames


def determine_run_date(run_date: Optional[str]) -> str:
    if run_date:
        if not re.fullmatch(r"\d{8}", run_date):
            raise ValueError("--run-date must follow YYYYMMDD.")
        return run_date
    return datetime.now().strftime("%Y%m%d")


def prepare_output_paths(
    input_tsv: Path,
    output_dir: Optional[Path],
    run_date: str,
    explicit_output: Optional[Path],
) -> Tuple[Path, Path, bool]:
    if explicit_output:
        output_tsv = explicit_output
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        if output_tsv.exists():
            logging.info("Using existing output TSV (resume mode): %s", output_tsv)
            created_new = False
        else:
            shutil.copy2(input_tsv, output_tsv)
            logging.info("Copied input TSV to %s", output_tsv)
            created_new = True
        output_xlsx = output_tsv.with_suffix(".xlsx")
        return output_tsv, output_xlsx, created_new

    base_dir = output_dir or DEFAULT_OUTPUT_DIR
    date_dir = base_dir / run_date
    date_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = date_dir / f"screening_results_{run_date}_{index:03d}.tsv"
        if not candidate.exists():
            shutil.copy2(input_tsv, candidate)
            logging.info("Copied input TSV to %s", candidate)
            output_tsv = candidate
            break
        index += 1
    output_xlsx = output_tsv.with_suffix(".xlsx")
    return output_tsv, output_xlsx, True


def build_prompt(
    *,
    title: str,
    abstract: str,
    heuristic_tags: str,
    reason: str,
    rq_definitions: Sequence[RQDefinition],
    allowed_codes: Sequence[str],
    extra_keywords: Sequence[str],
) -> str:
    rq_section = "\n\n".join(rq.format_block() for rq in rq_definitions)
    return PROMPT_TEMPLATE.format(
        title=title or "Unknown title",
        abstract=abstract or "No abstract provided.",
        heuristic_tags=heuristic_tags or "Not available",
        reason=reason or "Not provided",
        rq_section=rq_section,
        allowed_codes=", ".join(allowed_codes),
        extra_keywords=", ".join(extra_keywords) if extra_keywords else "(none)",
    )


def extract_json_payload(raw: str) -> Dict[str, object]:
    stripped = raw.strip()
    if stripped.startswith("```"):
        # Remove potential code fences from the response.
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object detected in LLM response.")
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def parse_llm_response(raw: str, allowed_codes: Sequence[str]) -> Tuple[str, str]:
    payload = extract_json_payload(raw)
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not a JSON object.")
    rqs_raw = str(payload.get("rqs", "")).strip()
    explanation = str(payload.get("explanation", "")).strip()
    if not explanation:
        raise ValueError("Missing 'explanation' field in LLM response.")
    allowed_set = list(allowed_codes)

    if not rqs_raw:
        normalized = ""
    else:
        seen: List[str] = []
        for token in rqs_raw.split(","):
            code = token.strip()
            if not code:
                continue
            if code not in allowed_set:
                raise ValueError(f"Invalid RQ code '{code}' in LLM response.")
            if code not in seen:
                seen.append(code)
        normalized = ",".join(seen)
    return normalized, explanation


def extract_reasoning_block(raw: str) -> Tuple[str, str]:
    match = REASONING_PATTERN.search(raw)
    if not match:
        return "", raw
    reasoning_block = raw[match.start() : match.end()].strip()
    remainder = (raw[: match.start()] + raw[match.end() :]).strip()
    return reasoning_block, remainder


def apply_result(row: Dict[str, str], result: WorkResult) -> None:
    row["rqs_llm"] = result.rqs
    row["llm_explanation"] = result.explanation
    row["llm_response"] = result.raw_response
    row["llm_reasoning"] = result.reasoning
    row["llm_status"] = result.status


def mark_simple_status(row: Dict[str, str], status: str, message: str) -> None:
    row["rqs_llm"] = ""
    row["llm_explanation"] = message
    row["llm_response"] = ""
    row["llm_reasoning"] = ""
    row["llm_status"] = status


def collect_work_items(
    rows: List[Dict[str, str]],
    bib_entries: Dict[str, BibEntry],
    rq_definitions: Sequence[RQDefinition],
    allowed_codes: Sequence[str],
    *,
    bib_id_filter: Optional[str],
    force: bool,
    extra_keywords: Sequence[str],
    prompt_dir: Optional[Path] = None,
) -> List[WorkItem]:
    work: List[WorkItem] = []
    if prompt_dir is not None:
        try:
            prompt_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to create prompt directory %s: %s", prompt_dir, exc)
            prompt_dir = None
    for index, row in enumerate(rows):
        row_id = (row.get("id") or "").strip()
        if not row_id:
            logging.warning("Row %s is missing 'id'; skipping.", index)
            continue
        if bib_id_filter and row_id != bib_id_filter:
            continue
        if not force and (row.get("llm_response") or "").strip():
            logging.debug("Skipping %s (already processed).", row_id)
            continue

        entry = bib_entries.get(row_id)
        if not entry:
            mark_simple_status(row, STATUS_ID_NO_MATCH, STATUS_ID_NO_MATCH)
            logging.warning("Missing BibTeX entry for id %s.", row_id)
            continue

        abstract = entry.abstract
        if not abstract:
            mark_simple_status(row, STATUS_NO_ABSTRACT, STATUS_NO_ABSTRACT)
            logging.warning("No abstract for id %s.", row_id)
            continue

        reason = (row.get("reason") or "").strip()
        heuristic_tags = (row.get("RQs") or "").strip()
        prompt = build_prompt(
            title=entry.title,
            abstract=abstract,
            heuristic_tags=heuristic_tags,
            reason=reason,
            rq_definitions=rq_definitions,
            allowed_codes=allowed_codes,
            extra_keywords=extra_keywords,
        )
        item = WorkItem(row_index=index, row_id=row_id, prompt=prompt)
        # Optionally dump prompt to disk for debugging
        if prompt_dir is not None:
            try:
                safe_id = re.sub(r"[^\w.-]+", "_", row_id) or f"row_{index}"
                pth = prompt_dir / f"{index:04d}_{safe_id}.txt"
                title_source = entry.title or (row.get("title") or "").strip()
                title_clean = re.sub(r"\s+", " ", title_source).strip() or "(sin título)"
                header = (
                    f"{PROMPT_SEPARATOR_LINE}\n"
                    f"ID: {row_id} | {title_clean}\n"
                    f"{PROMPT_SEPARATOR_LINE}\n\n"
                    f"{prompt}"
                )
                pth.write_text(header, encoding="utf-8")
            except Exception as exc:  # pragma: no cover
                logging.warning("Failed to write prompt for %s: %s", row_id, exc)
        work.append(item)
    return work


def run_work_items(
    client: LLMClient,
    items: Sequence[WorkItem],
    allowed_codes: Sequence[str],
    *,
    max_retries: int,
    concurrency: int,
    capture_reasoning: bool,
) -> List[WorkResult]:
    if not items:
        return []

    def _execute(item: WorkItem) -> WorkResult:
        for attempt in range(1, max_retries + 1):
            raw_response = ""
            reasoning_block = ""
            try:
                logging.info("Requesting LLM decision for %s (attempt %s/%s).", item.row_id, attempt, max_retries)
                raw_response, reasoning_block = client.generate(item.prompt, capture_reasoning=capture_reasoning)
                response_for_parsing = raw_response
                if capture_reasoning and not reasoning_block:
                    extracted_reasoning, remainder = extract_reasoning_block(raw_response)
                    if extracted_reasoning:
                        reasoning_block = extracted_reasoning
                        response_for_parsing = remainder
                rqs, explanation = parse_llm_response(response_for_parsing, allowed_codes)
                return WorkResult(
                    row_index=item.row_index,
                    status=STATUS_OK,
                    rqs=rqs,
                    explanation=explanation,
                    reasoning=reasoning_block,
                    raw_response=raw_response,
                )
            except LLMTimeoutError:
                logging.error("LLM timeout for %s (attempt %s/%s).", item.row_id, attempt, max_retries)
                if attempt == max_retries:
                    return WorkResult(row_index=item.row_index, status=STATUS_TIMEOUT, explanation=STATUS_TIMEOUT)
            except LLMClientError as exc:
                logging.error("LLM error for %s: %s", item.row_id, exc)
                if attempt == max_retries:
                    return WorkResult(row_index=item.row_index, status=STATUS_LLM_ERROR, explanation=str(exc))
            except ValueError as exc:
                logging.error("Failed to parse LLM response for %s: %s", item.row_id, exc)
                return WorkResult(
                    row_index=item.row_index,
                    status=STATUS_PARSE_ERROR,
                    explanation=str(exc),
                    reasoning=reasoning_block,
                    raw_response=raw_response,
                )
        return WorkResult(row_index=item.row_index, status=STATUS_LLM_ERROR, explanation="Unknown error")

    results: List[WorkResult] = []
    workers = max(1, concurrency)
    if workers == 1:
        for item in items:
            results.append(_execute(item))
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(_execute, item): item for item in items}
        for future in as_completed(future_map):
            results.append(future.result())
    return results


def summarize_status(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for row in rows:
        status = (row.get("llm_status") or "").strip() or "pending"
        summary[status] = summary.get(status, 0) + 1
    return summary


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    setup_logging(args.log_level)

    input_tsv = Path(args.input_tsv).resolve()
    bib_path = Path(args.bib_file).resolve()
    rq_config_path = Path(args.rqconfig).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    output_override = Path(args.output_tsv).resolve() if args.output_tsv else None

    run_date = determine_run_date(args.run_date)
    output_tsv, output_xlsx, _ = prepare_output_paths(input_tsv, output_dir, run_date, output_override)

    rows, fieldnames = read_tsv(output_tsv)
    fieldnames = ensure_columns(fieldnames, rows)

    bib_entries = load_bib_entries(bib_path)
    rq_definitions, extra_keywords = load_rq_config(rq_config_path)
    allowed_codes = [rq.code for rq in rq_definitions]

    logging.info("Loaded %s rows from %s.", len(rows), output_tsv)
    logging.info("Loaded %s BibTeX entries.", len(bib_entries))
    logging.info("Loaded RQ config: %s.", ", ".join(allowed_codes))

    if args.summary_only:
        summary = summarize_status(rows)
        logging.info("Summary only mode. Status counts: %s", summary)
        return 0

    prompt_dir = output_tsv.parent / "prompts" if args.dump_prompts else None

    items = collect_work_items(
        rows,
        bib_entries,
        rq_definitions,
        allowed_codes,
        bib_id_filter=args.bib_id,
        force=args.force,
        extra_keywords=extra_keywords,
        prompt_dir=prompt_dir,
    )

    logging.info("Prepared %s work items for LLM processing.", len(items))

    if items:
        with LLMClient(
            provider=args.provider,
            host=args.host,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
            request_timeout=args.request_timeout,
        ) as client:
            results = run_work_items(
                client,
                items,
                allowed_codes,
                max_retries=args.max_retries,
                concurrency=args.concurrency,
                capture_reasoning=args.capture_reasoning,
            )
        for result in results:
            apply_result(rows[result.row_index], result)
    else:
        logging.info("Nothing to process (all rows skipped or filtered).")

    write_tsv(output_tsv, fieldnames, rows)
    logging.info("Updated TSV saved to %s.", output_tsv)

    export_to_excel(output_xlsx, fieldnames, rows, args.sheet_name)
    logging.info("Excel export created at %s.", output_xlsx)

    summary = summarize_status(rows)
    logging.info("Final status counts: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(run())
