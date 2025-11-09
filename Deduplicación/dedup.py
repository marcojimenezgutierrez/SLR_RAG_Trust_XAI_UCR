#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Herramientas para la preparación y deduplicación de archivos BibTeX multi-fuente.

Características principales:
 - CLI basado en argparse con subcomandos `annotate` y `dedup`
 - Parseo robusto con bibtexparser y normalización extendida de títulos/DOI
 - Etiquetado automático del origen (`source`) dentro del campo keywords
 - Deduplicación configurable (prioridades externas, tolerancia en años, rapidfuzz)
 - Reportes enriquecidos (XLSX/CSV + JSON) con puntajes y criterios aplicados

Requiere: bibtexparser, rapidfuzz (u otra implementación compatible) y, opcionalmente, pandas.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import unicodedata
import copy
import glob
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent

DEFAULT_DEDUP_INPUTS = [
    HERE / "SLR - JoCiCi - 2025 - ACM Library.bib",
    HERE / "SLR - JoCiCi - 2025 -IEEE Xplore.bib",
    HERE / "SLR - JoCiCi - 2025 - Google Scholar.bib",
    HERE / "SLR - JoCiCi - 2025 - Arxiv.bib",
    HERE / "SRL - JoCiCi - 2025 - SCOPUS.bib",
]

DEFAULT_SEARCH_GLOB = HERE.parent / "Búsqueda" / "*.bib"

DEFAULT_OUT_BIB = HERE / "SLR_RAG_master_clean.bib"
DEFAULT_OUT_DUP = HERE / "SLR_RAG_duplicates_report.xlsx"
DEFAULT_OUT_SUM = HERE / "SLR_RAG_summary.txt"
DEFAULT_OUT_JSON = HERE / "SLR_RAG_duplicates_report.json"

DEFAULT_PRIORITY_CONFIG = {
    "type_priority": {
        "article": 0.0,
        "journal": 0.0,
        "inproceedings": 1.0,
        "conference": 1.0
    },
    "source_priority": {},
    "bonuses": {
        "has_doi": -0.25,
        "has_url": -0.1,
        "extra_fields": -0.005
    }
}

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - entorno sin pandas
    pd = None

try:
    import bibtexparser  # type: ignore
    from bibtexparser.bparser import BibTexParser  # type: ignore
    from bibtexparser.bwriter import BibTexWriter  # type: ignore
    from bibtexparser.bibdatabase import BibDatabase  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Se requiere bibtexparser. Instálalo vía 'pip install bibtexparser'.") from exc

try:
    from rapidfuzz import fuzz  # type: ignore
except ImportError:  # pragma: no cover - fallback
    from difflib import SequenceMatcher

    def fuzz_token_ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0
else:
    def fuzz_token_ratio(a: str, b: str) -> float:
        return float(fuzz.token_set_ratio(a, b))


@dataclass
class PriorityConfig:
    type_priority: Dict[str, float] = field(default_factory=lambda: DEFAULT_PRIORITY_CONFIG["type_priority"].copy())
    source_priority: Dict[str, float] = field(default_factory=lambda: DEFAULT_PRIORITY_CONFIG["source_priority"].copy())
    bonuses: Dict[str, float] = field(default_factory=lambda: DEFAULT_PRIORITY_CONFIG["bonuses"].copy())

    @classmethod
    def from_path(cls, path: Optional[Path]) -> "PriorityConfig":
        if not path:
            return cls()
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de configuración de prioridades: {path}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        cfg = copy.deepcopy(DEFAULT_PRIORITY_CONFIG)
        for key, value in loaded.items():
            if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
        return cls(
            type_priority=cfg.get("type_priority", {}).copy(),
            source_priority=cfg.get("source_priority", {}).copy(),
            bonuses=cfg.get("bonuses", {}).copy(),
        )


@dataclass
class Entry:
    source: str
    source_path: Path
    item_type: str
    cite_key: str
    fields: Dict[str, str]
    norm_title: str = ""
    norm_doi: str = ""
    norm_first_author: str = ""
    norm_venue: str = ""
    year_int: Optional[int] = None
    priority: float = 0.0

    def ensure_source_keyword(self):
        tag = f"source: {self.source}"
        value = self.fields.get("keywords", "")
        tokens = [t.strip() for t in re.split(r"[;,]", value) if t.strip()]
        if any(t.lower() == tag.lower() for t in tokens):
            self.fields["keywords"] = ", ".join(tokens) if tokens else tag
            return
        tokens.append(tag)
        self.fields["keywords"] = ", ".join(tokens)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if unicodedata.category(c) != "Mn"
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_title(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("&", " and ")
    text = strip_accents(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return normalize_whitespace(text)


def normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    doi = html.unescape(doi)
    doi = doi.strip().lower()
    doi = doi.replace("\\", "")
    doi = re.sub(r"https?://(dx\.)?doi\.org/", "", doi)
    doi = doi.replace("doi:", "")
    doi = doi.split("?")[0].split("#")[0]
    doi = doi.strip().strip(".").strip("/")
    return doi


def normalize_person(value: str) -> str:
    if not value:
        return ""
    parts = re.split(r"\s+and\s+|,\s*", value, flags=re.IGNORECASE)
    first = parts[0] if parts else ""
    first = re.sub(r"[^A-Za-zÀ-ÿ\s]", " ", first)
    return normalize_title(first)


def normalize_venue(entry_fields: Dict[str, str]) -> str:
    venue = entry_fields.get("journal") or entry_fields.get("booktitle") or ""
    return normalize_title(venue)


def extract_year(value: str) -> Optional[int]:
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    return int(match.group(0)) if match else None


def infer_source_from_name(path: Path, explicit: Optional[str] = None, source_map: Optional[Dict[str, str]] = None) -> str:
    if explicit:
        return explicit
    source_map = source_map or {}
    name = path.stem
    tokens = re.split(r"[_\-]", name)
    # prefer explicit mappings
    for token in tokens + [name]:
        clean = token.strip().upper()
        if clean in source_map:
            return source_map[clean]
    # heurística: prefijo antes del primer guion bajo (caso Búsqueda)
    prefix = name.split("_", 1)[0].strip()
    if prefix and prefix.isalpha():
        return source_map.get(prefix.upper(), prefix)
    # fallback: último token mayúscula (caso SLR - ... - ACM Library)
    dash_tokens = [t.strip() for t in name.split("-") if t.strip()]
    for token in reversed(dash_tokens):
        if token.isupper():
            return source_map.get(token, token)
        if token.lower() in {"acm library", "ieee xplore", "google scholar", "arxiv", "scopus"}:
            return source_map.get(token.upper(), token.title())
    return source_map.get("DEFAULT", "Unknown")


def build_bibtex_parser() -> BibTexParser:
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    parser.homogenize_fields = True
    return parser


def load_bib_entries(
    path: Path,
    source_label: Optional[str],
    source_map: Optional[Dict[str, str]],
) -> List[Entry]:
    parser = build_bibtex_parser()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        database = bibtexparser.load(handle, parser=parser)

    entries: List[Entry] = []
    inferred_source = infer_source_from_name(path, explicit=source_label, source_map=source_map)
    for raw in database.entries:
        fields = {k: v for k, v in raw.items() if k not in {"ENTRYTYPE", "ID"}}
        entry = Entry(
            source=inferred_source,
            source_path=path,
            item_type=raw.get("ENTRYTYPE", "article"),
            cite_key=raw.get("ID", ""),
            fields=fields,
        )
        entries.append(entry)
    return entries


def prepare_entries(entries: Iterable[Entry], priority_cfg: PriorityConfig):
    for entry in entries:
        entry.norm_title = normalize_title(entry.fields.get("title", ""))
        entry.norm_doi = normalize_doi(entry.fields.get("doi", ""))
        entry.norm_first_author = normalize_person(entry.fields.get("author", ""))
        entry.norm_venue = normalize_venue(entry.fields)
        entry.year_int = extract_year(entry.fields.get("year", ""))
        entry.priority = compute_priority(entry, priority_cfg)
        entry.ensure_source_keyword()


def compute_priority(entry: Entry, cfg: PriorityConfig) -> float:
    item_type = entry.item_type.lower()
    base = cfg.type_priority.get(item_type, 2.0)
    # intentar detectar "journal" en note/booktitle para mejorar heurística
    note = entry.fields.get("note", "").lower()
    booktitle = entry.fields.get("booktitle", "").lower()
    if "journal" in note or "journal" in booktitle:
        base = min(base, cfg.type_priority.get("journal", base))
    base += cfg.source_priority.get(entry.source, 0.0)
    if entry.norm_doi:
        base += cfg.bonuses.get("has_doi", 0.0)
    if entry.fields.get("url"):
        base += cfg.bonuses.get("has_url", 0.0)
    base += cfg.bonuses.get("extra_fields", 0.0) * len(entry.fields)
    return base


def bucket_signature(entry: Entry) -> Tuple[str, str]:
    tokens = entry.norm_title.split()
    prefix = "".join(tokens[:2])[:8]
    return (prefix or entry.norm_title[:2], str(entry.year_int or "unknown"))


def titles_match(
    e1: Entry,
    e2: Entry,
    threshold: float,
    year_tolerance: int,
    allow_author_mismatch: bool,
) -> Tuple[bool, float, str]:
    if e1.year_int and e2.year_int:
        if abs(e1.year_int - e2.year_int) > year_tolerance:
            return False, 0.0, "year_mismatch"
    score = fuzz_token_ratio(e1.norm_title, e2.norm_title) / 100.0
    bonus = 0.0
    if e1.norm_venue and e1.norm_venue == e2.norm_venue:
        bonus += 0.03
    if e1.norm_first_author and e1.norm_first_author == e2.norm_first_author:
        bonus += 0.02
    effective_score = min(score + bonus, 1.0)
    if effective_score < threshold:
        return False, effective_score, "title_threshold"
    if not allow_author_mismatch and e1.norm_first_author and e2.norm_first_author:
        if e1.norm_first_author != e2.norm_first_author:
            return False, effective_score, "author_mismatch"
    return True, effective_score, "duplicate_by_title"


def pick_best_entry(group: List[Entry]) -> Tuple[Entry, List[Entry]]:
    sorted_group = sorted(group, key=lambda e: (e.priority, -len(e.fields)))
    return sorted_group[0], sorted_group[1:]


def deduplicate_entries(
    entries: List[Entry],
    similarity_threshold: float,
    year_tolerance: int,
    allow_author_mismatch: bool,
) -> Tuple[List[Entry], List[Dict[str, object]]]:
    kept: List[Entry] = []
    removed_rows: List[Dict[str, object]] = []
    by_doi: Dict[str, List[Entry]] = defaultdict(list)
    no_doi: List[Entry] = []
    for entry in entries:
        if entry.norm_doi:
            by_doi[entry.norm_doi].append(entry)
        else:
            no_doi.append(entry)

    for doi, group in by_doi.items():
        best, rest = pick_best_entry(group)
        kept.append(best)
        for entry in rest:
            removed_rows.append(build_removed_row(entry, best, "duplicate_by_doi", 1.0, doi, None))

    buckets: Dict[Tuple[str, str], List[Entry]] = defaultdict(list)
    for entry in no_doi:
        buckets[bucket_signature(entry)].append(entry)

    visited = set()
    for items in buckets.values():
        for entry in items:
            if id(entry) in visited:
                continue
            group = [entry]
            visited.add(id(entry))
            pair_meta: Dict[int, Tuple[Entry, float, str]] = {}
            for candidate in items:
                if id(candidate) in visited:
                    continue
                match, score, reason = titles_match(entry, candidate, similarity_threshold, year_tolerance, allow_author_mismatch)
                if match:
                    group.append(candidate)
                    pair_meta[id(candidate)] = (entry, score, reason)
                    visited.add(id(candidate))
            if len(group) == 1:
                kept.append(entry)
            else:
                best, rest = pick_best_entry(group)
                kept.append(best)
                for candidate in rest:
                    origin, score, reason = pair_meta.get(id(candidate), (best, similarity_threshold, "duplicate_by_title"))
                    if origin is not best:
                        match, new_score, new_reason = titles_match(best, candidate, similarity_threshold, year_tolerance, allow_author_mismatch)
                        if match:
                            score, reason = new_score, new_reason
                    removed_rows.append(build_removed_row(candidate, best, reason, score, candidate.norm_title, None))

    return kept, removed_rows


def build_removed_row(entry: Entry, kept: Entry, reason: str, score: float, match_value: Optional[str], aux_reason: Optional[str]) -> Dict[str, object]:
    return {
        "removed_source": entry.source,
        "removed_path": str(entry.source_path),
        "removed_cite_key": entry.cite_key,
        "removed_title": entry.fields.get("title", ""),
        "removed_year": entry.fields.get("year", ""),
        "removed_doi": entry.fields.get("doi", ""),
        "removed_priority": entry.priority,
        "kept_source": kept.source,
        "kept_path": str(kept.source_path),
        "kept_cite_key": kept.cite_key,
        "kept_title": kept.fields.get("title", ""),
        "kept_year": kept.fields.get("year", ""),
        "kept_doi": kept.fields.get("doi", ""),
        "kept_priority": kept.priority,
        "reason": aux_reason or reason,
        "match_value": match_value or kept.norm_doi,
        "match_score": round(score, 3),
    }


def write_bib(entries: List[Entry], destination: Path):
    writer = BibTexWriter()
    writer.indent = "  "
    writer.order_entries_by = ("ID",)
    writer.align_values = True
    database = BibDatabase()
    serialized = []
    for entry in entries:
        record = {
            "ENTRYTYPE": entry.item_type,
            "ID": entry.cite_key or f"key_{abs(hash(entry.norm_title))}",
        }
        record.update(entry.fields)
        serialized.append(record)
    database.entries = serialized
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(writer.write(database))


def write_duplicates_report(rows: List[Dict[str, object]], destination: Path, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    if not rows:
        destination.write_text("No duplicates detected.", encoding="utf-8")
        return

    if pd is not None:
        df = pd.DataFrame(rows)
        try:
            df.to_excel(destination, index=False)
            return
        except Exception:
            pass

    csv_path = destination.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_summary(kept: List[Entry], removed_rows: List[Dict[str, object]], destination: Path, initial_counts: Counter):
    source_counts = Counter(entry.source for entry in kept)
    removed_counts = Counter(row["removed_source"] for row in removed_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("Duplicate Removal Summary\n\n")
        handle.write("Source, Initial_Count, Removed_Duplicates, Final_Count\n")
        for source in sorted(set(initial_counts.keys()) | set(source_counts.keys()) | set(removed_counts.keys())):
            handle.write(
                f"{source}, "
                f"{initial_counts.get(source,0)}, "
                f"{removed_counts.get(source,0)}, "
                f"{source_counts.get(source,0)}\n"
            )
        handle.write(f"\nFinal entries: {len(kept)}\nDuplicates removed: {len(removed_rows)}\n")


def resolve_input_paths(patterns: Optional[Sequence[str]], default_candidates: Sequence[Path]) -> List[Path]:
    resolved: List[Path] = []
    if patterns:
        for pattern in patterns:
            pattern_str = str(pattern)
            path = Path(pattern_str)
            if any(char in pattern_str for char in "*?[]"):
                resolved.extend(sorted(Path(p) for p in glob.glob(pattern_str)))
            elif path.exists():
                resolved.append(path)
            else:
                raise FileNotFoundError(f"No se encontró el archivo {pattern}")
    else:
        resolved = [Path(p) for p in default_candidates if Path(p).exists()]
    if not resolved:
        raise FileNotFoundError("No se encontraron archivos de entrada.")
    return resolved


def annotate_file(path: Path, source_map: Optional[Dict[str, str]], dry_run: bool = False) -> Dict[str, int]:
    parser = build_bibtex_parser()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        database = bibtexparser.load(handle, parser=parser)

    modified = 0
    source_label = infer_source_from_name(path, source_map=source_map)
    for entry in database.entries:
        keywords = entry.get("keywords", "")
        tag = f"source: {source_label}"
        tokens = [t.strip() for t in re.split(r"[;,]", keywords) if t.strip()]
        if any(t.lower() == tag.lower() for t in tokens):
            continue
        tokens.append(tag)
        entry["keywords"] = ", ".join(tokens)
        modified += 1

    if not dry_run and modified:
        writer = BibTexWriter()
        writer.indent = "  "
        writer.order_entries_by = ("ID",)
        writer.align_values = True
        with path.open("w", encoding="utf-8") as handle:
            handle.write(writer.write(database))

    return {"updated": modified}


def run_annotation(args: argparse.Namespace):
    source_map = load_source_map(args.source_map)
    inputs = resolve_annotation_inputs(args.inputs)
    totals = Counter()
    for path in inputs:
        stats = annotate_file(path, source_map=source_map, dry_run=args.dry_run)
        totals.update(stats)
        print(f"[ANNOTATE] {path}: {stats['updated']} registros etiquetados.")
    print(f"\nArchivos procesados: {len(inputs)} | Registros etiquetados: {totals['updated']}")
    if args.dry_run:
        print("Ejecutado en modo --dry-run (no se escribieron cambios).")


def resolve_annotation_inputs(patterns: Optional[Sequence[str]]) -> List[Path]:
    if patterns:
        inputs: List[Path] = []
        for pattern in patterns:
            inputs.extend(sorted(Path(p) for p in glob.glob(pattern)))
    else:
        inputs = sorted(Path(p) for p in glob.glob(str(DEFAULT_SEARCH_GLOB)))
    if not inputs:
        raise FileNotFoundError("No se hallaron archivos .bib para anotar.")
    return inputs


def load_source_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_dedup(args: argparse.Namespace):
    source_map = load_source_map(args.source_map)
    inputs = resolve_input_paths(args.inputs, DEFAULT_DEDUP_INPUTS)
    priority_cfg = PriorityConfig.from_path(Path(args.priority_config) if args.priority_config else None)

    all_entries: List[Entry] = []
    counters: Dict[str, int] = {}
    for path in inputs:
        entries = load_bib_entries(path, args.source_label, source_map)
        counters[str(path)] = len(entries)
        all_entries.extend(entries)
        preview_source = entries[0].source if entries else infer_source_from_name(path, args.source_label, source_map)
        print(f"[LOAD] {path} -> {len(entries)} registros (source: {preview_source})")

    prepare_entries(all_entries, priority_cfg)
    initial_counts = Counter(entry.source for entry in all_entries)
    kept, removed_rows = deduplicate_entries(
        all_entries,
        similarity_threshold=args.similarity_threshold,
        year_tolerance=args.year_tolerance,
        allow_author_mismatch=args.allow_author_mismatch,
    )

    write_bib(kept, Path(args.output_bib))
    write_duplicates_report(removed_rows, Path(args.duplicates_report), Path(args.duplicates_json))
    write_summary(kept, removed_rows, Path(args.summary_path), initial_counts)

    print(f"\nFinal entries count: {len(kept)}")
    print(f"Duplicates removed: {len(removed_rows)}")
    print(f"Clean bib saved to {args.output_bib}")
    print(f"Duplicates report saved to {args.duplicates_report} / {args.duplicates_json}")
    print(f"Summary saved to {args.summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Herramientas de deduplicación para BibTeX multi-fuente.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="Etiqueta cada entrada con el metadato source en keywords.")
    annotate.add_argument("--inputs", nargs="+", help="Archivos o patrones glob (por defecto Búsqueda/*.bib).")
    annotate.add_argument("--source-map", help="Ruta a JSON con mapeo de tokens a nombres de fuente.")
    annotate.add_argument("--dry-run", action="store_true", help="Simula los cambios sin escribir archivos.")
    annotate.set_defaults(func=run_annotation)

    dedup = subparsers.add_parser("dedup", help="Ejecuta la deduplicación con parámetros configurables.")
    dedup.add_argument("--inputs", nargs="+", help="Archivos o patrones glob. Si se omite usa entradas por defecto.")
    dedup.add_argument("--source-label", help="Sobrescribe la etiqueta de fuente para todos los archivos.")
    dedup.add_argument("--source-map", help="Ruta a JSON con mapeo de tokens a nombres de fuente.")
    dedup.add_argument("--output-bib", default=str(DEFAULT_OUT_BIB), help="Archivo .bib limpio resultante.")
    dedup.add_argument("--duplicates-report", default=str(DEFAULT_OUT_DUP), help="Reporte tabular de duplicados (XLSX/CSV).")
    dedup.add_argument("--duplicates-json", default=str(DEFAULT_OUT_JSON), help="Reporte JSON con detalles de duplicados.")
    dedup.add_argument("--summary-path", default=str(DEFAULT_OUT_SUM), help="Resumen de conteos por fuente.")
    dedup.add_argument("--priority-config", help="Archivo JSON con configuración de prioridades.")
    dedup.add_argument("--similarity-threshold", type=float, default=0.95, help="Umbral de similitud de título (0-1).")
    dedup.add_argument("--year-tolerance", type=int, default=0, help="Tolerancia permitida en años (ej. 1 para ±1).")
    dedup.add_argument("--allow-author-mismatch", action="store_true", help="Permite combinar duplicados aunque el primer autor difiera.")
    dedup.set_defaults(func=run_dedup)

    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
