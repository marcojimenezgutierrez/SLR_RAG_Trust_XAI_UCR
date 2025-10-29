#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limpieza y deduplicación de un archivo .bib único.

Características:
- Parseo robusto por llaves balanceadas.
- Normalización de DOI (lower, sin prefijos https://doi.org/ o doi:).
- Recuperación de DOI desde el campo url (p.ej., Scopus con "doi=").
- Normalización de título para detección de duplicados (lower, sin acentos/puntuación).
- Deduplicación primaria por DOI; secundaria por similitud de título + año (+ primer autor) con umbral alto.
- Selección del mejor registro (journal>conference>otros; más campos completos).
- Fusión de keywords y preservación de campos valiosos.
- Unificación de url hacia https://doi.org/<doi> cuando exista DOI.
- Resolución de colisiones de claves BibTeX, con mapeo removed->kept.

Salidas junto al .bib de entrada:
- <nombre>.dedup.bib
- <nombre>.duplicates.csv
- <nombre>.keymap.csv (mapa de claves eliminadas -> clave preservada)
- <nombre>.summary.txt

Uso:
  python clean_single_bib.py --input SLR_RAG_master_clean.bib
  (se generan archivos .dedup.* en el mismo directorio)
"""

from __future__ import annotations

import os
import re
import csv
import sys
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict
from typing import List, Dict, Tuple, Any
from urllib.parse import urlparse, parse_qs, unquote


def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize_title(t: str) -> str:
    if not t:
        return ''
    t = strip_accents(t)
    # quitar LaTeX básico
    t = re.sub(r"\\\\(textit|textbf|emph)\s*", "", t)
    t = t.replace('{', ' ').replace('}', ' ')
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_doi(doi: str) -> str:
    if not doi:
        return ''
    d = doi.strip().lower()
    d = d.replace('\\', '')
    d = re.sub(r'^(https?://(dx\.)?doi\.org/)', '', d)
    d = d.replace('doi:', '').strip()
    return d


def extract_doi_from_url(url: str) -> str:
    if not url:
        return ''
    u = url.strip()
    try:
        parsed = urlparse(u)
        # 1) parámetro query doi=...
        qs = parse_qs(parsed.query)
        if 'doi' in qs and qs['doi']:
            return normalize_doi(unquote(qs['doi'][0]))
        # 2) buscar patrón DOI en la URL completa
        m = re.search(r'(10\.\d{4,9}/[^\s&]+)', u, flags=re.I)
        if m:
            return normalize_doi(unquote(m.group(1)))
    except Exception:
        pass
    return ''


def first_author(authors_field: str) -> str:
    if not authors_field:
        return ''
    parts = re.split(r'\s+and\s+|,\s*', authors_field, flags=re.I)
    return parts[0].strip() if parts else ''


def item_priority(item_type: str, note_field: str, fields: Dict[str, str]) -> int:
    t = (item_type or '').lower()
    n = (note_field or '').lower()
    j = (fields.get('journal', '') or '').lower()
    # menor número = mayor prioridad
    if 'article' in t and (('journal' in n) or j):
        return 0
    if 'inproceedings' in t or 'conference' in t or 'proceedings' in n or 'conference' in n:
        return 1
    return 2


def parse_bibtex(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()

    out = []
    i, n = 0, len(txt)
    while i < n:
        if txt[i] != '@':
            i += 1
            continue
        j = i + 1
        while j < n and txt[j].isalpha():
            j += 1
        entry_type = txt[i+1:j].strip()
        while j < n and txt[j] != '{':
            j += 1
        if j >= n:
            break
        j += 1
        start = j
        brace = 1
        while j < n and brace > 0:
            if txt[j] == '{':
                brace += 1
            elif txt[j] == '}':
                brace -= 1
            j += 1
        block = txt[start:j-1]
        if ',' in block:
            cite_key, rest = block.split(',', 1)
            cite_key = cite_key.strip()
        else:
            cite_key, rest = block.strip(), ''

        fields: Dict[str, str] = {}
        for m in re.finditer(r'(\w+)\s*=\s*(\{.*?\}|\".*?\"|[^,]+)', rest, flags=re.S):
            k = m.group(1).strip().lower()
            v = m.group(2).strip().strip(',').strip()
            if v.startswith('{') and v.endswith('}'):
                v = v[1:-1]
            elif v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            fields[k] = v

        title = fields.get('title', '')
        author = fields.get('author', '')
        year = fields.get('year', '')
        doi = fields.get('doi', '')
        url = fields.get('url', '')
        note = fields.get('note', '')
        kws = fields.get('keywords', '')

        # completar DOI desde URL si no existe
        doi_norm = normalize_doi(doi)
        if not doi_norm:
            from_url = extract_doi_from_url(url)
            if from_url:
                doi_norm = from_url
                fields['doi'] = doi_norm
                # también preferimos URL canónica del DOI
                fields['url'] = f'https://doi.org/{doi_norm}'
        else:
            # normalizar URL a doi.org
            fields['url'] = f'https://doi.org/{doi_norm}'

        try:
            year_int = int(re.findall(r'\d{4}', year)[0]) if year else None
        except Exception:
            year_int = None

        out.append({
            'type': entry_type,
            'key': cite_key,
            'fields': fields,
            'title': title,
            'title_norm': normalize_title(title),
            'author': author,
            'first_author': first_author(author),
            'year': year,
            'year_int': year_int,
            'doi': fields.get('doi', ''),
            'doi_norm': doi_norm,
            'note': note,
            'keywords': kws,
            'priority': item_priority(entry_type, note, fields),
        })
        i = j

    return out


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def pick_best(group: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Prioridad luego cantidad de campos no vacíos
    def filled_fields_count(e: Dict[str, Any]) -> int:
        return sum(1 for v in e['fields'].values() if isinstance(v, str) and v.strip())

    sorted_group = sorted(group, key=lambda e: (e['priority'], -filled_fields_count(e)))
    return sorted_group[0], [g for g in sorted_group[1:]]


def merge_fields(keep: Dict[str, Any], other: Dict[str, Any]) -> None:
    f1, f2 = keep['fields'], other['fields']
    # completar vacíos en f1 con f2
    for k, v in f2.items():
        if not v:
            continue
        if k not in f1 or not f1[k]:
            f1[k] = v
    # fusionar keywords
    k1 = f1.get('keywords', '')
    k2 = f2.get('keywords', '')
    if k1 or k2:
        parts = [x.strip() for x in (k1 + ',' + k2).split(',') if x.strip()]
        # evitar duplicados (case-insensitive)
        seen = set()
        merged = []
        for p in parts:
            pl = p.lower()
            if pl not in seen:
                seen.add(pl)
                merged.append(p)
        f1['keywords'] = ', '.join(merged)
    # URL: preferir DOI
    doi_norm = normalize_doi(f1.get('doi', ''))
    if doi_norm:
        f1['url'] = f'https://doi.org/{doi_norm}'


def ensure_unique_keys(entries: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Asegura claves únicas. Devuelve lista de tuplas (old_key, new_key) aplicadas."""
    mapping: List[Tuple[str, str]] = []
    seen = {}
    for e in entries:
        k = e['key']
        if k not in seen:
            seen[k] = 1
            continue
        # colisión: generar sufijos -a, -b, ...
        base = k
        idx = seen[base]
        while True:
            candidate = f"{base}-{idx}"
            if candidate not in seen:
                e['key'] = candidate
                mapping.append((k, candidate))
                seen[candidate] = 1
                seen[base] = idx + 1
                break
            idx += 1
    return mapping


def write_bib(entries: List[Dict[str, Any]], path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            entry_type = e.get('type') or 'article'
            cite = e.get('key') or f"key_{abs(hash(e.get('title','')))}"
            fields = e.get('fields', {})

            f.write(f"@{entry_type}{{{cite},\n")
            ordered = [
                'title','author','year','doi','url','booktitle','journal','volume','number','pages',
                'publisher','note','keywords','abstract'
            ]
            done = set()
            for k in ordered:
                if k in fields and fields[k]:
                    f.write(f"  {k} = {{{fields[k]}}},\n")
                    done.add(k)
            for k, v in fields.items():
                if k in done or not v:
                    continue
                f.write(f"  {k} = {{{v}}},\n")
            f.write("}\n\n")


def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='Limpieza y deduplicación de un .bib único')
    ap.add_argument('--input', '-i', required=True, help='Ruta al archivo .bib a limpiar')
    ap.add_argument('--thr', type=float, default=0.95, help='Umbral de similitud de título [0-1]')
    args = ap.parse_args(argv)

    in_bib = os.path.abspath(args.input)
    if not os.path.exists(in_bib):
        print(f"[ERROR] No se encontró: {in_bib}")
        return 2

    base, ext = os.path.splitext(in_bib)
    out_bib = base + '.dedup.bib'
    out_dup = base + '.duplicates.csv'
    out_map = base + '.keymap.csv'
    out_sum = base + '.summary.txt'

    entries = parse_bibtex(in_bib)
    total = len(entries)

    # agrupar por DOI
    by_doi: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    no_doi: List[Dict[str, Any]] = []
    for e in entries:
        if e['doi_norm']:
            by_doi[e['doi_norm']].append(e)
        else:
            no_doi.append(e)

    kept: List[Dict[str, Any]] = []
    removed_rows: List[Dict[str, Any]] = []

    # resolver duplicados por DOI
    for doi, group in by_doi.items():
        best, rest = pick_best(group)
        # fusionar campos de los descartados hacia el mejor
        for r in rest:
            merge_fields(best, r)
            removed_rows.append({
                'reason': 'duplicate_by_doi',
                'removed_key': r['key'],
                'kept_key': best['key'],
                'doi': doi,
                'removed_title': r['title'],
                'kept_title': best['title'],
                'removed_year': r['year'],
                'kept_year': best['year'],
            })
        kept.append(best)

    # agrupar no DOI por buckets (primera letra de título + año) y similitud
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for e in no_doi:
        buckets[(e['title_norm'][:1], e.get('year_int'))].append(e)

    visited = set()
    for _, items in buckets.items():
        for i, e in enumerate(items):
            if id(e) in visited:
                continue
            group = [e]
            visited.add(id(e))
            for j in range(i+1, len(items)):
                x = items[j]
                if id(x) in visited:
                    continue
                # año si ambos lo tienen
                if e.get('year_int') and x.get('year_int') and e['year_int'] != x['year_int']:
                    continue
                if seq_ratio(e['title_norm'], x['title_norm']) < args.thr:
                    continue
                # validar primer autor si existe en ambos
                a1 = normalize_title(e.get('first_author') or '')
                a2 = normalize_title(x.get('first_author') or '')
                if a1 and a2 and a1 != a2:
                    continue
                group.append(x)
                visited.add(id(x))
            if len(group) == 1:
                kept.append(e)
            else:
                best, rest = pick_best(group)
                for r in rest:
                    merge_fields(best, r)
                    removed_rows.append({
                        'reason': 'duplicate_by_title',
                        'removed_key': r['key'],
                        'kept_key': best['key'],
                        'doi': best.get('doi', ''),
                        'removed_title': r['title'],
                        'kept_title': best['title'],
                        'removed_year': r['year'],
                        'kept_year': best['year'],
                    })
                kept.append(best)

    # asegurar claves únicas (registro mapping solo para renombrados por colisión)
    key_mapping = ensure_unique_keys(kept)

    # escribir salidas
    write_bib(kept, out_bib)
    with open(out_dup, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['reason','doi','removed_key','kept_key','removed_title','kept_title','removed_year','kept_year'])
        w.writeheader()
        for row in removed_rows:
            w.writerow(row)
    with open(out_map, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['old_key','new_key'])
        for old, new in key_mapping:
            w.writerow([old, new])
    with open(out_sum, 'w', encoding='utf-8') as f:
        f.write(f"Archivo: {in_bib}\n")
        f.write(f"Entradas totales: {total}\n")
        f.write(f"Entradas finales: {len(kept)}\n")
        f.write(f"Duplicados eliminados: {len(removed_rows)}\n")
        f.write(f"Claves renombradas por colisión: {len(key_mapping)}\n")
        # Campos faltantes (post-clean)
        missing_doi = sum(1 for e in kept if not normalize_doi(e['fields'].get('doi','')))
        missing_title = sum(1 for e in kept if not e['fields'].get('title'))
        missing_author = sum(1 for e in kept if not e['fields'].get('author'))
        missing_year = sum(1 for e in kept if not e['fields'].get('year'))
        f.write(f"Faltan DOI: {missing_doi}\n")
        f.write(f"Faltan título: {missing_title}\n")
        f.write(f"Faltan autor: {missing_author}\n")
        f.write(f"Faltan año: {missing_year}\n")

    print(f"OK. Escrito: {out_bib}")
    print(f"Reporte duplicados: {out_dup}")
    print(f"Mapa de claves: {out_map}")
    print(f"Resumen: {out_sum}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))

