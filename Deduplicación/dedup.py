#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deduplicación de archivos .bib multi-fuente (Kitchenham SLR)
Reglas:
 - Normalización de títulos (lower, quitar acentos/puntuación)
 - Unificación de DOI (lower, quitar prefijos, espacios)
 - Duplicado por DOI exacto
 - Duplicado por similitud de título >= 0.95 + año + primer autor (fallback)
 - Prioridad de conservación: Journal > Conference > Other
 - Se preserva 'source' (fuente) como tag en keywords

Entradas esperadas (mismo directorio):
 - SLR - JoCiCi - 2025 - ACM Library.bib
 - SLR - JoCiCi - 2025 -IEEE Xplore.bib
 - SLR - JoCiCi - 2025 - Google Scholar.bib
 - SLR - JoCiCi - 2025 - Arxiv.bib
 - SRL - JoCiCi - 2025 - SCOPUS.bib

Salidas:
 - SLR_RAG_master_clean.bib
 - SLR_RAG_duplicates_report.xlsx (o .csv si no hay engine de Excel)
 - SLR_RAG_summary.txt
"""

import os
import re
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict, OrderedDict
from datetime import datetime

# pandas es opcional para XLSX; si no está, usamos CSV
try:
    import pandas as pd
except Exception:
    pd = None

HERE = os.path.abspath(os.path.dirname(__file__)) if '__file__' in globals() else os.getcwd()

INPUT_FILES = OrderedDict({
    "ACM": os.path.join(HERE, "SLR - JoCiCi - 2025 - ACM Library.bib"),
    "IEEE": os.path.join(HERE, "SLR - JoCiCi - 2025 -IEEE Xplore.bib"),
    "Google Scholar": os.path.join(HERE, "SLR - JoCiCi - 2025 - Google Scholar.bib"),
    "Arxiv": os.path.join(HERE, "SLR - JoCiCi - 2025 - Arxiv.bib"),
    "Scopus": os.path.join(HERE, "SRL - JoCiCi - 2025 - SCOPUS.bib"),
})

OUT_BIB = os.path.join(HERE, "SLR_RAG_master_clean.bib")
OUT_DUP = os.path.join(HERE, "SLR_RAG_duplicates_report.xlsx")
OUT_SUM = os.path.join(HERE, "SLR_RAG_summary.txt")

# ---------------------------
# Utilidades
# ---------------------------

def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_title(t: str) -> str:
    if not t:
        return ''
    t = t.strip()
    t = strip_accents(t)
    t = t.lower()
    # quitar puntuación y espacios repetidos
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def normalize_doi(doi: str) -> str:
    if not doi:
        return ''
    d = doi.strip().lower()
    d = d.replace('\\', '')
    # quitar prefijos comunes
    d = re.sub(r'^(https?://(dx\.)?doi\.org/)', '', d)
    d = d.replace('doi:', '').strip()
    return d

def first_author(authors_field: str) -> str:
    if not authors_field:
        return ''
    # separar por 'and' o comas
    parts = re.split(r'\s+and\s+|,\s*', authors_field, flags=re.IGNORECASE)
    return parts[0].strip() if parts else ''

def item_priority(item_type: str, note_type: str) -> int:
    """
    Menor número = mayor prioridad
    Journal > Conference > Other
    """
    t = (item_type or '').lower()
    n = (note_type or '').lower()
    if 'article' in t and 'journal' in n:
        return 0
    if 'article' in t and 'journal' in t:  # algunos .bib ponen 'article' a secas
        return 0
    if 'inproceedings' in t or 'conference' in t or 'proceedings' in n or 'conference' in n:
        return 1
    return 2

def parse_bibtex_minimal(path: str, source: str):
    """
    Parser mínimo de BibTeX:
    - Separa entradas por '@'
    - Identifica el bloque por llaves balanceadas
    - Extrae campos 'title', 'author', 'year', 'doi', 'note', 'keywords'
    - Guarda item_type (@article, @inproceedings, etc.) y cite_key
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        txt = f.read()

    entries = []
    i = 0
    n = len(txt)
    while i < n:
        if txt[i] != '@':
            i += 1
            continue
        # tipo
        j = i + 1
        while j < n and txt[j].isalpha():
            j += 1
        item_type = txt[i+1:j].strip()
        # salto hasta '{'
        while j < n and txt[j] != '{':
            j += 1
        if j >= n:
            break
        j += 1  # después de '{'
        start = j
        brace = 1
        while j < n and brace > 0:
            if txt[j] == '{':
                brace += 1
            elif txt[j] == '}':
                brace -= 1
            j += 1
        block = txt[start:j-1]

        # cite_key es lo primero hasta la coma
        if ',' in block:
            cite_key, rest = block.split(',', 1)
            cite_key = cite_key.strip()
        else:
            cite_key, rest = block.strip(), ''

        fields = {}
        for m in re.finditer(r'(\w+)\s*=\s*(\{.*?\}|\".*?\"|[^,]+)', rest, flags=re.S):
            k = m.group(1).strip().lower()
            v = m.group(2).strip().strip(',').strip()
            # quitar llaves/quotes
            if v.startswith('{') and v.endswith('}'):
                v = v[1:-1]
            elif v.startswith('"') and v.endswith('"'):
                v = v[1:-1]
            fields[k] = v

        title = fields.get('title', '')
        author = fields.get('author', '')
        year = fields.get('year', '')
        doi = fields.get('doi', '')
        note = fields.get('note', '')
        kws = fields.get('keywords', '')

        entries.append({
            'source': source,
            'item_type': item_type,
            'cite_key': cite_key,
            'fields': fields,
            'title': title,
            'year': year,
            'author': author,
            'first_author': first_author(author),
            'doi': doi,
            'note': note,
            'keywords': kws
        })
        i = j
    return entries

def write_bib(entries, path):
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            item_type = e.get('item_type') or 'article'
            cite = e.get('cite_key') or f"key_{abs(hash(e['title']))}"
            fields = e.get('fields', {})
            # garantizar keyword de fuente
            kws = fields.get('keywords', '')
            tag = f"source: {e.get('source','Unknown')}"
            if kws:
                # evitar duplicar tag
                if tag.lower() not in kws.lower():
                    kws = kws + ', ' + tag
            else:
                kws = tag
            fields['keywords'] = kws

            f.write(f"@{item_type}{{{cite},\n")
            # escribir campos con formato ordenado
            ordered = ['title','author','year','doi','url','booktitle','journal','volume','number','pages','publisher','note','keywords']
            done = set()
            for k in ordered:
                if k in fields and fields[k]:
                    f.write(f"  {k} = {{{fields[k]}}},\n")
                    done.add(k)
            # resto
            for k,v in fields.items():
                if k in done or not v:
                    continue
                f.write(f"  {k} = {{{v}}},\n")
            f.write("}\n\n")

def ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ---------------------------
# Carga y normalización
# ---------------------------

all_entries = []
counters = {}
for src, fpath in INPUT_FILES.items():
    if not os.path.exists(fpath):
        print(f"[WARN] No encontrado: {fpath}")
        counters[src] = 0
        continue
    entries = parse_bibtex_minimal(fpath, src)
    counters[src] = len(entries)
    all_entries.extend(entries)
print("Counts by source:", counters)
print("Total entries loaded:", len(all_entries))

# Normalizar campos derivados
for e in all_entries:
    e['norm_title'] = normalize_title(e.get('title'))
    e['norm_doi'] = normalize_doi(e.get('doi'))
    yp = e.get('year')
    try:
        e['year_int'] = int(re.findall(r'\d{4}', yp)[0]) if yp else None
    except Exception:
        e['year_int'] = None
    e['priority'] = item_priority(e.get('item_type'), e.get('note'))

# ---------------------------
# Deduplicación por DOI
# ---------------------------

kept = []
removed_rows = []

by_doi = defaultdict(list)
no_doi = []
for e in all_entries:
    if e['norm_doi']:
        by_doi[e['norm_doi']].append(e)
    else:
        no_doi.append(e)

def pick_best(group):
    # elige por menor prioridad; si empata, el que tenga más campos
    group_sorted = sorted(group, key=lambda x: (x['priority'], -len(x['fields'])))
    return group_sorted[0], [g for g in group_sorted[1:]]

# grupos por DOI
for doi, group in by_doi.items():
    best, rest = pick_best(group)
    kept.append(best)
    for r in rest:
        removed_rows.append({
            'removed_source': r['source'],
            'removed_cite_key': r['cite_key'],
            'removed_title': r['title'],
            'removed_year': r['year'],
            'removed_doi': r['doi'],
            'kept_source': best['source'],
            'kept_cite_key': best['cite_key'],
            'kept_title': best['title'],
            'kept_year': best['year'],
            'kept_doi': best['doi'],
            'reason': 'duplicate_by_doi'
        })

# ---------------------------
# Deduplicación por título (~=) + año + primer autor
# ---------------------------

# index por (norm_title, year_int, first_author) aproximado
# usaremos un bucket por primera letra para reducir comparaciones
buckets = defaultdict(list)
for e in no_doi:
    key = (e['norm_title'][:1], e.get('year_int'))
    buckets[key].append(e)

def similar(e1, e2, thr=0.95):
    # match año (si ambos tienen), y primer autor si existe
    if e1.get('year_int') and e2.get('year_int'):
        if e1['year_int'] != e2['year_int']:
            return False
    # similitud de título
    if ratio(e1['norm_title'], e2['norm_title']) < thr:
        return False
    # si ambos tienen primer autor, preferimos coincidir
    a1 = normalize_title(e1.get('first_author') or '')
    a2 = normalize_title(e2.get('first_author') or '')
    if a1 and a2 and a1 != a2:
        # si autores difieren y títulos muy similares, aún podría ser distinto; mantenemos conservador
        return False
    return True

visited = set()
for bkey, items in buckets.items():
    # greedy grouping
    for i, e in enumerate(items):
        if id(e) in visited:
            continue
        group = [e]
        visited.add(id(e))
        for j in range(i+1, len(items)):
            x = items[j]
            if id(x) in visited:
                continue
            if similar(e, x, thr=0.95):
                group.append(x)
                visited.add(id(x))
        if len(group) == 1:
            kept.append(e)
        else:
            best, rest = pick_best(group)
            kept.append(best)
            for r in rest:
                removed_rows.append({
                    'removed_source': r['source'],
                    'removed_cite_key': r['cite_key'],
                    'removed_title': r['title'],
                    'removed_year': r['year'],
                    'removed_doi': r['doi'],
                    'kept_source': best['source'],
                    'kept_cite_key': best['cite_key'],
                    'kept_title': best['title'],
                    'kept_year': best['year'],
                    'kept_doi': best['doi'],
                    'reason': 'duplicate_by_title'
                })

# ---------------------------
# Salidas
# ---------------------------

# Escribir .bib limpio
write_bib(kept, OUT_BIB)

# Reporte de duplicados
if pd is not None:
    df = pd.DataFrame(removed_rows, columns=[
        'removed_source', 'removed_cite_key', 'removed_title', 'removed_year', 'removed_doi',
        'kept_source', 'kept_cite_key', 'kept_title', 'kept_year', 'kept_doi', 'reason'
    ])
    try:
        df.to_excel(OUT_DUP, index=False)
        dup_path = OUT_DUP
    except Exception:
        # Fallback a CSV si no hay engine para Excel
        dup_path = OUT_DUP.replace('.xlsx', '.csv')
        df.to_csv(dup_path, index=False, encoding='utf-8')
else:
    # Sin pandas: escribir CSV manual
    dup_path = OUT_DUP.replace('.xlsx', '.csv')
    import csv
    with open(dup_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'removed_source', 'removed_cite_key', 'removed_title', 'removed_year', 'removed_doi',
            'kept_source', 'kept_cite_key', 'kept_title', 'kept_year', 'kept_doi', 'reason'
        ])
        w.writeheader()
        for row in removed_rows:
            w.writerow(row)

# Resumen por fuente
initial_counts = {src: 0 for src in INPUT_FILES.keys()}
for src, cnt in counters.items():
    initial_counts[src] = cnt

final_counts = {src: 0 for src in INPUT_FILES.keys()}
for e in kept:
    final_counts[e['source']] += 1

removed_counts = {src: initial_counts.get(src,0) - final_counts.get(src,0) for src in initial_counts}

with open(OUT_SUM, 'w', encoding='utf-8') as f:
    f.write("Duplicate Removal Summary:\n\n")
    f.write("Source, Initial_Count, Removed_Duplicates, Final_Count\n\n")
    for src in INPUT_FILES.keys():
        f.write(f"{src}, {initial_counts.get(src,0)}, {removed_counts.get(src,0)}, {final_counts.get(src,0)}\n")

print(f"Final entries count after removing duplicates: {len(kept)}")
print(f"Duplicates found: {len(removed_rows)}")
print(f"Clean bib saved to {OUT_BIB}")
print(f"Duplicates report saved to {dup_path}")
print(f"Summary saved to {OUT_SUM}")

