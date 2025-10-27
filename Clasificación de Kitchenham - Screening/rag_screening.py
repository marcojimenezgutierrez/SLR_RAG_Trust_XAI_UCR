#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_screening.py
----------------
Script para el *screening* inicial de referencias bibliográficas en una
revisión sistemática sobre modelos de generación aumentada con recuperación
(RAG) frente a modelos de lenguaje grandes (LLM) sin RAG.

Se adapta la metodología de filtrado de título/resumen de Kitchenham & Charters (2007),
utilizando una serie de heurísticas basadas en palabras clave para evaluar la relevancia
de cada entrada BibTeX respecto a varias preguntas de investigación (RQ) como la
confianza, la factualidad/alucinaciones, la explicabilidad/calibración y los
métodos/métricas.

Entradas:
  - Un archivo ``.bib`` que contiene las entradas a evaluar (por ejemplo ``SLR_RAG_master_clean.bib``).

Salidas (se generan en la carpeta indicada por ``--outdir``):
  - ``screening_results.tsv``: listado completo de entradas con sus puntuaciones, etiquetas y decisiones.
  - ``screening_summary.txt``: resumen estadístico de los resultados.
  - ``screening_top20.tsv``: las 20 entradas más relevantes según la heurística (score y año).
  - ``screening_ambiguous.tsv``: entradas que mencionan RAG pero no activan ninguna RQ (decisión ``Maybe``).

Algoritmo general:
  1. Se lee y parsea el archivo BibTeX, identificando tipo de entrada, identificador y campos utilizando un contador de profundidad
     de llaves y un manejo sencillo de comillas.
  2. Para cada entrada se construye un "blob" de texto con título, resumen y palabras clave. Sobre este texto se busca la presencia
     de sinónimos de RAG, indicadores de LLM y listas de palabras clave asociadas a cada RQ.
  3. Se asigna un puntaje binario por RQ (1 si alguna palabra clave aparece, 0 en caso contrario); estos se suman para obtener un
     puntaje total. Dependiendo de este total se asigna una etiqueta de relevancia (Alta, Media, Baja o Excluir) y se decide si la
     entrada se incluye, se excluye o se marca como ambigua.
  4. Finalmente se generan los archivos de salida ordenando por puntuación y año, y se reportan estadísticas de cobertura de RQs.

Uso:
  ``python rag_screening.py --bib SLR_RAG_master_clean.bib --outdir .``
"""

import argparse
import os
import re
import csv
from typing import List, Dict, Tuple

# ---------------------------
# Utilidades de parsing BibTeX
# ---------------------------

def read_file(path: str) -> str:
    """Lee un archivo de texto y devuelve su contenido como una cadena.

    Se utiliza un manejador de archivo con codificación UTF-8 y se
    ignoran errores de decodificación para manejar archivos BibTeX
    potencialmente ruidosos.

    Args:
        path (str): Ruta al archivo a leer.

    Returns:
        str: Contenido del archivo como texto.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def split_top_level(content: str, sep: str = ',') -> List[str]:
    """Divide el contenido por el separador dado únicamente a nivel superior.

    Este analizador simple recorre la cadena carácter por carácter
    manteniendo un contador de profundidad de llaves ``{}`` y un
    indicador de si estamos dentro de comillas dobles. Sólo se separa
    cuando se encuentra el separador y no estamos dentro de llaves ni
    comillas. Esto permite dividir entradas de BibTeX donde los
    valores pueden contener comas internas.

    Args:
        content (str): Cadena de entrada a dividir.
        sep (str): Carácter separador utilizado para la división.

    Returns:
        List[str]: Lista de subcadenas separadas respetando el nivel
        de anidamiento.
    """
    parts = []
    buf = []
    depth = 0
    in_quotes = False
    for ch in content:
        # Cambia el estado de comillas cada vez que se encuentra '"'
        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
            continue
        # Ajusta el nivel de profundidad de llaves únicamente si no
        # estamos dentro de comillas
        if not in_quotes:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
        # Si encontramos el separador al nivel superior, cortamos
        if ch == sep and depth == 0 and not in_quotes:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    # Añade el último fragmento
    if buf:
        parts.append(''.join(buf).strip())
    return parts

def unbrace(value: str) -> str:
    """Normaliza un valor eliminando envoltorios y espacios extra.

    Elimina llaves exteriores ``{...}`` o comillas ``"..."`` de un
    valor y colapsa los espacios internos para simplificar su
    posterior análisis.

    Args:
        value (str): Texto que puede estar envuelto en llaves o comillas.

    Returns:
        str: Valor limpio sin envolturas ni espacios redundantes.
    """
    v = value.strip()
    # Elimina envoltorios { ... } o " ... "
    if (v.startswith('{') and v.endswith('}')) or (v.startswith('"') and v.endswith('"')):
        v = v[1:-1]
    # Compacta espacios
    v = re.sub(r'\s+', ' ', v).strip()
    return v

def parse_fields(block: str) -> Dict[str, str]:
    """Convierte un bloque de campos BibTeX en un diccionario clave-valor.

    Cada par ``clave=valor`` se separa usando :func:`split_top_level` para
    respetar llaves y comillas. Los valores se limpian con
    :func:`unbrace` para eliminar envoltorios y espacios redundantes.

    Args:
        block (str): Cadena con campos separados por comas, por ejemplo
            ``author = {Doe}, title = {Título}``.

    Returns:
        Dict[str, str]: Diccionario con claves normalizadas en minúsculas y
        valores limpios.
    """
    fields: Dict[str, str] = {}
    for kv in split_top_level(block):
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        key = k.strip().lower()
        val = unbrace(v)
        fields[key] = val
    return fields

def parse_bib_entries(text: str) -> List[Dict[str, str]]:
    """Analiza un texto BibTeX y produce una lista de entradas.

    El parser navega por el texto buscando el símbolo ``@`` que
    introduce cada entrada. A partir de ahí, extrae el tipo de
    entrada (p. ej. ``article``, ``inproceedings``) y el identificador.
    Para aislar el bloque de campos de una entrada, cuenta el
    balance de llaves ``{`` y ``}`` a medida que avanza, de modo que
    los campos anidados no rompan el límite. Una vez extraído el
    bloque, llama a :func:`parse_fields` para obtener un
    diccionario de claves y valores. Este método no utiliza
    ninguna librería externa y es robusto a comas dentro de los
    valores.

    Args:
        text (str): Contenido completo de un archivo ``.bib``.

    Returns:
        List[Dict[str, str]]: Lista de entradas, cada una con las
        claves ``type`` (tipo de entrada), ``id`` (identificador) y
        los campos presentes en la entrada.
    """
    entries: List[Dict[str, str]] = []
    i = 0
    n = len(text)
    while True:
        at = text.find('@', i)
        if at == -1:
            break
        lb = text.find('{', at)
        if lb == -1:
            break
        entry_type = text[at + 1 : lb].strip().lower()
        comma = text.find(',', lb)
        if comma == -1:
            break
        entry_id = text[lb + 1 : comma].strip()
        # Balancea llaves hasta el cierre del registro
        depth = 1
        j = comma + 1
        while j < n and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        block = text[comma + 1 : j - 1]
        i = j
        fields = parse_fields(block)
        e: Dict[str, str] = {'type': entry_type, 'id': entry_id}
        e.update(fields)
        entries.append(e)
    return entries

# ---------------------------
# Reglas y palabras clave
# ---------------------------

def lower(s: str) -> str:
    """Convierte una cadena a minúsculas manejando valores ``None``.

    Args:
        s (str): Texto de entrada o ``None``.

    Returns:
        str: Texto en minúsculas, o cadena vacía si la entrada es
        ``None``.
    """
    return (s or '').lower()

def any_in(text: str, terms: List[str]) -> bool:
    """Comprueba si alguna palabra clave aparece dentro del texto.

    Los términos se buscan como subcadenas en la versión en minúsculas
    del texto para realizar comparaciones insensibles a mayúsculas.

    Args:
        text (str): Texto completo donde buscar.
        terms (List[str]): Lista de términos a buscar.

    Returns:
        bool: ``True`` si al menos un término está presente, ``False``
        en caso contrario.
    """
    t = lower(text)
    return any(term in t for term in terms)

# Núcleo RAG / equivalentes
SYN_RAG = [
    'retrieval-augmented generation', 'retrieval-augmented', 'retrieval augmented',
    'rag', 'graph rag', 'graphrag',
    'grounded generation', 'document grounded', 'document-grounded',
    'context augmentation', 'context-augmented',
    'hybrid llm', 'hybrid llms', 'agentic rag',
    'provenance', 'source attribution'
]

# Indicadores LLM/Baseline (suaves; no estrictamente obligatorios)
SYN_LLM = [
    'large language model', 'large language models', 'llm', 'llms',
    'chatgpt', 'gpt-4', 'gpt4', 'gpt-3', 'gpt3', 'openai'
]

# RQ1 – Confianza
KW_RQ1 = [
    'trust', 'confidence', 'credibility', 'user perception', 'perceived reliability',
    'overtrust', 'trustworthiness'
]

# RQ2 – Calidad / factualidad / alucinaciones
KW_RQ2 = [
    'hallucination', 'hallucinations', 'factuality', 'faithfulness', 'accuracy',
    'grounding', 'grounded', 'correctness', 'consistency'
]

# RQ3 – XAI / calibración / citas / evidencia
KW_RQ3 = [
    'explainable ai', 'explainability', 'interpretable', 'interpretability',
    'transparency', 'attribution', 'saliency', 'attention',
    'citation', 'citations', 'reference', 'provenance', 'source attribution',
    'evidence', 'evidence highlighting', 'highlighting',
    'calibration', 'brier', 'ece'
]

# RQ4 – Métodos / métricas / instrumentos
KW_RQ4 = [
    'evaluation', 'evaluation metric', 'evaluation metrics', 'metric', 'metrics',
    'methodology', 'methodologies', 'method', 'methods', 'protocol',
    'instrument', 'questionnaire', 'scale', 'survey', 'user study',
    'benchmark', 'dataset', 'guideline', 'framework', 'pipeline'
]

WINDOW_MIN, WINDOW_MAX = 2020, 2025

def detect_year(entry: Dict[str, str]) -> int:
    """Detecta el año numérico de una entrada BibTeX.

    Extrae el primer grupo de cuatro dígitos del campo ``year`` si
    existe; de lo contrario devuelve ``-1``. Esta función tolera
    campos ``year`` con letras o formatos inusuales.

    Args:
        entry (Dict[str, str]): Entrada del registro con un posible
            campo ``year``.

    Returns:
        int: Año extraído o ``-1`` si no se encuentra uno válido.
    """
    y = lower(entry.get('year', '')).strip()
    m = re.search(r'(\d{4})', y)
    return int(m.group(1)) if m else -1

def venue_of(entry: Dict[str, str]) -> str:
    """Devuelve el lugar de publicación de una entrada.

    Busca en orden los campos comunes de BibTeX que indican la
    publicación: ``journal``, ``booktitle``, ``publisher`` o
    ``howpublished``. Si ninguno está presente, devuelve cadena
    vacía.

    Args:
        entry (Dict[str, str]): Entrada de BibTeX.

    Returns:
        str: Campo de publicación más relevante o cadena vacía.
    """
    return entry.get('journal') or entry.get('booktitle') or entry.get('publisher') or entry.get('howpublished') or ''

def text_blob(entry: Dict[str, str]) -> str:
    """Construye un ``blob`` de texto a partir de los campos clave.

    Se concatenan los campos ``title``, ``abstract`` y ``keywords`` para
    crear un texto único donde buscar coincidencias de palabras clave.

    Args:
        entry (Dict[str, str]): Entrada con posibles campos de título,
            resumen y palabras clave.

    Returns:
        str: Texto concatenado (puede ser cadena vacía si no existen campos).
    """
    parts = [
        entry.get('title', ''), entry.get('abstract', ''), entry.get('keywords', '')
    ]
    return ' '.join(p for p in parts if p)

def rq_score(text: str, kws: List[str]) -> float:
    """Calcula un puntaje binario para una lista de palabras clave.

    Devuelve ``1.0`` si al menos una de las palabras clave aparece
    dentro del texto proporcionado; de lo contrario, devuelve ``0.0``.
    Este puntaje se utiliza para sumarizar la presencia de cada
    pregunta de investigación (RQ).

    Args:
        text (str): Texto en el que buscar términos.
        kws (List[str]): Palabras clave asociadas a una RQ.

    Returns:
        float: Puntaje 1.0 si hay coincidencia, 0.0 en caso contrario.
    """
    return 1.0 if any_in(text, kws) else 0.0

def label_from_total(total: float) -> str:
    """Mapea el puntaje total a una etiqueta de relevancia.

    Los umbrales se inspiran en la guía de Kitchenham para screening
    título/abstract, clasificando cada entrada como ``Alta``, ``Media``,
    ``Baja`` o ``Excluir`` según su puntaje agregado. Los rangos son
    inclusivos en los extremos inferiores.

    Args:
        total (float): Puntaje sumado de todas las RQs.

    Returns:
        str: Etiqueta de relevancia correspondiente.
    """
    if total >= 3.0:
        return 'Alta'
    if 1.5 <= total <= 2.5:
        return 'Media'
    if 0.5 <= total <= 1.0:
        return 'Baja'
    return 'Excluir'

def reason_from_rqs(rqs: List[str]) -> str:
    """Genera una explicación legible de las RQs que dispararon una inclusión.

    Toma la lista de etiquetas de RQ detectadas y devuelve una cadena en
    español que describe, de manera concisa, cuáles fueron los
    aspectos activados: confianza, factualidad, XAI/calibración o
    métricas/metodologías. Si la lista está vacía, devuelve
    ``'Fuera de alcance'``.

    Args:
        rqs (List[str]): Lista de etiquetas de RQ activadas.

    Returns:
        str: Descripción de las dimensiones activadas.
    """
    pieces: List[str] = []
    if 'RQ1' in rqs:
        pieces.append('Confianza/percepción')
    if 'RQ2' in rqs:
        pieces.append('Factualidad/alucinaciones')
    if 'RQ3' in rqs:
        pieces.append('XAI/calibración')
    if 'RQ4' in rqs:
        pieces.append('Métricas/metodologías')
    return ' y '.join(pieces) if pieces else 'Fuera de alcance'

def decide(entry: Dict[str, str]) -> Tuple[List[str], float, str, str]:
    """Aplica heurísticas de screening para decidir la inclusión de una entrada.

    Para una entrada dada, se detecta primero si está dentro de la
    ventana temporal definida (``WINDOW_MIN`` a ``WINDOW_MAX``). A
    continuación se construye un texto consolidado a partir del título,
    resumen y palabras clave, y se evalúa la presencia de términos
    relacionados con RAG y LLMs. Cada pregunta de investigación (RQ)
    tiene un conjunto de palabras clave asociadas; se calcula un
    puntaje binario por RQ y se suman para obtener un total. Dependiendo
    de este total y la presencia de señales de RAG, se asigna una
    etiqueta de relevancia y una decisión:

    * ``Include``: al menos un término de RQ en la ventana.
    * ``Maybe``: menciona RAG pero no se detectan palabras clave de RQ.
    * ``Exclude``: fuera de la ventana temporal o sin señales de RAG/RQ.

    Args:
        entry (Dict[str, str]): Diccionario con los campos de una
            entrada BibTeX.

    Returns:
        Tuple[List[str], float, str, str]: Una tupla con (lista de RQs
        activadas, puntaje total, etiqueta de relevancia y decisión).
    """
    year = detect_year(entry)
    if year != -1 and (year < WINDOW_MIN or year > WINDOW_MAX):
        return [], 0.0, 'Excluir', 'Exclude'  # Fuera de ventana

    text = lower(text_blob(entry))
    venue = lower(venue_of(entry))
    lang = lower(entry.get('language', ''))

    # Relevancia a RAG (criterio CI principal)
    is_rag = any_in(text, SYN_RAG)

    # Marcadores mínimos de dominio (suaves)
    mentions_llm = any_in(text, SYN_LLM)

    # Puntaje por RQ
    s1 = rq_score(text, KW_RQ1)
    s2 = rq_score(text, KW_RQ2)
    s3 = rq_score(text, KW_RQ3)
    s4 = rq_score(text, KW_RQ4)
    total = s1 + s2 + s3 + s4

    rqs: List[str] = []
    if s1 > 0:
        rqs.append('RQ1')
    if s2 > 0:
        rqs.append('RQ2')
    if s3 > 0:
        rqs.append('RQ3')
    if s4 > 0:
        rqs.append('RQ4')

    label = label_from_total(total)

    # Decisión Kitchenham (screening título/abstract)
    if year == -1 and total == 0 and not is_rag:
        return rqs, total, 'Excluir', 'Exclude'  # registro insuficiente

    if not is_rag and total == 0:
        return rqs, total, 'Excluir', 'Exclude'  # irrelevante al tópico central

    if is_rag and total == 0:
        return rqs, total, 'Baja', 'Maybe'  # RAG sin evidencia clara a RQs (ambigua)

    # Dentro de ventana y con hits de RQ
    if total > 0:
        return rqs, total, label, 'Include'

    # Caso por defecto:
    return rqs, total, 'Excluir', 'Exclude'

# ---------------------------
# Escritura de salidas
# ---------------------------

def write_tsv(path: str, rows: List[Dict[str, str]], header: List[str]) -> None:
    """Escribe una lista de diccionarios en formato TSV.

    Crea un archivo de valores separados por tabuladores con los
    encabezados especificados. Cada fila del archivo corresponde a un
    diccionario de ``rows``; las claves faltantes se rellenan con
    cadenas vacías.

    Args:
        path (str): Ruta del archivo de salida.
        rows (List[Dict[str, str]]): Lista de registros a escribir.
        header (List[str]): Lista ordenada de encabezados de columna.

    Returns:
        None
    """
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(h, '') for h in header])

def main() -> None:
    """Punto de entrada cuando se ejecuta como script.

    Esta función procesa la línea de comandos para obtener la ruta del
    archivo ``.bib`` y la carpeta de salida. Luego ejecuta el pipeline
    completo de lectura, parsing, puntuación, etiquetado y exportación
    de resultados para la revisión sistemática. Se genera un archivo
    TSV con todas las entradas, un resumen de conteos, un Top 20
    ordenado y un listado de entradas ambiguas.

    Las estadísticas se imprimen en consola para inspección rápida.

    Returns:
        None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--bib', required=True, help='Ruta al archivo .bib')
    ap.add_argument('--outdir', default='.', help='Carpeta de salida')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    text = read_file(args.bib)
    entries = parse_bib_entries(text)

    results: List[Dict[str, str]] = []
    include_count = 0
    exclude_count = 0
    maybe_count = 0

    rq_counts: Dict[str, int] = {'RQ1': 0, 'RQ2': 0, 'RQ3': 0, 'RQ4': 0}
    rel_counts: Dict[str, int] = {'Alta': 0, 'Media': 0, 'Baja': 0, 'Excluir': 0}

    for e in entries:
        year = detect_year(e)
        title = e.get('title', '').strip()
        venue = venue_of(e)
        lang = e.get('language', '')
        rqs, total, label, decision = decide(e)
        reason = reason_from_rqs(rqs)

        if decision == 'Include':
            include_count += 1
        elif decision == 'Exclude':
            exclude_count += 1
        else:
            maybe_count += 1

        for rq in rqs:
            rq_counts[rq] += 1

        rel_counts[label] += 1

        results.append({
            'id': e.get('id', ''),
            'title': title,
            'year': str(year if year != -1 else ''),
            'venue': venue,
            'language': lang,
            'RQs': ','.join(rqs) if rqs else '-',
            'relevance': label,
            'score': f'{total:.1f}',
            'decision': decision,
            'reason': reason if reason else (
                'Fuera de ventana' if year != -1 and (year < WINDOW_MIN or year > WINDOW_MAX) else ''
            ),
        })

    # Orden para Top 20
    def sort_key(r: Dict[str, str]):
        try:
            y = int(r['year']) if r['year'] else -1
        except Exception:
            y = -1
        return (-float(r['score']), -y, r['id'])

    sorted_results = sorted(results, key=sort_key)

    # Escribir resultados completos
    out_all = os.path.join(args.outdir, 'screening_results.tsv')
    header = [
        'id',
        'title',
        'year',
        'venue',
        'language',
        'RQs',
        'relevance',
        'score',
        'decision',
        'reason',
    ]
    write_tsv(out_all, results, header)

    # Resumen
    out_sum = os.path.join(args.outdir, 'screening_summary.txt')
    with open(out_sum, 'w', encoding='utf-8') as f:
        f.write(f"Total entradas procesadas: {len(entries)}\n")
        f.write(f"Número de Include: {include_count}\n")
        f.write(f"Número de Exclude: {exclude_count}\n")
        f.write(f"Número de Maybe: {maybe_count}\n")
        f.write("Distribución por relevancia:\n")
        for k in ['Excluir', 'Baja', 'Media', 'Alta']:
            f.write(f"  {k}: {rel_counts.get(k, 0)}\n")
        f.write("Conteo de menciones por RQ:\n")
        for rq in ['RQ1', 'RQ2', 'RQ3', 'RQ4']:
            f.write(f"  {rq}: {rq_counts.get(rq, 0)}\n")

    # Top 20
    out_top = os.path.join(args.outdir, 'screening_top20.tsv')
    top_rows: List[Dict[str, str]] = []
    for r in sorted_results[:20]:
        top_rows.append(
            {
                'id': r['id'],
                'title': r['title'],
                'year': r['year'],
                'RQs': r['RQs'],
                'score': r['score'],
                'reason': r['reason'],
            }
        )
    write_tsv(out_top, top_rows, ['id', 'title', 'year', 'RQs', 'score', 'reason'])

    # Ambiguos
    out_amb = os.path.join(args.outdir, 'screening_ambiguous.tsv')
    amb_rows = [r for r in results if r['decision'] == 'Maybe']
    write_tsv(out_amb, amb_rows, header)

    print(f"Parsed entries: {len(entries)}")
    print(f"Total {len(entries)} Include {include_count} Exclude {exclude_count} Maybe {maybe_count}")
    print(rel_counts)
    print(rq_counts)
    print("Top 5 sample:")
    for r in sorted_results[:5]:
        print(r['id'], r['score'], r['reason'])

if __name__ == '__main__':
    main()
