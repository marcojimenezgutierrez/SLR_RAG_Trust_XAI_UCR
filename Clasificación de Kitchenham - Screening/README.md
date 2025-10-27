# SLR RAG — Screening inicial (Kitchenham)

Este kit contiene un script **Python** y un **wrapper Bash** para realizar el *screening* inicial (título + abstract + keywords) sobre un archivo **BibTeX** y clasificar automáticamente cada entrada según las **RQs**:

- **RQ1 — Confianza** (trust, confidence, credibility, user perception)
- **RQ2 — Calidad** (hallucinations, factuality, faithfulness, accuracy, grounding)
- **RQ3 — XAI & calibración** (attribution/citations, evidence highlighting, explanations, calibration, Brier, ECE)
- **RQ4 — Métodos & métricas** (evaluation metrics, methodology, user study, survey, benchmark, instrument)

## Contenido
- `rag_screening.py` — script principal (sin dependencias externas).
- `run.sh` — ejecutor de conveniencia (Bash).
- *(Tu archivo)* `SLR_RAG_master_clean.bib` — el BibTeX a analizar (colócalo en el mismo directorio o pasa la ruta).

## Salidas
- `screening_results.tsv` — tabla completa con columnas:
  `id, title, year, venue, language, RQs, relevance, score, decision, reason`
- `screening_summary.txt` — resumen con conteos (Include/Exclude/Maybe, distribución por relevancia y menciones por RQ).
- `screening_top20.tsv` — 20 entradas con mayor puntuación.
- `screening_ambiguous.tsv` — casos marcados como `Maybe` para revisión manual.

## Uso rápido

```bash
# Dar permisos (si hace falta) y ejecutar
chmod +x run.sh
./run.sh SLR_RAG_master_clean.bib .

# o directamente en Python
python rag_screening.py --bib SLR_RAG_master_clean.bib --outdir .
```

> Si tu .bib está en otra carpeta:
> `./run.sh /ruta/a/tu/archivo.bib ./salida`

## Cómo funciona (resumen técnico)

1. **Parseo BibTeX** robusto (sin bibtexparser): separa cada entrada `@type{key,...}` y extrae campos (`title`, `abstract`, `keywords`, `year`, `journal/booktitle`).
2. **Ventana temporal**: 2020–2025 (ajustable en `WINDOW_MIN`, `WINDOW_MAX`).
3. **Detección de RAG**: usa sinónimos (retrieval-augmented, graphRAG, grounded, context-augmented, hybrid LLM, provenance, source attribution, etc.).
4. **Scoring por RQ**: suma 0/1 por cada RQ si aparecen términos asociados.
5. **Etiqueta de relevancia**: 
   - 3.0–4.0 → *Alta*, 1.5–2.5 → *Media*, 0.5–1.0 → *Baja*, 0 → *Excluir*.
6. **Decisión** (*Include/Exclude/Maybe*): 
   - `Exclude` si fuera de ventana o sin señales de RAG y sin RQs.
   - `Maybe` si hay RAG pero no hay evidencia clara de RQs (ambigüedad típica por falta de abstract).
   - `Include` si tiene al menos un RQ detectado.
7. **Reason**: texto breve concatenando dimensiones detectadas (e.g., “Confianza/percepción y Factualidad/alucinaciones”).

## Ajustes recomendados

- **Dominios**: edita las listas `SYN_RAG`, `KW_RQ1..KW_RQ4` para afinar el cribado.
- **Criterios**: modifica `decide()` si quieres ser más estricto con el baseline sin RAG.
- **Salida**: el TSV está pensado para **Excel/Sheets** y para etiquetas/colecciones en **Zotero**.

## Flujo con Zotero

1. Importa el `.bib` a una colección.
2. Elimina duplicados y usa etiquetas de color `Included_TitleAbs`, `Excluded_TitleAbs`, `Maybe_TitleAbs`.
3. Exporta a CSV y sincroniza con `screening_results.tsv` para comparar decisiones.
4. Continúa con **Full‑Text Review** y **Quality Assessment (QA)** (Kitchenham).

## Licencia
MIT. Libre uso, modificación y distribución con atribución.
