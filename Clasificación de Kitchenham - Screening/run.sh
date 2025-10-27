#!/usr/bin/env bash
set -euo pipefail

# Ruta al .bib (ajústala si lo tienes en otra carpeta)
BIB="${1:-SLR_RAG_master_clean.bib}"
OUTDIR="${2:-.}"

echo "[INFO] Ejecutando screening inicial Kitchenham sobre: $BIB"
python rag_screening.py --bib "$BIB" --outdir "$OUTDIR"

echo
echo "== Head screening_results.tsv =="
head -n 5 "$OUTDIR/screening_results.tsv" || true

echo
echo "== Resumen =="
cat "$OUTDIR/screening_summary.txt" || true

echo
echo "== Top 20 =="
head -n 25 "$OUTDIR/screening_top20.tsv" || true

echo
echo "== Ambiguos (Maybe) =="
head -n 10 "$OUTDIR/screening_ambiguous.tsv" || true
