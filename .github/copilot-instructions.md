<!-- Project-specific Copilot instructions for AI coding agents -->
# Guidance for AI coding agents working on this repository

Keep suggestions focused, minimal, and validated by the project's small scripts.

- Big picture (what this repo does)
  - This repo implements an SLR (systematic literature review) pipeline for RAG + XAI trust topics.
    Core stages are: Deduplication (Deduplicación/) -> Screening (Clasificación de Kitchenham - Screening/) -> manual follow-up.
  - Key artifacts: cleaned master BibTeX `SLR_RAG_master_clean.bib`, screening TSV outputs (`screening_results.tsv`, `screening_top20.tsv`, `screening_ambiguous.tsv`).

- Where to look first (important files)
  - `Deduplicación/dedup.py` and `dedup.sh` — deduplication rules, DOI/title heuristics, output paths (`SLR_RAG_master_clean.bib`).
  - `Clasificación de Kitchenham - Screening/rag_screening.py` and `run.sh` — screening parser, RQ keyword lists (SYN_RAG, KW_RQ1..KW_RQ4), WINDOW_MIN/WINDOW_MAX, decision logic (`decide()`), and TSV output format.
  - `Búsqueda/` and top-level `.bib` files — input corpora. Outputs live next to scripts by default.

- Project-specific conventions and important invariants
  - BibTeX parsing intentionally uses a minimal in-repo parser (no external bibtexparser dependency). Preserve its robustness to braces and commas when editing.
  - Screening decisions are deterministic heuristics: do not change scoring thresholds globally without adding unit tests and updating `screening_summary.txt` expectations.
  - Keyword lists are the primary tunable surface: `SYN_RAG`, `SYN_LLM`, `KW_RQ1..KW_RQ4` in `rag_screening.py`. Prefer expanding synonyms to make rules less brittle.
  - Time window is controlled by `WINDOW_MIN`/`WINDOW_MAX`. Changing these affects many entries; document rationale in commit message.
  - Dedup prioritization: Journal > Conference > Other (see `item_priority()` in `dedup.py`). Title similarity threshold is ~0.95 — keep or change with caution and add tests.

- Typical developer workflow (how to run & validate)
  - Run deduplication: from `Deduplicación/` run `dedup.sh`; it writes `SLR_RAG_master_clean.bib` and a duplicates report (xlsx/csv).
  - Run screening: from `Clasificación de Kitchenham - Screening/` run `chmod +x run.sh; ./run.sh SLR_RAG_master_clean.bib .` or `python rag_screening.py --bib SLR_RAG_master_clean.bib --outdir .`.
  - Quick validation: check `screening_results.tsv` headers and `screening_summary.txt` counts.

- Editing guidance for AI agents (how to propose changes)
  - Small, self-contained PRs only. Each change must include: code change, a short test or quick run validating outputs, and an updated summary (if counts change).
  - When modifying heuristics, add a small unit or integration test (a handful of synthetic .bib entries) in the same folder that demonstrates the new behavior.
  - Preserve TSV headers exactly: `id,title,year,venue,language,RQs,relevance,score,decision,reason`.

- Integration/Dependencies notes
  - `dedup.py` optionally uses pandas (if available) to write Excel reports; code falls back to CSV if pandas is missing. Do not add heavy dependencies without justification.
  - Scripts are written for Python 3.x; prefer built-in stdlib facilities already used in the repo.

- Quick examples (do this when you change keywords)
  - To add a synonym to RAG detection: edit `SYN_RAG` in `rag_screening.py`, run `python rag_screening.py --bib SLR_RAG_master_clean.bib --outdir .`, and confirm `screening_summary.txt` updated counts.
  - To relax title duplicate threshold: edit the `similar(..., thr=0.95)` call in `dedup.py`, run `dedup.sh`, and inspect `SLR_RAG_duplicates_report.csv` for changed removals.

- When to ask the maintainers
  - If a change would alter the final inclusion counts materially (e.g., >5% of included entries), or adds external network or large dependencies, request review in the PR description.

If anything in these notes is unclear or you'd like me to include examples/tests in the repo, tell me which sections to expand and I'll iterate.
