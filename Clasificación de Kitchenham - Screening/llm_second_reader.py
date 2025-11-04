"""Herramientas para usar un LLM como segundo lector del screening.

Este módulo ofrece dos piezas principales:

* Un generador de *prompts* estructurados para solicitar a un modelo de
  lenguaje que evalúe título y resumen con respecto a las preguntas de
  investigación (RQs) de la revisión.
* Un pipeline de comparación que combina las salidas heurísticas del
  script :mod:`rag_screening` con las predicciones del LLM para entrenar
  un modelo supervisado ligero que cuantifica la alineación entre ambos.

La intención es facilitar un flujo *human-in-the-loop*: el heurístico de
Kitchenham actúa como primer lector, el LLM produce una segunda opinión y
el modelo entrenado permite analizar concordancias y discrepancias.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
import csv

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ResearchQuestion:
    """Representa una pregunta de investigación para el prompt del LLM."""

    code: str
    question: str
    keywords: Sequence[str]

    def format_for_prompt(self) -> str:
        """Devuelve una representación legible para incluir en el prompt."""

        keywords = ", ".join(self.keywords) if self.keywords else "(sin palabras clave)"
        return (
            f"- {self.code}: {self.question}\n"
            f"  Palabras clave indicativas: {keywords}"
        )


PROMPT_TEMPLATE = """Eres un revisor experto que apoya una revisión sistemática sobre\nRetrieval-Augmented Generation (RAG) frente a modelos de lenguaje grandes\nsin recuperación. Usa el título y el resumen provistos para decidir si cada\npregunta de investigación (RQ) está cubierta por el trabajo.\n\nContexto bibliográfico\n----------------------\nTítulo: "{title}"\nResumen: "{abstract}"\n\nPreguntas de investigación\n--------------------------\n{rq_section}\n\nInstrucciones\n-------------\n1. Considera únicamente la evidencia explícita del título y del resumen.\n2. Usa las palabras clave como pistas de búsqueda, no como reglas duras.\n3. Responde con 0 (no cubre) o 1 (sí cubre) para cada RQ.\n4. Añade al final una explicación breve (<60 palabras) justificando los 1.\n\nFormato de salida\n-----------------\nDevuelve una sola línea con el siguiente formato exacto (sin texto extra):\n{response_schema}\nExplicación: <texto libre en español>\n\nEjemplo de salida\n-----------------\nRQ1=1; RQ2=0; RQ3=0; RQ4=1\nExplicación: Se discute confianza y evaluación cuantitativa, pero no XAI ni factualidad.\n"""
"""Plantilla de prompt con marcadores ``{...}`` para rellenar desde Python."""


def build_prompt(title: str, abstract: str, research_questions: Sequence[ResearchQuestion]) -> str:
    """Construye el prompt para enviar a un LLM como segundo lector.

    Args:
        title: Título del artículo o registro.
        abstract: Resumen asociado.
        research_questions: Colección ordenada de preguntas de
            investigación con sus palabras clave.

    Returns:
        Cadena lista para ser enviada al LLM.
    """

    rq_section = "\n".join(rq.format_for_prompt() for rq in research_questions)
    schema_parts = [f"{rq.code}=<0|1>" for rq in research_questions]
    response_schema = "; ".join(schema_parts)
    return PROMPT_TEMPLATE.format(
        title=title,
        abstract=abstract,
        rq_section=rq_section,
        response_schema=response_schema,
    )


# ---------------------------------------------------------------------------
# Modelo de comparación heurístico vs LLM
# ---------------------------------------------------------------------------


def _parse_rq_columns(rq_field: str, expected: Sequence[str]) -> Dict[str, int]:
    """Convierte la columna ``RQs`` del heurístico en indicadores binarios."""

    tokens = {tok.strip() for tok in rq_field.split(',') if tok.strip()} if rq_field else set()
    return {rq: int(rq in tokens) for rq in expected}


def load_heuristic_results(path: Path, rq_labels: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Carga el TSV producido por ``rag_screening.py``.

    Se extraen el identificador, la decisión y el puntaje numérico junto
    con indicadores binarios por RQ para alimentar el modelo.
    """

    data: Dict[str, Dict[str, object]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entry_id = row.get("id", "").strip()
            if not entry_id:
                continue
            score = row.get("score", "0") or "0"
            try:
                numeric_score = float(score)
            except ValueError:
                numeric_score = 0.0
            rq_flags = _parse_rq_columns(row.get("RQs", ""), rq_labels)
            record: Dict[str, object] = {
                "score": numeric_score,
                "decision": row.get("decision", "").strip() or "Exclude",
                "relevance": row.get("relevance", "").strip(),
            }
            record.update({f"heur_{rq.lower()}": value for rq, value in rq_flags.items()})
            data[entry_id] = record
    return data


def load_llm_predictions(path: Path, rq_labels: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """Carga un TSV/CSV con las predicciones binarias del LLM por RQ.

    Se espera un encabezado ``id`` y columnas ``llm_rqX`` (0/1) además de
    una columna opcional ``llm_score`` (suma de activaciones).
    """

    data: Dict[str, Dict[str, object]] = {}
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            entry_id = row.get("id", "").strip()
            if not entry_id:
                continue
            record: Dict[str, object] = {}
            for rq in rq_labels:
                key = f"llm_{rq.lower()}"
                raw = row.get(key, "0")
                try:
                    record[key] = int(raw)
                except (TypeError, ValueError):
                    record[key] = 0
            score_raw = row.get("llm_score")
            if score_raw is not None:
                try:
                    record["llm_score"] = float(score_raw)
                except ValueError:
                    record["llm_score"] = sum(record[f"llm_{rq.lower()}"] for rq in rq_labels)
            else:
                record["llm_score"] = sum(record[f"llm_{rq.lower()}"] for rq in rq_labels)
            decision = row.get("llm_decision")
            if decision:
                record["llm_decision"] = decision.strip()
            data[entry_id] = record
    return data


def build_alignment_matrix(
    heuristic: Mapping[str, Mapping[str, object]],
    llm: Mapping[str, Mapping[str, object]],
    rq_labels: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Une los registros comunes y produce matrices de entrenamiento.

    Returns:
        X: Matriz de características.
        y: Etiquetas numéricas (0: Exclude, 1: Maybe, 2: Include).
        feature_names: Lista legible de las columnas en ``X``.
        ids: Identificadores de los registros en el mismo orden que ``X``.
    """

    label_map = {"Exclude": 0, "Maybe": 1, "Include": 2}
    feature_names: List[str] = ["heur_score"]
    feature_names += [f"heur_{rq.lower()}" for rq in rq_labels]
    feature_names += [f"llm_{rq.lower()}" for rq in rq_labels]
    feature_names.append("llm_score")

    rows: List[List[float]] = []
    labels: List[int] = []
    ids: List[str] = []

    for entry_id in sorted(set(heuristic) & set(llm)):
        h = heuristic[entry_id]
        l = llm[entry_id]
        try:
            label = label_map[h["decision"]]
        except KeyError:
            # Se descartan filas con decisiones fuera del esquema estándar.
            continue
        row: List[float] = [float(h.get("score", 0.0))]
        for rq in rq_labels:
            row.append(float(h.get(f"heur_{rq.lower()}", 0)))
        for rq in rq_labels:
            row.append(float(l.get(f"llm_{rq.lower()}", 0)))
        row.append(float(l.get("llm_score", 0.0)))
        rows.append(row)
        labels.append(label)
        ids.append(entry_id)

    if not rows:
        raise ValueError("No hay intersección entre heurístico y LLM para comparar.")

    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), feature_names, ids


def train_alignment_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Entrena una regresión logística multinomial como modelo de comparación."""

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    max_iter=500,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def evaluate_alignment(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    target_names: Sequence[str] = ("Exclude", "Maybe", "Include"),
) -> str:
    """Genera un reporte legible de la comparación modelo vs heurístico."""

    preds = model.predict(X)
    return classification_report(y, preds, target_names=target_names, digits=3)


def cli(
    heuristic_path: Path,
    llm_path: Path,
    rq_labels: Sequence[str] = ("RQ1", "RQ2", "RQ3", "RQ4"),
) -> None:
    """Ejecuta el flujo completo desde la línea de comandos."""

    heuristic = load_heuristic_results(heuristic_path, rq_labels)
    llm = load_llm_predictions(llm_path, rq_labels)
    X, y, feature_names, ids = build_alignment_matrix(heuristic, llm, rq_labels)
    model = train_alignment_model(X, y)
    report = evaluate_alignment(model, X, y)

    print("Entradas utilizadas:", len(ids))
    print("Características:")
    for name in feature_names:
        print(f"  - {name}")
    print("\nReporte de comparación (predicción vs heurístico):")
    print(report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comparación heurístico vs LLM segundo lector")
    parser.add_argument("--heuristic", type=Path, required=True, help="TSV generado por rag_screening.py")
    parser.add_argument("--llm", type=Path, required=True, help="TSV/CSV con columnas llm_rq1..llm_rq4")
    args = parser.parse_args()
    cli(args.heuristic, args.llm)
