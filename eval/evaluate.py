"""Evaluate retrieval and grounded generation against hand-written QA pairs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.generation import answer_question
from src.prompts import FALLBACK_PHRASE
from src.retrieval import retrieve


ROOT = Path(__file__).resolve().parent
QA_PATH = ROOT / "qa_pairs.json"
RESULTS_DIR = ROOT / "results"
LATEST_PATH = RESULTS_DIR / "latest.json"


def _contains_phrases(answer: str, phrases: list[str]) -> bool:
    answer_lower = answer.casefold()
    return all(phrase.casefold() in answer_lower for phrase in phrases)


def _percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def evaluate(qa_pairs: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    """Run evaluation and return metrics plus detailed per-question results."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    details = []
    for pair in qa_pairs:
        question = pair["question"]
        answerable = bool(pair["answerable"])
        expected_source = pair["expected_source"]
        expected_phrases = pair["expected_answer_contains"]
        detail: dict[str, Any] = {
            "question": question,
            "expected": {
                "answer_contains": expected_phrases,
                "source": expected_source,
                "answerable": answerable,
            },
            "actual": {},
            "retrieval_hit": None,
            "answer_correct": None,
            "fallback_correct": None,
            "passed": False,
        }

        try:
            retrieved = retrieve(question, k=k)
            actual_sources = [chunk.get("source", "") for chunk in retrieved]
            detail["actual"]["retrieved_sources"] = actual_sources
            detail["retrieval_hit"] = expected_source in actual_sources if answerable else None

            generated = answer_question(question, k=k)
            answer = str(generated.get("answer", ""))
            used_fallback = bool(generated.get("used_fallback")) or FALLBACK_PHRASE in answer
            detail["actual"].update({
                "answer": answer,
                "citations": generated.get("citations", []),
                "used_fallback": used_fallback,
            })
            if answerable:
                detail["answer_correct"] = not used_fallback and _contains_phrases(answer, expected_phrases)
                detail["passed"] = bool(detail["retrieval_hit"] and detail["answer_correct"])
            else:
                detail["fallback_correct"] = used_fallback
                detail["passed"] = bool(detail["fallback_correct"])
        except Exception as exc:
            detail["actual"]["error"] = str(exc)
            detail["passed"] = False
        details.append(detail)

    answerable_pairs = [item for item in details if item["expected"]["answerable"]]
    unanswerable_pairs = [item for item in details if not item["expected"]["answerable"]]
    retrieval_hits = sum(bool(item["retrieval_hit"]) for item in answerable_pairs)
    correct_answers = sum(bool(item["answer_correct"]) for item in answerable_pairs)
    correct_fallbacks = sum(bool(item["fallback_correct"]) for item in unanswerable_pairs)
    false_positives = sum(item["fallback_correct"] is False for item in unanswerable_pairs)
    metrics = {
        "k": k,
        "retrieval_hit_rate_at_k_percent": _percentage(retrieval_hits, len(answerable_pairs)),
        "answer_accuracy_percent": _percentage(correct_answers, len(answerable_pairs)),
        "fallback_precision_percent": _percentage(correct_fallbacks, len(unanswerable_pairs)),
        "false_positive_rate_percent": _percentage(false_positives, len(unanswerable_pairs)),
        "answerable_count": len(answerable_pairs),
        "unanswerable_count": len(unanswerable_pairs),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "results": details,
    }


def print_summary(report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    rows = [
        (f"Retrieval Hit Rate @{metrics['k']}", metrics["retrieval_hit_rate_at_k_percent"]),
        ("Answer Accuracy", metrics["answer_accuracy_percent"]),
        ("Fallback Precision", metrics["fallback_precision_percent"]),
        ("FALSE POSITIVE RATE (most important)", metrics["false_positive_rate_percent"]),
    ]
    width = max(len(label) for label, _ in rows)
    print("\nRAG Evaluation Summary")
    print("-" * (width + 14))
    print(f"{'Metric':<{width}}  Value")
    print("-" * (width + 14))
    for label, value in rows:
        print(f"{label:<{width}}  {value:>6.2f}%")
    print("-" * (width + 14))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and generation quality.")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved chunks to evaluate")
    parser.add_argument("--qa-file", type=Path, default=QA_PATH)
    args = parser.parse_args()
    qa_pairs = json.loads(args.qa_file.read_text(encoding="utf-8"))
    report = evaluate(qa_pairs, k=args.k)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"\nDetailed results: {LATEST_PATH}")


if __name__ == "__main__":
    main()
