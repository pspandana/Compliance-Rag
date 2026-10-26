"""Regression gate: fail if retrieval or answer quality drops below the saved baseline.

  uv run gate               retrieval metrics only (offline, no API key; used in CI)
  uv run gate --answers     also run the full answer evaluation (needs OPENAI_API_KEY)
  uv run gate --update      overwrite eval/baseline.json with the current numbers

Run it after any change to chunking, retrieval, prompts or models.
"""

import argparse
import contextlib
import io
import json
import sys

from compliance_rag.evaluate import load_golden, score, summarize
from compliance_rag.ingest import PROJECT_ROOT
from compliance_rag.retrieval import build_retrievers

BASELINE_PATH = PROJECT_ROOT / "eval" / "baseline.json"

# How far a metric may fall below its baseline before the gate fails.
# Retrieval is deterministic, so its tolerance is tight; LLM answers vary run to run.
TOLERANCE = {"retrieval": 0.02, "answers": 0.06}
# Metrics where a lower number is better.
LOWER_IS_BETTER = {"wrongly_auto"}
RETRIEVAL_KEYS = ("recall@1", "recall@3", "mrr")
ANSWER_KEYS = ("cites_relevant", "supported_rate", "correct", "abstained", "wrongly_auto")


def compare(baseline: dict, current: dict, tolerance: float) -> list[str]:
    """Return a message for every metric that got worse than the baseline allows."""
    failures = []
    for key, base in baseline.items():
        now = current[key]
        worse = now > base if key in LOWER_IS_BETTER else now < base - tolerance
        if worse:
            failures.append(f"{key}: {now:.3f} (baseline {base:.3f})")
    return failures


def retrieval_metrics() -> dict:
    retriever = build_retrievers(["rerank"], kind="article")["rerank"]
    summary = summarize(score(retriever, load_golden(), 50))
    return {k: summary[k] for k in RETRIEVAL_KEYS}


def answer_metrics() -> dict:
    from compliance_rag.answer import AnswerPipeline, run_golden, run_unanswerable
    from compliance_rag.judge import Judge

    pipeline, judge = AnswerPipeline(), Judge()
    with contextlib.redirect_stdout(io.StringIO()):  # skip the per-question lines
        golden = run_golden(pipeline, judge, 50)
        unanswerable = run_unanswerable(pipeline, judge)
    merged = {**golden, **unanswerable}
    return {k: merged[k] for k in ANSWER_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", action="store_true", help="also evaluate answers (calls the OpenAI API)")
    parser.add_argument("--update", action="store_true", help="save the current numbers as the new baseline")
    args = parser.parse_args()

    current = {"retrieval": retrieval_metrics()}
    if args.answers or args.update:
        current["answers"] = answer_metrics()

    if args.update:
        BASELINE_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
        print(f"Baseline updated: {BASELINE_PATH}\n{json.dumps(current, indent=2)}")
        return

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    failures = []
    for section, metrics in current.items():
        print(f"\n{section}")
        for key, value in metrics.items():
            print(f"  {key:<16}{value:>8.3f}   baseline {baseline[section][key]:>6.3f}")
        failures += [f"{section} {msg}" for msg in compare(baseline[section], metrics, TOLERANCE[section])]

    if failures:
        print("\nGATE FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nGate passed.")


if __name__ == "__main__":
    main()
