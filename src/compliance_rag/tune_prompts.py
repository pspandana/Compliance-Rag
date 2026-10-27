"""Compare generator prompt variants on answer quality (calls the OpenAI API).

  uv run tune-prompts v0 v0 v1 v2

Each name is run once on all 70 answerable questions (50 EU + 20 jurisdiction), scored by the
LLM judge. Naming a variant twice reruns it, which measures the run-to-run noise: the AI is not
repeatable, so a difference smaller than that noise means nothing. Results are shown for a tune
half and a holdout half (evaluate.tuning_split) and as question-level flips against the first run.

Decision rule, fixed before running: adopt a variant only if its judged-correct rate beats the
first run on BOTH halves by more than the replicate-to-replicate difference, and its share of
supported claims does not fall by more than 3 points.
"""

import argparse
import contextlib
import io
import json

from compliance_rag.answer import (
    GENERATOR_VARIANTS,
    RESULTS_DIR,
    AnswerPipeline,
    run_golden,
)
from compliance_rag.evaluate import GOLDEN_PATH, JURISDICTION_PATH, tuning_split
from compliance_rag.judge import Judge


def run_variant(pipeline: AnswerPipeline, judge: Judge, variant: str, label: str, resume: bool) -> list[dict]:
    pipeline.set_prompt(variant)
    rows = []
    with contextlib.redirect_stdout(io.StringIO()):  # skip the per-question lines
        for path, part in ((GOLDEN_PATH, "golden"), (JURISDICTION_PATH, "jurisdiction")):
            saved = RESULTS_DIR / f"tune_{label}_{part}.json"
            if not (resume and saved.exists()):  # --resume reuses a part that already finished
                run_golden(pipeline, judge, 1000, path, saved.name)
            rows += json.loads(saved.read_text(encoding="utf-8"))
    return rows


def rate(rows: list[dict], test) -> float:
    return sum(test(r) for r in rows) / len(rows)


def supported_share(rows: list[dict]) -> float:
    claims = sum(r["supported"] + r["unsupported"] for r in rows)
    return sum(r["supported"] for r in rows) / max(claims, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variants", nargs="+", choices=list(GENERATOR_VARIANTS))
    parser.add_argument("--resume", action="store_true", help="reuse runs already saved in data/eval/")
    args = parser.parse_args()

    tune, holdout = tuning_split()
    halves = {"tune": {q["id"] for q in tune}, "holdout": {q["id"] for q in holdout}}

    pipeline, judge = AnswerPipeline(), Judge()
    runs: dict[str, list[dict]] = {}
    for variant in args.variants:
        label = variant if variant not in runs else f"{variant}#{sum(k.split('#')[0] == variant for k in runs) + 1}"
        runs[label] = run_variant(pipeline, judge, variant, label.replace("#", "_"), args.resume)
        print(f"finished {label}", flush=True)

    first = next(iter(runs))
    base = {r["id"]: r["correctness"] for r in runs[first]}
    print(f"\n{'run':<7}{'correct: tune':>14}{'holdout':>9}{'all':>7}{'partial':>9}{'incorrect':>10}"
          f"{'supported':>10}{'cites':>7}{'fixed':>7}{'broke':>7}")
    for label, rows in runs.items():
        def correct(r):
            return r["correctness"] == "correct"

        by_half = {name: [r for r in rows if r["id"] in ids] for name, ids in halves.items()}
        fixed = sum(base[r["id"]] != "correct" and correct(r) for r in rows)
        broke = sum(base[r["id"]] == "correct" and not correct(r) for r in rows)
        print(f"{label:<7}{rate(by_half['tune'], correct):>14.1%}{rate(by_half['holdout'], correct):>9.1%}"
              f"{rate(rows, correct):>7.1%}{rate(rows, lambda r: r['correctness'] == 'partial'):>9.1%}"
              f"{rate(rows, lambda r: r['correctness'] == 'incorrect'):>10.1%}{supported_share(rows):>10.1%}"
              f"{rate(rows, lambda r: r['cites_relevant']):>7.1%}{fixed:>7}{broke:>7}")
    print(f"\n'fixed' / 'broke' = questions that became correct / stopped being correct versus {first}.")
    print("Per-question results: data/eval/tune_<run>_<set>.json")


if __name__ == "__main__":
    main()
