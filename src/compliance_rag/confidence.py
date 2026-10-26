"""Bayesian answer confidence and the auto-answer / human-review / abstain decision.

Every piece of evidence is a yes/no observation about "is this answer trustworthy?":

  verifier   each claim the fact-checker labels supported / unsupported
  grader     for each of the top-3 retrieved chunks, did the grader agree it is relevant?
  retrieval  the reranker's best score, calibrated to a probability p (see calibrate.py),
             counted as RETRIEVAL_WEIGHT fractional observations (p successes, 1 - p failures)

Starting from a uniform Beta(1, 1) prior ("no idea yet"), the posterior is
Beta(1 + successes, 1 + failures). We report its mean and a 90% credible interval:
little evidence gives a wide interval, lots of agreeing evidence a narrow one.
"""

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta

from compliance_rag.ingest import PROJECT_ROOT

PRIOR = (1.0, 1.0)
RETRIEVAL_WEIGHT = 2.0
CALIBRATION_PATH = PROJECT_ROOT / "eval" / "calibration.json"

# Routing thresholds, chosen on the golden set (see docs/GUIDE.md).
AUTO_MEAN = 0.80
AUTO_LOW = 0.55


@dataclass
class Confidence:
    supported: int
    unsupported: int
    mean: float
    low: float
    high: float

    @property
    def label(self) -> str:
        if self.supported == 0:
            return "none"
        if self.mean >= 0.8:
            return "high"
        return "medium" if self.mean >= 0.6 else "low"


def beta_confidence(
    supported: int,
    unsupported: int,
    level: float = 0.9,
    *,
    grader_yes: int = 0,
    grader_no: int = 0,
    retrieval_p: float | None = None,
) -> Confidence:
    a = PRIOR[0] + supported + grader_yes
    b = PRIOR[1] + unsupported + grader_no
    if retrieval_p is not None:
        a += RETRIEVAL_WEIGHT * retrieval_p
        b += RETRIEVAL_WEIGHT * (1 - retrieval_p)
    dist = beta(a, b)
    low, high = dist.interval(level)
    return Confidence(supported, unsupported, float(dist.mean()), float(low), float(high))


def route(conf: Confidence, n_claims: int, n_relevant_chunks: int) -> str:
    """"abstain" (nothing to answer from), "auto" (answer directly) or "review" (send to a human)."""
    if n_claims == 0 or n_relevant_chunks == 0:
        return "abstain"
    if conf.mean >= AUTO_MEAN and conf.low >= AUTO_LOW:
        return "auto"
    return "review"


def fit_platt(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Fit p = sigmoid(a * score + b) by regularised logistic regression."""
    s, y = np.array(scores), np.array(labels, dtype=float)

    def loss(params):
        z = params[0] * s + params[1]
        return float(np.sum(np.logaddexp(0, z) - y * z) + 1e-3 * params[0] ** 2)

    a, b = minimize(loss, [1.0, 0.0]).x
    return float(a), float(b)


def retrieval_probability(score: float, a: float, b: float) -> float:
    return 1 / (1 + math.exp(-(a * score + b)))


def load_calibration() -> tuple[float, float]:
    if not CALIBRATION_PATH.exists():
        raise SystemExit("Missing eval/calibration.json. Run `uv run calibrate` first.")
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    return data["a"], data["b"]
