"""Bayesian answer confidence from fact-check results.

Each claim the fact-checker labels supported / unsupported is treated as a coin
flip with unknown probability p of being supported. Starting from a uniform
Beta(1, 1) prior ("no idea yet"), s supported and u unsupported claims give the
posterior Beta(1 + s, 1 + u). We report its mean and a 90% credible interval: few
checks give a wide interval, many agreeing checks a narrow one.
"""

from dataclasses import dataclass

from scipy.stats import beta

PRIOR = (1.0, 1.0)


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


def beta_confidence(supported: int, unsupported: int, level: float = 0.9) -> Confidence:
    dist = beta(PRIOR[0] + supported, PRIOR[1] + unsupported)
    low, high = dist.interval(level)
    return Confidence(supported, unsupported, float(dist.mean()), float(low), float(high))
