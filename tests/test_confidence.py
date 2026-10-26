from pytest import approx

from compliance_rag.confidence import beta_confidence


def test_posterior_mean_counts_the_uniform_prior():
    # Beta(1 + 3, 1 + 0): mean 4 / 5
    assert beta_confidence(3, 0).mean == approx(0.8)


def test_more_evidence_narrows_the_interval():
    few, many = beta_confidence(1, 0), beta_confidence(8, 0)
    assert many.mean > few.mean
    assert (many.high - many.low) < (few.high - few.low)


def test_labels():
    assert beta_confidence(8, 0).label == "high"
    assert beta_confidence(2, 0).label == "medium"
    assert beta_confidence(1, 3).label == "low"
    assert beta_confidence(0, 0).label == "none"
