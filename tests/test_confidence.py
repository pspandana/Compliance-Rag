from pytest import approx

from compliance_rag.confidence import (
    beta_confidence,
    fit_platt,
    retrieval_probability,
    route,
)


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


def test_grader_and_retrieval_evidence_move_the_posterior():
    base = beta_confidence(3, 0)
    assert beta_confidence(3, 0, grader_no=3).mean < base.mean
    assert beta_confidence(3, 0, retrieval_p=0.9).mean > beta_confidence(3, 0, retrieval_p=0.1).mean


def test_route():
    strong, weak = beta_confidence(8, 0, grader_yes=3, retrieval_p=0.9), beta_confidence(1, 2)
    assert route(strong, n_claims=8, n_relevant_chunks=3) == "auto"
    assert route(weak, n_claims=3, n_relevant_chunks=3) == "review"
    assert route(strong, n_claims=0, n_relevant_chunks=3) == "abstain"
    assert route(strong, n_claims=8, n_relevant_chunks=0) == "abstain"


def test_platt_scaling_is_increasing_and_separates_the_classes():
    a, b = fit_platt([-11, -10, -9, -5, -4, -3], [0, 0, 0, 1, 1, 1])
    assert a > 0
    assert retrieval_probability(-10, a, b) < 0.5 < retrieval_probability(-4, a, b)
