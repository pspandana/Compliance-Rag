from compliance_rag.gate import compare


def test_small_drop_within_tolerance_passes():
    assert compare({"mrr": 0.88}, {"mrr": 0.87}, tolerance=0.02) == []


def test_drop_beyond_tolerance_fails():
    failures = compare({"mrr": 0.88}, {"mrr": 0.80}, tolerance=0.02)
    assert len(failures) == 1 and failures[0].startswith("mrr")


def test_improvement_passes():
    assert compare({"mrr": 0.88}, {"mrr": 0.95}, tolerance=0.02) == []


def test_lower_is_better_metrics_allow_no_increase():
    assert compare({"wrongly_auto": 0}, {"wrongly_auto": 0}, tolerance=0.06) == []
    assert compare({"wrongly_auto": 0}, {"wrongly_auto": 1}, tolerance=0.06) != []
