"""Tests for the scoring pipeline used by the Streamlit app.

Checks the pieces line up and the cost-based threshold lands where the report
says (t* = 0.09 at the baseline cost ratio).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import scoring


@pytest.fixture(scope="module")
def bundle():
    return scoring.load_bundle()


def test_bundle_shapes(bundle):
    assert len(bundle["X"]) == len(bundle["y"])
    assert len(bundle["features"]) == 204
    assert "TARGET" not in bundle["features"]


def test_pd_in_range(bundle):
    pd_all = scoring.predicted_pd(bundle["calibrated"], bundle["X"])
    assert pd_all.min() >= 0.0
    assert pd_all.max() <= 1.0
    assert 0.05 < pd_all.mean() < 0.12  # base rate is ~8%


def test_threshold_matches_report(bundle):
    pd_all = scoring.predicted_pd(bundle["calibrated"], bundle["X"])
    c_fn, c_fp = scoring.build_costs(bundle["X"], cost_ratio=1.0)
    t_star, approval = scoring.choose_threshold(bundle["y"], pd_all, c_fn, c_fp)
    assert t_star == pytest.approx(0.09, abs=1e-9)
    assert 0.68 < approval < 0.72


def test_higher_cost_declines_more(bundle):
    pd_all = scoring.predicted_pd(bundle["calibrated"], bundle["X"])
    base_t, base_appr = scoring.choose_threshold(
        bundle["y"], pd_all, *scoring.build_costs(bundle["X"], cost_ratio=1.0))
    high_t, high_appr = scoring.choose_threshold(
        bundle["y"], pd_all, *scoring.build_costs(bundle["X"], cost_ratio=3.0))
    assert high_t < base_t
    assert high_appr < base_appr


def test_decision_rule():
    assert scoring.decide(0.05, 0.09) == "APPROVE"
    assert scoring.decide(0.09, 0.09) == "DECLINE"
    assert scoring.decide(0.50, 0.09) == "DECLINE"


def test_reason_codes_shape(bundle):
    x_row = bundle["X"].iloc[[0]]
    reasons = scoring.explain_applicant(
        bundle["raw"], x_row, bundle["features"], top_n=3)
    assert len(reasons) <= 3
    for label, contrib in reasons:
        assert isinstance(label, str)
        assert contrib > 0  # reason codes only list risk-increasing factors


def test_reason_code_labels_are_plain(bundle):
    x_row = bundle["X"].iloc[[0]]
    reasons = scoring.explain_applicant(
        bundle["raw"], x_row, bundle["features"], top_n=3)
    labels = [label for label, _ in reasons]
    assert all("_" not in lbl and lbl != lbl.upper() for lbl in labels)
    assert len(labels) == len(set(labels))


GOOD_INPUTS = {
    "income": 150000, "loan": 500000, "annuity": 25000, "goods_price": 450000,
    "age": 40, "years_employed": 5, "ext_score": 0.5,
    "missed_payment_rate": 0.0, "card_usage": 0.25,
}


@pytest.fixture(scope="module")
def medians():
    return scoring.training_medians()


def test_build_new_row_shape_and_order(bundle, medians):
    row = scoring.build_new_row(GOOD_INPUTS, medians)
    assert row.shape == (1, 204)
    assert list(row.columns) == bundle["features"]


def test_build_new_row_recomputes_ratios(medians):
    row = scoring.build_new_row(GOOD_INPUTS, medians)
    expected = GOOD_INPUTS["loan"] / GOOD_INPUTS["income"]
    assert row["CREDIT_INCOME_RATIO"].iloc[0] == pytest.approx(expected)


def test_new_applicant_scores_and_reacts(bundle, medians):
    safe = scoring.predicted_pd(
        bundle["calibrated"], scoring.build_new_row(GOOD_INPUTS, medians))[0]
    risky_inputs = {**GOOD_INPUTS, "ext_score": 0.1,
                    "missed_payment_rate": 0.4, "income": 60000, "loan": 900000}
    risky = scoring.predicted_pd(
        bundle["calibrated"], scoring.build_new_row(risky_inputs, medians))[0]
    assert 0.0 <= safe <= 1.0
    assert risky > safe


def test_validation_flags_bad_inputs():
    assert scoring.validate_new_inputs(GOOD_INPUTS) == []
    bad = {**GOOD_INPUTS, "income": 0, "ext_score": 1.5, "annuity": 999999}
    problems = scoring.validate_new_inputs(bad)
    assert len(problems) >= 3
