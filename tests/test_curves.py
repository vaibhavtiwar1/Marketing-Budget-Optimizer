import numpy as np
import pytest

from budget_engine.curves import CURVES, fit_curve, guess_curve, sample_model, select_curve, shape_is_learnable


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Google Paid Search", "log"),
        ("Bing Ads", "log"),
        ("Meta Paid Social", "hill"),
        ("Google Display", "hill"),
        ("Programmatic", "hill"),  # used to match "pr" -> threshold
        ("YouTube (Google)", "hill"),
        ("LinkedIn Sponsored Content", "hill"),
        ("HubSpot Email Blast", "negexp"),
        ("SEO Blog Content", "power"),
        ("PR Agency", "threshold"),
        ("Trade Show Events", "threshold"),
        ("Something Else", "log"),
    ],
)
def test_guess_curve(name, expected):
    assert guess_curve(name) == expected


@pytest.mark.parametrize("key", list(CURVES))
def test_curves_start_at_zero_and_increase(key):
    spec = CURVES[key]
    u = np.linspace(0, 5, 200)
    v = spec.fn(u, 1.0, *spec.shape_default)
    assert v[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(v) >= -1e-12)


def test_recovers_hill_shape_from_history():
    rng = np.random.default_rng(1)
    spend = np.linspace(2000, 12000, 10)
    true = 400 * (spend / 6000) ** 2.5 / (1 + (spend / 6000) ** 2.5)
    leads = true * rng.normal(1, 0.02, len(spend))
    m = fit_curve("hill", spend, leads)
    assert m.method == "fitted"
    assert m.r2 > 0.98
    assert m.theta[1] * m.x_scale == pytest.approx(6000, rel=0.15)  # half-saturation in dollars
    assert m.confidence == "High"


def test_shape_depends_on_scale_not_hardcoded():
    """The same shape at 100x the spend should give a 100x half-saturation point."""
    spend = np.linspace(2000, 12000, 10)
    leads = 400 * (spend / 6000) ** 2 / (1 + (spend / 6000) ** 2)
    small = fit_curve("hill", spend, leads)
    big = fit_curve("hill", spend * 100, leads)
    assert big.theta[1] * big.x_scale == pytest.approx(100 * small.theta[1] * small.x_scale, rel=0.02)


def test_single_point_is_anchored_and_low_confidence():
    m = fit_curve("log", [5000], [250])
    assert m.method == "anchored"
    assert m.confidence == "Low"
    assert m.response(5000) == pytest.approx(250, rel=1e-6)
    assert "distinct spend" in m.note


def test_flat_spend_cannot_learn_shape():
    assert not shape_is_learnable([5000, 5010, 4995, 5005, 5000])
    assert shape_is_learnable([2000, 4000, 6000, 8000])


def test_fixed_cost_is_subtracted():
    m = fit_curve("log", [3000], [100], fixed_cost=1000)
    assert m.spend == (2000.0,)


def test_select_curve_picks_true_family():
    spend = np.linspace(1000, 9000, 10)
    leads = 150 / (1 + np.exp(-(spend - 4500) / 500))
    model, scores = select_curve(spend, leads, preferred="log")
    assert model.curve in ("threshold", "hill")
    assert scores["log"] is not None


def test_select_curve_keeps_preferred_when_close():
    spend = np.linspace(1000, 9000, 10)
    leads = 80 * np.log1p(spend / 1500)
    model, _ = select_curve(spend, leads, preferred="log")
    assert model.curve == "log"


def test_sample_model_varies_anchored_shape():
    m = fit_curve("hill", [5000], [200])
    rng = np.random.default_rng(0)
    shapes = {sample_model(m, rng).theta[1] for _ in range(10)}
    assert len(shapes) > 1
