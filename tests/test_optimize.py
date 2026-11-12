import numpy as np
import pytest

from budget_engine.curves import fit_curve
from budget_engine.optimize import Channel, Scenario, channel_bounds, optimize, plan_uncertainty


def _ch(name, curve, spend, leads, current, **kw):
    return Channel(name, fit_curve(curve, spend, leads), current, **kw)


@pytest.fixture
def channels():
    s = np.linspace(2000, 10000, 8)
    return [
        _ch("Search", "log", s, 90 * np.log1p(s / 1200), 6000, value_per_lead=50),
        _ch("Social", "hill", s, 400 * (s / 6000) ** 2 / (1 + (s / 6000) ** 2), 4000, value_per_lead=200),
        _ch("SEO", "power", s, 2.5 * s**0.55, 3000, value_per_lead=100),
    ]


def test_budget_mode_spends_budget_and_beats_current(channels):
    sc = Scenario(mode="budget", budget=13000)
    plan = optimize(channels, sc)
    assert plan.feasible
    assert plan.total_spend == pytest.approx(13000, rel=1e-4)
    current = sum(ch.leads(ch.current_spend) for ch in channels)
    assert plan.total_leads >= current - 1e-6


def test_budget_mode_equalizes_marginal_returns(channels):
    plan = optimize(channels, Scenario(mode="budget", budget=13000, extrapolation_limit=5))
    marg = [ch.leads_per_extra_1k(b) for ch, b in zip(channels, plan.budgets) if b > 1]
    assert max(marg) - min(marg) < 0.05 * max(marg)


def test_target_mode_hits_target_at_lower_cost(channels):
    current_leads = sum(ch.leads(ch.current_spend) for ch in channels)
    plan = optimize(channels, Scenario(mode="target", target=current_leads))
    assert plan.feasible
    assert plan.total_leads >= current_leads * (1 - 1e-3)
    assert plan.total_spend <= sum(ch.current_spend for ch in channels) + 1


def test_max_change_is_respected(channels):
    plan = optimize(channels, Scenario(mode="budget", budget=13000, max_change_pct=20))
    for ch, b in zip(channels, plan.budgets):
        assert ch.current_spend * 0.8 - 1e-6 <= b <= ch.current_spend * 1.2 + 1e-6


def test_locked_channel_stays_put(channels):
    channels[0] = Channel(**{**channels[0].__dict__, "locked": True})
    plan = optimize(channels, Scenario(mode="budget", budget=13000))
    assert plan.budgets[0] == pytest.approx(6000)
    assert plan.limit_flags()[0] == "Locked"


def test_min_and_max_spend(channels):
    channels[1] = Channel(**{**channels[1].__dict__, "min_spend": 5000.0})
    channels[2] = Channel(**{**channels[2].__dict__, "max_spend": 1000.0})
    plan = optimize(channels, Scenario(mode="budget", budget=13000))
    assert plan.budgets[1] >= 5000 - 1e-6
    assert plan.budgets[2] <= 1000 + 1e-6


def test_extrapolation_limit_caps_spend(channels):
    lo, hi = channel_bounds(channels[0], Scenario(extrapolation_limit=1.5))
    assert hi == pytest.approx(15000)


def test_value_goal_shifts_money_to_valuable_channel(channels):
    leads_plan = optimize(channels, Scenario(mode="budget", budget=13000))
    value_plan = optimize(channels, Scenario(mode="budget", goal="value", budget=13000))
    assert value_plan.budgets[1] > leads_plan.budgets[1]  # Social leads are worth 4x Search leads
    assert value_plan.total_value >= leads_plan.total_value - 1e-6


def test_unreachable_target_explains_why(channels):
    plan = optimize(channels, Scenario(mode="target", target=1e6))
    assert not plan.feasible
    assert "isn't reachable" in plan.message
    assert "Search" in plan.message


def test_budget_below_minimums_explains_why(channels):
    channels[0] = Channel(**{**channels[0].__dict__, "min_spend": 20000.0, "max_spend": 30000.0})
    plan = optimize(channels, Scenario(mode="budget", budget=10000))
    assert not plan.feasible
    assert "Search" in plan.message


def test_conflicting_limits(channels):
    channels[0] = Channel(**{**channels[0].__dict__, "min_spend": 5000.0, "max_spend": 1000.0})
    plan = optimize(channels, Scenario(mode="budget", budget=13000))
    assert not plan.feasible
    assert "conflict" in plan.message


def test_threshold_channel_is_funded_when_it_wins():
    """A great channel with a minimum-spend threshold must not be starved (old SLSQP bug)."""
    s = np.linspace(1000, 9000, 9)
    event = _ch("Event", "threshold", s, 600 / (1 + np.exp(-(s - 4000) / 300)), 0.0)
    search = _ch("Search", "log", s, 40 * np.log1p(s / 2000), 5000)
    plan = optimize([event, search], Scenario(mode="budget", budget=10000))
    assert plan.budgets[0] > 4000


def test_uncertainty_ranges_bracket_plan(channels):
    sc = Scenario(mode="budget", budget=13000)
    plan = optimize(channels, sc)
    unc = plan_uncertainty(channels, sc, plan, n_samples=15)
    assert np.all(unc.budget_low <= unc.budget_high + 1e-9)
    assert unc.leads_low <= unc.leads_mid <= unc.leads_high
