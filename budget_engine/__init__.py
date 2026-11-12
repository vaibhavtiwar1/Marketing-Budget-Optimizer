"""Core math for the Marketing Budget Optimizer, independent of the Streamlit UI."""

from budget_engine.curves import CURVES, ChannelModel, fit_curve, guess_curve, select_curve
from budget_engine.data import clean_history, current_spend, sample_history
from budget_engine.optimize import Channel, PlanResult, Scenario, optimize, plan_uncertainty

__all__ = [
    "CURVES",
    "Channel",
    "ChannelModel",
    "PlanResult",
    "Scenario",
    "clean_history",
    "current_spend",
    "fit_curve",
    "guess_curve",
    "optimize",
    "plan_uncertainty",
    "sample_history",
    "select_curve",
]
