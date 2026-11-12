"""Budget allocation: hit a lead/value target at least cost, or get the most from a fixed budget."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize

from budget_engine.curves import ChannelModel, sample_model

FEAS_TOL = 1e-4


@dataclass(frozen=True)
class Channel:
    name: str
    model: ChannelModel
    current_spend: float
    fixed_cost: float = 0.0
    min_spend: float = 0.0
    max_spend: float | None = None
    locked: bool = False
    value_per_lead: float = 0.0

    def leads(self, budget):
        return self.model.response(np.maximum(np.asarray(budget, dtype=float) - self.fixed_cost, 0.0))

    def leads_per_extra_1k(self, budget: float) -> float:
        """Marginal leads from the next $1,000 at this budget."""
        h = max(1.0, 0.01 * max(budget, self.current_spend, 100.0))
        return (self.leads(budget + h) - self.leads(budget)) / h * 1000.0


@dataclass(frozen=True)
class Scenario:
    mode: str = "budget"  # "budget": maximize output for a fixed spend; "target": hit output at min spend
    goal: str = "leads"  # "leads" or "value" (pipeline $ = leads x value per lead)
    budget: float = 0.0
    target: float = 0.0
    max_change_pct: float | None = None  # e.g. 30 -> each channel may move at most +/-30% vs current
    extrapolation_limit: float = 2.0  # max spend as a multiple of the highest spend ever observed


@dataclass
class PlanResult:
    feasible: bool
    names: list[str]
    budgets: np.ndarray
    leads: np.ndarray
    value: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    message: str = ""
    issues: list[str] = field(default_factory=list)

    @property
    def total_spend(self) -> float:
        return float(self.budgets.sum())

    @property
    def total_leads(self) -> float:
        return float(self.leads.sum())

    @property
    def total_value(self) -> float:
        return float(self.value.sum())

    def limit_flags(self) -> list[str]:
        flags = []
        for b, lo, hi in zip(self.budgets, self.lower, self.upper):
            tol = max(1.0, 1e-4 * hi)
            if hi - lo <= tol:
                flags.append("Locked")
            elif b >= hi - tol:
                flags.append("At max")
            elif b <= lo + tol and lo > 0:
                flags.append("At min")
            else:
                flags.append("")
        return flags


def channel_bounds(ch: Channel, sc: Scenario) -> tuple[float, float]:
    """Lowest and highest spend the optimizer may assign to a channel."""
    observed_max = ch.model.max_observed + ch.fixed_cost
    hi = ch.max_spend if ch.max_spend is not None else sc.extrapolation_limit * max(observed_max, ch.current_spend, 1.0)
    lo = ch.min_spend
    if sc.max_change_pct is not None and ch.current_spend > 0:
        p = sc.max_change_pct / 100.0
        lo = max(lo, ch.current_spend * (1 - p))
        hi = min(hi, ch.current_spend * (1 + p))
    if ch.locked:
        lo = hi = ch.current_spend
    return float(lo), float(hi)


def _weights(channels: list[Channel], sc: Scenario) -> np.ndarray:
    if sc.goal == "value":
        return np.array([max(ch.value_per_lead, 0.0) for ch in channels], dtype=float)
    return np.ones(len(channels))


def _result(channels, sc, b, lo, hi, feasible=True, message="", issues=None) -> PlanResult:
    b = np.clip(np.asarray(b, dtype=float), lo, hi)
    leads = np.array([ch.leads(x) for ch, x in zip(channels, b)], dtype=float)
    value = leads * np.array([ch.value_per_lead for ch in channels], dtype=float)
    return PlanResult(feasible, [c.name for c in channels], b, leads, value, lo, hi, message, issues or [])


def _greedy(outputs_i, lo, hi, mode, amount, steps=300):
    """Marginal-return allocation with look-ahead, so threshold curves aren't starved.

    Repeatedly gives the next slice of money to the channel with the best average
    return over the next 1, 3, 10 or 30 slices.
    """
    b = lo.copy()
    n = len(b)
    if mode == "budget":
        step = max((amount - lo.sum()) / steps, 1e-9)
    else:
        step = max((hi - lo).sum() / steps, 1e-9)
    current = np.array([outputs_i(i, b[i]) for i in range(n)])
    for _ in range(steps * 6):
        if mode == "budget":
            remaining = amount - b.sum()
            if remaining <= 1e-9:
                break
        else:
            remaining = np.inf
            if current.sum() >= amount:
                break
        best_gain, best = 0.0, None
        for i in range(n):
            room = min(hi[i] - b[i], remaining)
            if room <= 1e-12:
                continue
            for j in (1, 3, 10, 30):
                d = min(j * step, room)
                gain = (outputs_i(i, b[i] + d) - current[i]) / d
                if gain > best_gain:
                    best_gain, best = gain, (i, d)
                if d >= room:
                    break
        if best is None:
            break
        i, d = best
        b[i] += d
        current[i] = outputs_i(i, b[i])
    return b


def optimize(channels: list[Channel], sc: Scenario, n_random_starts: int = 3, seed: int = 0) -> PlanResult:
    """Find the best allocation for a scenario.

    Uses a greedy warm start plus several SLSQP runs from different starting points
    (the S-curve and threshold families make the problem non-convex), and keeps the
    best feasible answer. Infeasible scenarios come back with a specific explanation.
    """
    n = len(channels)
    if n == 0:
        empty = np.zeros(0)
        return PlanResult(False, [], empty, empty, empty, empty, empty, "No channels to optimize.")

    bounds = [channel_bounds(ch, sc) for ch in channels]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    conflicts = [ch.name for ch, a, b in zip(channels, lo, hi) if a > b + 1e-9]
    if conflicts:
        return _result(
            channels,
            sc,
            lo,
            lo,
            np.maximum(hi, lo),
            False,
            "Channel limits conflict: the minimum is above the maximum for "
            + ", ".join(conflicts)
            + ". Check min/max spend and the change limit.",
        )

    w = _weights(channels, sc)
    if sc.goal == "value" and not np.any(w > 0):
        return _result(
            channels,
            sc,
            lo,
            lo,
            hi,
            False,
            "Set a value per lead for at least one channel to optimize for pipeline value.",
        )

    def outputs_i(i, x):
        return w[i] * channels[i].leads(x)

    def total(b):
        return float(sum(outputs_i(i, b[i]) for i in range(n)))

    unit = "pipeline value" if sc.goal == "value" else "leads"
    fmt = (lambda v: f"${v:,.0f}") if sc.goal == "value" else (lambda v: f"{v:,.0f}")

    if sc.mode == "budget":
        B = float(sc.budget)
        if lo.sum() > B + 1e-6:
            drivers = sorted(zip(lo, [c.name for c in channels]), reverse=True)[:3]
            detail = ", ".join(f"{name} ${v:,.0f}" for v, name in drivers if v > 0)
            return _result(
                channels,
                sc,
                lo,
                lo,
                hi,
                False,
                f"Minimums, locks and change limits already need ${lo.sum():,.0f}, more than the "
                f"${B:,.0f} budget. Largest: {detail}.",
            )
        if hi.sum() <= B + 1e-6:
            left = B - hi.sum()
            issues = []
            if left > 1:
                issues.append(
                    f"Every channel is at its maximum, so ${left:,.0f} of the budget is left unspent. "
                    "Raise channel maximums or the extrapolation limit to use it."
                )
            return _result(channels, sc, hi, lo, hi, issues=issues)
    else:
        T = float(sc.target)
        reachable = total(hi)
        if reachable < T * (1 - FEAS_TOL):
            capped = sorted(((outputs_i(i, hi[i]), channels[i].name, hi[i]) for i in range(n)), reverse=True)
            detail = "; ".join(f"{name} tops out at {fmt(v)} at ${h:,.0f}" for v, name, h in capped[:4])
            return _result(
                channels,
                sc,
                hi,
                lo,
                hi,
                False,
                f"The target of {fmt(T)} {unit} isn't reachable within the current limits; the most "
                f"these channels can deliver is {fmt(reachable)}. {detail}. Lower the target, raise "
                "channel maximums, loosen the change limit, or raise the extrapolation limit.",
            )
        if total(lo) >= T:
            return _result(channels, sc, lo, lo, hi, issues=["Channel minimums alone already reach the target."])

    free = hi - lo > 1e-9
    span = np.where(free, hi - lo, 1.0)

    def to_b(z):
        b = lo.copy()
        b[free] = lo[free] + np.clip(z, 0, 1) * span[free]
        return b

    def to_z(b):
        return np.clip((b[free] - lo[free]) / span[free], 0, 1)

    amount = sc.budget if sc.mode == "budget" else sc.target
    greedy = _greedy(outputs_i, lo, hi, sc.mode, float(amount))

    rng = np.random.default_rng(seed)
    starts = [greedy]
    if n_random_starts > 0:
        cur = np.clip(np.array([c.current_spend for c in channels]), lo, hi)
        starts += [cur, (lo + hi) / 2]
        starts += [lo + rng.uniform(0, 1, n) * (hi - lo) for _ in range(n_random_starts)]

    if sc.mode == "budget":
        out_scale = max(total(hi), 1e-9)
        objective = lambda z: -total(to_b(z)) / out_scale  # noqa: E731
        cons = {"type": "ineq", "fun": lambda z: (B - to_b(z).sum()) / max(B, 1.0)}
        feasible = lambda b: b.sum() <= B * (1 + 1e-6) + 1e-6  # noqa: E731
        better = lambda b, best: total(b) > total(best) + 1e-9  # noqa: E731
    else:
        spend_scale = max(hi.sum(), 1.0)
        objective = lambda z: to_b(z).sum() / spend_scale  # noqa: E731
        cons = {"type": "ineq", "fun": lambda z: (total(to_b(z)) - T) / max(T, 1e-9)}
        feasible = lambda b: total(b) >= T * (1 - FEAS_TOL)  # noqa: E731
        better = lambda b, best: b.sum() < best.sum() - 1e-6  # noqa: E731

    best = greedy if feasible(greedy) else None
    if not free.any():
        return _result(channels, sc, lo, lo, hi)
    for start in starts:
        if sc.mode == "budget" and start.sum() > B:
            # Scale the free part down so the start respects the budget.
            excess_room = start[free] - lo[free]
            if excess_room.sum() > 0:
                start = start.copy()
                start[free] = lo[free] + excess_room * max(B - lo.sum(), 0) / excess_room.sum()
        res = minimize(
            objective,
            to_z(start),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * int(free.sum()),
            constraints=[cons],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        cand = to_b(res.x)
        if feasible(cand) and (best is None or better(cand, best)):
            best = cand

    if best is None:
        return _result(
            channels,
            sc,
            greedy,
            lo,
            hi,
            False,
            "The optimizer couldn't find an allocation that meets every constraint. "
            "Try loosening channel limits or the target.",
        )
    return _result(channels, sc, best, lo, hi)


@dataclass
class PlanUncertainty:
    budget_low: np.ndarray  # P10 of each channel's recommended budget
    budget_high: np.ndarray  # P90
    leads_low: float  # P10 / P50 / P90 of total leads the recommended plan delivers
    leads_mid: float
    leads_high: float
    value_low: float
    value_high: float
    n_samples: int


def plan_uncertainty(
    channels: list[Channel], sc: Scenario, plan: PlanResult, n_samples: int = 40, seed: int = 0
) -> PlanUncertainty:
    """Re-run the scenario on plausible alternative curves to show how sure we are.

    Gives (1) the P10-P90 range of each channel's recommended budget and (2) the
    P10-P90 range of leads/value the *recommended* plan actually delivers.
    """
    rng = np.random.default_rng(seed)
    budgets, leads, values = [], [], []
    for _ in range(n_samples):
        alt = [replace(ch, model=sample_model(ch.model, rng)) for ch in channels]
        out = np.array([ch.leads(b) for ch, b in zip(alt, plan.budgets)])
        leads.append(out.sum())
        values.append(float(out @ np.array([ch.value_per_lead for ch in channels])))
        res = optimize(alt, sc, n_random_starts=0)
        if res.feasible:
            budgets.append(res.budgets)
    arr = np.array(budgets) if budgets else plan.budgets[None, :]
    return PlanUncertainty(
        budget_low=np.percentile(arr, 10, axis=0),
        budget_high=np.percentile(arr, 90, axis=0),
        leads_low=float(np.percentile(leads, 10)),
        leads_mid=float(np.percentile(leads, 50)),
        leads_high=float(np.percentile(leads, 90)),
        value_low=float(np.percentile(values, 10)),
        value_high=float(np.percentile(values, 90)),
        n_samples=n_samples,
    )
