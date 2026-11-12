"""Channel response curves: how leads respond to spend.

Every curve is defined in *normalized* units: spend ``u = x / x_scale`` and
leads ``v = y / y_scale``, where the scales are the median observed spend and
leads for that channel. That keeps the shape parameters in a sane numeric range
whatever the channel's size, so a $500/month channel and a $500K/month channel
are fitted the same way.

When a channel has enough history (several distinct spend levels), both the
height *and* the shape of its curve are fitted from the data. When it does not,
the shape falls back to a default scaled to the channel's own spend level and
only the height is fitted, and the model is flagged as low confidence.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit

MIN_POINTS_FOR_SHAPE = 4
MIN_SPEND_VARIATION = 0.10  # coefficient of variation of spend needed to learn a shape


# ---------------------------------------------------------------------------
# Curve families (normalized units)
# ---------------------------------------------------------------------------
def _log(u, a, k):
    return a * np.log1p(np.maximum(u, 0.0) / k)


def _hill(u, a, k, s):
    r = np.power(np.maximum(u, 0.0) / k, s)
    return a * r / (1.0 + r)


def _negexp(u, a, k):
    return -a * np.expm1(-np.maximum(u, 0.0) / k)


def _power(u, a, b):
    return a * np.power(np.maximum(u, 0.0), b)


def _threshold(u, a, k, w):
    base = expit(-k / w)
    return a * (expit((np.maximum(u, 0.0) - k) / w) - base) / (1.0 - base)


@dataclass(frozen=True)
class CurveSpec:
    key: str
    label: str
    description: str
    fn: Callable
    shape_names: tuple[str, ...]
    shape_default: tuple[float, ...]
    shape_lower: tuple[float, ...]
    shape_upper: tuple[float, ...]
    saturates: bool


CURVES: dict[str, CurveSpec] = {
    "log": CurveSpec(
        "log",
        "Fast saturation",
        "Efficient first dollars, flattens quickly. Typical for search and retargeting.",
        _log,
        ("k",),
        (0.25,),
        (0.01,),
        (100.0,),
        False,
    ),
    "hill": CurveSpec(
        "hill",
        "S-curve",
        "Slow start while the platform learns, then scales, then flattens. Typical for paid social, video and display.",
        _hill,
        ("k", "s"),
        (0.8, 2.0),
        (0.05, 1.0),
        (20.0, 5.0),
        True,
    ),
    "negexp": CurveSpec(
        "negexp",
        "Hard ceiling",
        "Grows almost linearly, then stops when the audience runs out. Typical for email and SMS.",
        _negexp,
        ("k",),
        (1.0,),
        (0.05,),
        (50.0,),
        True,
    ),
    "power": CurveSpec(
        "power",
        "Steady diminishing returns",
        "Returns shrink slowly with no hard ceiling. Typical for SEO and content.",
        _power,
        ("b",),
        (0.5,),
        (0.05,),
        (1.0,),
        False,
    ),
    "threshold": CurveSpec(
        "threshold",
        "Needs minimum spend",
        "Little or nothing until a minimum spend is crossed. Typical for events, PR and sponsorships.",
        _threshold,
        ("k", "w"),
        (0.5, 0.15),
        (0.01, 0.02),
        (10.0, 5.0),
        True,
    ),
}

CURVE_KEYS = tuple(CURVES)


# ---------------------------------------------------------------------------
# Keyword-based starting guess
# ---------------------------------------------------------------------------
# Checked in order; whole words only, so "PR" no longer matches "Programmatic".
_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "threshold",
        (
            "event",
            "events",
            "trade show",
            "tradeshow",
            "conference",
            "sponsorship",
            "sponsorships",
            "billboard",
            "billboards",
            "pr",
            "press",
            "ooh",
        ),
    ),
    ("negexp", ("email", "emails", "newsletter", "sms", "hubspot", "mailchimp", "klaviyo", "push")),
    (
        "hill",
        (
            "social",
            "meta",
            "facebook",
            "instagram",
            "tiktok",
            "linkedin",
            "snapchat",
            "pinterest",
            "reddit",
            "twitter",
            "x",
            "youtube",
            "video",
            "display",
            "programmatic",
            "tv",
            "ctv",
        ),
    ),
    ("power", ("seo", "content", "organic", "blog")),
    ("log", ("search", "google", "bing", "sem", "ppc", "retargeting", "remarketing", "shopping")),
]


def guess_curve(channel_name: str) -> str:
    """Best-guess curve family from a channel's name. Defaults to fast saturation."""
    words = re.findall(r"[a-z0-9]+", str(channel_name).lower())
    tokens, text = set(words), " ".join(words)
    for key, keywords in _KEYWORDS:
        for kw in keywords:
            if (" " in kw and kw in text) or kw in tokens:
                return key
    return "log"


# ---------------------------------------------------------------------------
# Fitted channel model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelModel:
    """A curve fitted to one channel's history. Spend here is *effective* spend
    (spend minus any fixed cost)."""

    curve: str
    theta: tuple[float, ...]  # (a, *shape) in normalized units
    x_scale: float
    y_scale: float
    spend: tuple[float, ...]  # effective spend observations used for fitting
    leads: tuple[float, ...]
    method: str  # "fitted" (shape learned) or "anchored" (default shape, height fitted)
    r2: float | None = None
    aicc: float | None = None
    note: str = ""

    @property
    def spec(self) -> CurveSpec:
        return CURVES[self.curve]

    @property
    def n_points(self) -> int:
        return len(self.spend)

    @property
    def max_observed(self) -> float:
        return max(self.spend) if self.spend else 0.0

    @property
    def confidence(self) -> str:
        if self.method != "fitted":
            return "Low"
        if self.r2 is not None and self.r2 < 0.3:
            return "Low"
        if self.n_points >= 8 and self.r2 is not None and self.r2 >= 0.6:
            return "High"
        return "Medium"

    def response(self, effective_spend):
        x = np.asarray(effective_spend, dtype=float)
        out = self.y_scale * self.spec.fn(np.maximum(x, 0.0) / self.x_scale, *self.theta)
        return float(out) if out.ndim == 0 else out

    def shape_summary(self) -> str:
        """Plain-English description of the fitted shape, in dollars."""
        shape = dict(zip(self.spec.shape_names, self.theta[1:]))
        a = self.theta[0] * self.y_scale
        if self.curve in ("hill", "negexp", "threshold") and shape["k"] * self.x_scale > 1.5 * self.max_observed:
            return "Still scaling at your highest spend; the saturation point is beyond your data"
        if self.curve == "hill":
            return f"Half of max (~{a:,.0f} leads) reached at ${shape['k'] * self.x_scale:,.0f}"
        if self.curve == "negexp":
            return f"Ceiling ~{a:,.0f} leads; 63% reached at ${shape['k'] * self.x_scale:,.0f}"
        if self.curve == "threshold":
            return f"Kicks in around ${shape['k'] * self.x_scale:,.0f}; ceiling ~{a:,.0f} leads"
        if self.curve == "power":
            return f"Doubling spend gives {2 ** shape['b']:.2f}x leads"
        return f"Returns halve roughly every ${shape['k'] * self.x_scale:,.0f} of extra spend"


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _scales(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xs = float(np.median(x[x > 0])) if np.any(x > 0) else 1.0
    ys = float(np.median(y[y > 0])) if np.any(y > 0) else 1.0
    return xs, ys


def _anchor_height(spec: CurveSpec, shape: tuple[float, ...], u: np.ndarray, v: np.ndarray) -> float:
    """Least-squares height for a fixed shape (closed form)."""
    g = spec.fn(u, 1.0, *shape)
    denom = float(np.dot(g, g))
    return max(float(np.dot(v, g)) / denom, 0.0) if denom > 0 else 0.0


def _r2(y: np.ndarray, yhat: np.ndarray) -> float | None:
    if len(y) < 2:
        return None
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return None
    return 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot


def _aicc(n: int, k: int, rss: float) -> float | None:
    if n - k - 1 <= 0:
        return None
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def shape_is_learnable(effective_spend) -> bool:
    """True if the history has enough distinct spend levels to learn a curve's shape."""
    x = np.asarray(effective_spend, dtype=float)
    x = x[x > 0]
    if len(np.unique(np.round(x, 6))) < MIN_POINTS_FOR_SHAPE:
        return False
    return float(x.std() / x.mean()) >= MIN_SPEND_VARIATION


def fit_curve(curve: str, spend, leads, fixed_cost: float = 0.0) -> ChannelModel:
    """Fit one curve family to a channel's history.

    ``spend`` is total spend per period; ``fixed_cost`` is subtracted first because it
    produces no leads. Periods whose effective spend is zero are ignored.
    """
    spec = CURVES[curve]
    x = np.maximum(np.asarray(spend, dtype=float) - fixed_cost, 0.0)
    y = np.asarray(leads, dtype=float)
    keep = x > 0
    x, y = x[keep], y[keep]
    if len(x) == 0:
        raise ValueError("No periods with spend above the fixed cost.")

    xs, ys = _scales(x, y)
    u, v = x / xs, y / ys
    note = ""
    method = "anchored"
    theta: tuple[float, ...]

    if shape_is_learnable(x):
        best = None
        for mult in (1.0, 0.3, 3.0):
            shape0 = tuple(
                float(np.clip(d * mult, lo, hi))
                for d, lo, hi in zip(spec.shape_default, spec.shape_lower, spec.shape_upper)
            )
            a0 = max(_anchor_height(spec, shape0, u, v), 1e-6)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        spec.fn,
                        u,
                        v,
                        p0=(a0, *shape0),
                        bounds=((0.0, *spec.shape_lower), (np.inf, *spec.shape_upper)),
                        maxfev=20000,
                    )
            except (RuntimeError, ValueError):
                continue
            rss = float(np.sum((spec.fn(u, *popt) - v) ** 2))
            if np.isfinite(rss) and (best is None or rss < best[0]):
                best = (rss, tuple(float(p) for p in popt))
        if best is not None:
            theta, method = best[1], "fitted"
        else:
            note = "Shape fit did not converge; using the default shape."
    else:
        n_distinct = len(np.unique(np.round(x, 6)))
        if n_distinct < MIN_POINTS_FOR_SHAPE:
            note = f"Only {n_distinct} distinct spend level(s); need {MIN_POINTS_FOR_SHAPE}+ to learn the curve shape."
        else:
            note = "Spend barely varied over time, so the curve shape can't be learned."

    if method == "anchored":
        theta = (_anchor_height(spec, spec.shape_default, u, v), *spec.shape_default)

    yhat = ys * spec.fn(u, *theta)
    rss = float(np.sum(((yhat - y) / ys) ** 2))
    n_params = 1 + (len(spec.shape_names) if method == "fitted" else 0)
    return ChannelModel(
        curve=curve,
        theta=theta,
        x_scale=xs,
        y_scale=ys,
        spend=tuple(float(a) for a in x),
        leads=tuple(float(b) for b in y),
        method=method,
        r2=_r2(y, yhat),
        aicc=_aicc(len(x), n_params, rss),
        note=note,
    )


def select_curve(
    spend, leads, fixed_cost: float = 0.0, preferred: str | None = None
) -> tuple[ChannelModel, dict[str, float | None]]:
    """Fit every curve family and keep the best by AICc.

    The ``preferred`` family (usually the keyword guess) wins ties: it is kept if its
    AICc is within 2 of the best, since the data can't really tell them apart then.
    Returns the chosen model and the AICc of every family (``None`` if not fitted).
    """
    preferred = preferred or "log"
    x = np.maximum(np.asarray(spend, dtype=float) - fixed_cost, 0.0)
    if not shape_is_learnable(x):
        return fit_curve(preferred, spend, leads, fixed_cost), {}

    fits = {key: fit_curve(key, spend, leads, fixed_cost) for key in CURVE_KEYS}
    scores = {k: (m.aicc if m.method == "fitted" else None) for k, m in fits.items()}
    valid = {k: s for k, s in scores.items() if s is not None}
    if not valid:
        return fits[preferred], scores
    best = min(valid, key=valid.get)
    if preferred in valid and valid[preferred] <= valid[best] + 2.0:
        best = preferred
    return fits[best], scores


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def sample_model(model: ChannelModel, rng: np.random.Generator) -> ChannelModel:
    """Draw a plausible alternative model, used to build confidence ranges.

    Fitted models are refitted on a bootstrap resample of their history. Anchored
    models (shape unknown) get their default shape scaled by a random factor
    between 0.5x and 2x, then their height is refitted to the observed data.
    """
    x = np.asarray(model.spend)
    y = np.asarray(model.leads)
    spec = model.spec

    if model.method == "fitted":
        idx = rng.integers(0, len(x), len(x))
        if shape_is_learnable(x[idx]):
            return fit_curve(model.curve, x[idx], y[idx])

    shape = tuple(
        float(np.clip(s * np.exp(rng.uniform(np.log(0.5), np.log(2.0))), lo, hi))
        for s, lo, hi in zip(model.theta[1:], spec.shape_lower, spec.shape_upper)
    )
    u, v = x / model.x_scale, y / model.y_scale
    return replace(model, theta=(_anchor_height(spec, shape, u, v), *shape))
