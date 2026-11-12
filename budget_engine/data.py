"""Loading, validating and summarizing channel history."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUMNS = ["Channel", "Period", "Spend", "Leads"]

_ALIASES = {
    "channel": "Channel",
    "source": "Channel",
    "medium": "Channel",
    "period": "Period",
    "date": "Period",
    "month": "Period",
    "week": "Period",
    "spend": "Spend",
    "cost": "Spend",
    "past spend": "Spend",
    "amount spent": "Spend",
    "leads": "Leads",
    "past leads": "Leads",
    "conversions": "Leads",
    "results": "Leads",
}


@dataclass(frozen=True)
class Issue:
    level: str  # "error" or "warning"
    message: str


def _canonical_column(name: str) -> str | None:
    text = re.sub(r"\(.*?\)", "", str(name)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return _ALIASES.get(text)


def clean_history(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[Issue]]:
    """Validate and normalize a history table.

    Accepts flexible column names (e.g. "Spend ($)", "Cost", "Conversions", "Date").
    Returns a frame with columns Channel, Period, Spend, Leads plus a list of issues
    explaining anything that was dropped, merged or looks suspicious.
    """
    issues: list[Issue] = []
    rename = {}
    for col in raw.columns:
        canon = _canonical_column(col)
        if canon and canon not in rename.values():
            rename[col] = canon
    df = raw.rename(columns=rename)

    missing = [c for c in ("Channel", "Spend", "Leads") if c not in df.columns]
    if missing:
        issues.append(
            Issue(
                "error",
                f"Missing column(s): {', '.join(missing)}. Expected Channel, Spend, Leads and optionally Period.",
            )
        )
        return pd.DataFrame(columns=COLUMNS), issues
    if "Period" not in df.columns:
        df["Period"] = ""
    df = df[COLUMNS].copy()

    df["Channel"] = df["Channel"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    blank = df["Channel"].isna() | (df["Channel"] == "")
    df = df[~blank]

    for col in ("Spend", "Leads"):
        cleaned = df[col].astype(str).str.replace(r"[$,\s]", "", regex=True)
        df[col] = pd.to_numeric(cleaned, errors="coerce")
    bad = df["Spend"].isna() | df["Leads"].isna()
    if bad.any():
        issues.append(Issue("warning", f"Ignored {int(bad.sum())} row(s) with missing or non-numeric spend/leads."))
    df = df[~bad]

    neg = (df["Spend"] < 0) | (df["Leads"] < 0)
    if neg.any():
        issues.append(Issue("warning", f"Ignored {int(neg.sum())} row(s) with negative spend or leads."))
    df = df[~neg]

    free = (df["Spend"] == 0) & (df["Leads"] > 0)
    if free.any():
        chans = ", ".join(sorted(df.loc[free, "Channel"].unique()))
        issues.append(
            Issue(
                "warning",
                f"Ignored {int(free.sum())} row(s) with leads but no spend ({chans}). "
                "Those leads are organic/unattributed and can't be modeled against spend.",
            )
        )
    df = df[~free]

    # Merge channels that differ only in capitalization.
    df["Period"] = df["Period"].fillna("").astype(str).str.strip()
    canonical = {}
    for name in df["Channel"]:
        canonical.setdefault(name.lower(), name)
    merged = sorted({n for n in df["Channel"].unique() if canonical[n.lower()] != n})
    if merged:
        issues.append(
            Issue("warning", f"Merged channel names that differ only in capitalization: {', '.join(merged)}.")
        )
    df["Channel"] = df["Channel"].map(lambda n: canonical[n.lower()])

    has_period = df["Period"] != ""
    dupes = df[has_period].duplicated(["Channel", "Period"], keep=False)
    if dupes.any():
        issues.append(Issue("warning", "Summed duplicate rows for the same channel and period."))
        dated = df[has_period].groupby(["Channel", "Period"], as_index=False, sort=False)[["Spend", "Leads"]].sum()
        df = pd.concat([dated, df[~has_period]], ignore_index=True)

    for chan, grp in df.groupby("Channel", sort=False):
        if (grp["Spend"] <= 0).all():
            issues.append(Issue("warning", f"'{chan}' has no periods with spend, so it was left out."))
    df = df[df.groupby("Channel")["Spend"].transform("max") > 0]

    if df.empty:
        issues.append(Issue("error", "No usable rows. Each channel needs at least one period with spend and leads."))
    return df.reset_index(drop=True)[COLUMNS], issues


def _period_order(periods: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(periods, errors="coerce", format="mixed")
    if parsed.notna().all() and len(periods) > 0:
        return parsed.rank(method="first")
    return pd.Series(np.arange(len(periods)), index=periods.index, dtype=float)


def current_spend(history: pd.DataFrame) -> dict[str, float]:
    """Current monthly spend per channel: the latest period if periods are given,
    otherwise the average across rows."""
    out = {}
    for chan, grp in history.groupby("Channel", sort=False):
        if (grp["Period"] != "").all() and len(grp) > 1:
            out[chan] = float(grp.loc[_period_order(grp["Period"]).idxmax(), "Spend"])
        else:
            out[chan] = float(grp["Spend"].mean())
    return out


def sample_history(months: int = 8, seed: int = 7) -> pd.DataFrame:
    """Realistic demo data: monthly spend and leads for five channels."""
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2026-01", periods=months, freq="M").astype(str)
    specs = {
        # name: (typical spend, response function in $ -> leads)
        "Google Paid Search": (6000, lambda x: 95 * np.log1p(x / 1200)),
        "Meta Paid Social": (5000, lambda x: 420 * (x / 6500) ** 2.2 / (1 + (x / 6500) ** 2.2)),
        "HubSpot Email": (1200, lambda x: 160 * (1 - np.exp(-x / 900))),
        "SEO Blog Content": (3000, lambda x: 2.4 * x**0.55),
        "Trade Show Events": (4000, lambda x: 150 / (1 + np.exp(-(x - 3500) / 500))),
    }
    rows = []
    for name, (base, fn) in specs.items():
        spends = base * rng.uniform(0.55, 1.45, months)
        for period, spend in zip(periods, spends):
            spend = round(float(spend), -1)
            leads = max(0, round(float(fn(spend)) * rng.normal(1.0, 0.07)))
            rows.append((name, period, spend, leads))
    return pd.DataFrame(rows, columns=COLUMNS)


def template_csv() -> str:
    return (
        "Channel,Period,Spend,Leads\n"
        "Google Paid Search,2026-01,5000,250\n"
        "Google Paid Search,2026-02,6200,282\n"
        "Meta Paid Social,2026-01,4000,180\n"
        "Meta Paid Social,2026-02,4800,231\n"
    )
