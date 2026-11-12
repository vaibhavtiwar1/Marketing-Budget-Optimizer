"""Tables and exports built from an optimization result."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pandas as pd

from budget_engine.curves import CURVES
from budget_engine.optimize import Channel, PlanResult, PlanUncertainty, Scenario


def plan_table(channels: list[Channel], plan: PlanResult, unc: PlanUncertainty | None, sc: Scenario) -> pd.DataFrame:
    """Before/after table: one row per channel."""
    flags = plan.limit_flags()
    rows = []
    for i, ch in enumerate(channels):
        cur, rec = ch.current_spend, float(plan.budgets[i])
        cur_leads, rec_leads = float(ch.leads(cur)), float(plan.leads[i])
        row = {
            "Channel": ch.name,
            "Curve": CURVES[ch.model.curve].label,
            "Confidence": ch.model.confidence,
            "Current spend": cur,
            "Recommended spend": rec,
            "Change": rec - cur,
            "Change %": (rec - cur) / cur * 100 if cur > 0 else None,
        }
        if unc is not None:
            row["Range low"] = float(unc.budget_low[i])
            row["Range high"] = float(unc.budget_high[i])
        row.update(
            {
                "Leads now": cur_leads,
                "Leads after": rec_leads,
                "CAC after": rec / rec_leads if rec_leads > 0.5 else None,
                "Leads per next $1K": ch.leads_per_extra_1k(rec),
                "Limit": flags[i],
            }
        )
        if sc.goal == "value":
            row["Value after"] = float(plan.value[i])
        rows.append(row)
    return pd.DataFrame(rows)


def totals(channels: list[Channel], plan: PlanResult) -> dict[str, float]:
    cur_spend = sum(ch.current_spend for ch in channels)
    cur_leads = float(sum(ch.leads(ch.current_spend) for ch in channels))
    cur_value = float(sum(ch.leads(ch.current_spend) * ch.value_per_lead for ch in channels))
    return {
        "current_spend": cur_spend,
        "current_leads": cur_leads,
        "current_value": cur_value,
        "current_cac": cur_spend / cur_leads if cur_leads > 0 else 0.0,
        "spend": plan.total_spend,
        "leads": plan.total_leads,
        "value": plan.total_value,
        "cac": plan.total_spend / plan.total_leads if plan.total_leads > 0 else 0.0,
    }


def scenario_record(name: str, sc: Scenario, table: pd.DataFrame, tot: dict[str, float]) -> dict:
    """A JSON-serializable snapshot of a plan, for saving and comparing."""
    return {
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": {
            "mode": sc.mode,
            "goal": sc.goal,
            "budget": sc.budget,
            "target": sc.target,
            "max_change_pct": sc.max_change_pct,
            "extrapolation_limit": sc.extrapolation_limit,
        },
        "totals": {k: round(float(v), 2) for k, v in tot.items()},
        "allocation": [
            {
                "channel": r["Channel"],
                "current_spend": round(float(r["Current spend"]), 2),
                "recommended_spend": round(float(r["Recommended spend"]), 2),
                "expected_leads": round(float(r["Leads after"]), 2),
            }
            for _, r in table.iterrows()
        ],
    }


def to_excel(table: pd.DataFrame, history: pd.DataFrame, settings: pd.DataFrame, record: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="Plan", index=False)
        summary = pd.DataFrame(
            [(k, v) for k, v in {**record["scenario"], **record["totals"]}.items()], columns=["Item", "Value"]
        )
        summary.to_excel(xw, sheet_name="Summary", index=False)
        settings.to_excel(xw, sheet_name="Channel settings", index=False)
        history.to_excel(xw, sheet_name="History", index=False)
    return buf.getvalue()


def to_json(record: dict) -> str:
    return json.dumps(record, indent=2)


def comparison_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        sc, t = r["scenario"], r["totals"]
        rows.append(
            {
                "Scenario": r["name"],
                "Mode": "Fixed budget" if sc["mode"] == "budget" else "Hit target",
                "Goal": "Pipeline value" if sc["goal"] == "value" else "Leads",
                "Total spend": t["spend"],
                "Expected leads": t["leads"],
                "Blended CAC": t["cac"],
                "Pipeline value": t.get("value", 0.0),
                "Max change %": sc["max_change_pct"],
            }
        )
    return pd.DataFrame(rows)


def comparison_allocations(records: list[dict]) -> pd.DataFrame:
    frames = []
    for r in records:
        df = pd.DataFrame(r["allocation"])[["channel", "recommended_spend"]]
        frames.append(df.set_index("channel").rename(columns={"recommended_spend": r["name"]}))
    return pd.concat(frames, axis=1).reset_index().rename(columns={"index": "Channel", "channel": "Channel"})
