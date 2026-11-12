import json

import pandas as pd

from budget_engine import report
from budget_engine.advisor import build_system_prompt
from budget_engine.curves import fit_curve
from budget_engine.data import clean_history, current_spend, sample_history
from budget_engine.optimize import Channel, Scenario, optimize


def test_flexible_column_names_and_currency_strings():
    raw = pd.DataFrame(
        {
            "Date": ["2026-01", "2026-02"],
            "Source": ["Search", "Search"],
            "Cost ($)": ["$1,000", "2,000"],
            "Conversions": [10, 18],
        }
    )
    df, issues = clean_history(raw)
    assert list(df.columns) == ["Channel", "Period", "Spend", "Leads"]
    assert df["Spend"].tolist() == [1000, 2000]
    assert not [i for i in issues if i.level == "error"]


def test_missing_columns_is_an_error():
    df, issues = clean_history(pd.DataFrame({"Channel": ["A"], "Spend": [1]}))
    assert df.empty
    assert issues[0].level == "error"
    assert "Leads" in issues[0].message


def test_bad_rows_dropped_with_warnings():
    raw = pd.DataFrame(
        {
            "Channel": ["Search", "Search", "", "Search", "Social", "social"],
            "Period": ["1", "2", "3", "4", "1", "2"],
            "Spend": [100, -5, 10, 0, 50, 60],
            "Leads": [10, 1, 1, 7, 5, 6],
        }
    )
    df, issues = clean_history(raw)
    text = " ".join(i.message for i in issues)
    assert "negative" in text
    assert "no spend" in text
    assert "capitalization" in text
    assert set(df["Channel"]) == {"Search", "Social"}
    assert len(df) == 3


def test_duplicate_periods_are_summed():
    raw = pd.DataFrame({"Channel": ["A", "A"], "Period": ["2026-01", "2026-01"], "Spend": [100, 50], "Leads": [5, 3]})
    df, issues = clean_history(raw)
    assert df["Spend"].tolist() == [150]
    assert any("duplicate" in i.message for i in issues)


def test_current_spend_uses_latest_period():
    raw = pd.DataFrame(
        {
            "Channel": ["A", "A", "A"],
            "Period": ["2026-03", "2026-01", "2026-02"],
            "Spend": [300, 100, 200],
            "Leads": [3, 1, 2],
        }
    )
    df, _ = clean_history(raw)
    assert current_spend(df) == {"A": 300}


def test_current_spend_without_periods_is_mean():
    df, _ = clean_history(pd.DataFrame({"Channel": ["A", "A"], "Spend": [100, 300], "Leads": [1, 3]}))
    assert current_spend(df) == {"A": 200}


def test_sample_history_is_clean_and_deterministic():
    a, issues = clean_history(sample_history())
    assert not issues
    assert a.equals(clean_history(sample_history())[0])
    assert a["Channel"].nunique() == 5


def _plan():
    ch = [
        Channel("Search", fit_curve("log", [4000, 6000, 8000, 10000], [120, 150, 170, 185]), 6000, value_per_lead=80),
        Channel("Email", fit_curve("negexp", [1000], [90]), 1000, value_per_lead=40),
    ]
    sc = Scenario(mode="budget", budget=7000, max_change_pct=50)
    return ch, sc, optimize(ch, sc)


def test_plan_table_and_exports():
    ch, sc, plan = _plan()
    table = report.plan_table(ch, plan, None, sc)
    assert {"Current spend", "Recommended spend", "Change", "Leads per next $1K", "Confidence"} <= set(table.columns)
    assert table.loc[1, "Confidence"] == "Low"
    tot = report.totals(ch, plan)
    record = report.scenario_record("Test", sc, table, tot)
    assert json.loads(report.to_json(record))["name"] == "Test"
    xlsx = report.to_excel(table, pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [2]}), record)
    assert xlsx[:2] == b"PK"
    comp = report.comparison_table([record, {**record, "name": "B"}])
    assert len(comp) == 2
    alloc = report.comparison_allocations([record, {**record, "name": "B"}])
    assert list(alloc.columns) == ["Channel", "Test", "B"]


def test_advisor_prompt_mentions_low_confidence_and_goal():
    ch, sc, plan = _plan()
    table = report.plan_table(ch, plan, None, sc)
    prompt = build_system_prompt(pd.DataFrame({"x": [1]}), table, report.totals(ch, plan), sc, None)
    assert "Low-confidence channels: Email" in prompt
    assert "fixed budget of $7,000" in prompt
    assert "+/-50%" in prompt
