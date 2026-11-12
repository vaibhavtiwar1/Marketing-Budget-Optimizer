import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from budget_engine import report
from budget_engine.advisor import GEMINI_MODEL, build_system_prompt
from budget_engine.curves import CURVES, fit_curve, guess_curve, select_curve
from budget_engine.data import clean_history, current_spend, sample_history, template_csv
from budget_engine.optimize import Channel, Scenario, channel_bounds, optimize, plan_uncertainty

# ==========================================
# 1. Page configuration & theme
# ==========================================
st.set_page_config(page_title="Marketing Budget Optimizer", layout="wide")

# Categorical palette, validated for colorblind separation and contrast on the app's dark surface.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
MUTED = "#8a83a6"
GRID = "rgba(255,255,255,0.08)"
AUTO = "Auto (best fit)"
LABEL_TO_KEY = {spec.label: key for key, spec in CURVES.items()}
UNCERTAINTY_SAMPLES = 30

st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(circle at 50% 0%, #431c7a 0%, #130f26 50%, #0b0817 100%); }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        text-shadow: 0 0 10px rgba(0, 255, 170, 0.8), 0 0 20px rgba(0, 255, 170, 0.3);
    }
    [data-testid="stMetricLabel"] { color: #b8b1ce !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.dialog("📖 How to use the optimizer", width="large")
def guide_modal():
    st.markdown(
        """
Most budget calculators assume returns are linear: double the money, double the leads. Real channels
saturate. This tool learns a **response curve** for each channel from your history and finds the split
where the next dollar works hardest everywhere.

### Steps
1. **Add your history** (sidebar): one row per channel per month with spend and leads. Edit the
   sample table or upload a CSV. **6+ months with varied spend** lets the tool learn each curve's shape;
   with less, it assumes a typical shape and marks the channel *Low confidence*.
2. **Pick a goal**: spend a fixed budget as well as possible, or hit a lead/pipeline target at the lowest
   cost. Choose *Pipeline value* to weight leads by what they're worth (set value per lead in
   *Channel settings*).
3. **Set guardrails**: limit how far any channel can move from current spend, and how far beyond your
   highest observed spend the model may extrapolate.
4. **Fine-tune channels** (*Channel settings & limits*): min/max spend, lock a channel, fixed costs
   (e.g. tool subscriptions) and the curve type.
5. **Read the plan**: before/after table, confidence ranges and the return on the next $1K per channel.
   Save scenarios to compare, export to Excel/CSV, or ask the AI advisor.

### Curve types
"""
        + "\n".join(f"* **{s.label}**: {s.description}" for s in CURVES.values())
    )


col_title, col_btn = st.columns([8.5, 1.5])
with col_title:
    st.title("📊 Marketing Budget Optimizer")
    st.markdown("*Learns each channel's diminishing returns from your history and finds the best budget split.*")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📖 How to use", width="stretch"):
        guide_modal()


# ==========================================
# 2. Sidebar: data
# ==========================================
@st.cache_data(show_spinner=False)
def cached_sample() -> pd.DataFrame:
    return sample_history()


st.sidebar.header("1 · Your data")
source = st.sidebar.radio("Data source", ["Edit table", "Upload CSV"], horizontal=True, label_visibility="collapsed")
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("CSV with Channel, Period, Spend, Leads", type=["csv"])
    st.sidebar.download_button(
        "⬇️ Download CSV template", template_csv(), "budget_history_template.csv", "text/csv", width="stretch"
    )
    if uploaded is None:
        st.info(
            "Upload a CSV of monthly spend and leads per channel to get started, "
            "or switch the sidebar to **Edit table** to try the sample data."
        )
        st.stop()
    try:
        raw = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001 - show any parse error to the user
        st.error(f"Couldn't read that CSV: {exc}")
        st.stop()
else:
    st.sidebar.caption("One row per channel per month. Pre-filled with 8 months of sample data; replace it with yours.")
    raw = st.sidebar.data_editor(
        cached_sample(),
        num_rows="dynamic",
        hide_index=True,
        height=320,
        key="history_editor",
        column_config={
            "Period": st.column_config.TextColumn(help="Month, e.g. 2026-03. Used to find your current spend."),
            "Spend": st.column_config.NumberColumn("Spend ($)", min_value=0, format="%d"),
            "Leads": st.column_config.NumberColumn(min_value=0, format="%d"),
        },
    )

history, data_issues = clean_history(raw)
for issue in data_issues:
    if issue.level == "error":
        st.error(issue.message)
if any(i.level == "error" for i in data_issues):
    st.stop()

current = current_spend(history)
names = list(current)

# ==========================================
# 3. Channel settings
# ==========================================
SETTINGS_COLS = ["Channel", "Curve", "Fixed cost", "Min spend", "Max spend", "Lock", "Value per lead"]


def default_settings(channel_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Channel": channel_names,
            "Curve": [AUTO] * len(channel_names),
            "Fixed cost": [0.0] * len(channel_names),
            "Min spend": [0.0] * len(channel_names),
            "Max spend": [np.nan] * len(channel_names),
            "Lock": [False] * len(channel_names),
            "Value per lead": [100.0] * len(channel_names),
        }
    )


settings_key = "settings_" + hashlib.md5("|".join(names).encode()).hexdigest()[:10]
with st.expander("⚙️ Channel settings & limits", expanded=False):
    st.caption(
        "**Curve**: leave on *Auto* to pick the best fit from your data. **Fixed cost**: monthly cost that "
        "produces no leads (e.g. a tool subscription). **Max spend**: blank = up to the extrapolation limit. "
        "**Lock**: keep at current spend. **Value per lead**: pipeline $ per lead (close rate × deal size), "
        "used when optimizing for pipeline value."
    )
    settings = st.data_editor(
        default_settings(names),
        key=settings_key,
        hide_index=True,
        width="stretch",
        disabled=["Channel"],
        column_config={
            "Curve": st.column_config.SelectboxColumn(options=[AUTO, *LABEL_TO_KEY], required=True),
            "Fixed cost": st.column_config.NumberColumn("Fixed cost ($)", min_value=0.0, format="%d"),
            "Min spend": st.column_config.NumberColumn("Min spend ($)", min_value=0.0, format="%d"),
            "Max spend": st.column_config.NumberColumn("Max spend ($)", min_value=0.0, format="%d"),
            "Lock": st.column_config.CheckboxColumn("Lock at current"),
            "Value per lead": st.column_config.NumberColumn("Value per lead ($)", min_value=0.0, format="%d"),
        },
    )


def _num(value, default=0.0):
    return default if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


@st.cache_data(show_spinner=False)
def fit_models(history_df: pd.DataFrame, choices: tuple, fixed_costs: tuple):
    models, scores, errors = {}, {}, {}
    for (name, choice), fixed in zip(choices, fixed_costs):
        grp = history_df[history_df["Channel"] == name]
        try:
            if choice == AUTO:
                models[name], scores[name] = select_curve(grp["Spend"], grp["Leads"], fixed, guess_curve(name))
            else:
                models[name], scores[name] = fit_curve(LABEL_TO_KEY[choice], grp["Spend"], grp["Leads"], fixed), {}
        except ValueError:
            errors[name] = (
                f"'{name}': the fixed cost (${fixed:,.0f}) is at or above every month's spend, "
                "so there's nothing left to model. Lower the fixed cost."
            )
    return models, scores, errors


def build_channels(history_df: pd.DataFrame, settings_df: pd.DataFrame):
    rows = settings_df.set_index("Channel").reindex(names)
    choices = tuple((n, rows.loc[n, "Curve"] or AUTO) for n in names)
    fixed = tuple(_num(rows.loc[n, "Fixed cost"]) for n in names)
    models, scores, errors = fit_models(history_df, choices, fixed)
    channels = []
    for n, fx in zip(names, fixed):
        if n not in models:
            continue
        max_spend = rows.loc[n, "Max spend"]
        channels.append(
            Channel(
                name=n,
                model=models[n],
                current_spend=current[n],
                fixed_cost=fx,
                min_spend=_num(rows.loc[n, "Min spend"]),
                max_spend=None if max_spend is None or pd.isna(max_spend) else float(max_spend),
                locked=bool(rows.loc[n, "Lock"]),
                value_per_lead=_num(rows.loc[n, "Value per lead"]),
            )
        )
    return channels, scores, errors


channels, fit_scores, fit_errors = build_channels(history, settings)
for msg in fit_errors.values():
    st.error(msg)
if not channels:
    st.stop()

cur_total_spend = sum(ch.current_spend for ch in channels)
cur_total_leads = float(sum(ch.leads(ch.current_spend) for ch in channels))
cur_total_value = float(sum(ch.leads(ch.current_spend) * ch.value_per_lead for ch in channels))

# ==========================================
# 4. Sidebar: goal & guardrails
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("2 · Goal")
mode_label = st.sidebar.radio("What do you want to do?", ["Spend a fixed budget", "Hit a target"])
goal_label = st.sidebar.radio(
    "Optimize for",
    ["Leads", "Pipeline value"],
    help="Pipeline value = leads × value per lead (set per channel in Channel settings).",
)
mode = "budget" if mode_label.startswith("Spend") else "target"
goal = "value" if goal_label == "Pipeline value" else "leads"

budget, target = 0.0, 0.0
if mode == "budget":
    budget = st.sidebar.number_input(
        "Monthly budget ($)",
        min_value=0.0,
        step=500.0,
        format="%.0f",
        value=float(round(cur_total_spend)),
        key="budget_input",
        help=f"Current monthly spend is ${cur_total_spend:,.0f}.",
    )
elif goal == "leads":
    target = st.sidebar.number_input(
        "Target leads per month",
        min_value=1.0,
        step=10.0,
        format="%.0f",
        value=float(max(10, round(cur_total_leads * 1.1, -1))),
        key="target_leads",
        help=f"The model estimates ~{cur_total_leads:,.0f} leads at current spend.",
    )
else:
    target = st.sidebar.number_input(
        "Target pipeline value per month ($)",
        min_value=1.0,
        step=1000.0,
        format="%.0f",
        value=float(max(1000, round(cur_total_value * 1.1, -3))),
        key="target_value",
        help=f"The model estimates ~${cur_total_value:,.0f} at current spend.",
    )

st.sidebar.markdown("---")
st.sidebar.header("3 · Guardrails")
limit_change = st.sidebar.toggle(
    "Limit change vs. current spend",
    value=True,
    help="Avoid recommending overnight swings that disrupt campaigns and platform learning.",
)
max_change = st.sidebar.slider("Max change per channel (%)", 5, 200, 30, step=5) if limit_change else None
extrapolation = st.sidebar.slider(
    "Max spend vs. highest observed (×)",
    1.0,
    5.0,
    2.0,
    step=0.25,
    help="The model is only reliable near spend levels you've actually run. 2× means no channel is pushed past "
    "twice its highest month. Override per channel with Max spend.",
)
show_ranges = st.sidebar.toggle(
    "Show confidence ranges",
    value=True,
    help=f"Re-runs the plan on {UNCERTAINTY_SAMPLES} plausible alternative curve fits.",
)

scenario = Scenario(
    mode=mode,
    goal=goal,
    budget=float(budget),
    target=float(target),
    max_change_pct=float(max_change) if max_change is not None else None,
    extrapolation_limit=float(extrapolation),
)


@st.cache_data(show_spinner=False)
def solve(history_df: pd.DataFrame, settings_df: pd.DataFrame, sc: Scenario):
    chans, _, _ = build_channels(history_df, settings_df)
    return optimize(chans, sc)


@st.cache_data(show_spinner=False)
def solve_uncertainty(history_df: pd.DataFrame, settings_df: pd.DataFrame, sc: Scenario):
    chans, _, _ = build_channels(history_df, settings_df)
    return plan_uncertainty(chans, sc, optimize(chans, sc), n_samples=UNCERTAINTY_SAMPLES)


with st.spinner("Optimizing…"):
    plan = solve(history, settings, scenario)
unc = None
if plan.feasible and show_ranges:
    with st.spinner("Computing confidence ranges…"):
        unc = solve_uncertainty(history, settings, scenario)

table = report.plan_table(channels, plan, unc, scenario)
tot = report.totals(channels, plan)
warnings_ = [i.message for i in data_issues if i.level == "warning"]
if warnings_:
    with st.expander(f"⚠️ Data notes ({len(warnings_)})"):
        for w in warnings_:
            st.markdown(f"- {w}")


# ==========================================
# 5. Charts
# ==========================================
def _layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6e1f5"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#1f183b"),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def allocation_chart() -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        y=table["Channel"],
        x=table["Current spend"],
        name="Current",
        orientation="h",
        marker=dict(color=MUTED, cornerradius=4),
        hovertemplate="%{y}<br>Current: $%{x:,.0f}<extra></extra>",
    )
    error = None
    if unc is not None:
        error = dict(
            type="data",
            symmetric=False,
            color="#e6e1f5",
            thickness=1.5,
            width=4,
            array=np.maximum(table["Range high"] - table["Recommended spend"], 0),
            arrayminus=np.maximum(table["Recommended spend"] - table["Range low"], 0),
        )
    fig.add_bar(
        y=table["Channel"],
        x=table["Recommended spend"],
        name="Recommended",
        orientation="h",
        marker=dict(color=PALETTE[0], cornerradius=4),
        error_x=error,
        hovertemplate="%{y}<br>Recommended: $%{x:,.0f}<extra></extra>",
    )
    fig = _layout(fig, 60 + 70 * len(table), barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(title="Monthly spend ($)", tickprefix="$", tickformat=",.0f")
    return fig


def _curve_segments(ch: Channel, x_max: float):
    xs = np.linspace(0, x_max, 160)
    edge = ch.model.max_observed + ch.fixed_cost
    ys = ch.leads(xs)
    inside = xs <= edge
    return xs, ys, inside


def curves_chart() -> go.Figure:
    fig = go.Figure()
    x_max = max(channel_bounds(ch, scenario)[1] for ch in channels) * 1.05
    for i, ch in enumerate(channels):
        color = PALETTE[i % len(PALETTE)]
        xs, ys, inside = _curve_segments(ch, x_max)
        fig.add_scatter(
            x=xs[inside],
            y=ys[inside],
            mode="lines",
            name=ch.name,
            legendgroup=ch.name,
            line=dict(color=color, width=2),
            hovertemplate=f"{ch.name}<br>$%{{x:,.0f}} → %{{y:,.0f}} leads<extra></extra>",
        )
        k = max(int(inside.sum()) - 1, 0)
        fig.add_scatter(
            x=xs[k:],
            y=ys[k:],
            mode="lines",
            legendgroup=ch.name,
            showlegend=False,
            line=dict(color=color, width=2, dash="dot"),
            hovertemplate=f"{ch.name} (extrapolated)<br>$%{{x:,.0f}} → %{{y:,.0f}} leads<extra></extra>",
        )
        fig.add_scatter(
            x=[ch.current_spend],
            y=[ch.leads(ch.current_spend)],
            mode="markers",
            legendgroup=ch.name,
            showlegend=False,
            marker=dict(size=10, color="#130f26", line=dict(width=2, color=color)),
            hovertemplate=f"{ch.name} current<br>$%{{x:,.0f}} → %{{y:,.0f}} leads<extra></extra>",
        )
        fig.add_scatter(
            x=[plan.budgets[i]],
            y=[plan.leads[i]],
            mode="markers",
            legendgroup=ch.name,
            showlegend=False,
            marker=dict(size=11, color=color, line=dict(width=2, color="#130f26")),
            hovertemplate=f"{ch.name} recommended<br>$%{{x:,.0f}} → %{{y:,.0f}} leads<extra></extra>",
        )
    fig = _layout(fig, 420)
    fig.update_xaxes(title="Monthly spend ($)", tickprefix="$", tickformat=",.0f")
    fig.update_yaxes(title="Leads per month", rangemode="tozero")
    return fig


def fit_chart() -> go.Figure:
    cols = 2
    rows = math.ceil(len(channels) / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[ch.name for ch in channels],
        horizontal_spacing=0.08,
        vertical_spacing=0.14 if rows > 1 else 0.1,
    )
    for i, ch in enumerate(channels):
        r, c = i // cols + 1, i % cols + 1
        color = PALETTE[i % len(PALETTE)]
        x_max = channel_bounds(ch, scenario)[1] * 1.05
        xs, ys, inside = _curve_segments(ch, max(x_max, ch.model.max_observed + ch.fixed_cost))
        k = max(int(inside.sum()) - 1, 0)
        fig.add_scatter(
            x=xs[inside],
            y=ys[inside],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            row=r,
            col=c,
            hovertemplate="$%{x:,.0f} → %{y:,.0f} (model)<extra></extra>",
        )
        fig.add_scatter(
            x=xs[k:],
            y=ys[k:],
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            showlegend=False,
            row=r,
            col=c,
            hovertemplate="$%{x:,.0f} → %{y:,.0f} (extrapolated)<extra></extra>",
        )
        fig.add_scatter(
            x=np.array(ch.model.spend) + ch.fixed_cost,
            y=ch.model.leads,
            mode="markers",
            marker=dict(size=8, color="#f0efec", line=dict(width=1.5, color=color)),
            showlegend=False,
            row=r,
            col=c,
            hovertemplate="$%{x:,.0f} → %{y:,.0f} (actual)<extra></extra>",
        )
    fig = _layout(fig, 280 * rows)
    fig.update_xaxes(tickprefix="$", tickformat="~s")
    fig.update_yaxes(rangemode="tozero")
    fig.update_annotations(font=dict(size=13, color="#e6e1f5"))
    return fig


def money(v):
    return "–" if v is None or pd.isna(v) else f"${v:,.0f}"


def styled_plan(df: pd.DataFrame):
    fmt = {
        "Current spend": money,
        "Recommended spend": money,
        "Change": lambda v: "$0" if abs(v) < 0.5 else f"{'+' if v > 0 else '−'}${abs(v):,.0f}",
        "Change %": lambda v: "new" if v is None or pd.isna(v) else ("0%" if abs(v) < 0.5 else f"{v:+.0f}%"),
        "Range low": money,
        "Range high": money,
        "Leads now": "{:,.0f}",
        "Leads after": "{:,.0f}",
        "CAC after": lambda v: "–" if v is None or pd.isna(v) else f"${v:,.2f}",
        "Leads per next $1K": "{:,.1f}",
        "Value after": money,
    }
    return df.style.format({k: v for k, v in fmt.items() if k in df.columns})


# ==========================================
# 6. Workspace tabs
# ==========================================
tab_plan, tab_fit, tab_scen, tab_ai = st.tabs(["📊 Plan", "🔬 Model fit", "🗂️ Scenarios", "✨ AI Advisor"])

with tab_plan:
    if not plan.feasible:
        st.error(f"**No feasible plan.** {plan.message}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Recommended spend",
            f"${tot['spend']:,.0f}",
            f"{tot['spend'] - tot['current_spend']:+,.0f} vs current",
            delta_color="off",
        )
        lead_help = (
            f"P10–P90 range across {unc.n_samples} alternative fits: {unc.leads_low:,.0f}–{unc.leads_high:,.0f}"
            if unc
            else None
        )
        c2.metric(
            "Expected leads",
            f"{tot['leads']:,.0f}",
            f"{tot['leads'] - tot['current_leads']:+,.0f} vs current",
            help=lead_help,
        )
        c3.metric(
            "Blended CAC",
            f"${tot['cac']:,.2f}",
            f"{tot['cac'] - tot['current_cac']:+,.2f} vs current",
            delta_color="inverse",
        )
        if goal == "value":
            c4.metric(
                "Pipeline value", f"${tot['value']:,.0f}", f"{tot['value'] - tot['current_value']:+,.0f} vs current"
            )
        elif unc is not None:
            c4.metric(
                "Likely lead range",
                f"{unc.leads_low:,.0f}–{unc.leads_high:,.0f}",
                help="80% of plausible model fits put this plan's results in this range.",
            )
        else:
            c4.metric("Current modeled leads", f"{tot['current_leads']:,.0f}")

        low = table.loc[table["Confidence"] == "Low", "Channel"].tolist()
        if low:
            st.warning(
                f"**Low confidence:** {', '.join(low)}. There isn't enough spend variation in the history to learn "
                "these curves, so a typical shape was assumed. Treat their recommendations as directional "
                "and consider a 4-week spend test to learn more.",
                icon="⚠️",
            )
        for msg in plan.issues:
            st.info(msg)

        st.subheader("Before → after")
        st.dataframe(styled_plan(table), hide_index=True, width="stretch")
        st.caption(
            "**Leads now** is the model's estimate at current spend. **Leads per next $1K** is the marginal "
            "return at the recommended spend: roughly equal across unconstrained channels means the split is "
            "balanced. **Limit** shows channels held back by a guardrail."
            + (" **Range** is the P10–P90 recommended spend across alternative fits." if unc else "")
        )

        col_a, col_b = st.columns([1, 1.4], gap="large")
        with col_a:
            st.subheader("Spend by channel")
            st.plotly_chart(allocation_chart(), width="stretch")
        with col_b:
            st.subheader("Response curves")
            st.plotly_chart(curves_chart(), width="stretch")
            st.caption("○ current spend · ● recommended · dotted = beyond your highest observed spend (less certain)")

        st.markdown("---")
        st.subheader("Save & export")
        record_name = st.text_input(
            "Scenario name",
            value=f"{mode_label} · {goal_label}" + (f" · ±{max_change}%" if max_change is not None else ""),
        )
        record = report.scenario_record(record_name, scenario, table, tot)
        e1, e2, e3, e4 = st.columns(4)
        if e1.button("💾 Save scenario", width="stretch"):
            st.session_state.setdefault("scenarios", []).append(record)
            st.toast(f"Saved “{record_name}”. Compare it in the Scenarios tab.")
        e2.download_button(
            "⬇️ Excel",
            report.to_excel(table, history, settings, record),
            "budget_plan.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
        e3.download_button("⬇️ CSV", table.to_csv(index=False), "budget_plan.csv", "text/csv", width="stretch")
        e4.download_button(
            "⬇️ Scenario JSON", report.to_json(record), "budget_scenario.json", "application/json", width="stretch"
        )

with tab_fit:
    st.subheader("How well each curve fits your history")
    fit_rows = []
    for ch in channels:
        m = ch.model
        chosen = settings.set_index("Channel").loc[ch.name, "Curve"]
        how = (
            "Default shape (not enough data)"
            if m.method == "anchored"
            else "Best fit (auto)"
            if chosen == AUTO
            else "Chosen manually"
        )
        fit_rows.append(
            {
                "Channel": ch.name,
                "Curve": m.spec.label,
                "How chosen": how,
                "Months": m.n_points,
                "R²": m.r2,
                "Confidence": m.confidence,
                "What it means": m.shape_summary(),
                "Note": m.note,
            }
        )
    st.dataframe(
        pd.DataFrame(fit_rows).style.format({"R²": lambda v: "–" if v is None or pd.isna(v) else f"{v:.2f}"}),
        hide_index=True,
        width="stretch",
    )
    st.plotly_chart(fit_chart(), width="stretch")
    st.caption(
        "Dots are your actual months; the line is the fitted curve; dotted = extrapolated beyond observed spend."
    )
    with st.expander("Curve comparison (AICc, lower is better)"):
        comp = pd.DataFrame({n: {CURVES[k].label: v for k, v in s.items()} for n, s in fit_scores.items() if s}).T
        if comp.empty:
            st.caption("Not enough varied history to compare curve families yet.")
        else:
            st.dataframe(comp.style.format(lambda v: "–" if v is None or pd.isna(v) else f"{v:.1f}"), width="stretch")
            st.caption(
                "Auto keeps the curve suggested by the channel's name unless another fits clearly better (AICc lower by more than 2)."
            )

with tab_scen:
    saved = st.session_state.setdefault("scenarios", [])
    with st.expander("Load a saved scenario file"):
        up = st.file_uploader("Scenario JSON", type=["json"], key="scenario_upload")
        if up is not None and st.button("Add to comparison"):
            try:
                loaded = json.load(up)
                saved.extend(loaded if isinstance(loaded, list) else [loaded])
                st.rerun()
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                st.error(f"Couldn't read that file: {exc}")
    if not saved:
        st.info("Save a plan from the **Plan** tab to compare scenarios side by side here.")
    else:
        try:
            st.subheader("Scenario comparison")
            st.dataframe(
                report.comparison_table(saved).style.format(
                    {
                        "Total spend": money,
                        "Expected leads": "{:,.0f}",
                        "Blended CAC": "${:,.2f}",
                        "Pipeline value": money,
                        "Max change %": lambda v: "–" if v is None or pd.isna(v) else f"±{v:.0f}%",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            st.subheader("Spend by channel")
            alloc = report.comparison_allocations(saved)
            st.dataframe(
                alloc.style.format({c: money for c in alloc.columns if c != "Channel"}),
                hide_index=True,
                width="stretch",
            )
        except (KeyError, TypeError, ValueError):
            st.error("One of the saved scenarios is in an unexpected format. Clear the list and try again.")
        s1, s2 = st.columns(2)
        s1.download_button(
            "⬇️ Download all (JSON)",
            json.dumps(saved, indent=2),
            "budget_scenarios.json",
            "application/json",
            width="stretch",
        )
        if s2.button("🗑️ Clear scenarios", width="stretch"):
            st.session_state["scenarios"] = []
            st.rerun()


def gemini_key() -> str | None:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:  # noqa: BLE001 - no secrets file configured
        key = None
    return key or os.environ.get("GEMINI_API_KEY")


with tab_ai:
    st.subheader("✨ AI Strategy Advisor")
    st.caption(
        "Ask about the shifts, risks or what to test next. Your channel history and the plan are sent to "
        f"Google Gemini ({GEMINI_MODEL}) with each question; don't use it with data you aren't allowed to share."
    )
    api_key = gemini_key()
    if not plan.feasible:
        st.info("Fix the plan in the Plan tab first; the advisor needs a feasible plan to discuss.")
    elif not api_key:
        st.info(
            "The advisor is off because no Gemini API key is configured. Add `GEMINI_API_KEY` to "
            "`.streamlit/secrets.toml` (or the app's Secrets in Streamlit Cloud) to turn it on."
        )
    else:
        if "messages" not in st.session_state or st.button("↺ New conversation"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "I've read your history and the recommended plan. Ask me why a channel "
                    "moved, how much to trust it, or what to test next.",
                }
            ]
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("e.g. Why did Meta get more budget? What should we test first?"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                contents = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model", parts=[types.Part.from_text(text=m["content"])]
                    )
                    for m in st.session_state.messages
                ]
                config = types.GenerateContentConfig(
                    system_instruction=build_system_prompt(history, table, tot, scenario, unc), temperature=0.5
                )
                with st.spinner("Thinking…"):
                    reply = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config).text
            except Exception as exc:  # noqa: BLE001 - surface any API error in the chat
                reply = f"🚨 The advisor couldn't respond: {exc}"
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
