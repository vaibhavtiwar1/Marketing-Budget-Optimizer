# 📊 Marketing Budget Optimizer

> Learns each channel's diminishing returns from your history and finds the budget split where the next dollar works hardest everywhere.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)

**Live app:** https://vaibhavtiwar1-marketing-budget-optimizer-app-srt8k5.streamlit.app/

---

## 🧠 About

Most marketing calculators assume returns are linear (*"$100 gets 10 leads, so $10,000 gets 1,000"*). Real channels
saturate, need learning budgets, or don't work at all below a minimum spend. This app fits a **response curve per
channel to your own monthly history**, then uses constrained optimization to find the best split, with ranges that
show how sure the model is.

### ✨ Features
* **Learns from your history.** Upload a CSV (or edit the table) with one row per channel per month. With 4+ distinct
  spend levels, both the height *and* the shape of each curve are fitted; the best curve family is picked by AICc.
* **Honest about confidence.** Every channel is rated High / Medium / Low. Plans come with P10–P90 ranges from
  re-running the optimization on bootstrapped alternative fits.
* **Two goals.** *Spend a fixed budget* as well as possible, or *hit a target* at the lowest cost.
* **Optimize for value, not just volume.** Set a pipeline value per lead for each channel (close rate × deal size).
* **Real-world guardrails.** Per-channel min/max spend, lock a channel, fixed costs, a cap on change vs. current
  spend, and a cap on how far beyond observed spend the model may extrapolate.
* **Before → after view.** Current vs. recommended spend, expected leads, CAC and the marginal return of the next $1K.
* **Clear failure messages.** Infeasible plans say which channel or limit is the problem.
* **Scenarios & export.** Save and compare scenarios, export to Excel, CSV or JSON.
* **AI Strategy Advisor.** A Gemini 2.5 Flash chat that reads the plan *and* its confidence, so it can say when a
  recommendation is shaky. Optional; the app works without an API key.

---

## 🧮 Curve families

| Curve | Typical channels | Behavior |
|---|---|---|
| **Fast saturation** (logarithmic) | Search, retargeting | Efficient first dollars, flattens quickly |
| **S-curve** (Hill) | Paid social, video, display | Slow learning phase, then scales, then flattens |
| **Hard ceiling** (negative exponential) | Email, SMS | Near-linear until the audience runs out |
| **Steady diminishing returns** (power law) | SEO, content | Shrinking returns with no hard ceiling |
| **Needs minimum spend** (threshold sigmoid) | Events, PR, sponsorships | Little or nothing below a minimum spend |

On *Auto*, the app starts from the family suggested by the channel's name and switches only if another fits the data
clearly better. With too little data it keeps the suggested family, uses a default shape scaled to the channel's spend,
and marks it Low confidence.

---

## 📄 Data format

```csv
Channel,Period,Spend,Leads
Google Paid Search,2026-01,5000,250
Google Paid Search,2026-02,6200,282
Meta Paid Social,2026-01,4000,180
```

Column names are flexible (`Date`/`Month`, `Cost`/`Spend ($)`, `Conversions` etc.), and `$1,000`-style values are
accepted. The latest period is used as current spend. Bad rows are dropped with an explanation.

---

## 🛠️ Run locally

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

To enable the AI advisor, add your key to `.streamlit/secrets.toml` (not committed) or set it as an environment variable:

```toml
GEMINI_API_KEY = "your-key"
```

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
```

```bash
pytest -q && ruff check . && ruff format --check .
```

CI runs the same checks on every push.

## 🗂️ Project layout

```
app.py                  Streamlit UI
budget_engine/
  curves.py             curve families, fitting, model selection, uncertainty sampling
  optimize.py           constraints, multi-start optimizer, confidence ranges
  data.py               CSV cleaning/validation, current spend, sample data
  report.py             before/after table, scenario records, Excel/CSV/JSON export
  advisor.py            AI advisor system prompt
tests/                  unit tests + headless Streamlit app tests
sandbox.py              earlier UI experiment (not used by the app)
```
