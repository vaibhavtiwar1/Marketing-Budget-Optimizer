import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types

# ==========================================
# 1. Page Configuration & Custom SaaS Theme
# ==========================================
st.set_page_config(
    page_title="Enterprise Allocator", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS to force a high-end dark SaaS aesthetic
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #090a0f !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #f3f4f6 !important;
    }
    
    /* Remove default Streamlit header bar and footer */
    header, footer {
        visibility: hidden !important;
        height: 0 !important;
    }
    
    /* Navigation/Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0d0e15 !important;
        border-right: 1px solid #1f2232 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Clean Custom Cards */
    div[data-testid="stVerticalBlock"] > div[dir="ltr"] {
        gap: 1.5rem !important;
    }
    
    div.stMetric {
        background-color: #0d0e15 !important;
        border: 1px solid #1f2232 !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.03em !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #9ca3af !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Premium Buttons and Inputs */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important;
        box-shadow: 0 0 12px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Customizing Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem !important;
        background-color: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0d0e15 !important;
        border: 1px solid #1f2232 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        color: #9ca3af !important;
        font-weight: 500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f2232 !important;
        color: #ffffff !important;
        border-color: #6366f1 !important;
    }
    
    /* Data Editor container background adjustment */
    div[data-testid="stDataEditor"] {
        border: 1px solid #1f2232 !important;
        border-radius: 8px !important;
        background-color: #0d0e15 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- How-to-use Guide Modal ---
@st.dialog("📖 System Mechanics & Methodology")
def guide_modal():
    st.markdown("""
    This system replaces linear allocation models with non-linear diminishing return curves. 
    It runs continuous calculus simulations to identify the point where additional budget allocation yields sub-optimal returns.
    
    ### ⚙️ Operating Procedures
    1. **Baseline Inputs:** Use the sidebar to enter past spend data and actual conversions.
    2. **Targets:** Set the goal metric using the scale input fields.
    3. **Adjust Parameters:** Toggle controls to input fixed setup fees or algorithm requirements.
    4. **Analyze Output:** Review recommendations, channel spend distributions, and target trajectories.
    
    ### 📐 Mathematical Models
    * **Logarithmic:** Fast saturation ceiling (typically Search).
    * **Hill Function:** S-curve with initial friction followed by rapid scale (Paid Social).
    * **Negative Exponential:** Soft ceiling asymptotic to audience limit (Email).
    * **Power Law:** Compounding trajectory without absolute limits (SEO).
    * **Shifted Sigmoid:** Absolute minimum cost threshold required (Events).
    """)

# --- Top Header Workspace ---
header_col, button_col = st.columns([7.5, 2.5])
with header_col:
    st.markdown("<h1 style='margin:0; font-weight:800; font-size:2.5rem; letter-spacing:-0.04em;'>Media Allocation Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af; margin-top:0.25rem; font-size:1.1rem;'>Multi-model distribution solver powered by continuous curves and live context analysis.</p>", unsafe_allow_html=True)

with button_col:
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if st.button("System Documentation", use_container_width=True):
        guide_modal()

# ==========================================
# 2. Sidebar Layout
# ==========================================
st.sidebar.markdown("<h2 style='margin-bottom:1rem;'>Control Panel</h2>", unsafe_allow_html=True)
target_leads = st.sidebar.number_input("Target Conversions", min_value=10, max_value=10000, value=500, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3>Historical Performance</h3>", unsafe_allow_html=True)
st.sidebar.caption("Provide baseline figures for the allocation formulas.")

default_data = pd.DataFrame({
    "Channel": ["Google Paid Search", "Meta Paid Social", "HubSpot Email Blast", "SEO Blog Content"],
    "Past Spend (K)": [5.0, 4.0, 1.0, 3.0],
    "Past Leads": [250, 180, 95, 120]
})
edited_df = st.sidebar.data_editor(default_data, num_rows="dynamic", hide_index=True)

# ==========================================
# 3. Model Matching
# ==========================================
def guess_default_model(channel_name):
    name = str(channel_name).lower()
    if any(k in name for k in ["search", "google", "bing"]):
        return "Logarithmic (Search)"
    elif any(k in name for k in ["social", "meta", "tiktok", "linkedin", "instagram", "facebook"]):
        return "Hill Function (S-Curve)"
    elif any(k in name for k in ["email", "newsletter", "sms", "hubspot", "mailchimp"]):
        return "Negative Exponential (Email Ceiling)"
    elif any(k in name for k in ["seo", "content", "organic", "blog"]):
        return "Power Law (Compounding)"
    elif any(k in name for k in ["show", "event", "billboard", "pr", "sponsorship"]):
        return "Shifted Sigmoid (Threshold)"
    else:
        return "Logarithmic (Search)"

# ==========================================
# 4. Advanced Controls
# ==========================================
st.sidebar.markdown("---")
advanced_mode = st.sidebar.toggle("Advanced Model Tuning")

adv_settings = {}
model_options = [
    "Logarithmic (Search)", 
    "Hill Function (S-Curve)", 
    "Negative Exponential (Email Ceiling)", 
    "Power Law (Compounding)", 
    "Shifted Sigmoid (Threshold)"
]

for index, row in edited_df.iterrows():
    channel_name = row["Channel"]
    if pd.notna(channel_name) and str(channel_name).strip() != "":
        default_guessed = guess_default_model(channel_name)
        
        if advanced_mode:
            with st.sidebar.expander(f"⚙️ {channel_name}"):
                idx = model_options.index(default_guessed)
                model = st.selectbox("Formula Type", model_options, index=idx, key=f"mod_{channel_name}")
                fixed_cost = st.number_input("Platform Base Cost (K)", min_value=0.0, value=0.0, step=0.5, key=f"fix_{channel_name}")
                learning = st.checkbox("Apply Warmup Budget Floors", key=f"lrn_{channel_name}")
                adv_settings[channel_name] = {"model": model, "fixed_cost": fixed_cost, "learning": learning}
        else:
            adv_settings[channel_name] = {"model": default_guessed, "fixed_cost": 0.0, "learning": False}

# ==========================================
# 5. Core Math Engine
# ==========================================
channels = {}
for index, row in edited_df.iterrows():
    channel_name = row["Channel"]
    spend = float(row["Past Spend (K)"])
    leads = float(row["Past Leads"])
    
    if spend > 0 and leads > 0 and pd.notna(channel_name):
        settings = adv_settings[channel_name]
        var_spend = max(0.01, spend - settings["fixed_cost"])
        model_type = settings["model"]
        
        # Isolate baseline metrics based on curve types
        if "Logarithmic" in model_type:
            alpha = leads / np.log(var_spend + 1)
        elif "Hill Function" in model_type:
            alpha = leads / (var_spend**2 / (var_spend**2 + 16))
        elif "Negative Exponential" in model_type:
            alpha = leads / (1 - np.exp(-0.2 * var_spend))
        elif "Power Law" in model_type:
            alpha = leads / (var_spend**0.5)
        elif "Shifted Sigmoid" in model_type:
            alpha = leads / (1 / (1 + np.exp(-2 * (var_spend - 4))))
            
        channels[channel_name] = {"alpha": alpha, "settings": settings}

if len(channels) == 0:
    st.info("Provide active channel metrics in the control panel to view analysis.")
    st.stop()

# Diminishing return logic formulas
def lead_function(budget, channel_name):
    ch_data = channels[channel_name]
    alpha = ch_data["alpha"]
    model_type = ch_data["settings"]["model"]
    fixed_cost = ch_data["settings"]["fixed_cost"]
    
    eff_budget = max(0.0, budget - fixed_cost)
    if eff_budget == 0:
        return 0.0
        
    if "Logarithmic" in model_type:
        return alpha * np.log(eff_budget + 1)
    elif "Hill Function" in model_type:
        return alpha * (eff_budget**2 / (eff_budget**2 + 16))
    elif "Negative Exponential" in model_type:
        return alpha * (1 - np.exp(-0.2 * eff_budget))
    elif "Power Law" in model_type:
        return alpha * (eff_budget**0.5)
    elif "Shifted Sigmoid" in model_type:
        return alpha * (1 / (1 + np.exp(-2 * (eff_budget - 4))))

# Solver Constraints
def objective_function(budgets):
    return np.sum(budgets)

def constraint_target_leads(budgets):
    total_leads = sum(lead_function(b, ch) for b, ch in zip(budgets, channels.keys()))
    return total_leads - target_leads

init_guess = []
bounds = []
for ch_name, ch_data in channels.items():
    settings = ch_data["settings"]
    min_bound = settings["fixed_cost"]
    if settings["learning"]:
        min_bound += 2.0  # Apply absolute bottom limits for stability
    init_guess.append(max(1.0, min_bound))
    bounds.append((min_bound, 1000.0))

constraints = {'type': 'ineq', 'fun': constraint_target_leads}

# Limit boundaries validation
theoretical_max_leads = sum(lead_function(1000.0, ch) for ch in channels.keys())
if target_leads > theoretical_max_leads:
    st.error(f"Target exceeds maximum system output. The system caps at {int(theoretical_max_leads)} conversions. Lower your targets or improve historical performance metrics.")
    st.stop()

result = minimize(objective_function, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# ==========================================
# 6. Output Panel
# ==========================================
if result.success:
    optimized_budgets = result.x
    channel_leads = [lead_function(b, ch) for b, ch in zip(optimized_budgets, channels.keys())]
    total_spend = np.sum(optimized_budgets)
    blended_cac = total_spend / target_leads

    # Display Metrics Grid
    col1, col2, col3 = st.columns(3)
    col1.metric("Calculated Budget", f"${total_spend:,.2f} K")
    col2.metric("Expected Conversions", f"{int(round(sum(channel_leads)))}")
    col3.metric("System CAC", f"${blended_cac:,.2f} K")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Workspace Tabs
    tab1, tab2 = st.tabs(["📊 Performance Matrix", "💬 Strategy Assistant"])
    
    with tab1:
        df_results = pd.DataFrame({
            "Channel": list(channels.keys()),
            "Curve": [channels[ch]["settings"]["model"].split(" ")[0] for ch in channels.keys()],
            "Allocation (K)": optimized_budgets,
            "Target Yield": channel_leads,
            "Unit Cost (K)": [b/l if l > 0 else 0 for b, l in zip(optimized_budgets, channel_leads)]
        })
        
        # Allocations and Chart Breakouts
        table_col, chart_col = st.columns([1.3, 1], gap="large")
        
        with table_col:
            st.markdown("<h3 style='margin-bottom:1rem;'>Calculated Allocation Matrix</h3>", unsafe_allow_html=True)
            styled_df = df_results.style.format({
                "Allocation (K)": "${:,.2f}",
                "Target Yield": "{:.1f}",
                "Unit Cost (K)": "${:,.2f}"
            })
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
            
        with chart_col:
            st.markdown("<h3 style='margin-bottom:1rem;'>Distribution Share</h3>", unsafe_allow_html=True)
            fig_donut = px.pie(
                df_results, 
                values="Allocation (K)", 
                names="Channel", 
                hole=0.6,
                color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
            )
            fig_donut.update_traces(
                textposition='inside', 
                textinfo='percent', 
                showlegend=True,
                marker=dict(line=dict(color='#090a0f', width=2))
            )
            fig_donut.update_layout(
                margin=dict(t=0, b=0, l=0, r=0), 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#f3f4f6"))
            )
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")
        st.markdown("<h3 style='margin-bottom:0.5rem;'>Model Saturation Curves</h3>", unsafe_allow_html=True)
        st.caption("Visual representation of curve trajectories and diminishing returns.")
        
        fig_curve = go.Figure()
        max_b = max(optimized_budgets) * 1.6 if max(optimized_budgets) > 0 else 15
        budget_range = np.linspace(0, max_b, 150)
        
        colors = ["#6366f1", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"]
        for i, ch_name in enumerate(channels.keys()):
            opt_b = optimized_budgets[i]
            opt_l = channel_leads[i]
            color = colors[i % len(colors)]
            
            fig_curve.add_trace(go.Scatter(
                x=budget_range, 
                y=[lead_function(b, ch_name) for b in budget_range], 
                mode='lines', 
                name=ch_name, 
                line=dict(width=3, color=color)
            ))
            fig_curve.add_trace(go.Scatter(
                x=[opt_b], 
                y=[opt_l], 
                mode='markers', 
                marker=dict(size=12, color='#ffffff', line=dict(width=2, color=color)), 
                showlegend=False, 
                hoverinfo="skip"
            ))

        fig_curve.update_layout(
            xaxis=dict(title="Budget (K)", gridcolor="#1f2232", titlefont=dict(color="#9ca3af"), tickfont=dict(color="#9ca3af")),
            yaxis=dict(title="Yield (Conversions)", gridcolor="#1f2232", titlefont=dict(color="#9ca3af"), tickfont=dict(color="#9ca3af")),
            hovermode="x unified", 
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(color="#f3f4f6"))
        )
        st.plotly_chart(fig_curve, use_container_width=True)

    with tab2:
        st.markdown("<h3 style='margin-bottom:0.5rem;'>CMO Partner Agent</h3>", unsafe_allow_html=True)
        st.caption("Active model analysis based on your current budget outputs.")
        
        client = genai.Client(api_key=st.secrets.get("GEMINI_API_KEY", ""))
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant", 
                "content": "Allocation results evaluated. I have reviewed the mathematical distribution profiles. Let me know what operational changes or budget shifts we should challenge."
            }]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about allocation distribution..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            dashboard_context = f"""
            You are a rigorous, highly-paid Fractional CMO.
            The user is viewing future allocation recommendations from a non-linear calculus engine.
            
            BASELINE:
            {edited_df.to_string(index=False)}
            
            RECOMMENDATION:
            Total Budget: ${total_spend:,.2f}K
            Target Goal: {int(round(sum(channel_leads)))} conversions
            Calculated Blended Unit Cost: ${blended_cac:,.2f}K
            
            PROPOSAL DETAIL:
            {df_results.to_string()}
            
            Guidelines:
            1. Compare past baseline directly to the recommended matrix. Call out exact changes.
            2. Challenge any operational issues indicated by the changes.
            3. End with a singular, high-friction diagnostic question about operational context.
            4. Keep answers short. 3 paragraphs max. Do not sound like a template.
            """
            
            api_messages = []
            for msg in st.session_state.messages:
                role = "user" if msg["role"] == "user" else "model"
                api_messages.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
                
            config = types.GenerateContentConfig(system_instruction=dashboard_context, temperature=0.6)
            
            with st.spinner("Analyzing allocation profiles..."):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=api_messages,
                        config=config
                    )
                    ai_reply = response.text
                except Exception as e:
                    ai_reply = f"System Error: Connection issue. Check your API configurations."
            
            with st.chat_message("assistant"):
                st.markdown(ai_reply)
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})

else:
    st.error("The solver could not find a convergent solution. Adjust target inputs or lower base cost barriers.")
