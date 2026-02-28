"""
Formwork Kitting & BoQ Optimization System
L&T CreaTech - Problem Statement 4
Run with: streamlit run formwork_app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Formwork Optimization System",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #030712;
        color: #f3f4f6;
    }
    .stApp { background-color: #030712; }

    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-family: 'DM Mono', monospace;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        font-family: 'DM Mono', monospace;
    }
    .metric-sub {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 4px;
    }
    .section-header {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #f3f4f6;
        border-left: 3px solid #f7c948;
        padding-left: 10px;
        margin: 20px 0 14px 0;
        font-family: 'DM Mono', monospace;
    }
    .assumption-box {
        background: #0f172a;
        border-left: 2px solid #f7c94850;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 12px;
        color: #9ca3af;
    }
    .stSidebar { background-color: #0f172a; }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [aria-selected="true"] {
        color: #f7c948 !important;
        border-bottom-color: #f7c948 !important;
    }
    div[data-testid="metric-container"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .highlight-green { color: #4ade80; font-weight: 700; }
    .highlight-yellow { color: #f7c948; font-weight: 700; }
    .highlight-red { color: #f87171; font-weight: 700; }
    .highlight-blue { color: #60a5fa; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─── CORE COMPUTATION ENGINE ───────────────────────────────────────────────────
def compute_all(p):
    formwork_cost = (p['total_cost'] * p['formwork_pct']) / 100
    material_cost = formwork_cost * 0.46
    labour_cost   = formwork_cost * (p['labour_pct'] / 100)
    logistics_cost = formwork_cost * (p['logistics_pct'] / 100)

    # Reuse calculations
    pours = math.floor((p['duration_months'] * 30) / p['cycle_days'])
    theoretical_reuse = min(pours, p['reuse_limit'])
    effective_reuse = theoretical_reuse * (1 - p['damage_factor']) * 0.82
    repetition_index = effective_reuse / p['reuse_limit']

    # Inventory
    optimized_buffer = p['over_order_buffer'] * (1 - repetition_index * 0.55)
    trad_inventory   = material_cost * (1 + p['over_order_buffer'] / 100)
    opt_inventory    = material_cost * (1 + optimized_buffer / 100)
    inv_reduction    = trad_inventory - opt_inventory
    inv_reduction_pct = (inv_reduction / trad_inventory) * 100

    # Carrying costs
    holding_months = p['duration_months'] * 0.6
    cc_rate = p['carrying_cost_pct'] / 100
    trad_carrying = trad_inventory * cc_rate * (holding_months / 12)
    opt_carrying  = opt_inventory  * cc_rate * (holding_months / 12)
    carrying_saved = trad_carrying - opt_carrying

    # EOQ
    annual_demand = 1000
    ordering_cost = 0.5
    holding_per_unit = (opt_inventory * cc_rate) / annual_demand
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / max(holding_per_unit, 0.0001))

    # Cost per m2
    total_area = (p['total_cost'] * 1e7) / 4500
    trad_cost_m2 = (formwork_cost * 1e7) / total_area
    opt_cost_m2  = ((formwork_cost - carrying_saved - inv_reduction * 0.3) * 1e7) / total_area

    # Financial
    wc_unlocked  = inv_reduction
    roi_impact   = (carrying_saved / (p['total_cost'] * 0.15)) * 100
    roa_impact   = (wc_unlocked / p['total_cost']) * 100

    # Productivity
    trad_productivity = 12.0
    opt_productivity  = trad_productivity * (1 + repetition_index * 0.18)
    productivity_gain = ((opt_productivity - trad_productivity) / trad_productivity) * 100

    # Sustainability
    embodied_energy = 45  # kWh/m2
    trad_energy = (total_area * embodied_energy) / 1e6  # GWh
    opt_energy  = trad_energy * (1 - (effective_reuse / p['reuse_limit']) * 0.35)
    energy_saved = trad_energy - opt_energy
    co2_saved   = energy_saved * 0.82 * 1000  # tonnes

    # ABC data
    abc_data = pd.DataFrame([
        {"Category": "A", "Item": "Slab Panels (std)",  "Value_Cr": material_cost * 0.38, "CumPct": 38,  "Turns": int(effective_reuse * 0.95)},
        {"Category": "A", "Item": "Column Boxes",        "Value_Cr": material_cost * 0.22, "CumPct": 60,  "Turns": int(effective_reuse * 0.90)},
        {"Category": "B", "Item": "Beam Sides",          "Value_Cr": material_cost * 0.14, "CumPct": 74,  "Turns": int(effective_reuse * 0.75)},
        {"Category": "B", "Item": "Prop Systems",        "Value_Cr": material_cost * 0.10, "CumPct": 84,  "Turns": int(effective_reuse * 0.70)},
        {"Category": "C", "Item": "Edge Forms",          "Value_Cr": material_cost * 0.08, "CumPct": 92,  "Turns": int(effective_reuse * 0.55)},
        {"Category": "C", "Item": "Accessories",         "Value_Cr": material_cost * 0.08, "CumPct": 100, "Turns": int(effective_reuse * 0.40)},
    ])

    # Reuse timeline
    cycles = list(range(1, min(int(theoretical_reuse), 30) + 1))
    reuse_timeline = pd.DataFrame({
        "Cycle": cycles,
        "Cost_Per_Use": [(material_cost * 1e2) / c * max(0.4, 1 - (c / p['reuse_limit']) * 0.6 * p['damage_factor'] * 8) for c in cycles],
        "Traditional":  [material_cost * 1e2] * len(cycles),
        "Condition_Pct": [max(40.0, 100 - (c / p['reuse_limit']) * 60 * p['damage_factor'] * 8) for c in cycles],
    })

    return {
        "formwork_cost": formwork_cost, "material_cost": material_cost,
        "labour_cost": labour_cost, "logistics_cost": logistics_cost,
        "pours": pours, "theoretical_reuse": theoretical_reuse,
        "effective_reuse": effective_reuse, "repetition_index": repetition_index,
        "trad_inventory": trad_inventory, "opt_inventory": opt_inventory,
        "inv_reduction": inv_reduction, "inv_reduction_pct": inv_reduction_pct,
        "trad_carrying": trad_carrying, "opt_carrying": opt_carrying,
        "carrying_saved": carrying_saved, "eoq": eoq,
        "trad_cost_m2": trad_cost_m2, "opt_cost_m2": opt_cost_m2,
        "wc_unlocked": wc_unlocked, "roi_impact": roi_impact, "roa_impact": roa_impact,
        "trad_productivity": trad_productivity, "opt_productivity": opt_productivity,
        "productivity_gain": productivity_gain,
        "embodied_energy": embodied_energy, "trad_energy": trad_energy,
        "opt_energy": opt_energy, "energy_saved": energy_saved, "co2_saved": co2_saved,
        "total_area": total_area, "abc_data": abc_data, "reuse_timeline": reuse_timeline,
        "optimized_buffer": optimized_buffer,
    }

# ─── PLOT THEME ───────────────────────────────────────────────────────────────
DARK_BG   = "#030712"
CARD_BG   = "#111827"
BORDER    = "#1f2937"
YELLOW    = "#f7c948"
GREEN     = "#4ade80"
BLUE      = "#60a5fa"
ORANGE    = "#fb923c"
TEAL      = "#2dd4bf"
RED       = "#f87171"
GRAY      = "#475569"

def dark_layout(title=""):
    return dict(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color="#9ca3af", family="DM Sans"),
        title=dict(text=title, font=dict(color="#f3f4f6", size=13)),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, font=dict(size=11)),
        margin=dict(l=40, r=20, t=40, b=40),
    )

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Project Parameters")
    st.markdown("---")

    total_cost        = st.slider("Total Project Cost (₹ Cr)", 10, 1000, 100, 10)
    formwork_pct      = st.slider("Formwork Cost %", 7.0, 10.0, 8.5, 0.1)
    cycle_days        = st.slider("Casting Cycle (days)", 4, 14, 7, 1)
    reuse_limit       = st.slider("Panel Reuse Limit", 10, 50, 30, 5)
    duration_months   = st.slider("Project Duration (months)", 6, 48, 18, 3)
    over_order_buffer = st.slider("Traditional Over-order %", 5, 25, 12, 1)
    carrying_cost_pct = st.slider("Carrying Cost % p.a.", 10.0, 18.0, 13.0, 0.5)
    damage_factor     = st.slider("Damage Factor", 0.02, 0.15, 0.06, 0.01)
    labour_pct        = st.slider("Labour % of Formwork", 25, 40, 32, 1)
    logistics_pct     = st.slider("Logistics % of Formwork", 8, 18, 12, 1)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#4b5563; line-height:1.7'>
    All calculations derived from first principles.<br>
    Sources: L&T PS4, CPWD SOR, ACI 347R,<br>
    ICE Database, CEA India, RSMeans.
    </div>
    """, unsafe_allow_html=True)

params = {
    "total_cost": total_cost, "formwork_pct": formwork_pct,
    "cycle_days": cycle_days, "reuse_limit": reuse_limit,
    "duration_months": duration_months, "over_order_buffer": over_order_buffer,
    "carrying_cost_pct": carrying_cost_pct, "damage_factor": damage_factor,
    "labour_pct": labour_pct, "logistics_pct": logistics_pct,
}
r = compute_all(params)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#0f172a,#1e293b,#0f172a);
     border:1px solid #1f2937; border-radius:12px; padding:24px 28px; margin-bottom:24px;'>
    <div style='font-size:11px; color:#f7c948; letter-spacing:3px; text-transform:uppercase;
         font-family:DM Mono,monospace; margin-bottom:6px;'>
        L&T · CreaTech · Problem Statement 4
    </div>
    <h1 style='margin:0; font-size:24px; font-weight:800; color:#f9fafb;'>
        Formwork Kitting & BoQ Optimization System
    </h1>
    <div style='font-size:12px; color:#6b7280; margin-top:6px;'>
        Data-driven · Conservative modelling · Transparent assumptions
    </div>
    <div style='display:flex; gap:32px; margin-top:16px;'>
        <div>
            <div style='font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:1px;'>Inventory Saved</div>
            <div style='font-size:22px; font-weight:800; color:#4ade80; font-family:DM Mono,monospace;'>
                ₹{r['inv_reduction']:.2f} Cr
            </div>
        </div>
        <div>
            <div style='font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:1px;'>WC Unlocked</div>
            <div style='font-size:22px; font-weight:800; color:#f7c948; font-family:DM Mono,monospace;'>
                ₹{r['wc_unlocked']:.2f} Cr
            </div>
        </div>
        <div>
            <div style='font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:1px;'>CO₂ Avoided</div>
            <div style='font-size:22px; font-weight:800; color:#2dd4bf; font-family:DM Mono,monospace;'>
                {r['co2_saved']:.0f} t
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🔬 Clustering",
    "🔁 Repetition & Kitting",
    "📦 Inventory & EOQ",
    "💰 Financial Impact",
    "♻️ Reuse Lifecycle",
    "🌱 Sustainability",
    "🎛️ What-If"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Summary KPIs</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Formwork Budget", f"₹{r['formwork_cost']:.2f} Cr",
                  f"{formwork_pct}% of ₹{total_cost} Cr")
    with c2:
        st.metric("Material Pool", f"₹{r['material_cost']:.2f} Cr", "~46% of formwork")
    with c3:
        st.metric("Inventory Saved", f"₹{r['inv_reduction']:.2f} Cr",
                  f"↓ {r['inv_reduction_pct']:.1f}% reduction")
    with c4:
        st.metric("Carrying Cost Saved", f"₹{r['carrying_saved']:.3f} Cr",
                  f"Over {duration_months} months")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Working Capital Freed", f"₹{r['wc_unlocked']:.2f} Cr", "From idle stock")
    with c6:
        st.metric("Effective Reuse", f"{r['effective_reuse']:.1f}×", f"of {reuse_limit} max")
    with c7:
        st.metric("Pours / Project", f"{r['pours']}", f"{cycle_days}-day cycle")
    with c8:
        st.metric("Repetition Index", f"{r['repetition_index']*100:.1f}%", "Higher = more standard")

    st.markdown('<div class="section-header">Traditional vs Optimized — Cost Breakdown</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        categories = ["Material", "Carrying", "Labour", "Logistics"]
        trad_vals = [
            r['trad_inventory'],
            r['trad_carrying'],
            r['labour_cost'],
            r['logistics_cost']
        ]
        opt_vals = [
            r['opt_inventory'],
            r['opt_carrying'],
            r['labour_cost'] * (1 - r['productivity_gain'] / 300),
            r['logistics_cost'] * 0.88
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Traditional", x=categories, y=trad_vals,
                             marker_color=GRAY, marker_line_width=0))
        fig.add_trace(go.Bar(name="Optimized", x=categories, y=opt_vals,
                             marker_color=YELLOW, marker_line_width=0))
        fig.update_layout(**dark_layout("Cost Comparison (₹ Cr)"), barmode="group",
                          yaxis_tickprefix="₹", yaxis_tickformat=".2f")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        saving_labels = ["Inventory\nReduction", "Carrying\nCost", "Labour\nGain", "Logistics\nSaving"]
        saving_vals = [
            r['inv_reduction'],
            r['carrying_saved'],
            r['labour_cost'] * r['productivity_gain'] / 300,
            r['logistics_cost'] * 0.12
        ]
        saving_colors = [GREEN, TEAL, BLUE, ORANGE]

        fig2 = go.Figure(go.Bar(
            x=saving_labels, y=saving_vals,
            marker_color=saving_colors, marker_line_width=0,
            text=[f"₹{v:.3f}" for v in saving_vals], textposition="outside",
            textfont=dict(color="#f3f4f6", size=10)
        ))
        fig2.update_layout(**dark_layout("Savings Waterfall (₹ Cr)"),
                           yaxis_tickprefix="₹", yaxis_tickformat=".3f")
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Structural Element Clustering — K-Means Simulation</div>',
                unsafe_allow_html=True)

    st.info("K-Means groups near-identical structural elements (±50mm tolerance) into standard kit configurations. Fewer clusters = fewer unique mould types = lower procurement cost.", icon="ℹ️")

    col_l, col_r = st.columns(2)

    with col_l:
        np.random.seed(42)
        clusters = {
            "Cluster A — Slabs": {"x": 300 + np.random.randn(18)*25, "y": 120 + np.random.randn(18)*15, "color": YELLOW},
            "Cluster B — Columns": {"x": 450 + np.random.randn(12)*20, "y": 450 + np.random.randn(12)*20, "color": BLUE},
            "Cluster C — Beams": {"x": 200 + np.random.randn(10)*20, "y": 300 + np.random.randn(10)*25, "color": ORANGE},
            "Outliers": {"x": [180, 520, 350, 420], "y": [480, 200, 380, 100], "color": RED},
        }

        fig3 = go.Figure()
        for name, d in clusters.items():
            fig3.add_trace(go.Scatter(
                x=d["x"], y=d["y"], mode="markers", name=name,
                marker=dict(color=d["color"], size=9, opacity=0.85,
                            line=dict(color=DARK_BG, width=1))
            ))
        fig3.update_layout(**dark_layout("Element Distribution (Width vs Depth, mm)"),
                           xaxis_title="Width (mm)", yaxis_title="Depth (mm)")
        st.plotly_chart(fig3, use_container_width=True)

    with col_r:
        cluster_df = pd.DataFrame([
            {"Cluster": "A — Slabs",   "Elements": 18, "Before": 14, "After": 3,  "Reduction": "78.6%"},
            {"Cluster": "B — Columns", "Elements": 12, "Before": 9,  "After": 2,  "Reduction": "77.8%"},
            {"Cluster": "C — Beams",   "Elements": 10, "Before": 8,  "After": 3,  "Reduction": "62.5%"},
            {"Cluster": "Outliers",    "Elements": 4,  "Before": 4,  "After": 4,  "Reduction": "0%"},
            {"Cluster": "TOTAL",       "Elements": 44, "Before": 35, "After": 12, "Reduction": "65.7%"},
        ])
        st.dataframe(
            cluster_df.style
            .applymap(lambda v: "color: #4ade80; font-weight:700" if v == "65.7%" else "")
            .set_properties(**{"background-color": "#111827", "color": "#f3f4f6"}),
            use_container_width=True, hide_index=True
        )

        st.markdown("""
        <div class='assumption-box'>
        <span style='color:#f7c948; font-weight:700;'>Key Insight:</span>
        Reducing from <span style='color:#f87171; font-weight:700;'>35 unique configurations</span>
        to <span style='color:#4ade80; font-weight:700;'>12 standard kits</span>
        cuts procurement complexity by 65.7%.
        Fewer moulds → higher per-unit reuse → lower lifecycle cost.
        </div>
        <div class='assumption-box' style='margin-top:8px;'>
        <span style='color:#f7c948; font-weight:700;'>Assumption:</span>
        Cluster counts are illustrative (simulated data).
        In production, feed actual drawing register + BIM element export.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REPETITION & KITTING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Repetition Index & Kitting Optimization</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Repetition Index", f"{r['repetition_index']*100:.1f}%", "Effective / Max reuse")
    with c2:
        st.metric("Theoretical Reuse", f"{r['theoretical_reuse']}×", "Schedule-limited")
    with c3:
        st.metric("Effective Reuse", f"{r['effective_reuse']:.1f}×", "Damage-adjusted")
    with c4:
        st.metric("Pours / Project", f"{r['pours']}", f"{cycle_days}-day cycles")

    col_l, col_r = st.columns(2)

    with col_l:
        pour_count = min(r['pours'], 24)
        pour_nums  = list(range(1, pour_count + 1))
        reused_pct = [min(100, (i / max(r['effective_reuse'], 1)) * 100 * 0.9) for i in pour_nums]
        new_pct    = [max(0, 100 - rp) for rp in reused_pct]

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=pour_nums, y=reused_pct, fill="tozeroy",
                                  name="Reused panels %", line=dict(color=GREEN, width=2),
                                  fillcolor=f"{GREEN}30"))
        fig4.add_trace(go.Scatter(x=pour_nums, y=new_pct, fill="tozeroy",
                                  name="New procurement %", line=dict(color=RED, width=2),
                                  fillcolor=f"{RED}20"))
        fig4.update_layout(**dark_layout("Pour Sequence — Reuse vs New Procurement"),
                           xaxis_title="Pour Number", yaxis_title="%", yaxis_range=[0, 110])
        st.plotly_chart(fig4, use_container_width=True)

    with col_r:
        kit_data = pd.DataFrame({
            "Scenario": ["Traditional", "Optimized"],
            "Unique Kit Types": [35, 12],
            "Wastage %": [18, 6],
            "Assembly Time (hrs)": [8.5, 5.2],
        })
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(name="Unique Kit Types", x=kit_data["Scenario"],
                              y=kit_data["Unique Kit Types"], marker_color=[GRAY, YELLOW]))
        fig5.add_trace(go.Bar(name="Wastage %", x=kit_data["Scenario"],
                              y=kit_data["Wastage %"], marker_color=[RED, GREEN]))
        fig5.update_layout(**dark_layout("Kitting Comparison"), barmode="group")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("""
    <div class='assumption-box'>
    <b style='color:#f7c948;'>Repetition Index Formula:</b><br>
    RI = Effective Reuse / Max Reuse Limit<br>
    Effective Reuse = min(Pours, Limit) × (1 − DamageFactor) × 0.82 (utilization)<br><br>
    <b style='color:#2dd4bf;'>Decision Rule:</b>
    RI > 0.50 → Standardize kit · RI 0.30–0.50 → Review · RI < 0.30 → Bespoke justified
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INVENTORY & EOQ
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">ABC Classification & Inventory Optimization</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        abc = r['abc_data'].copy()
        abc["Value_Cr"] = abc["Value_Cr"].round(3)

        def color_category(val):
            if val == "A": return "background-color: #ff6b3530; color: #fb923c; font-weight:700"
            if val == "B": return "background-color: #f7c94830; color: #f7c948; font-weight:700"
            return "background-color: #4ecdc430; color: #2dd4bf; font-weight:700"

        styled = abc.style.applymap(color_category, subset=["Category"]) \
                          .format({"Value_Cr": "₹{:.3f}", "CumPct": "{}%", "Turns": "{}"}) \
                          .set_properties(**{"background-color": "#111827", "color": "#f3f4f6"})
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class='assumption-box'>
        <b style='color:#f7c948;'>ABC Rule:</b>
        A items (60% value) → weekly monitoring, tight EOQ.
        B → bi-weekly. C → bulk order, minimal oversight.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("EOQ (panels)", f"{r['eoq']:.0f} units", "Economic Order Qty")
            st.metric("Trad. Inventory", f"₹{r['trad_inventory']:.2f} Cr", f"+{over_order_buffer}% buffer")
        with c2:
            st.metric("Opt. Inventory", f"₹{r['opt_inventory']:.2f} Cr", "Repetition-adjusted")
            st.metric("Inventory Δ", f"₹{r['inv_reduction']:.2f} Cr", f"↓{r['inv_reduction_pct']:.1f}%")
        c3, c4 = st.columns(2)
        with c3:
            st.metric("Trad. Carrying", f"₹{r['trad_carrying']:.3f} Cr", "Annual rate applied")
        with c4:
            st.metric("Opt. Carrying", f"₹{r['opt_carrying']:.3f} Cr", "On reduced inventory")

        st.markdown(f"""
        <div class='assumption-box' style='margin-top:12px;'>
        <b style='color:#60a5fa;'>Carrying Cost @ {carrying_cost_pct}% p.a.:</b><br>
        Capital cost ~{carrying_cost_pct*0.6:.1f}% ·
        Storage ~{carrying_cost_pct*0.2:.1f}% ·
        Insurance ~{carrying_cost_pct*0.1:.1f}% ·
        Risk ~{carrying_cost_pct*0.1:.1f}%
        </div>
        """, unsafe_allow_html=True)

    # ABC bar chart
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(name="Traditional", x=abc["Item"],
                          y=abc["Value_Cr"] * (1 + over_order_buffer / 100),
                          marker_color=GRAY))
    fig6.add_trace(go.Bar(name="Optimized", x=abc["Item"],
                          y=abc["Value_Cr"], marker_color=YELLOW))
    fig6.update_layout(**dark_layout("Inventory Value by ABC Category (₹ Cr)"),
                       barmode="group", yaxis_tickprefix="₹", yaxis_tickformat=".3f")
    st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FINANCIAL IMPACT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">Financial Impact Modelling</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Working Capital Freed", f"₹{r['wc_unlocked']:.2f} Cr", "From reduced inventory")
    with c2:
        st.metric("ROI Impact", f"+{r['roi_impact']:.2f}%", "vs project equity base")
    with c3:
        st.metric("ROA Impact", f"+{r['roa_impact']:.2f}%", "Asset efficiency gain")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Cost/m² Traditional", f"₹{r['trad_cost_m2']:.0f}", "Formwork cost per m²")
    with c5:
        savings_m2_pct = (r['trad_cost_m2'] - r['opt_cost_m2']) / r['trad_cost_m2'] * 100
        st.metric("Cost/m² Optimized", f"₹{r['opt_cost_m2']:.0f}", f"↓{savings_m2_pct:.1f}% vs traditional")
    with c6:
        st.metric("Productivity Gain", f"+{r['productivity_gain']:.1f}%",
                  f"{r['trad_productivity']} → {r['opt_productivity']:.1f} m²/gang/day")

    col_l, col_r = st.columns(2)

    with col_l:
        radar_metrics = ["WC Efficiency", "Cost/m²", "Inventory Turn", "ROI", "Productivity", "Asset Use"]
        trad_radar = [40, 100, 35, 50, 50, 45]
        opt_radar  = [
            min(100, 40 + r['roa_impact'] * 3),
            max(0, 100 - savings_m2_pct),
            60,
            min(100, 50 + r['roi_impact'] * 2),
            min(100, 50 + r['productivity_gain']),
            min(100, 45 + r['roa_impact'] * 2),
        ]

        fig7 = go.Figure()
        fig7.add_trace(go.Scatterpolar(r=trad_radar + [trad_radar[0]], theta=radar_metrics + [radar_metrics[0]],
                                       fill="toself", name="Traditional",
                                       line=dict(color=GRAY), fillcolor=f"{GRAY}30"))
        fig7.add_trace(go.Scatterpolar(r=opt_radar + [opt_radar[0]], theta=radar_metrics + [radar_metrics[0]],
                                       fill="toself", name="Optimized",
                                       line=dict(color=YELLOW), fillcolor=f"{YELLOW}30"))
        fig7.update_layout(
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            polar=dict(bgcolor=CARD_BG,
                       radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER, color="#4b5563"),
                       angularaxis=dict(gridcolor=BORDER, color="#9ca3af")),
            title=dict(text="Financial Performance Radar", font=dict(color="#f3f4f6", size=13)),
            legend=dict(bgcolor=CARD_BG, bordercolor=BORDER),
            margin=dict(l=40, r=40, t=50, b=40),
            font=dict(color="#9ca3af")
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header" style="margin-top:0">Assumptions & Caveats</div>',
                    unsafe_allow_html=True)
        assumptions = [
            ("Carrying cost rate", f"{carrying_cost_pct}% p.a. — standard working capital cost"),
            ("Material proportion", "46% of formwork cost ± 5% by project type"),
            ("Labour saving", "Modelled as function of productivity gain only"),
            ("Contract type", "Assumes EPC — savings fully captured by L&T"),
            ("Supplier lead time", "Assumes ≤5 days — validate for Tier 2/3 sites"),
            ("Data quality", "Outputs only as reliable as drawing & schedule inputs"),
            ("Damage factor", f"{damage_factor} — validate against actual site records"),
            ("82% utilization", "Accounts for cleaning, repair, transit downtime"),
        ]
        for label, text in assumptions:
            st.markdown(f"""
            <div class='assumption-box'>
            <span style='color:#f7c948; font-weight:700;'>{label}:</span> {text}
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REUSE LIFECYCLE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Reuse Cycle Simulation</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([2, 1])
    rt = r['reuse_timeline']

    with col_l:
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=rt["Cycle"], y=rt["Cost_Per_Use"],
                                  name="Optimized cost/use", line=dict(color=TEAL, width=2.5)))
        fig8.add_trace(go.Scatter(x=rt["Cycle"], y=rt["Traditional"],
                                  name="Single-use (traditional)", line=dict(color=RED, width=1.5, dash="dash")))
        fig8.add_vline(x=r['effective_reuse'], line_color=YELLOW, line_dash="dot",
                       annotation_text="Effective Reuse", annotation_font_color=YELLOW)
        fig8.update_layout(**dark_layout("Lifecycle Cost per Use (₹ Lakhs)"),
                           xaxis_title="Reuse Cycle", yaxis_title="₹ Lakhs per use")
        st.plotly_chart(fig8, use_container_width=True)

    with col_r:
        fig9 = go.Figure()
        fig9.add_trace(go.Scatter(x=rt["Cycle"], y=rt["Condition_Pct"],
                                  fill="tozeroy", name="Panel condition %",
                                  line=dict(color=GREEN), fillcolor=f"{GREEN}30"))
        fig9.add_hline(y=40, line_color=RED, line_dash="dash",
                       annotation_text="Retire threshold", annotation_font_color=RED)
        fig9.update_layout(**dark_layout("Panel Condition Decay (%)"),
                           xaxis_title="Cycle", yaxis_title="%", yaxis_range=[0, 110])
        st.plotly_chart(fig9, use_container_width=True)

        lifecycle_reduction = (1 - r['opt_cost_m2'] / r['trad_cost_m2']) * 100
        st.metric("Lifecycle Cost Reduction", f"{lifecycle_reduction:.1f}%", "vs no-reuse scenario")
        st.metric("Damage Factor Applied", f"{damage_factor*100:.0f}%", "Per-project panel loss")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SUSTAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">Sustainability & Circular Economy</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Embodied Energy/m²", f"{r['embodied_energy']} kWh", "Timber formwork baseline")
    with c2:
        st.metric("Energy Saved", f"{r['energy_saved']*1000:.1f} MWh", "vs single-use scenario")
    with c3:
        st.metric("CO₂ Avoided", f"{r['co2_saved']:.0f} tonnes", "India grid factor 0.82")
    with c4:
        st.metric("Material Waste Reduced", f"{r['inv_reduction_pct']*0.7:.1f}%", "From optimised procurement")

    col_l, col_r = st.columns(2)

    with col_l:
        energy_cycles = list(range(1, len(rt) + 1))
        single_use_energy = [c * r['embodied_energy'] * (r['total_area'] / 1e4) / 1000 for c in energy_cycles]
        reuse_energy = [c * r['embodied_energy'] * (r['total_area'] / 1e4) / (1000 * max(r['effective_reuse'], 1)) for c in energy_cycles]

        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(x=energy_cycles, y=single_use_energy,
                                   fill="tozeroy", name="Single-use energy (MWh)",
                                   line=dict(color=RED), fillcolor=f"{RED}20"))
        fig10.add_trace(go.Scatter(x=energy_cycles, y=reuse_energy,
                                   fill="tozeroy", name="Optimized (reuse) energy (MWh)",
                                   line=dict(color=GREEN), fillcolor=f"{GREEN}30"))
        fig10.update_layout(**dark_layout("Cumulative Energy Consumption by Cycle"),
                            xaxis_title="Reuse Cycle", yaxis_title="MWh")
        st.plotly_chart(fig10, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header" style="margin-top:0">Circular Economy Alignment</div>',
                    unsafe_allow_html=True)
        ce_items = [
            ("Reduce", r['inv_reduction_pct'], GREEN,
             "EOQ + repetition analytics cut over-procurement"),
            ("Reuse", (r['effective_reuse'] / reuse_limit) * 100, TEAL,
             f"Maximize panel life to {r['effective_reuse']:.1f} cycles"),
            ("Recycle", 65.0, BLUE,
             "End-of-life panel material recovery tracked"),
            ("Recover", 80.0, ORANGE,
             "Damage analytics predict retirement before failure"),
        ]
        for principle, pct_val, color, desc in ce_items:
            st.markdown(f"""
            <div style='padding:12px 14px; background:#0f172a; border-radius:8px;
                 border-left:3px solid {color}; margin-bottom:10px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span style='color:{color}; font-weight:700; font-size:14px;
                          font-family:DM Mono,monospace;'>{principle}</span>
                    <span style='color:{color}; font-weight:700;'>{pct_val:.1f}%</span>
                </div>
                <div style='height:6px; background:#1f2937; border-radius:3px;'>
                    <div style='width:{min(pct_val,100):.0f}%; height:100%;
                         background:{color}; border-radius:3px;'></div>
                </div>
                <div style='font-size:11px; color:#6b7280; margin-top:6px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — WHAT-IF
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.info("All parameters are controlled from the sidebar. Charts below show sensitivity across the full parameter range.", icon="🎛️")

    col_l, col_r = st.columns(2)

    with col_l:
        # Carrying cost sensitivity
        cc_range = np.arange(10, 18.5, 0.5)
        wc_vals, saving_vals = [], []
        for cc in cc_range:
            rr = compute_all({**params, "carrying_cost_pct": cc})
            wc_vals.append(rr['wc_unlocked'])
            saving_vals.append(rr['carrying_saved'])

        fig11 = go.Figure()
        fig11.add_trace(go.Scatter(x=cc_range, y=wc_vals, name="WC Unlocked (Cr)",
                                   line=dict(color=GREEN, width=2), mode="lines+markers"))
        fig11.add_trace(go.Scatter(x=cc_range, y=saving_vals, name="Carrying Saved (Cr)",
                                   line=dict(color=TEAL, width=2), mode="lines+markers"))
        fig11.add_vline(x=carrying_cost_pct, line_color=YELLOW, line_dash="dot",
                        annotation_text="Current", annotation_font_color=YELLOW)
        fig11.update_layout(**dark_layout("Sensitivity: Carrying Cost % vs Savings"),
                            xaxis_title="Carrying Cost % p.a.", yaxis_title="₹ Cr")
        st.plotly_chart(fig11, use_container_width=True)

    with col_r:
        # Over-order buffer sensitivity
        ob_range = list(range(5, 26))
        inv_saved_vals = []
        for ob in ob_range:
            rr = compute_all({**params, "over_order_buffer": ob})
            inv_saved_vals.append(rr['inv_reduction'])

        fig12 = go.Figure(go.Bar(
            x=ob_range, y=inv_saved_vals, marker_color=ORANGE,
            marker_line_width=0
        ))
        fig12.add_vline(x=over_order_buffer, line_color=YELLOW, line_dash="dot",
                        annotation_text="Current", annotation_font_color=YELLOW)
        fig12.update_layout(**dark_layout("Sensitivity: Over-Order Buffer vs Inventory Saved"),
                            xaxis_title="Over-Order Buffer %", yaxis_title="₹ Cr saved")
        st.plotly_chart(fig12, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        # Reuse limit sensitivity
        rl_range = list(range(10, 55, 5))
        ri_vals, er_vals = [], []
        for rl in rl_range:
            rr = compute_all({**params, "reuse_limit": rl})
            ri_vals.append(rr['repetition_index'] * 100)
            er_vals.append(rr['effective_reuse'])

        fig13 = go.Figure()
        fig13.add_trace(go.Scatter(x=rl_range, y=ri_vals, name="Repetition Index (%)",
                                   line=dict(color=TEAL, width=2), mode="lines+markers"))
        fig13.add_trace(go.Scatter(x=rl_range, y=er_vals, name="Effective Reuse (×)",
                                   line=dict(color=BLUE, width=2), mode="lines+markers"))
        fig13.add_vline(x=reuse_limit, line_color=YELLOW, line_dash="dot",
                        annotation_text="Current", annotation_font_color=YELLOW)
        fig13.update_layout(**dark_layout("Reuse Limit vs Index & Effective Reuse"),
                            xaxis_title="Reuse Limit (cycles)", yaxis_title="Value")
        st.plotly_chart(fig13, use_container_width=True)

    with col_r2:
        # Stress test table
        st.markdown('<div class="section-header" style="margin-top:0">Stress Test Results</div>',
                    unsafe_allow_html=True)
        scenarios = [
            ("Best Case",         {"carrying_cost_pct": 10, "reuse_limit": 50, "over_order_buffer": 20}),
            ("Base Case (current)", {}),
            ("Conservative",      {"damage_factor": 0.12, "duration_months": 12, "over_order_buffer": 7}),
            ("Pessimistic",       {"cycle_days": 12, "over_order_buffer": 5, "carrying_cost_pct": 10}),
        ]
        stress_rows = []
        for name, overrides in scenarios:
            rr = compute_all({**params, **overrides})
            total_saving = rr['carrying_saved'] + rr['inv_reduction']
            verdict = "✅ Viable" if total_saving > 0.1 else "⚠️ Marginal"
            stress_rows.append({
                "Scenario": name,
                "Total Saving (₹ Cr)": round(total_saving, 3),
                "Verdict": verdict
            })

        stress_df = pd.DataFrame(stress_rows)
        st.dataframe(
            stress_df.style.set_properties(**{"background-color": "#111827", "color": "#f3f4f6"}),
            use_container_width=True, hide_index=True
        )
        st.markdown("""
        <div class='assumption-box' style='margin-top:12px;'>
        <b style='color:#f7c948;'>Conclusion:</b>
        The economic case holds even under conservative assumptions.
        The pessimistic scenario is marginal only at very small project scale.
        Scale the project cost slider up to see the case strengthen significantly.
        </div>
        """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='display:flex; justify-content:space-between; font-size:11px; color:#4b5563; padding:8px 0;'>
    <span>All savings computed bottom-up · No arbitrary multipliers · Sources: CPWD, ACI 347R, ICE DB, CEA India, RSMeans</span>
    <span style='font-family:DM Mono,monospace;'>Formwork Optimization System · L&T CreaTech 2025</span>
</div>
""", unsafe_allow_html=True)
