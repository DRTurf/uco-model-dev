"""
Climate Risk Capital Impact Model -- Streamlit Dashboard
=========================================================
Interactive dashboard for climate risk capital impact analysis.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import (
    BANK_TICKERS, BANK_FINANCIALS, SECTOR_CONFIGS,
    PRIMARY_COLOR, ACCENT_COLOR, WARNING_COLOR, CHART_TEMPLATE,
)
from data_loader import fetch_market_data, build_synthetic_loan_portfolio
from climate_factors import construct_climate_factors, calibrate_stress_level, get_factor_summary
from climate_beta import estimate_climate_betas_ewma, smooth_betas, compute_loan_portfolio_climate_beta
from crisk_engine import (
    compute_crisk, compute_mcrisk, compute_lrmes, compute_scrisk,
    compute_snapshot, compute_crisk_timeseries, compute_all_banks_crisk,
    compute_aggregate_crisk,
)
from rwa_engine import compute_sector_rwa_summary, compute_total_rwa_impact
from icaap_engine import (
    compute_icaap_assessment, generate_capital_waterfall,
    generate_multi_bank_comparison, sensitivity_analysis,
)

# --- Page Config ---
st.set_page_config(
    page_title="Climate Risk Capital Impact Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { background: linear-gradient(135deg, #1A1D23 0%, #2D3139 100%);
                padding: 16px; border-radius: 12px; border: 1px solid #333; }
    .metric-card { background: linear-gradient(135deg, #1A1D23, #2D3139);
                   padding: 20px; border-radius: 12px; border: 1px solid #333;
                   margin-bottom: 10px; }
    h1 { background: linear-gradient(90deg, #00D4AA, #00A3FF);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-size: 2.2rem !important; }
    h2 { color: #00D4AA !important; border-bottom: 2px solid #00D4AA;
         padding-bottom: 8px; }
    .stTabs [data-baseweb="tab"] { color: #888; font-size: 1.05rem; }
    .stTabs [aria-selected="true"] { color: #00D4AA !important;
         border-bottom-color: #00D4AA !important; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0E1117 0%, #1A1D23 100%); }
</style>
""", unsafe_allow_html=True)


# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("#  Climate Risk Model")
    st.markdown("*CRISK Framework (Jung et al., JFE 2025)*")
    st.markdown("---")

    st.markdown("###  Bank Selection")
    selected_banks = st.multiselect(
        "Select Banks",
        options=list(BANK_TICKERS.keys()),
        default=["JPM", "BAC", "C", "WFC"],
        format_func=lambda x: f"{x} -- {BANK_TICKERS[x]}",
    )

    st.markdown("###  Scenario Parameters")
    stress_level = st.slider("Climate Stress Level (θ)", 0.10, 0.90, 0.50, 0.05,
                             help="Fraction decline in climate factor (paper default: 50%)")
    market_stress = st.slider("Market Stress Level (θ_Mkt)", 0.10, 0.60, 0.40, 0.05,
                              help="For compound S&CRISK calculation")
    pd_sensitivity = st.slider("PD Sensitivity", 0.1, 2.0, 0.5, 0.1,
                               help="Climate beta → PD transmission strength")

    st.markdown("###  Data Period")
    start_year = st.selectbox("Start Year", [2010, 2012, 2015, 2018, 2020], index=2)

    st.markdown("### ⚙️ Climate Factor")
    factor_choice = st.selectbox("Primary Climate Factor",
                                 ["stranded_asset", "emission", "bmg", "cep"],
                                 format_func=lambda x: {
                                     "stranded_asset": "Stranded Asset (XLE short)",
                                     "emission": "Emission (ICLN − XLE)",
                                     "bmg": "Brown-Minus-Green",
                                     "cep": "Climate Efficient Portfolio",
                                 }[x])

    st.markdown("---")
    run_model = st.button(" Run Model", type="primary", width='stretch')


# --- Main Content ---
st.markdown("#  Climate Risk Capital Impact Model")
st.markdown("*Quantifying climate transition risk impact on bank capital adequacy (ICAAP) and Risk-Weighted Assets (RWA)*")

if not run_model and "model_run" not in st.session_state:
    # Landing page
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3> CRISK Framework</h3>
        <p>Market-based expected capital shortfall under climate stress scenarios, using dynamic climate betas.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3> RWA Impact</h3>
        <p>Basel IRB credit risk weights adjusted for climate-induced PD stress across loan portfolio sectors.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>️ ICAAP Integration</h3>
        <p>Pillar 2 climate capital buffer combining market risk and credit risk channels.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(" Configure parameters in the sidebar and click **Run Model** to start the analysis.")
    st.stop()


# --- Run the model ---
st.session_state["model_run"] = True

with st.spinner(" Fetching market data…"):
    returns = fetch_market_data(selected_banks, start=f"{start_year}-01-01")

with st.spinner(" Constructing climate factors…"):
    factors = construct_climate_factors(returns)
    calibrated_theta = calibrate_stress_level(factors, factor_choice)

with st.spinner(" Estimating climate betas…"):
    bank_cols = [c for c in selected_banks if c in returns.columns]
    bank_rets = returns[bank_cols] if bank_cols else returns[[c for c in returns.columns if c not in ["SPY","XLE","ICLN","KOL"]]]
    climate_betas, market_betas = estimate_climate_betas_ewma(
        bank_rets, factors[factor_choice], factors["mkt"]
    )
    smoothed_betas = smooth_betas(climate_betas)

with st.spinner(" Computing CRISK & RWA…"):
    loan_portfolios = {}
    assessments = {}
    for ticker in bank_cols:
        if ticker in BANK_FINANCIALS:
            loans = build_synthetic_loan_portfolio(ticker)
            loan_portfolios[ticker] = loans
            cb = smoothed_betas[ticker].dropna().iloc[-1] if ticker in smoothed_betas.columns and not smoothed_betas[ticker].dropna().empty else 0.3
            mb = market_betas[ticker].dropna().iloc[-1] if ticker in market_betas.columns and not market_betas[ticker].dropna().empty else 1.0
            assessments[ticker] = compute_icaap_assessment(
                ticker, cb, mb, loans, stress_level, market_stress, pd_sensitivity
            )


# --- Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Overview", " Climate Betas", " CRISK Analysis",
    " RWA Impact", "️ ICAAP Summary"
])

# ===========================================
# TAB 1: Overview
# ===========================================
with tab1:
    st.markdown("## Executive Summary")

    if assessments:
        # Top metrics
        cols = st.columns(4)
        total_mcrisk = sum(a["mcrisk_bn"] for a in assessments.values())
        total_delta_rwa = sum(a["rwa_delta_bn"] for a in assessments.values())
        avg_cet1_post = np.mean([a["cet1_ratio_post"] for a in assessments.values()])
        max_buffer = max(a["pillar2_buffer_bps"] for a in assessments.values())

        cols[0].metric("Aggregate mCRISK", f"${total_mcrisk:.1f}B", delta=None)
        cols[1].metric("Total ΔRWA", f"${total_delta_rwa:.1f}B", delta=f"{total_delta_rwa/sum(a['rwa_baseline_bn'] for a in assessments.values())*100:.1f}%")
        cols[2].metric("Avg CET1 Post-Stress", f"{avg_cet1_post:.1f}%")
        cols[3].metric("Max Pillar 2 Buffer", f"{max_buffer:.0f} bps")

        st.markdown("---")

        # Multi-bank comparison table
        comp_data = []
        for t, a in assessments.items():
            comp_data.append({
                "Bank": a["bank_name"],
                "Climate β": a["climate_beta"],
                "LRMES": f"{a['lrmes']:.1%}",
                "mCRISK ($B)": a["mcrisk_bn"],
                "ΔRWA ($B)": a["rwa_delta_bn"],
                "CET1 Base (%)": a["cet1_ratio_baseline"],
                "CET1 Stressed (%)": a["cet1_ratio_post"],
                "Buffer (bps)": int(a["pillar2_buffer_bps"]),
                "Status": "[OK]" if a["is_adequate"] else "[FAIL]",
            })
        st.dataframe(pd.DataFrame(comp_data), width='stretch', hide_index=True)

        # Bar chart: mCRISK by bank
        fig = go.Figure()
        banks = [a["bank_name"] for a in assessments.values()]
        mcrisk_vals = [a["mcrisk_bn"] for a in assessments.values()]
        fig.add_trace(go.Bar(x=banks, y=mcrisk_vals, marker_color=ACCENT_COLOR, name="mCRISK"))
        fig.update_layout(template=CHART_TEMPLATE, title="Marginal CRISK by Bank ($B)",
                          yaxis_title="mCRISK ($B)", height=400)
        st.plotly_chart(fig, width='stretch')


# ===========================================
# TAB 2: Climate Betas
# ===========================================
with tab2:
    st.markdown("## Climate Beta Evolution")
    st.markdown("*6-month smoothed climate betas (Eq. 1: r_it = β_Mkt × MKT + β_Climate × CF + ε)*")

    if not smoothed_betas.empty:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, col in enumerate(smoothed_betas.columns):
            fig.add_trace(go.Scatter(
                x=smoothed_betas.index, y=smoothed_betas[col],
                mode="lines", name=BANK_TICKERS.get(col, col),
                line=dict(width=2, color=colors[i % len(colors)]),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(template=CHART_TEMPLATE, height=500,
                          title=f"Time-Varying Climate Beta ({factor_choice.replace('_',' ').title()} Factor)",
                          yaxis_title="Climate Beta (β_Climate)", xaxis_title="Date")
        st.plotly_chart(fig, width='stretch')

        # Latest betas table
        st.markdown("### Latest Climate Betas")
        latest = pd.DataFrame({
            "Bank": [BANK_TICKERS.get(c, c) for c in smoothed_betas.columns],
            "Climate Beta (raw)": [climate_betas[c].dropna().iloc[-1] if not climate_betas[c].dropna().empty else 0 for c in smoothed_betas.columns],
            "Climate Beta (6m avg)": [smoothed_betas[c].dropna().iloc[-1] if not smoothed_betas[c].dropna().empty else 0 for c in smoothed_betas.columns],
            "Market Beta": [market_betas[c].dropna().iloc[-1] if c in market_betas.columns and not market_betas[c].dropna().empty else 0 for c in smoothed_betas.columns],
        }).round(4)
        st.dataframe(latest, width='stretch', hide_index=True)

    # Climate factor summary
    st.markdown("### Climate Factor Statistics")
    factor_stats = get_factor_summary(factors[["stranded_asset", "emission", "bmg", "cep"]])
    st.dataframe(factor_stats, width='stretch')


# ===========================================
# TAB 3: CRISK Analysis
# ===========================================
with tab3:
    st.markdown("## CRISK Capital Shortfall Analysis")
    st.markdown(f"*Stress level θ = {stress_level:.0%} | Prudential ratio k = 8%*")

    if assessments:
        selected_bank = st.selectbox("Select Bank for Detailed Analysis",
                                     list(assessments.keys()),
                                     format_func=lambda x: BANK_TICKERS.get(x, x))
        a = assessments[selected_bank]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CRISK", f"${a['crisk_bn']:.1f}B")
        col2.metric("mCRISK", f"${a['mcrisk_bn']:.1f}B")
        col3.metric("S&CRISK", f"${a['scrisk_bn']:.1f}B")
        col4.metric("LRMES", f"{a['lrmes']:.1%}")

        # CRISK time series
        if selected_bank in BANK_FINANCIALS and selected_bank in smoothed_betas.columns:
            fin = BANK_FINANCIALS[selected_bank]
            cb_ts = smoothed_betas[selected_bank].dropna()
            mb_ts = market_betas[selected_bank].dropna() if selected_bank in market_betas.columns else pd.Series(1.0, index=cb_ts.index)
            crisk_ts = compute_crisk_timeseries(cb_ts, mb_ts, fin, stress_level)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=["CRISK ($B)", "mCRISK ($B)"],
                                vertical_spacing=0.08)
            fig.add_trace(go.Scatter(x=crisk_ts.index, y=crisk_ts["CRISK"],
                                     fill="tozeroy", fillcolor="rgba(255,107,107,0.2)",
                                     line=dict(color=ACCENT_COLOR, width=2), name="CRISK"), row=1, col=1)
            fig.add_trace(go.Scatter(x=crisk_ts.index, y=crisk_ts["mCRISK"],
                                     fill="tozeroy", fillcolor="rgba(0,212,170,0.2)",
                                     line=dict(color=PRIMARY_COLOR, width=2), name="mCRISK"), row=2, col=1)
            fig.update_layout(template=CHART_TEMPLATE, height=600,
                              title=f"CRISK Time Series -- {BANK_TICKERS.get(selected_bank, selected_bank)}")
            st.plotly_chart(fig, width='stretch')

        # CRISK Snapshot table
        st.markdown("### CRISK Snapshot")
        snap = compute_snapshot(selected_bank, a["climate_beta"], a["market_beta"])
        snap_df = pd.DataFrame(list(snap.items()), columns=["Metric", "Value"])
        st.dataframe(snap_df, width='stretch', hide_index=True)


# ===========================================
# TAB 4: RWA Impact
# ===========================================
with tab4:
    st.markdown("## Climate-Adjusted RWA Analysis")
    st.markdown("*Basel IRB capital requirements with climate PD stress overlay*")

    if assessments:
        rwa_bank = st.selectbox("Select Bank", list(assessments.keys()),
                                format_func=lambda x: BANK_TICKERS.get(x, x),
                                key="rwa_bank")
        if rwa_bank in loan_portfolios:
            loans = loan_portfolios[rwa_bank]
            sector_summary = compute_sector_rwa_summary(loans, stress_level, pd_sensitivity)
            rwa_total = compute_total_rwa_impact(loans, stress_level, pd_sensitivity)

            # Top metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline RWA", f"${rwa_total['Baseline RWA']:.1f}B")
            c2.metric("Stressed RWA", f"${rwa_total['Stressed RWA']:.1f}B")
            c3.metric("ΔRWA", f"${rwa_total['ΔRWA (Climate)']:.1f}B", delta=f"{rwa_total['ΔRWA %']:.1f}%")

            # Sector waterfall chart
            fig = go.Figure()
            sectors = sector_summary.index.tolist()
            deltas = sector_summary["delta_rwa"].tolist()
            colors = [ACCENT_COLOR if d > 0 else PRIMARY_COLOR for d in deltas]
            fig.add_trace(go.Bar(x=sectors, y=deltas, marker_color=colors, name="ΔRWA"))
            fig.update_layout(template=CHART_TEMPLATE, height=450,
                              title="ΔRWA by Sector (Climate Stress)",
                              yaxis_title="ΔRWA ($B)", xaxis_tickangle=-45)
            st.plotly_chart(fig, width='stretch')

            # Sector detail table
            st.markdown("### Sector Detail")
            display_cols = ["avg_climate_beta", "avg_base_pd", "avg_stressed_pd",
                            "rw_base", "rw_stressed", "delta_rwa", "delta_rwa_pct"]
            rename = {
                "avg_climate_beta": "Climate β", "avg_base_pd": "Base PD (%)",
                "avg_stressed_pd": "Stressed PD (%)", "rw_base": "RW Base (%)",
                "rw_stressed": "RW Stressed (%)", "delta_rwa": "ΔRWA ($B)",
                "delta_rwa_pct": "ΔRWA (%)",
            }
            st.dataframe(sector_summary[display_cols].rename(columns=rename),
                         width='stretch')

            # PD stress scatter
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=sector_summary["avg_climate_beta"], y=sector_summary["delta_rwa_pct"],
                mode="markers+text", text=sector_summary.index,
                textposition="top center", textfont=dict(size=9),
                marker=dict(size=sector_summary["base_ead"].clip(lower=1) / sector_summary["base_ead"].max() * 40 + 8,
                            color=sector_summary["avg_climate_beta"], colorscale="RdYlGn_r",
                            showscale=True, colorbar=dict(title="Climate β")),
            ))
            fig2.update_layout(template=CHART_TEMPLATE, height=450,
                               title="Climate Beta vs. RWA Increase by Sector",
                               xaxis_title="Avg Climate Beta", yaxis_title="ΔRWA (%)")
            st.plotly_chart(fig2, width='stretch')


# ===========================================
# TAB 5: ICAAP Summary
# ===========================================
with tab5:
    st.markdown("## ICAAP Pillar 2 Climate Capital Assessment")

    if assessments:
        icaap_bank = st.selectbox("Select Bank", list(assessments.keys()),
                                  format_func=lambda x: BANK_TICKERS.get(x, x),
                                  key="icaap_bank")
        a = assessments[icaap_bank]

        # Capital adequacy metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CET1 Baseline", f"{a['cet1_ratio_baseline']:.1f}%")
        c2.metric("CET1 Post-Stress", f"{a['cet1_ratio_post']:.1f}%",
                  delta=f"{a['cet1_ratio_post']-a['cet1_ratio_baseline']:.1f}%")
        c3.metric("Pillar 2 Buffer", f"{a['pillar2_buffer_bps']:.0f} bps")
        c4.metric("Capital Adequacy", "[OK] Adequate" if a["is_adequate"] else "[FAIL] Shortfall")

        # Waterfall chart
        waterfall = generate_capital_waterfall(a)
        fig = go.Figure(go.Waterfall(
            x=waterfall["Step"],
            y=waterfall["Impact (%)"],
            measure=["absolute", "relative", "relative", "total", "absolute", "absolute"],
            connector=dict(line=dict(color="#444")),
            increasing=dict(marker=dict(color=PRIMARY_COLOR)),
            decreasing=dict(marker=dict(color=ACCENT_COLOR)),
            totals=dict(marker=dict(color=WARNING_COLOR)),
            text=[f"{v:.2f}%" for v in waterfall["Impact (%)"]],
            textposition="outside",
        ))
        fig.add_hline(y=7.0, line_dash="dash", line_color="red", opacity=0.7,
                      annotation_text="Min CET1 + CCB (7.0%)")
        fig.update_layout(template=CHART_TEMPLATE, height=500,
                          title="Capital Adequacy Waterfall (Climate Stress Impact)",
                          yaxis_title="CET1 Ratio (%)")
        st.plotly_chart(fig, width='stretch')

        # Full assessment table
        st.markdown("### Full ICAAP Assessment")
        assessment_df = pd.DataFrame([
            {"Category": "Balance Sheet", "Metric": "Total Assets", "Value": f"${a['total_assets_bn']:.0f}B"},
            {"Category": "Balance Sheet", "Metric": "CET1 Capital", "Value": f"${a['cet1_capital_bn']:.1f}B"},
            {"Category": "Balance Sheet", "Metric": "RWA (Baseline)", "Value": f"${a['rwa_baseline_bn']:.0f}B"},
            {"Category": "Risk Metrics", "Metric": "Climate Beta", "Value": f"{a['climate_beta']:.4f}"},
            {"Category": "Risk Metrics", "Metric": "Market Beta", "Value": f"{a['market_beta']:.4f}"},
            {"Category": "Risk Metrics", "Metric": "LRMES", "Value": f"{a['lrmes']:.2%}"},
            {"Category": "Market Channel", "Metric": "CRISK", "Value": f"${a['crisk_bn']:.1f}B"},
            {"Category": "Market Channel", "Metric": "mCRISK", "Value": f"${a['mcrisk_bn']:.1f}B"},
            {"Category": "Market Channel", "Metric": "S&CRISK", "Value": f"${a['scrisk_bn']:.1f}B"},
            {"Category": "Credit Channel", "Metric": "ΔRWA", "Value": f"${a['rwa_delta_bn']:.1f}B ({a['rwa_delta_pct']:.1f}%)"},
            {"Category": "Post-Stress", "Metric": "CET1 Post-Stress", "Value": f"${a['cet1_post_stress_bn']:.1f}B"},
            {"Category": "Post-Stress", "Metric": "CET1 Ratio Post", "Value": f"{a['cet1_ratio_post']:.2f}%"},
            {"Category": "Pillar 2", "Metric": "Climate Buffer", "Value": f"{a['pillar2_buffer_pct']:.2f}% ({a['pillar2_buffer_bps']:.0f} bps)"},
            {"Category": "Pillar 2", "Metric": "Capital for Buffer", "Value": f"${a['capital_for_buffer_bn']:.1f}B"},
            {"Category": "Pillar 2", "Metric": "Headroom", "Value": f"{a['headroom_pct']:.2f}%"},
        ])
        st.dataframe(assessment_df, width='stretch', hide_index=True)

        # Sensitivity heatmap
        st.markdown("### Sensitivity Analysis")
        if icaap_bank in loan_portfolios:
            sens = sensitivity_analysis(
                icaap_bank, loan_portfolios[icaap_bank],
                beta_range=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5],
                stress_range=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                market_beta=a["market_beta"],
            )
            pivot = sens.pivot(index="Climate Beta", columns="Stress Level (θ)", values="CET1 Post-Stress (%)")
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot.values, x=[f"{c:.0%}" for c in pivot.columns],
                y=[f"{r:.1f}" for r in pivot.index],
                colorscale="RdYlGn", text=np.round(pivot.values, 1),
                texttemplate="%{text}%", colorbar=dict(title="CET1 %"),
            ))
            fig_heat.update_layout(template=CHART_TEMPLATE, height=400,
                                   title="CET1 Post-Stress Sensitivity (β × θ)",
                                   xaxis_title="Stress Level (θ)", yaxis_title="Climate Beta")
            st.plotly_chart(fig_heat, width='stretch')

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666; font-size:0.85rem;'>"
    "Climate Risk Capital Impact Model | Based on CRISK (Jung, Engle, Giglio -- JFE 2025) "
    "& BIS Staff Report sr977 | For research and educational purposes"
    "</div>", unsafe_allow_html=True
)
