"""
Climate Risk Capital Impact Model -- ICAAP Integration Engine
==============================================================
Combines CRISK market-based measures with credit-risk RWA analysis
to produce an integrated ICAAP Pillar 2 climate capital add-on.

Components:
  1. CRISK-based capital impact (market risk channel)
  2. RWA-based capital impact (credit risk channel)
  3. Combined Pillar 2 climate buffer recommendation
  4. Capital adequacy waterfall (pre → post climate stress)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from config import (
    BANK_FINANCIALS, BankFinancials,
    PRUDENTIAL_RATIO_US, DEFAULT_STRESS_LEVEL,
)
from crisk_engine import compute_crisk, compute_mcrisk, compute_lrmes, compute_scrisk
from rwa_engine import compute_total_rwa_impact


def compute_icaap_assessment(
    ticker: str,
    climate_beta: float,
    market_beta: float,
    loan_portfolio: pd.DataFrame,
    stress_level: float = DEFAULT_STRESS_LEVEL,
    market_stress: float = 0.40,
    pd_sensitivity: float = 0.5,
) -> Dict[str, object]:
    """
    Comprehensive ICAAP climate risk assessment for a single bank.

    Integrates:
      - CRISK / mCRISK (market-based capital shortfall)
      - Climate-adjusted RWA (credit risk channel)
      - Pillar 2 buffer recommendation

    Parameters
    ----------
    ticker : str
        Bank ticker (must be in BANK_FINANCIALS).
    climate_beta : float
        Estimated climate beta for the bank.
    market_beta : float
        Estimated market beta for the bank.
    loan_portfolio : pd.DataFrame
        Synthetic loan portfolio from data_loader.
    stress_level : float
        θ -- climate stress level.
    market_stress : float
        θ_Mkt -- market stress level for compound S&CRISK.
    pd_sensitivity : float
        Calibration for the PD stress overlay.

    Returns
    -------
    dict with comprehensive ICAAP assessment metrics.
    """
    fin = BANK_FINANCIALS[ticker]

    # --- 1. CRISK Measures (Market Risk Channel) ---
    D = fin.debt_book_bn
    W = fin.equity_market_bn
    k = PRUDENTIAL_RATIO_US

    lrmes = compute_lrmes(climate_beta, stress_level)
    crisk = compute_crisk(D, W, climate_beta, stress_level, k)
    mcrisk = compute_mcrisk(W, climate_beta, stress_level, k)
    scrisk = compute_scrisk(D, W, climate_beta, market_beta, stress_level, market_stress, k)

    # Post-stress equity
    equity_post_climate = W * (1 - lrmes)
    total_assets = D + W
    assets_post_climate = D + equity_post_climate

    # --- 2. RWA Impact (Credit Risk Channel) ---
    rwa_impact = compute_total_rwa_impact(loan_portfolio, stress_level, pd_sensitivity)
    delta_rwa = rwa_impact["ΔRWA (Climate)"]
    baseline_rwa = rwa_impact["Baseline RWA"]
    stressed_rwa = rwa_impact["Stressed RWA"]

    # --- 3. Capital Ratios ---
    cet1_capital = fin.cet1_ratio / 100 * fin.rwa_bn  # CET1 capital in $B
    baseline_cet1_ratio = fin.cet1_ratio

    # Post climate-stress CET1 ratio
    # Climate stress reduces equity → reduces CET1 capital
    cet1_post_stress = cet1_capital - mcrisk
    cet1_post_stress = max(cet1_post_stress, 0)

    # RWA increases under stress
    rwa_post_stress = fin.rwa_bn + delta_rwa

    cet1_ratio_post = (cet1_post_stress / rwa_post_stress * 100) if rwa_post_stress > 0 else 0

    # --- 4. Pillar 2 Climate Buffer ---
    # Method 1: CRISK-based (as % of total assets)
    crisk_ratio = (mcrisk / total_assets * 100) if total_assets > 0 else 0

    # Method 2: RWA-based (as % of baseline RWA)
    rwa_ratio = (delta_rwa / fin.rwa_bn * 100) if fin.rwa_bn > 0 else 0

    # Combined buffer = max of the two channels
    pillar2_buffer_pct = max(crisk_ratio, rwa_ratio)

    # In basis points
    pillar2_buffer_bps = pillar2_buffer_pct * 100

    # Capital needed for buffer (in $B)
    capital_for_buffer = pillar2_buffer_pct / 100 * fin.rwa_bn

    # --- 5. Buffer adequacy check ---
    min_cet1 = 4.5  # Basel III minimum
    ccb = 2.5       # Capital conservation buffer
    total_min = min_cet1 + ccb  # 7.0%

    headroom = cet1_ratio_post - total_min
    is_adequate = headroom > 0

    return {
        # Bank info
        "bank_name":              fin.name,
        "ticker":                 ticker,
        "total_assets_bn":        fin.total_assets_bn,

        # Baseline position
        "cet1_capital_bn":        round(cet1_capital, 2),
        "cet1_ratio_baseline":    round(baseline_cet1_ratio, 2),
        "rwa_baseline_bn":        round(fin.rwa_bn, 2),

        # Climate betas
        "climate_beta":           round(climate_beta, 4),
        "market_beta":            round(market_beta, 4),

        # Stress parameters
        "stress_level":           stress_level,
        "market_stress":          market_stress,

        # CRISK measures (market channel)
        "lrmes":                  round(lrmes, 4),
        "crisk_bn":               round(crisk, 2),
        "mcrisk_bn":              round(mcrisk, 2),
        "scrisk_bn":              round(scrisk, 2),
        "equity_post_stress_bn":  round(equity_post_climate, 2),

        # RWA impact (credit channel)
        "rwa_delta_bn":           round(delta_rwa, 2),
        "rwa_delta_pct":          round(rwa_impact["ΔRWA %"], 2),
        "rwa_stressed_bn":        round(rwa_post_stress, 2),

        # Post-stress capital
        "cet1_post_stress_bn":    round(cet1_post_stress, 2),
        "cet1_ratio_post":        round(cet1_ratio_post, 2),

        # Pillar 2 buffer
        "pillar2_buffer_pct":     round(pillar2_buffer_pct, 2),
        "pillar2_buffer_bps":     round(pillar2_buffer_bps, 0),
        "capital_for_buffer_bn":  round(capital_for_buffer, 2),

        # Adequacy
        "min_regulatory_ratio":   total_min,
        "headroom_pct":           round(headroom, 2),
        "is_adequate":            is_adequate,
    }


def generate_capital_waterfall(assessment: Dict) -> pd.DataFrame:
    """
    Generate a capital adequacy waterfall showing the step-by-step
    impact from baseline to post-climate-stress.

    Returns
    -------
    pd.DataFrame with waterfall steps.
    """
    steps = [
        {
            "Step":              "Baseline CET1 Ratio",
            "Impact (%)":        assessment["cet1_ratio_baseline"],
            "Cumulative (%)":    assessment["cet1_ratio_baseline"],
            "Type":              "baseline",
        },
        {
            "Step":              "Climate CRISK Impact",
            "Impact (%)":        -round(assessment["mcrisk_bn"] / assessment["rwa_baseline_bn"] * 100, 2),
            "Cumulative (%)":    None,  # calculated below
            "Type":              "negative",
        },
        {
            "Step":              "Climate RWA Increase",
            "Impact (%)":        -round(assessment["rwa_delta_pct"] * assessment["cet1_ratio_baseline"] / 100, 2),
            "Cumulative (%)":    None,
            "Type":              "negative",
        },
        {
            "Step":              "Post-Stress CET1 Ratio",
            "Impact (%)":        assessment["cet1_ratio_post"],
            "Cumulative (%)":    assessment["cet1_ratio_post"],
            "Type":              "result",
        },
        {
            "Step":              "Min Regulatory (CET1 + CCB)",
            "Impact (%)":        assessment["min_regulatory_ratio"],
            "Cumulative (%)":    assessment["min_regulatory_ratio"],
            "Type":              "threshold",
        },
        {
            "Step":              "Pillar 2 Climate Buffer",
            "Impact (%)":        assessment["pillar2_buffer_pct"],
            "Cumulative (%)":    assessment["pillar2_buffer_pct"],
            "Type":              "buffer",
        },
    ]

    # Fill cumulative
    cum = assessment["cet1_ratio_baseline"]
    for s in steps:
        if s["Type"] == "negative":
            cum += s["Impact (%)"]
            s["Cumulative (%)"] = round(cum, 2)

    return pd.DataFrame(steps)


def generate_multi_bank_comparison(
    tickers: list,
    climate_betas: Dict[str, float],
    market_betas: Dict[str, float],
    loan_portfolios: Dict[str, pd.DataFrame],
    stress_level: float = DEFAULT_STRESS_LEVEL,
) -> pd.DataFrame:
    """
    Compare ICAAP climate risk metrics across multiple banks.

    Returns
    -------
    pd.DataFrame -- one row per bank with key metrics.
    """
    rows = []
    for ticker in tickers:
        if ticker not in BANK_FINANCIALS:
            continue
        cb = climate_betas.get(ticker, 0.3)
        mb = market_betas.get(ticker, 1.0)
        portfolio = loan_portfolios.get(ticker)
        if portfolio is None:
            from data_loader import build_synthetic_loan_portfolio
            portfolio = build_synthetic_loan_portfolio(ticker)

        assessment = compute_icaap_assessment(
            ticker, cb, mb, portfolio, stress_level
        )
        rows.append({
            "Bank":                     assessment["bank_name"],
            "Climate β":                assessment["climate_beta"],
            "LRMES":                    f"{assessment['lrmes']:.1%}",
            "mCRISK ($B)":              assessment["mcrisk_bn"],
            "ΔRWA ($B)":               assessment["rwa_delta_bn"],
            "CET1 Base (%)":            assessment["cet1_ratio_baseline"],
            "CET1 Post-Stress (%)":     assessment["cet1_ratio_post"],
            "Pillar 2 Buffer (bps)":    int(assessment["pillar2_buffer_bps"]),
            "Adequate":                 "[OK]" if assessment["is_adequate"] else "[FAIL]",
        })

    return pd.DataFrame(rows)


def sensitivity_analysis(
    ticker: str,
    loan_portfolio: pd.DataFrame,
    beta_range: Optional[list] = None,
    stress_range: Optional[list] = None,
    market_beta: float = 1.0,
) -> pd.DataFrame:
    """
    Run sensitivity analysis varying climate beta and/or stress level.

    Returns
    -------
    pd.DataFrame with one row per (beta, stress) combination.
    """
    if beta_range is None:
        beta_range = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if stress_range is None:
        stress_range = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    rows = []
    for beta in beta_range:
        for theta in stress_range:
            assessment = compute_icaap_assessment(
                ticker, beta, market_beta, loan_portfolio, theta
            )
            rows.append({
                "Climate Beta":          beta,
                "Stress Level (θ)":      theta,
                "mCRISK ($B)":           assessment["mcrisk_bn"],
                "ΔRWA ($B)":            assessment["rwa_delta_bn"],
                "CET1 Post-Stress (%)":  assessment["cet1_ratio_post"],
                "Pillar 2 Buffer (bps)": int(assessment["pillar2_buffer_bps"]),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import build_synthetic_loan_portfolio

    loans = build_synthetic_loan_portfolio("JPM")

    print("=" * 60)
    print("  ICAAP CLIMATE RISK ASSESSMENT -- JPMorgan Chase")
    print("=" * 60)

    assessment = compute_icaap_assessment(
        "JPM", climate_beta=0.45, market_beta=1.1, loan_portfolio=loans
    )
    for k, v in assessment.items():
        print(f"  {k:30s}: {v}")

    print("\n--- Capital Waterfall ---")
    waterfall = generate_capital_waterfall(assessment)
    print(waterfall.to_string(index=False))

    print("\n--- Sensitivity Analysis ---")
    sens = sensitivity_analysis("JPM", loans, beta_range=[0.1, 0.3, 0.5, 0.7, 1.0])
    print(sens.to_string(index=False))
