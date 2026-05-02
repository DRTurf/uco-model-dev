"""
Climate Risk Capital Impact Model -- CRISK Engine
==================================================
Computes the core capital shortfall measures from the CRISK paper:

  CRISK   -- Expected capital shortfall under climate stress (Eq. 5, 7)
  mCRISK  -- Marginal CRISK isolating climate effect (Eq. 8)
  S&CRISK -- Compound stress: climate + market stress combined
  LRMES   -- Long-run marginal expected shortfall (Eq. 6)

Reference: Jung, Engle, Giglio -- JFE 2025, Section 6.1
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from config import (
    PRUDENTIAL_RATIO_US, PRUDENTIAL_RATIO_EU,
    DEFAULT_STRESS_LEVEL, BANK_FINANCIALS, BankFinancials,
)


def compute_lrmes(
    climate_beta: float,
    stress_level: float = DEFAULT_STRESS_LEVEL,
) -> float:
    """
    Long-Run Marginal Expected Shortfall (Eq. 6).

    LRMES_it = −E[R_i | R_CF < C]
             = 1 − exp(β_Climate × log(1 − θ))

    Parameters
    ----------
    climate_beta : float
        Estimated climate beta of the bank.
    stress_level : float
        θ -- fraction of decline in climate factor (e.g. 0.50 = 50%).

    Returns
    -------
    float
        LRMES as a positive fraction (expected equity loss).
    """
    if stress_level <= 0 or stress_level >= 1:
        raise ValueError(f"Stress level θ must be in (0, 1), got {stress_level}")

    log_factor = climate_beta * np.log(1 - stress_level)
    lrmes = 1 - np.exp(log_factor)
    return lrmes


def compute_crisk(
    debt: float,
    equity_market: float,
    climate_beta: float,
    stress_level: float = DEFAULT_STRESS_LEVEL,
    prudential_ratio: float = PRUDENTIAL_RATIO_US,
) -> float:
    """
    CRISK -- Expected capital shortfall under climate stress (Eq. 7).

    CRISK_it = k × D_it − (1 − k) × W_it × exp(β_Climate × log(1 − θ))

    Parameters
    ----------
    debt : float
        Book value of debt D (in $B).
    equity_market : float
        Market value of equity W (in $B).
    climate_beta : float
        Estimated climate beta.
    stress_level : float
        θ -- stress level.
    prudential_ratio : float
        k -- required equity-to-assets ratio.

    Returns
    -------
    float
        CRISK in $B (positive = shortfall, negative = surplus).
    """
    k = prudential_ratio
    log_factor = climate_beta * np.log(1 - stress_level)

    crisk = k * debt - (1 - k) * equity_market * np.exp(log_factor)
    return crisk


def compute_mcrisk(
    equity_market: float,
    climate_beta: float,
    stress_level: float = DEFAULT_STRESS_LEVEL,
    prudential_ratio: float = PRUDENTIAL_RATIO_US,
) -> float:
    """
    Marginal CRISK -- isolates the climate stress effect (Eq. 8).

    mCRISK = (1 − k) × W × LRMES

    This is the ADDITIONAL capital shortfall specifically due to
    climate stress, above and beyond current undercapitalization.

    Returns
    -------
    float
        mCRISK in $B (always ≥ 0 by construction with max(0, ...)).
    """
    k = prudential_ratio
    lrmes = compute_lrmes(climate_beta, stress_level)
    mcrisk = (1 - k) * equity_market * lrmes
    return max(0, mcrisk)


def compute_scrisk(
    debt: float,
    equity_market: float,
    climate_beta: float,
    market_beta: float,
    climate_stress: float = DEFAULT_STRESS_LEVEL,
    market_stress: float = 0.40,
    prudential_ratio: float = PRUDENTIAL_RATIO_US,
) -> float:
    """
    S&CRISK -- Compound risk: simultaneous climate + market stress.

    S&CRISK = k × D − (1 − k) × W × exp(β_Climate × log(1 − θ_Climate)
                                        + β_Mkt × log(1 − θ_Mkt))

    Returns
    -------
    float
        S&CRISK in $B.
    """
    k = prudential_ratio
    log_factor = (
        climate_beta * np.log(1 - climate_stress) +
        market_beta * np.log(1 - market_stress)
    )
    scrisk = k * debt - (1 - k) * equity_market * np.exp(log_factor)
    return scrisk


def decompose_crisk_change(
    debt_t0: float,
    debt_t1: float,
    equity_t0: float,
    equity_t1: float,
    lrmes_t0: float,
    lrmes_t1: float,
    prudential_ratio: float = PRUDENTIAL_RATIO_US,
) -> Dict[str, float]:
    """
    Decompose the change in CRISK into three components (Eq. 9):

    dCRISK = dDEBT + dEQUITY + dRISK

    where:
      dDEBT   = k × ΔD
      dEQUITY = −(1 − k)(1 − LRMES_avg) × ΔW
      dRISK   = (1 − k) × W_avg × ΔLRMES

    Returns
    -------
    dict with keys: 'dDEBT', 'dEQUITY', 'dRISK', 'total'
    """
    k = prudential_ratio
    delta_d = debt_t1 - debt_t0
    delta_w = equity_t1 - equity_t0
    delta_lrmes = lrmes_t1 - lrmes_t0

    avg_lrmes = (lrmes_t0 + lrmes_t1) / 2
    avg_w = (equity_t0 + equity_t1) / 2

    d_debt = k * delta_d
    d_equity = -(1 - k) * (1 - avg_lrmes) * delta_w
    d_risk = (1 - k) * avg_w * delta_lrmes

    return {
        "dDEBT":   round(d_debt, 2),
        "dEQUITY": round(d_equity, 2),
        "dRISK":   round(d_risk, 2),
        "total":   round(d_debt + d_equity + d_risk, 2),
    }


def compute_crisk_timeseries(
    climate_betas: pd.Series,
    market_betas: pd.Series,
    financials: BankFinancials,
    stress_level: float = DEFAULT_STRESS_LEVEL,
    market_stress: float = 0.40,
) -> pd.DataFrame:
    """
    Compute time series of CRISK, mCRISK, and S&CRISK for a single bank.

    Uses fixed balance-sheet data (quasi-static assumption)
    with time-varying betas.

    Returns
    -------
    pd.DataFrame with columns: ['CRISK', 'mCRISK', 'SCRISK', 'LRMES']
    """
    D = financials.debt_book_bn
    W = financials.equity_market_bn
    k = PRUDENTIAL_RATIO_US

    results = []
    for date, cb in climate_betas.items():
        if np.isnan(cb):
            results.append({
                "date": date, "CRISK": np.nan, "mCRISK": np.nan,
                "SCRISK": np.nan, "LRMES": np.nan,
            })
            continue

        mb = market_betas.get(date, 1.0)
        if np.isnan(mb):
            mb = 1.0

        lrmes = compute_lrmes(cb, stress_level)
        crisk = compute_crisk(D, W, cb, stress_level, k)
        mcrisk = compute_mcrisk(W, cb, stress_level, k)
        scrisk = compute_scrisk(D, W, cb, mb, stress_level, market_stress, k)

        results.append({
            "date":   date,
            "CRISK":  round(crisk, 2),
            "mCRISK": round(mcrisk, 2),
            "SCRISK": round(scrisk, 2),
            "LRMES":  round(lrmes, 4),
        })

    df = pd.DataFrame(results).set_index("date")
    return df


def compute_all_banks_crisk(
    climate_betas: pd.DataFrame,
    market_betas: pd.DataFrame,
    stress_level: float = DEFAULT_STRESS_LEVEL,
) -> Dict[str, pd.DataFrame]:
    """
    Compute CRISK time series for all configured banks.

    Returns
    -------
    dict: ticker -> DataFrame of CRISK metrics over time.
    """
    results = {}
    for ticker in climate_betas.columns:
        if ticker not in BANK_FINANCIALS:
            continue
        fin = BANK_FINANCIALS[ticker]
        cb = climate_betas[ticker]
        mb = market_betas[ticker] if ticker in market_betas.columns else pd.Series(1.0, index=cb.index)

        ts = compute_crisk_timeseries(cb, mb, fin, stress_level)
        results[ticker] = ts

    print(f"[OK] Computed CRISK time series for {len(results)} banks")
    return results


def compute_aggregate_crisk(
    all_bank_crisk: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate CRISK and mCRISK across all banks (systemic measure).

    From the paper:
      CRISK_t = Σ_i max(CRISK_it, 0)
      mCRISK_t = Σ_i mCRISK_it
    """
    crisk_list = []
    mcrisk_list = []

    for ticker, df in all_bank_crisk.items():
        crisk_list.append(df["CRISK"].clip(lower=0).rename(ticker))
        mcrisk_list.append(df["mCRISK"].rename(ticker))

    agg_crisk = pd.concat(crisk_list, axis=1).sum(axis=1)
    agg_mcrisk = pd.concat(mcrisk_list, axis=1).sum(axis=1)

    return pd.DataFrame({
        "Aggregate CRISK ($B)":  agg_crisk.round(2),
        "Aggregate mCRISK ($B)": agg_mcrisk.round(2),
    })


def compute_snapshot(
    ticker: str,
    climate_beta: float,
    market_beta: float = 1.0,
    stress_level: float = DEFAULT_STRESS_LEVEL,
    market_stress: float = 0.40,
) -> Dict[str, float]:
    """
    Single point-in-time CRISK computation for a bank.
    Returns a summary dictionary.
    """
    fin = BANK_FINANCIALS[ticker]
    D = fin.debt_book_bn
    W = fin.equity_market_bn

    lrmes = compute_lrmes(climate_beta, stress_level)
    crisk = compute_crisk(D, W, climate_beta, stress_level)
    mcrisk = compute_mcrisk(W, climate_beta, stress_level)
    scrisk = compute_scrisk(D, W, climate_beta, market_beta, stress_level, market_stress)

    equity_post_stress = W * (1 - lrmes)
    assets_post = D + equity_post_stress
    capital_ratio_post = equity_post_stress / assets_post if assets_post > 0 else 0

    return {
        "Bank":                   fin.name,
        "Total Assets ($B)":      fin.total_assets_bn,
        "Market Equity ($B)":     W,
        "Debt ($B)":              D,
        "Climate Beta":           round(climate_beta, 4),
        "Market Beta":            round(market_beta, 4),
        "Stress Level (θ)":       f"{stress_level:.0%}",
        "LRMES":                  f"{lrmes:.2%}",
        "Equity Post-Stress ($B)": round(equity_post_stress, 2),
        "Capital Ratio Post":     f"{capital_ratio_post:.2%}",
        "CRISK ($B)":             round(crisk, 2),
        "mCRISK ($B)":            round(mcrisk, 2),
        "S&CRISK ($B)":           round(scrisk, 2),
    }


if __name__ == "__main__":
    # Quick test with stylized values
    print("=== CRISK Snapshot: JPMorgan Chase ===")
    snapshot = compute_snapshot("JPM", climate_beta=0.45, market_beta=1.1)
    for k, v in snapshot.items():
        print(f"  {k:30s}: {v}")

    print("\n=== CRISK Snapshot: Citigroup ===")
    snapshot = compute_snapshot("C", climate_beta=0.60, market_beta=1.3)
    for k, v in snapshot.items():
        print(f"  {k:30s}: {v}")
