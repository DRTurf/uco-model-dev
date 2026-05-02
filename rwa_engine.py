"""
Climate Risk Capital Impact Model -- RWA Engine
================================================
Translates climate risk into regulatory capital impact through the
credit risk channel using the Basel IRB framework.

Key components:
  1. Basel IRB capital requirement function K(PD)
  2. Climate PD stress overlay (based on CRISK paper Fig. 7 findings)
  3. Climate-adjusted RWA computation
  4. Sector-level waterfall decomposition

Reference: Basel II IRB approach + CRISK paper Section 5.2
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
from config import SECTOR_CONFIGS, BANK_FINANCIALS, BankFinancials


def compute_asset_correlation(pd_val: float) -> float:
    """
    Basel IRB asset correlation formula for corporate exposures.

    R = 0.12 × (1 − exp(−50 × PD)) / (1 − exp(−50))
      + 0.24 × [1 − (1 − exp(−50 × PD)) / (1 − exp(−50))]
    """
    exp_factor = (1 - np.exp(-50 * pd_val)) / (1 - np.exp(-50))
    R = 0.12 * exp_factor + 0.24 * (1 - exp_factor)
    return R


def compute_maturity_adjustment(pd_val: float, maturity: float) -> float:
    """
    Basel IRB maturity adjustment factor.

    b(PD) = (0.11852 − 0.05478 × ln(PD))²
    MA = (1 + (M − 2.5) × b) / (1 − 1.5 × b)
    """
    pd_val = max(pd_val, 1e-8)  # avoid log(0)
    b = (0.11852 - 0.05478 * np.log(pd_val)) ** 2
    ma = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    return max(ma, 0.0)


def compute_irb_capital_requirement(
    pd_val: float,
    lgd: float,
    maturity: float,
) -> float:
    """
    Basel IRB capital requirement K(PD, LGD, M).

    K = [LGD × Φ(√(1/(1-R)) × Φ⁻¹(PD) + √(R/(1-R)) × Φ⁻¹(0.999))
         − LGD × PD] × Maturity_Adj

    Parameters
    ----------
    pd_val : float
        Probability of default (as decimal, e.g. 0.02 = 2%).
    lgd : float
        Loss-given-default (as decimal, e.g. 0.45 = 45%).
    maturity : float
        Effective maturity in years.

    Returns
    -------
    float
        Capital requirement K as a fraction of EAD.
    """
    pd_val = max(pd_val, 1e-8)
    pd_val = min(pd_val, 0.9999)

    R = compute_asset_correlation(pd_val)
    ma = compute_maturity_adjustment(pd_val, maturity)

    # Vasicek formula
    conditional_pd = norm.cdf(
        np.sqrt(1 / (1 - R)) * norm.ppf(pd_val)
        + np.sqrt(R / (1 - R)) * norm.ppf(0.999)
    )

    K = (lgd * conditional_pd - lgd * pd_val) * ma
    return max(K, 0.0)


def compute_rwa_single_exposure(
    pd_val: float,
    lgd: float,
    ead: float,
    maturity: float,
) -> float:
    """
    RWA for a single exposure.
    RWA = 12.5 × K(PD, LGD, M) × EAD
    """
    K = compute_irb_capital_requirement(pd_val, lgd, maturity)
    rwa = 12.5 * K * ead
    return rwa


def apply_climate_pd_stress(
    base_pd: float,
    sector_climate_beta: float,
    stress_level: float = 0.50,
    sensitivity: float = 0.5,
) -> float:
    """
    Apply climate PD stress overlay based on CRISK paper findings.

    From Fig. 7: PD increases nonlinearly with climate beta.
    We model this as:

    PD_stressed = PD_base × (1 + climate_multiplier)

    where:
    climate_multiplier = sensitivity × β_climate × θ × (1 + 0.5 × β_climate)

    The quadratic term captures the nonlinearity documented in the paper:
    "PD nonlinearly increases in the climate beta"

    Parameters
    ----------
    base_pd : float
        Baseline probability of default (decimal).
    sector_climate_beta : float
        Sector-level climate beta.
    stress_level : float
        θ -- climate stress severity.
    sensitivity : float
        Calibration parameter for PD-climate beta relationship.

    Returns
    -------
    float
        Stressed PD (capped at 100%).
    """
    if sector_climate_beta <= 0:
        return base_pd

    # Nonlinear multiplier (per paper's Fig. 7)
    multiplier = sensitivity * sector_climate_beta * stress_level * (1 + 0.5 * sector_climate_beta)
    stressed_pd = base_pd * (1 + multiplier)

    return min(stressed_pd, 0.9999)  # cap at ~100%


def compute_portfolio_rwa(
    loan_portfolio: pd.DataFrame,
    stress_level: float = 0.0,
    pd_sensitivity: float = 0.5,
) -> pd.DataFrame:
    """
    Compute RWA for each loan in the portfolio, optionally with climate stress.

    Parameters
    ----------
    loan_portfolio : pd.DataFrame
        From build_synthetic_loan_portfolio().
    stress_level : float
        θ -- if 0, compute baseline RWA; if >0, apply climate PD stress.
    pd_sensitivity : float
        Calibration for the PD stress overlay.

    Returns
    -------
    pd.DataFrame
        Original dataframe augmented with RWA columns.
    """
    results = loan_portfolio.copy()
    rwas = []
    stressed_pds = []
    k_values = []

    for _, row in results.iterrows():
        base_pd = row["base_pd_pct"] / 100.0
        lgd = row["lgd_pct"] / 100.0
        ead = row["committed_amount"]
        maturity = row["maturity_years"]
        climate_beta = row["climate_beta"]

        if stress_level > 0:
            pd_val = apply_climate_pd_stress(base_pd, climate_beta, stress_level, pd_sensitivity)
        else:
            pd_val = base_pd

        K = compute_irb_capital_requirement(pd_val, lgd, maturity)
        rwa = 12.5 * K * ead

        stressed_pds.append(pd_val * 100)
        k_values.append(K * 100)
        rwas.append(rwa)

    results["stressed_pd_pct"] = stressed_pds
    results["capital_req_pct"] = k_values
    results["rwa"] = rwas

    return results


def compute_sector_rwa_summary(
    loan_portfolio: pd.DataFrame,
    stress_level: float = 0.50,
    pd_sensitivity: float = 0.5,
) -> pd.DataFrame:
    """
    Compute sector-level RWA summary: baseline vs. stressed.

    Returns
    -------
    pd.DataFrame
        Sector-level: base_rwa, stressed_rwa, delta_rwa, etc.
    """
    # Baseline
    base = compute_portfolio_rwa(loan_portfolio, stress_level=0.0)
    # Stressed
    stressed = compute_portfolio_rwa(loan_portfolio, stress_level=stress_level,
                                     pd_sensitivity=pd_sensitivity)

    # Aggregate by sector
    base_agg = base.groupby("sector").agg(
        base_ead=("committed_amount", "sum"),
        base_rwa=("rwa", "sum"),
        avg_base_pd=("base_pd_pct", "mean"),
        avg_climate_beta=("climate_beta", "mean"),
        num_loans=("sector", "count"),
    )

    stressed_agg = stressed.groupby("sector").agg(
        stressed_rwa=("rwa", "sum"),
        avg_stressed_pd=("stressed_pd_pct", "mean"),
    )

    summary = base_agg.join(stressed_agg)
    summary["delta_rwa"] = summary["stressed_rwa"] - summary["base_rwa"]
    summary["delta_rwa_pct"] = (summary["delta_rwa"] / summary["base_rwa"] * 100).round(2)
    summary["rw_base"] = (summary["base_rwa"] / summary["base_ead"] * 100).round(1)
    summary["rw_stressed"] = (summary["stressed_rwa"] / summary["base_ead"] * 100).round(1)

    # Sort by delta_rwa descending
    summary = summary.sort_values("delta_rwa", ascending=False)

    return summary.round(2)


def compute_total_rwa_impact(
    loan_portfolio: pd.DataFrame,
    stress_level: float = 0.50,
    pd_sensitivity: float = 0.5,
) -> Dict[str, float]:
    """
    Compute total portfolio RWA impact from climate stress.

    Returns
    -------
    dict with total baseline RWA, stressed RWA, delta, etc.
    """
    base = compute_portfolio_rwa(loan_portfolio, stress_level=0.0)
    stressed = compute_portfolio_rwa(loan_portfolio, stress_level=stress_level,
                                     pd_sensitivity=pd_sensitivity)

    total_ead = base["committed_amount"].sum()
    total_base_rwa = base["rwa"].sum()
    total_stressed_rwa = stressed["rwa"].sum()
    delta = total_stressed_rwa - total_base_rwa

    return {
        "Total EAD":              round(total_ead, 2),
        "Baseline RWA":           round(total_base_rwa, 2),
        "Stressed RWA":           round(total_stressed_rwa, 2),
        "ΔRWA (Climate)":         round(delta, 2),
        "ΔRWA %":                 round(delta / total_base_rwa * 100, 2) if total_base_rwa > 0 else 0,
        "Avg Risk Weight Base":   round(total_base_rwa / total_ead * 100, 1) if total_ead > 0 else 0,
        "Avg Risk Weight Stress": round(total_stressed_rwa / total_ead * 100, 1) if total_ead > 0 else 0,
    }


if __name__ == "__main__":
    from data_loader import build_synthetic_loan_portfolio

    loans = build_synthetic_loan_portfolio("JPM")

    print("=== Sector RWA Summary (θ=50%) ===")
    sector_summary = compute_sector_rwa_summary(loans, stress_level=0.50)
    print(sector_summary[["avg_climate_beta", "avg_base_pd", "avg_stressed_pd",
                          "rw_base", "rw_stressed", "delta_rwa_pct"]].to_string())

    print("\n=== Total RWA Impact ===")
    impact = compute_total_rwa_impact(loans, stress_level=0.50)
    for k, v in impact.items():
        print(f"  {k:30s}: {v}")
