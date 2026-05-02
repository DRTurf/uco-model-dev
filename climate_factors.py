"""
Climate Risk Capital Impact Model -- Climate Factor Construction
================================================================
Constructs the four climate transition risk factors from the
CRISK paper (Jung, Engle, Giglio -- JFE 2025):

1. Stranded Asset Factor  -- short fossil fuel energy
2. Emission Factor        -- low-emission minus high-emission
3. Brown-Minus-Green (BMG) -- brown portfolio minus green portfolio
4. CEP Factor             -- climate efficient portfolio (orthogonalized)

Also calibrates the climate stress level θ from the tail of
the factor return distribution.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from config import (
    MARKET_ETF, ENERGY_ETF, CLEAN_ETF,
    STRESS_QUANTILE, STRESS_HORIZON_MONTHS,
)


def construct_climate_factors(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Build the four climate transition risk factors from daily return data.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-returns. Must include columns for ENERGY_ETF,
        CLEAN_ETF, and MARKET_ETF at minimum.

    Returns
    -------
    pd.DataFrame
        Columns: ['stranded_asset', 'emission', 'bmg', 'cep', 'mkt']
    """
    mkt = returns[MARKET_ETF].copy()

    # 1. Stranded Asset Factor = −R(XLE)
    #    Falls when fossil fuel stocks decline → rises with transition risk
    #    We negate: factor DECLINES as transition risk RISES (per paper convention)
    stranded = -returns[ENERGY_ETF].copy()

    # 2. Emission Factor = R(ICLN) − R(XLE)
    #    Long clean energy, short dirty energy
    #    Declines when transition risk rises (brown outperforms in reversal)
    emission = returns[CLEAN_ETF] - returns[ENERGY_ETF]

    # 3. Brown-Minus-Green (BMG) Factor
    #    R(Brown) − R(Green) = R(XLE) − R(ICLN)
    #    We negate to match paper convention (factor falls as transition risk rises)
    bmg = -(returns[ENERGY_ETF] - returns[CLEAN_ETF])

    # 4. Climate Efficient Portfolio (CEP) Factor
    #    Orthogonalize ICLN to market and stranded asset factor
    cep = _orthogonalize_cep(returns[CLEAN_ETF], mkt, stranded)
    cep = -cep  # Negate: factor falls as transition risk rises

    factors = pd.DataFrame({
        "stranded_asset": stranded,
        "emission":       emission,
        "bmg":            bmg,
        "cep":            cep,
        "mkt":            mkt,
    })

    print(f"[OK] Constructed 4 climate factors + MKT ({len(factors)} days)")
    return factors


def _orthogonalize_cep(
    clean_ret: pd.Series,
    mkt_ret: pd.Series,
    stranded_ret: pd.Series,
) -> pd.Series:
    """
    Orthogonalize clean energy returns to market and stranded asset factor.
    CEP captures climate news AFTER removing standard financial risks.
    """
    X = pd.DataFrame({"mkt": mkt_ret, "stranded": stranded_ret}).dropna()
    y = clean_ret.reindex(X.index).dropna()
    X = X.reindex(y.index)

    # OLS regression: ICLN = a + b1*MKT + b2*STRANDED + residual
    X_with_const = np.column_stack([np.ones(len(X)), X.values])
    try:
        betas = np.linalg.lstsq(X_with_const, y.values, rcond=None)[0]
        residuals = y.values - X_with_const @ betas
    except np.linalg.LinAlgError:
        residuals = y.values  # fallback: use raw returns

    cep = pd.Series(residuals, index=y.index, name="cep")
    return cep.reindex(clean_ret.index).fillna(0)


def calibrate_stress_level(
    factors: pd.DataFrame,
    factor_name: str = "stranded_asset",
    quantile: float = STRESS_QUANTILE,
    horizon_months: int = STRESS_HORIZON_MONTHS,
) -> float:
    """
    Calibrate the climate stress level θ as the 1st percentile of
    the h-month return distribution of the climate factor.

    From the paper: "We take the lowest one percentile of the 6-month
    return distribution of a climate risk factor to calibrate the
    stress level."

    Parameters
    ----------
    factors : pd.DataFrame
        Climate factor daily returns.
    factor_name : str
        Which factor to use for calibration.
    quantile : float
        Quantile for stress calibration (default 0.01 = 1st percentile).
    horizon_months : int
        Horizon in months (default 6).

    Returns
    -------
    float
        Calibrated stress level θ (as a positive fraction, e.g. 0.50 = 50% loss).
    """
    factor_series = factors[factor_name].dropna()

    # Approximate h-month returns: ~21 trading days per month
    trading_days = horizon_months * 21
    cumulative_returns = factor_series.rolling(window=trading_days).sum()
    cumulative_returns = cumulative_returns.dropna()

    if len(cumulative_returns) < 50:
        print(f"[WARN]️  Insufficient data for calibration. Using default θ = 0.50")
        return 0.50

    # 1st percentile of cumulative return distribution
    tail_return = cumulative_returns.quantile(quantile)

    # Convert log-return to simple return loss fraction
    # θ = 1 − exp(tail_return)  [tail_return is negative]
    theta = 1 - np.exp(tail_return)
    theta = max(0.05, min(theta, 0.95))  # bound to [5%, 95%]

    print(f"[OK] Calibrated stress level θ = {theta:.2%} "
          f"(factor={factor_name}, quantile={quantile}, horizon={horizon_months}mo)")
    return theta


def get_factor_summary(factors: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for climate factors.
    Mimics Table B.1 from the CRISK paper.
    """
    stats = pd.DataFrame({
        "Mean (ann %)":     factors.mean() * 252 * 100,
        "Std (ann %)":      factors.std() * np.sqrt(252) * 100,
        "Skewness":         factors.skew(),
        "Kurtosis":         factors.kurtosis(),
        "Min (daily %)":    factors.min() * 100,
        "Max (daily %)":    factors.max() * 100,
        "Observations":     factors.count(),
    }).round(3)
    return stats


def get_factor_correlations(factors: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of climate factors (Table B.2 analogue)."""
    return factors.corr().round(3)


if __name__ == "__main__":
    from data_loader import fetch_market_data

    returns = fetch_market_data(["JPM", "BAC"], start="2015-01-01")
    factors = construct_climate_factors(returns)
    print("\n--- Factor Summary ---")
    print(get_factor_summary(factors))
    print("\n--- Factor Correlations ---")
    print(get_factor_correlations(factors))
    print(f"\nCalibrated θ = {calibrate_stress_level(factors):.2%}")
