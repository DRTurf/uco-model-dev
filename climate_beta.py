"""
Climate Risk Capital Impact Model -- Climate Beta Estimation
=============================================================
Estimates time-varying climate betas using:
  1. DCC-GARCH (primary method, per Engle 2002/2009)
  2. Rolling-window OLS (fallback)

Factor model (Eq. 1 from CRISK paper):
    r_it = β_Mkt_it × MKT_t + β_Climate_it × CF_t + ε_it

The climate beta β_Climate_it measures bank i's stock return
sensitivity to the climate risk factor CF on day t.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from config import ROLLING_WINDOW_DAYS


def estimate_climate_betas_rolling(
    bank_returns: pd.DataFrame,
    climate_factor: pd.Series,
    market_factor: pd.Series,
    window: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Estimate time-varying climate betas via rolling-window OLS.

    Model: r_it = β_Mkt × MKT_t + β_Climate × CF_t + ε_it

    Parameters
    ----------
    bank_returns : pd.DataFrame
        Daily returns for each bank (columns = tickers).
    climate_factor : pd.Series
        Daily climate factor returns.
    market_factor : pd.Series
        Daily market factor returns.
    window : int
        Rolling window size in trading days.

    Returns
    -------
    pd.DataFrame
        Time-varying climate betas (columns = bank tickers).
    """
    # Align all data
    data = pd.DataFrame({
        "mkt": market_factor,
        "cf":  climate_factor,
    })
    for col in bank_returns.columns:
        data[col] = bank_returns[col]
    data = data.dropna()

    bank_cols = [c for c in bank_returns.columns if c in data.columns]

    climate_betas = {}
    market_betas = {}

    for col in bank_cols:
        cb_series = pd.Series(np.nan, index=data.index, name=col)
        mb_series = pd.Series(np.nan, index=data.index, name=col)

        for i in range(window, len(data)):
            window_data = data.iloc[i - window:i]
            y = window_data[col].values
            X = np.column_stack([
                window_data["mkt"].values,
                window_data["cf"].values,
            ])

            # Add constant
            X_c = np.column_stack([np.ones(len(X)), X])

            try:
                betas = np.linalg.lstsq(X_c, y, rcond=None)[0]
                mb_series.iloc[i] = betas[1]  # market beta
                cb_series.iloc[i] = betas[2]  # climate beta
            except np.linalg.LinAlgError:
                pass

        climate_betas[col] = cb_series
        market_betas[col] = mb_series

    climate_beta_df = pd.DataFrame(climate_betas)
    market_beta_df = pd.DataFrame(market_betas)

    print(f"[OK] Estimated rolling climate betas for {len(bank_cols)} banks "
          f"(window={window} days)")

    return climate_beta_df, market_beta_df


def estimate_climate_betas_ewma(
    bank_returns: pd.DataFrame,
    climate_factor: pd.Series,
    market_factor: pd.Series,
    halflife: int = 126,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate time-varying climate betas using EWMA covariance
    (lighter-weight approximation to DCC-GARCH).

    β_Climate_it = Cov_ewma(r_i, CF) / Var_ewma(CF)

    Parameters
    ----------
    bank_returns : pd.DataFrame
        Daily bank stock returns.
    climate_factor : pd.Series
        Daily climate factor returns.
    market_factor : pd.Series
        Daily market factor returns.
    halflife : int
        EWMA halflife in days (default 126 ≈ 6 months).

    Returns
    -------
    climate_betas : pd.DataFrame
    market_betas : pd.DataFrame
    """
    # Align
    data = pd.DataFrame({"mkt": market_factor, "cf": climate_factor})
    for col in bank_returns.columns:
        data[col] = bank_returns[col]
    data = data.dropna()

    bank_cols = [c for c in bank_returns.columns if c in data.columns]

    # Demean using expanding mean for each series
    cf = data["cf"]
    mkt = data["mkt"]

    # EWMA decay factor
    alpha = 1 - np.exp(-np.log(2) / halflife)

    climate_betas = {}
    market_betas = {}

    # EWMA variance of climate factor
    var_cf = cf.ewm(halflife=halflife).var()
    var_mkt = mkt.ewm(halflife=halflife).var()

    for col in bank_cols:
        ri = data[col]

        # EWMA covariance: Cov(r_i, CF) and Cov(r_i, MKT)
        cov_ri_cf = (ri * cf).ewm(halflife=halflife).mean() - \
                    ri.ewm(halflife=halflife).mean() * cf.ewm(halflife=halflife).mean()
        cov_ri_mkt = (ri * mkt).ewm(halflife=halflife).mean() - \
                     ri.ewm(halflife=halflife).mean() * mkt.ewm(halflife=halflife).mean()

        # Beta = Cov / Var
        cb = cov_ri_cf / var_cf.replace(0, np.nan)
        mb = cov_ri_mkt / var_mkt.replace(0, np.nan)

        climate_betas[col] = cb
        market_betas[col] = mb

    climate_beta_df = pd.DataFrame(climate_betas)
    market_beta_df = pd.DataFrame(market_betas)

    print(f"[OK] Estimated EWMA climate betas for {len(bank_cols)} banks "
          f"(halflife={halflife} days)")

    return climate_beta_df, market_beta_df


def smooth_betas(
    betas: pd.DataFrame,
    window_months: int = 6,
) -> pd.DataFrame:
    """
    Apply 6-month moving average to betas as in the paper.
    "Figure 3 presents the 6-month moving average climate betas"
    """
    trading_days = window_months * 21
    smoothed = betas.rolling(window=trading_days, min_periods=trading_days // 2).mean()
    return smoothed


def get_latest_betas(
    climate_betas: pd.DataFrame,
    market_betas: pd.DataFrame,
) -> pd.DataFrame:
    """Get the most recent climate and market betas for each bank."""
    last_cb = climate_betas.iloc[-1]
    last_mb = market_betas.iloc[-1]

    summary = pd.DataFrame({
        "Climate Beta": last_cb,
        "Market Beta":  last_mb,
    })

    # Also get 6-month smoothed
    smoothed_cb = smooth_betas(climate_betas)
    summary["Climate Beta (6m avg)"] = smoothed_cb.iloc[-1]

    return summary.round(4)


def compute_loan_portfolio_climate_beta(
    loan_portfolio: pd.DataFrame,
) -> float:
    """
    Compute the loan-size-weighted average climate beta of a bank's
    loan portfolio (Eq. 2 from CRISK paper).

    β_Portfolio = Σ_j (w_j × β_Climate_j)

    where w_j = committed_amount_j / Σ committed_amount
    """
    total_committed = loan_portfolio["committed_amount"].sum()
    if total_committed == 0:
        return 0.0

    weights = loan_portfolio["committed_amount"] / total_committed
    portfolio_beta = (weights * loan_portfolio["climate_beta"]).sum()

    return round(portfolio_beta, 4)


if __name__ == "__main__":
    from data_loader import fetch_market_data, build_synthetic_loan_portfolio
    from climate_factors import construct_climate_factors

    # Fetch data
    returns = fetch_market_data(["JPM", "BAC", "C", "WFC"], start="2015-01-01")
    factors = construct_climate_factors(returns)

    # Estimate betas
    bank_cols = [c for c in returns.columns if c not in ["SPY", "XLE", "ICLN", "KOL"]]
    bank_rets = returns[bank_cols]

    cb, mb = estimate_climate_betas_ewma(bank_rets, factors["stranded_asset"], factors["mkt"])
    print("\n--- Latest Betas ---")
    print(get_latest_betas(cb, mb))

    # Loan portfolio beta
    loans = build_synthetic_loan_portfolio("JPM")
    print(f"\nLoan portfolio climate beta: {compute_loan_portfolio_climate_beta(loans):.4f}")
