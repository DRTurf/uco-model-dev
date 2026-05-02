"""
Climate Risk Capital Impact Model -- Data Loader
=================================================
Fetches market data via yfinance and generates synthetic
loan portfolio data for demonstration.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from config import (
    BANK_TICKERS, MARKET_ETF, ENERGY_ETF, COAL_ETF, CLEAN_ETF,
    DATA_START_DATE, DATA_END_DATE, SECTOR_CONFIGS, SectorConfig,
    BANK_FINANCIALS, BankFinancials,
)


def fetch_market_data(
    bank_tickers: Optional[List[str]] = None,
    start: str = DATA_START_DATE,
    end: Optional[str] = DATA_END_DATE,
) -> pd.DataFrame:
    """
    Download adjusted-close daily prices for banks + factor ETFs.

    Returns
    -------
    pd.DataFrame
        Daily log-returns, columns = tickers.
    """
    if bank_tickers is None:
        bank_tickers = list(BANK_TICKERS.keys())

    all_tickers = bank_tickers + [MARKET_ETF, ENERGY_ETF, CLEAN_ETF]

    # Deduplicate while preserving order
    seen = set()
    unique_tickers = []
    for t in all_tickers:
        if t not in seen:
            unique_tickers.append(t)
            seen.add(t)

    print(f" Downloading market data for {len(unique_tickers)} tickers…")

    try:
        data = yf.download(
            unique_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        print(f"[WARN]️  yfinance download failed: {e}")
        print("   Generating synthetic market data as fallback.")
        return _generate_synthetic_market_data(unique_tickers, start, end)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    if prices.empty:
        print("[WARN]️  Empty data returned. Generating synthetic data.")
        return _generate_synthetic_market_data(unique_tickers, start, end)

    # Flatten MultiIndex columns if present
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)

    # Compute log-returns and drop NaN rows
    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"[OK] Loaded {len(returns)} trading days, {returns.shape[1]} series")
    return returns


def _generate_synthetic_market_data(
    tickers: List[str],
    start: str,
    end: Optional[str],
) -> pd.DataFrame:
    """Generate realistic synthetic daily returns when live data unavailable."""
    dates = pd.bdate_range(start=start, end=end or "2024-12-31")
    np.random.seed(42)
    n = len(dates)
    k = len(tickers)

    # Correlated returns via Cholesky decomposition
    mean_ret = 0.0003  # ~7.5% annualized
    daily_vol = 0.015

    # Create correlation structure: banks correlated ~0.6, ETFs ~0.3
    corr = np.eye(k) * 0.4 + 0.6
    np.fill_diagonal(corr, 1.0)
    cov = corr * daily_vol**2

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < 0):
        cov += np.eye(k) * (abs(eigvals.min()) + 1e-6)

    L = np.linalg.cholesky(cov)
    raw = np.random.randn(n, k)
    returns = raw @ L.T + mean_ret

    df = pd.DataFrame(returns, index=dates, columns=tickers)
    print(f"[OK] Generated synthetic data: {n} days, {k} series")
    return df


def build_synthetic_loan_portfolio(
    bank_ticker: str = "JPM",
    num_loans_per_sector: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic loan portfolio mimicking Y-14 structure.

    Each loan has:
      - sector, NAICS code
      - committed_amount, utilized_amount
      - base_pd, lgd, maturity
      - climate_beta (sector-level)
      - interest_rate_spread

    Returns
    -------
    pd.DataFrame with one row per loan.
    """
    np.random.seed(seed)
    records = []

    financials = BANK_FINANCIALS.get(bank_ticker)
    if financials is None:
        total_loans_bn = 500.0  # fallback
    else:
        total_loans_bn = financials.rwa_bn * 0.6  # ~60% of RWA from loans

    for sector in SECTOR_CONFIGS:
        sector_alloc = total_loans_bn * (sector.weight / 100.0)

        for _ in range(num_loans_per_sector):
            committed = np.random.lognormal(
                mean=np.log(sector_alloc / num_loans_per_sector),
                sigma=0.5,
            )
            utilization = np.random.uniform(0.5, 1.0)
            utilized = committed * utilization

            # PD with some noise around the base
            pd_val = max(0.01, np.random.normal(sector.base_pd, sector.base_pd * 0.2))

            # Interest spread correlated with PD
            spread_bps = max(50, np.random.normal(150 + pd_val * 80, 30))

            records.append({
                "bank":               bank_ticker,
                "sector":             sector.name,
                "naics_code":         sector.naics_code,
                "committed_amount":   round(committed, 4),
                "utilized_amount":    round(utilized, 4),
                "base_pd_pct":        round(pd_val, 4),
                "lgd_pct":            sector.lgd,
                "maturity_years":     round(np.random.normal(sector.maturity_years, 0.5), 1),
                "climate_beta":       round(sector.climate_beta + np.random.normal(0, 0.05), 4),
                "interest_spread_bps": round(spread_bps, 1),
            })

    df = pd.DataFrame(records)
    print(f"[OK] Built synthetic loan portfolio for {bank_ticker}: "
          f"{len(df)} loans across {len(SECTOR_CONFIGS)} sectors")
    return df


def get_bank_financials(ticker: str) -> BankFinancials:
    """Return stylized financials for a given bank ticker."""
    if ticker in BANK_FINANCIALS:
        return BANK_FINANCIALS[ticker]
    raise ValueError(f"No financials configured for {ticker}")


if __name__ == "__main__":
    # Quick test
    returns = fetch_market_data(["JPM", "BAC"], start="2020-01-01")
    print(returns.tail())
    print()
    loans = build_synthetic_loan_portfolio("JPM")
    print(loans.head(10))
