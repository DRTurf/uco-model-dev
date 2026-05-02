"""
Climate Risk Capital Impact Model -- Configuration
===================================================
Central configuration for all model parameters, based on:
  - CRISK (Jung, Engle, Giglio -- JFE 2025)
  - BIS Staff Report sr977

All calibration defaults are from the papers unless noted.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# -------------------------------------------------------------
# Bank Universe
# -------------------------------------------------------------
BANK_TICKERS: Dict[str, str] = {
    "JPM":  "JPMorgan Chase",
    "BAC":  "Bank of America",
    "C":    "Citigroup",
    "WFC":  "Wells Fargo",
    "GS":   "Goldman Sachs",
    "MS":   "Morgan Stanley",
    "USB":  "US Bancorp",
    "PNC":  "PNC Financial",
    "TFC":  "Truist Financial",
    "COF":  "Capital One",
}

# -------------------------------------------------------------
# ETF Tickers for Climate Factor Construction
# -------------------------------------------------------------
MARKET_ETF   = "SPY"       # S&P 500 proxy for MKT factor
ENERGY_ETF   = "XLE"       # Energy Sector SPDR -- stranded assets
COAL_ETF     = "KOL"       # VanEck Coal ETF
CLEAN_ETF    = "ICLN"      # iShares Global Clean Energy
SP500_ETF    = "SPY"       # Market benchmark

# -------------------------------------------------------------
# Model Parameters (from papers)
# -------------------------------------------------------------
PRUDENTIAL_RATIO_US     = 0.08    # k = 8% equity-to-assets (US banks)
PRUDENTIAL_RATIO_EU     = 0.055   # k = 5.5% (European banks)
STRESS_HORIZON_MONTHS   = 6       # h = 6 months
STRESS_QUANTILE         = 0.01    # 1st percentile of factor return dist.
DEFAULT_STRESS_LEVEL    = 0.50    # θ = 50% decline in climate factor

# GARCH / DCC parameters
GARCH_P = 1
GARCH_Q = 1
DCC_A   = 0.05   # DCC intercept (initial guess)
DCC_B   = 0.93   # DCC persistence (initial guess)

# Rolling OLS fallback window
ROLLING_WINDOW_DAYS = 252  # 1-year trading days

# -------------------------------------------------------------
# Data Settings
# -------------------------------------------------------------
DATA_START_DATE = "2005-01-01"
DATA_END_DATE   = None  # None = today

# -------------------------------------------------------------
# Synthetic Loan Portfolio -- Sector Definitions
# -------------------------------------------------------------
@dataclass
class SectorConfig:
    """Configuration for a single loan-portfolio sector."""
    name: str
    naics_code: str
    base_pd: float          # Annual probability of default (%)
    lgd: float              # Loss-given-default (%)
    climate_beta: float     # Sector-level climate beta (stylized)
    weight: float           # Share of total loan portfolio (%)
    maturity_years: float   # Average loan maturity

# Calibrated to approximate large US bank loan composition
# Brown sectors have higher climate betas, consistent with Fig. 6 of CRISK paper
SECTOR_CONFIGS: List[SectorConfig] = [
    SectorConfig("Oil & Gas Extraction",        "211", 2.50, 45, 1.80, 8.0,  4.0),
    SectorConfig("Mining Support Activities",    "213", 3.00, 50, 2.20, 3.0,  3.5),
    SectorConfig("Coal & Petroleum Mfg",        "324", 2.80, 45, 1.60, 4.0,  4.0),
    SectorConfig("Primary Metal Mfg",           "331", 2.00, 40, 1.30, 3.0,  4.5),
    SectorConfig("Mining (non O&G)",             "212", 2.20, 42, 1.50, 2.0,  4.0),
    SectorConfig("Utilities",                    "221", 1.20, 35, 0.70, 7.0,  6.0),
    SectorConfig("Transportation",               "481", 1.50, 38, 0.50, 6.0,  5.0),
    SectorConfig("Manufacturing",                "333", 1.00, 35, 0.30, 12.0, 4.0),
    SectorConfig("Real Estate",                  "531", 0.90, 30, 0.20, 15.0, 7.0),
    SectorConfig("Technology",                   "511", 0.60, 30, -0.10, 10.0, 3.0),
    SectorConfig("Healthcare",                   "621", 0.50, 25, 0.05, 8.0,  4.0),
    SectorConfig("Financial Services",           "522", 0.70, 30, 0.15, 10.0, 3.5),
    SectorConfig("Retail Trade",                 "441", 1.10, 35, 0.10, 7.0,  3.0),
    SectorConfig("Professional Services",        "541", 0.40, 25, 0.00, 5.0,  3.0),
]

# -------------------------------------------------------------
# Bank Financial Data (stylized, for demonstration)
# -------------------------------------------------------------
@dataclass
class BankFinancials:
    """Stylized balance-sheet data for a bank."""
    ticker: str
    name: str
    total_assets_bn: float      # Total assets ($B)
    equity_market_bn: float     # Market cap ($B)
    debt_book_bn: float         # Book value of debt ($B)
    cet1_ratio: float           # CET1 ratio (%)
    rwa_bn: float               # Risk-weighted assets ($B)

# Approximate 2023-era financials for demonstration
BANK_FINANCIALS: Dict[str, BankFinancials] = {
    "JPM": BankFinancials("JPM", "JPMorgan Chase",   3900, 490, 3410, 15.2, 1700),
    "BAC": BankFinancials("BAC", "Bank of America",   3200, 310, 2890, 13.5, 1600),
    "C":   BankFinancials("C",   "Citigroup",         2400, 100, 2300, 13.0, 1300),
    "WFC": BankFinancials("WFC", "Wells Fargo",       1900, 180, 1720, 13.8, 1200),
    "GS":  BankFinancials("GS",  "Goldman Sachs",     1600, 130, 1470, 15.5,  700),
    "MS":  BankFinancials("MS",  "Morgan Stanley",    1200, 155, 1045, 16.0,  500),
    "USB": BankFinancials("USB", "US Bancorp",         680,  72,  608, 10.5,  475),
    "PNC": BankFinancials("PNC", "PNC Financial",      560,  68,  492, 10.2,  400),
    "TFC": BankFinancials("TFC", "Truist Financial",   540,  50,  490,  9.8,  395),
    "COF": BankFinancials("COF", "Capital One",        480,  55,  425, 13.1,  370),
}

# -------------------------------------------------------------
# Dashboard / Visualization
# -------------------------------------------------------------
CHART_TEMPLATE = "plotly_dark"
PRIMARY_COLOR  = "#00D4AA"
ACCENT_COLOR   = "#FF6B6B"
WARNING_COLOR  = "#FFD93D"
BG_COLOR       = "#0E1117"
CARD_BG        = "#1A1D23"
