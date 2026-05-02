# Climate Risk Capital Impact Model

A Python-based model for quantifying the capital impact of climate risk factors on bank capital adequacy (**ICAAP**) and **Risk-Weighted Assets (RWA)**.

## Based On

- **CRISK Framework** — Jung, Engle, Giglio (*Journal of Financial Economics*, 2025)
- **BIS Staff Report sr977** — Climate risk and financial stability

## Key Features

| Module | Description |
|--------|-------------|
| `config.py` | Bank universe (10 banks), sector definitions (14 sectors), model parameters |
| `data_loader.py` | Market data via yfinance + synthetic Y-14-style loan portfolios |
| `climate_factors.py` | 4 climate transition risk factors (Stranded Asset, Emission, BMG, CEP) |
| `climate_beta.py` | EWMA + rolling OLS time-varying climate beta estimation |
| `crisk_engine.py` | CRISK, mCRISK, S&CRISK, LRMES computation (Equations 5-9) |
| `rwa_engine.py` | Basel IRB capital requirements with climate PD stress overlay |
| `icaap_engine.py` | ICAAP Pillar 2 climate buffer integration |
| `app.py` | Streamlit dashboard with 5 interactive tabs |

## Core Formulas

**CRISK** (Eq. 7):
```
CRISK = k * D - (1 - k) * W * exp(beta_climate * log(1 - theta))
```

**mCRISK** (Eq. 8):
```
mCRISK = (1 - k) * W * LRMES
```

**LRMES** (Eq. 6):
```
LRMES = 1 - exp(beta_climate * log(1 - theta))
```

**Climate-Adjusted RWA**:
```
PD_stressed = PD_base * (1 + sensitivity * beta * theta * (1 + 0.5 * beta))
RWA = 12.5 * K(PD_stressed, LGD, M) * EAD
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the Dashboard
```bash
streamlit run app.py
```

### Use as a Library
```python
from crisk_engine import compute_snapshot
from rwa_engine import compute_total_rwa_impact
from data_loader import build_synthetic_loan_portfolio
from icaap_engine import compute_icaap_assessment

# CRISK snapshot for a bank
snapshot = compute_snapshot("JPM", climate_beta=0.45, market_beta=1.1)

# RWA impact
loans = build_synthetic_loan_portfolio("JPM")
impact = compute_total_rwa_impact(loans, stress_level=0.50)

# Full ICAAP assessment
assessment = compute_icaap_assessment("JPM", 0.45, 1.1, loans)
```

## Dashboard Tabs

1. **Overview** — Multi-bank comparison with aggregate metrics
2. **Climate Betas** — Time-varying beta evolution charts
3. **CRISK Analysis** — Per-bank capital shortfall time series
4. **RWA Impact** — Sector-level PD stress and RWA waterfall
5. **ICAAP Summary** — Capital adequacy waterfall, sensitivity heatmap

## License

For research and educational purposes.
