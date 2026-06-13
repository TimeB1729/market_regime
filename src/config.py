"""
Configuration and constants for the Market Regime Detection project.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
FIGURES_DIR  = OUTPUT_DIR / "figures"
REPORTS_DIR  = OUTPUT_DIR / "reports"
DATA_DIR     = OUTPUT_DIR / "data"

for _d in (FIGURES_DIR, REPORTS_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
TICKER         = "INFY.NS"
START_DATE     = "2010-01-01"
TRAIN_END_DATE = "2020-12-31"

# ── Model ────────────────────────────────────────────────────────────────────
N_COMPONENTS       = 4
RANDOM_SEED        = 42
N_ITER             = 200
COV_TYPE           = "full"     # "diag" | "full" | "tied" | "spherical"

# ── Feature engineering ──────────────────────────────────────────────────────
ROLLING_VOL_WINDOW = 20
MOMENTUM_WINDOW    = 20
VOL_ZSCORE_WINDOW  = 20

# ── Regime labels ─────────────────────────────────────────────────────────────
REGIME_LABELS = {
    0: "Trending Bull",
    1: "Trending Bear",
    2: "Ranging / Sideways",
    3: "High Volatility / Crisis",
}

REGIME_COLORS = {
    "Trending Bull":            "#2ecc71",   # green
    "Trending Bear":            "#e74c3c",   # red
    "Ranging / Sideways":       "#3498db",   # blue
    "High Volatility / Crisis": "#e67e22",   # orange
}

# ── Changepoint detection ────────────────────────────────────────────────────
MIN_REGIME_DURATION   = 10     # trading days
OCCUPANCY_THRESHOLD   = 0.05   # minimum regime frequency to be considered valid
CONFIDENCE_THRESHOLD  = 0.70   # minimum posterior probability for a "confident" day

# ── Annualisation ─────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
