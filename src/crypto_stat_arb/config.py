from __future__ import annotations

# Hourly bars
BARS_PER_DAY = 24
ANNUALIZATION = 24 * 365

# Trading costs (bps → rate)
COST_BPS = 7
COST_RATE = COST_BPS / 10_000

# Default benchmark symbol for residualization / alpha-beta
BENCH_DEFAULT = "BTCUSDT"