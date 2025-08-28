from __future__ import annotations
import numpy as np
import pandas as pd
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]          # this is .../src
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_stat_arb.config import BENCH_DEFAULT # type: ignore

# ----------------- helpers -----------------

def residualize_to_bench(R: pd.DataFrame, bench: str | None, beta_win: int | None) -> pd.DataFrame:
    """
    Rolling regression on bench to remove alpha/beta; returns residuals. Drops bench column.
    If bench missing or beta_win=None, returns R unchanged.
    """
    if bench is None or beta_win is None or bench not in R.columns:
        return R.copy()
    x = R[bench]
    cov = R.rolling(beta_win, min_periods=beta_win).cov(x)
    var = x.rolling(beta_win, min_periods=beta_win).var()
    beta = cov.div(var, axis=0)
    alpha = R.rolling(beta_win, min_periods=beta_win).mean() - beta.mul(
        x.rolling(beta_win, min_periods=beta_win).mean(), axis=0
    )
    resid = R - (alpha + beta.mul(x, axis=0))
    return resid.drop(columns=[bench], errors="ignore")

def _zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)

def _neutral_l1(w: pd.DataFrame) -> pd.DataFrame:
    w = w.sub(w.mean(axis=1), axis=0)         
    l1 = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(l1, axis=0).fillna(0.0)       

# ----------------- strategies -----------------

def cs_reversal_weights(
    R: pd.DataFrame,
    k: int = 4,
    band: float = 2.5,
    beta_win: int | None = 168,
    bench: str | None = BENCH_DEFAULT,
    vol_win: int | None = 24,
) -> pd.DataFrame:
    """
    Cross-sectional reversal (mean reversion over k bars).
    - signal = - rolling k-bar cumulative return
    - z-score across assets
    - hard band: zero weights where |z| < band
    - optional inverse-vol scaling (rolling std with window vol_win)
    - L1-normalize & neutralize each bar
    """
    X = residualize_to_bench(R, bench, beta_win)
    mom = X.rolling(k, min_periods=k).sum()
    sig = -mom
    z = _zscore_xs(sig)
    if band and band > 0:
        z = z.where(z.abs() >= band, 0.0)
    w = z.copy()
    if vol_win and vol_win > 1:
        vol = X.rolling(vol_win, min_periods=vol_win).std().replace(0, np.nan)
        w = w.div(vol, axis=1)
    return _neutral_l1(w)

def cs_momentum_weights(
    R: pd.DataFrame,
    k: int = 400,
    band: float = 2.5,
    beta_win: int | None = None,
    bench: str | None = BENCH_DEFAULT,
    vol_win: int | None = None,
) -> pd.DataFrame:
    """
    Cross-sectional momentum (trend over k bars).
    - signal = rolling k-bar cumulative return of residualized returns, but on X.shift(1)
    - z-score across assets
    - hard band: zero weights where |z| < band
    - optional inverse-vol scaling
    - L1-normalize & neutralize each bar
    """
    X = residualize_to_bench(R, bench, beta_win)
    X_lag = X.shift(24)
    mom = X_lag.rolling(k, min_periods=k).sum()
    sig = mom
    z = _zscore_xs(sig)
    if band and band > 0:
        z = z.where(z.abs() >= band, 0.0)
    w = z.copy()
    if vol_win and vol_win > 1:
        vol = X.rolling(vol_win, min_periods=vol_win).std().replace(0, np.nan)
        w = w.div(vol, axis=1)
    return _neutral_l1(w)
