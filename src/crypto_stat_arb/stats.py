from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]          
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_stat_arb.config import ANNUALIZATION

def perf_summary_from_series(r: pd.Series, label: str, freq_per_year=ANNUALIZATION):
    """
    Summary stats on an arithmetic return series r (per bar).
    """
    r = r.dropna()
    if len(r) == 0:
        return {"label": label, "ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    ann_ret = (1 + r).prod()**(freq_per_year / len(r)) - 1
    ann_vol = r.std() * np.sqrt(freq_per_year)
    sharpe  = np.nan if ann_vol == 0 else ann_ret / ann_vol
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    max_dd = (eq / peak - 1.0).min()
    return {"label": label, "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}

def compute_alpha_beta(strategy_ret: pd.Series, bench_ret: pd.Series, freq=ANNUALIZATION, lag_bars=24):
    """
    OLS with HAC (Newey–West) standard errors.
    Returns annualized alpha, alpha t, beta, beta t, R², and n.
    """
    df = pd.concat([strategy_ret, bench_ret], axis=1).dropna()
    if df.empty:
        raise ValueError("No overlapping data between strategy and benchmark.")
    df.columns = ["y", "x"]
    X = sm.add_constant(df["x"])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": int(lag_bars)})

    return {
        "alpha_ann": model.params["const"] * freq,
        "alpha_t":   float(model.tvalues["const"]),
        "beta":      float(model.params["x"]),
        "beta_t":    float(model.tvalues["x"]),
        "R2":        float(model.rsquared),
        "n_obs":     int(model.nobs),
    }

def nw_mean_tstat(r: pd.Series, lag_bars: int = 24, freq=ANNUALIZATION):
    """
    Newey–West t-stat for the mean of r (per-bar arithmetic returns).
    Returns (mean_per_bar, t_stat, ann_mean).
    """
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan, np.nan

    X = np.ones((len(r), 1))
    model = sm.OLS(r.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(lag_bars)})
    mean_per_bar = float(model.params[0])
    t_stat = float(model.tvalues[0])
    ann_mean = mean_per_bar * freq
    return mean_per_bar, t_stat, ann_mean


def rolling_sharpe(x: pd.Series, window_bars: int = 90) -> pd.Series:
    """Rolling annualized Sharpe using trailing window (in bars)."""
    w = window_bars * 24
    mu = x.rolling(w).mean()
    sd = x.rolling(w).std()
    rs = (mu / sd) * np.sqrt(ANNUALIZATION)
    return rs

def perf_summary_from_series_exact(r: pd.Series, label: str, freq_per_year=ANNUALIZATION):
    """
      - ann_ret = geometric annualized return over the whole series
      - ann_vol = std * sqrt(freq_per_year)
      - sharpe  = ann_ret / ann_vol
      - max_dd  = min drawdown of cumulative simple return curve
    """
    r = r.dropna()
    if len(r) == 0:
        return {"label": label, "ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_dd": np.nan}

    ann_ret = (1.0 + r).prod() ** (freq_per_year / len(r)) - 1.0
    ann_vol = r.std() * np.sqrt(freq_per_year)
    sharpe  = np.nan if ann_vol == 0 else ann_ret / ann_vol

    eq = (1.0 + r).cumprod()
    max_dd = (eq / eq.cummax() - 1.0).min()

    return {"label": label, "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}
