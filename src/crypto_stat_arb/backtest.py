from __future__ import annotations
import numpy as np
import pandas as pd
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]          
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_stat_arb.config import ANNUALIZATION # type: ignore

def backtest(w: pd.DataFrame, R: pd.DataFrame, cost_rate: float):
    """
    Vectorized backtest with no lookahead (weights applied with shift()).
    Returns: (net_series, gross_series, summary_series)
    """
    cols = w.columns.intersection(R.columns)
    w, R = w[cols], R[cols]

    gross = (w.shift().fillna(0.0) * R).sum(axis=1)  # P&L per bar
    turnover = (w - w.shift()).abs().sum(axis=1).fillna(0.0)
    cost = cost_rate * turnover
    net = gross - cost

    def ann_sharpe(x: pd.Series) -> float:
        sd = x.std()
        return np.nan if sd == 0 or np.isnan(sd) else (x.mean() / sd) * np.sqrt(ANNUALIZATION)

    summary = {
        "sharpe_gross": ann_sharpe(gross),
        "sharpe_net":   ann_sharpe(net),
        "ann_ret_gross": gross.mean() * ANNUALIZATION,
        "ann_ret_net":   net.mean()   * ANNUALIZATION,
        "ann_vol_gross": gross.std() * np.sqrt(ANNUALIZATION),
        "ann_vol_net":   net.std()   * np.sqrt(ANNUALIZATION),
        "turnover_bar":  turnover.mean(),
        "turnover_py":   turnover.mean() * ANNUALIZATION,
        "cost_py":       cost.mean() * ANNUALIZATION,
    }
    return net, gross, pd.Series(summary)

def perf_stats(x: pd.Series) -> dict:
    mu, sd = x.mean(), x.std()
    sh = np.nan if sd == 0 or np.isnan(sd) else (mu / sd) * np.sqrt(ANNUALIZATION)
    return {"ann_ret": mu * ANNUALIZATION, "ann_vol": sd * np.sqrt(ANNUALIZATION), "sharpe": sh}

def max_drawdown(x: pd.Series) -> float:
    """
    Max drawdown on cumulative arithmetic equity (not log).
    Input is a return series per bar (arithmetic).
    """
    eq = (1 + x.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())