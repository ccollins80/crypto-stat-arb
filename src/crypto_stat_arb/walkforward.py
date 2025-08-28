from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]         
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_stat_arb.config import BARS_PER_DAY, COST_BPS, ANNUALIZATION # type: ignore
from crypto_stat_arb.backtest import backtest, perf_stats # type: ignore

@dataclass
class WFConfig:
    train_days: int = 365
    test_days: int = 90
    mode: str = "expanding"  # "expanding" or "rolling"

def _wf_splits(index: pd.DatetimeIndex, train_days=365, test_days=90, mode="expanding"):
    train_bars = train_days * BARS_PER_DAY
    test_bars  = test_days  * BARS_PER_DAY
    N = len(index)
    splits = []
    train_end = train_bars
    while True:
        test_end = train_end + test_bars
        if test_end > N:
            break
        tr_start = 0 if mode == "expanding" else max(0, train_end - train_bars)
        tr_end   = train_end
        te_start = train_end
        te_end   = test_end
        splits.append((slice(tr_start, tr_end), slice(te_start, te_end)))
        train_end = test_end
    return splits

def _global_rebalance_mask(index: pd.DatetimeIndex, every: int | None) -> pd.Series:
    every = int(every or 1)
    if every <= 1:
        return pd.Series(True, index=index)  # rebalance every bar
    return pd.Series((np.arange(len(index)) % every) == 0, index=index)

def _apply_schedule(w: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    out = w.copy()
    out.loc[~mask] = np.nan
    return out.ffill()

def run_walk_forward(
    R: pd.DataFrame,
    params: dict,
    cost_bps: int = COST_BPS,
    cfg: WFConfig = WFConfig(),
    weight_func=None,
):
    """
    Single-strategy walk-forward.
    Build weights from TRAIN+TEST window (rolling ops prevent lookahead in signals),
    then apply in TEST slice with chosen rebalance cadence (params['every']).
    Returns (summary_df, stitched_oos_net).
    """
    assert weight_func is not None, "Provide weight_func (e.g., cs_reversal_weights)"
    splits = _wf_splits(R.index, cfg.train_days, cfg.test_days, cfg.mode)

    # build global mask once
    every = int(params.get("every", 1) or 1)
    mask_global = _global_rebalance_mask(R.index, every)

    folds, oos_segments = [], []
    for i, (tr_slice, te_slice) in enumerate(splits, 1):
        R_train = R.iloc[tr_slice]
        R_test  = R.iloc[te_slice]

        R_fold = R.iloc[tr_slice.start:te_slice.stop]
        w_fold = weight_func(R_fold, **{k: v for k, v in params.items() if k != "every"})

        # apply GLOBAL schedule within the fold window, then slice test
        mask_fold = mask_global.loc[R_fold.index]
        w_fold = _apply_schedule(w_fold, mask_fold).loc[R_test.index]

        net_te, _, _ = backtest(w_fold, R_test, cost_rate=cost_bps / 10_000)
        stats_te = perf_stats(net_te)

        folds.append({
            "fold": i, "mode": cfg.mode,
            "train_start": R_train.index[0], "train_end": R_train.index[-1],
            "test_start":  R_test.index[0],  "test_end":  R_test.index[-1],
            "train_days": (R_train.index[-1] - R_train.index[0]).days + 1,
            "test_days":  (R_test.index[-1]  - R_test.index[0]).days  + 1,
            "cost_bps": cost_bps,
            "test_ann_ret": stats_te["ann_ret"], "test_ann_vol": stats_te["ann_vol"], "test_sharpe": stats_te["sharpe"],
        })
        net_te.index = R_test.index
        oos_segments.append(net_te)

    return pd.DataFrame(folds), pd.concat(oos_segments).sort_index()

def run_walk_forward_mixed(
    R: pd.DataFrame,
    params_rev: dict,
    params_mom: dict,
    every_rev: int = 24,
    every_mom: int = 720,
    mix_mode: str = "equal_vol",   # "equal_vol" | "5050" | "static" | "train_opt"
    train_days: int = 365,
    test_days: int = 90,
    mode: str = "expanding",
    cost_bps: int = COST_BPS,
    reversal_func=None,
    momentum_func=None,
    w_mom_static: float | None = None,            # <— NEW (for mix_mode="static")
    opt_grid: np.ndarray | None = None,           # <— NEW (for mix_mode="train_opt")
):
    """
    Multi-sleeve walk-forward with equal-vol, 50/50 blend and static optimizer.
    - Train sleeves on TRAIN to estimate blend weights (if equal_vol)
    - Recompute sleeves on TRAIN+TEST (no lookahead in rolling ops), slice TEST
    - Blend sleeves in TEST using train-estimated weights
    Returns (summary_df, stitched_oos_net).
    """
    assert reversal_func is not None and momentum_func is not None, "Provide reversal_func and momentum_func"
    splits = _wf_splits(R.index, train_days, test_days, mode)

    mask_rev_global = _global_rebalance_mask(R.index, every_rev)
    mask_mom_global = _global_rebalance_mask(R.index, every_mom)

    folds, oos_segments = [], []
    for i, (tr_slice, te_slice) in enumerate(splits, 1):
        R_train = R.iloc[tr_slice]
        R_test  = R.iloc[te_slice]

        # Train-only sleeves (use GLOBAL schedule sliced to the train window)
        w_rev_tr = reversal_func(R_train, **{k: v for k, v in params_rev.items() if k != "every"})
        w_rev_tr = _apply_schedule(w_rev_tr, mask_rev_global.loc[R_train.index])
        w_mom_tr = momentum_func(R_train, **{k: v for k, v in params_mom.items() if k != "every"})
        w_mom_tr = _apply_schedule(w_mom_tr, mask_mom_global.loc[R_train.index])

        net_rev_tr, _, _ = backtest(w_rev_tr, R_train, cost_rate=cost_bps / 10_000)
        net_mom_tr, _, _ = backtest(w_mom_tr, R_train, cost_rate=cost_bps / 10_000)

        if mix_mode == "equal_vol":
            vol_rev = net_rev_tr.std() * (ANNUALIZATION ** 0.5)
            vol_mom = net_mom_tr.std() * (ANNUALIZATION ** 0.5)
            w_rev_mix = 0.5 if vol_rev == 0 else 0.5 / vol_rev
            w_mom_mix = 0.5 if vol_mom == 0 else 0.5 / vol_mom
            s = w_rev_mix + w_mom_mix
            w_rev_mix /= s; w_mom_mix /= s

        elif mix_mode == "5050":
            w_rev_mix = w_mom_mix = 0.5

        elif mix_mode == "static":
            assert w_mom_static is not None, "Provide w_mom_static when mix_mode='static'"
            w_mom_mix = float(w_mom_static)
            w_rev_mix = 1.0 - w_mom_mix

        elif mix_mode == "train_opt":
            grid = opt_grid if opt_grid is not None else np.linspace(0.0, 1.0, 51)
            best_w, best_s = 0.5, -np.inf
            for w in grid:
                mix_tr = (1 - w) * net_rev_tr + w * net_mom_tr
                s = perf_stats(mix_tr.dropna())["sharpe"]
                if np.isfinite(s) and s > best_s:
                    best_s, best_w = s, float(w)
            w_mom_mix = best_w
            w_rev_mix = 1.0 - best_w

        else:
            raise ValueError(f"Unknown mix_mode: {mix_mode}")


        R_fold = R.iloc[tr_slice.start:te_slice.stop]
        w_rev_fold = reversal_func(R_fold, **{k: v for k, v in params_rev.items() if k != "every"})
        w_rev_fold = _apply_schedule(w_rev_fold, mask_rev_global.loc[R_fold.index]).loc[R_test.index]

        w_mom_fold = momentum_func(R_fold, **{k: v for k, v in params_mom.items() if k != "every"})
        w_mom_fold = _apply_schedule(w_mom_fold, mask_mom_global.loc[R_fold.index]).loc[R_test.index]

        w_mix_test = (w_rev_mix * w_rev_fold) + (w_mom_mix * w_mom_fold)
        net_te, _, _ = backtest(w_mix_test, R_test, cost_rate=cost_bps / 10_000)
        stats_te = perf_stats(net_te)

        folds.append({
            "fold": i, "mode": mode,
            "train_start": R_train.index[0], "train_end": R_train.index[-1],
            "test_start":  R_test.index[0],  "test_end":  R_test.index[-1],
            "train_days": (R_train.index[-1] - R_train.index[0]).days + 1,
            "test_days":  (R_test.index[-1]  - R_test.index[0]).days  + 1,
            "mix_mode": mix_mode, "cost_bps": cost_bps,
            "w_rev_mix": w_rev_mix, "w_mom_mix": w_mom_mix,
            "test_ann_ret": stats_te["ann_ret"], "test_ann_vol": stats_te["ann_vol"], "test_sharpe": stats_te["sharpe"],
        })
        net_te.index = R_test.index
        oos_segments.append(net_te)

    return pd.DataFrame(folds), pd.concat(oos_segments).sort_index()
