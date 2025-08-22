from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SRC = HERE.parents[1]         
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_stat_arb.config import BARS_PER_DAY, COST_BPS, ANNUALIZATION
from crypto_stat_arb.backtest import backtest, perf_stats
from crypto_stat_arb.portfolio import downsample_weights

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

    folds, oos_segments = [], []
    for i, (tr_slice, te_slice) in enumerate(splits, 1):
        R_train = R.iloc[tr_slice]
        R_test  = R.iloc[te_slice]

        R_fold = R.iloc[tr_slice.start:te_slice.stop]
        w_fold = weight_func(R_fold, **{k: v for k, v in params.items() if k != "every"})
        w_fold = downsample_weights(w_fold, every=params.get("every", 1)).loc[R_test.index]

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
    mix_mode: str = "equal_vol",   # "equal_vol" or "5050"
    train_days: int = 365,
    test_days: int = 90,
    mode: str = "expanding",
    cost_bps: int = COST_BPS,
    reversal_func=None,
    momentum_func=None,
):
    """
    Two-sleeve walk-forward with equal-vol or 50/50 blend.
    - Train sleeves on TRAIN to estimate blend weights (if equal_vol)
    - Recompute sleeves on TRAIN+TEST (no lookahead in rolling ops), slice TEST
    - Blend sleeves in TEST using train-estimated weights
    Returns (summary_df, stitched_oos_net).
    """
    assert reversal_func is not None and momentum_func is not None, "Provide reversal_func and momentum_func"
    splits = _wf_splits(R.index, train_days, test_days, mode)

    folds, oos_segments = [], []
    for i, (tr_slice, te_slice) in enumerate(splits, 1):
        R_train = R.iloc[tr_slice]
        R_test  = R.iloc[te_slice]

        w_rev_tr = reversal_func(R_train, **{k: v for k, v in params_rev.items() if k != "every"})
        w_rev_tr = downsample_weights(w_rev_tr, every=every_rev)
        w_mom_tr = momentum_func(R_train, **{k: v for k, v in params_mom.items() if k != "every"})
        w_mom_tr = downsample_weights(w_mom_tr, every=every_mom)

        net_rev_tr, _, _ = backtest(w_rev_tr, R_train, cost_rate=cost_bps / 10_000)
        net_mom_tr, _, _ = backtest(w_mom_tr, R_train, cost_rate=cost_bps / 10_000)

        if mix_mode == "equal_vol":
            vol_rev = net_rev_tr.std() * (ANNUALIZATION ** 0.5)
            vol_mom = net_mom_tr.std() * (ANNUALIZATION ** 0.5)
            w_rev_mix = 0.5 if vol_rev == 0 else 0.5 / vol_rev
            w_mom_mix = 0.5 if vol_mom == 0 else 0.5 / vol_mom
            s = w_rev_mix + w_mom_mix
            w_rev_mix /= s; w_mom_mix /= s
        else:
            w_rev_mix = w_mom_mix = 0.5

        R_fold = R.iloc[tr_slice.start:te_slice.stop]
        w_rev_fold = reversal_func(R_fold, **{k: v for k, v in params_rev.items() if k != "every"})
        w_rev_fold = downsample_weights(w_rev_fold, every=every_rev).loc[R_test.index]
        w_mom_fold = momentum_func(R_fold, **{k: v for k, v in params_mom.items() if k != "every"})
        w_mom_fold = downsample_weights(w_mom_fold, every=every_mom).loc[R_test.index]

        w_mix_test = (w_rev_mix * w_rev_fold) + (w_mom_mix * w_mom_fold)

        # Backtest in TEST
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
