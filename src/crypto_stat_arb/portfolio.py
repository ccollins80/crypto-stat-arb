from __future__ import annotations
import numpy as np
import pandas as pd

def downsample_weights(w: pd.DataFrame, every: int | None = None) -> pd.DataFrame:
    """
    Hold weights constant between rebalances (every N bars).
    If every is None or <=1, return original w.
    """
    if every is None or every <= 1:
        return w
    out = w.copy()
    mask = np.arange(len(out)) % every != 0
    out.iloc[mask] = np.nan
    return out.ffill()