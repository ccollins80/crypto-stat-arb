from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def load_panels(data_dir: Path, px_name="px_1h.csv", ret_name="ret_1h.csv"):
    """
    Load price and return panels (RAW).
    - Reads CSVs
    - Intersects columns so both frames share the same symbols
    - Sorts indexes
    - Does NOT align indexes, forward-fill, or drop rows

    Returns
    -------
    px : pd.DataFrame
        Raw prices with common columns, sorted index.
    ret : pd.DataFrame
        Raw returns with common columns, sorted index.
    """
    px = pd.read_csv(data_dir / px_name, index_col=0, parse_dates=True).sort_index()
    ret = pd.read_csv(data_dir / ret_name, index_col=0, parse_dates=True).sort_index()

    common = px.columns.intersection(ret.columns)
    px = px.loc[:, common]
    ret = ret.loc[:, common]

    return px, ret
 # type: ignore