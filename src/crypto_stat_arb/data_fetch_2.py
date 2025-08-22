#!/usr/bin/env python3
"""
Full historical hourly (1h) crypto prices from Binance / Binance.US
with pagination from START to END (or now), universe selection by
Top N quote volume (USDT pairs), and CSV output for prices & returns.
"""

from binance.client import Client as bnb_client
import pandas as pd, numpy as np, time, os, math, sys
from typing import Optional, Union

# ─────────────────────────────── Config ───────────────────────────────
USE_BINANCE_US = True
ALLOWED_QUOTES = {"USDT"}        # keep only USDT-quoted pairs
TOP_N = 20
INTERVAL = "1h"                  # e.g., "15m","1h","4h","1d",...
START = "2023-01-01"
END   = None                     # None => up to exchange "now"
PAUSE = 0.25                     # pause between API calls (rate-limit friendly)
LIMIT = 1000                     # klines per page (API max is usually 1000)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
# ---------------------------------------------------------------------

# Map Binance intervals → pandas frequencies (for reindexing)
PD_FREQ_MAP = {
    "1m":"1min","3m":"3min","5m":"5min","15m":"15min","30m":"30min",
    "1h":"1h","2h":"2h","4h":"4h","6h":"6h","8h":"8h","12h":"12h",
    "1d":"1D","3d":"3D","1w":"1W","1M":"1MS"
}

client = bnb_client(tld='us' if USE_BINANCE_US else 'com')

STABLE_BASES = {"USDT","USDC","BUSD","DAI","TUSD","FDUSD","USD"}
BAD_SUFFIXES = ("UPUSDT","DOWNUSDT","BULLUSDT","BEARUSDT","UP","DOWN","BULL","BEAR")

def to_ms(ts: Optional[Union[str, pd.Timestamp]]) -> Optional[int]:
    if ts is None:
        return None
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)

def get_spot_symbols(client, allowed_quotes):
    """All SPOT symbols that are TRADING, USDT-quoted, exclude leveraged/multi-token suffixes & stablecoin bases."""
    info = client.get_exchange_info()
    syms = []
    for s in info["symbols"]:
        if s.get("status") != "TRADING":
            continue
        perms = set(s.get("permissions", []))
        if "SPOT" not in perms:
            continue
        base, quote, symbol = s["baseAsset"], s["quoteAsset"], s["symbol"]
        if quote not in allowed_quotes:
            continue
        if any(symbol.endswith(suf) for suf in BAD_SUFFIXES):
            continue
        if base in STABLE_BASES:
            continue
        syms.append(symbol)
    return syms

def top_by_quote_volume(client, candidates, top_n):
    """Rank by 24h quoteVolume and return top_n symbols."""
    stats = client.get_ticker()
    qv = {}
    for row in stats:
        sym = row.get("symbol")
        if sym in candidates:
            try:
                qv[sym] = float(row["quoteVolume"])
            except Exception:
                pass
    ranked = sorted(qv.items(), key=lambda kv: kv[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]

def get_klines_df(client, symbol, interval, start_ts, end_ts=None, pause=0.25, limit=1000, max_pages=200000):
    """
    Robust kline fetcher that paginates from start_ts to end_ts (or now),
    returning a full DataFrame indexed by open_time.
    """
    start_ms = to_ms(start_ts)
    end_ms   = to_ms(end_ts)  # None -> open-ended to exchange "now"
    rows = []
    pages = 0

    while True:
        batch = client.get_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ms,
            endTime=end_ms,
            limit=limit
        )
        if not batch:
            break

        rows.extend(batch)
        pages += 1

        last_open_ms = batch[-1][0]

        # If we specified an end and we've reached/passed it, stop
        if end_ms is not None and last_open_ms >= end_ms:
            break

        # If fewer than limit returned, likely caught up to end/now
        if len(batch) < limit:
            # Try one more pull; if empty, we'll exit next loop
            start_ms = last_open_ms + 1
            time.sleep(pause)
            continue

        # Advance the window by 1ms past the last open_time to avoid duplicates
        start_ms = last_open_ms + 1

        time.sleep(pause)

        if pages >= max_pages:
            print(f"[WARN] Stopping pagination for {symbol}: reached max_pages={max_pages}", file=sys.stderr)
            break

    cols = [
        'open_time','open','high','low','close','volume','close_time',
        'quote_volume','num_trades','taker_base_volume','taker_quote_volume','ignore'
    ]
    if not rows:
        return pd.DataFrame(columns=cols).set_index('open_time')

    df = pd.DataFrame(rows, columns=cols)
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    for c in ["open","high","low","close","volume","quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index("open_time").sort_index()
    return df

def build_px_ret(client, symbols, interval, start_ts, end_ts=None, pause=0.25, limit=1000):
    """
    Fetch closes for each symbol, DROP any symbol that doesn't have data at or before start_ts,
    align on a common window, forward-fill gaps within the window, and compute log returns.
    """
    # Normalize START as UTC timestamp for comparisons
    min_start = pd.Timestamp(start_ts)
    if min_start.tzinfo is None:
        min_start = min_start.tz_localize("UTC")
    else:
        min_start = min_start.tz_convert("UTC")

    series = {}
    starts, ends = {}, {}

    # 1) Fetch close series per symbol
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] Fetching {sym} ...", flush=True)
        df = get_klines_df(client, sym, interval, start_ts, end_ts, pause=pause, limit=limit)
        if df.empty:
            print(f"  -> No data for {sym} (skipping).")
            continue

        s = df["close"].astype(float).rename(sym).sort_index()
        starts[sym] = s.index.min()
        ends[sym]   = s.index.max()
        series[sym] = s
        print(f"  -> {sym} range: {starts[sym]} → {ends[sym]}  ({len(s)} rows)")

    if not series:
        raise ValueError("No data returned for any symbol. Check interval/range/symbols/network.")

    # 2) Keep only symbols whose history includes (or precedes) START
    keep = [sym for sym, s in series.items() if s.index.min() <= min_start]
    drop = sorted(set(series.keys()) - set(keep))

    if drop:
        print(f"\nDropping {len(drop)} late-start symbols (first candle after {min_start}): {drop}")
    if not keep:
        raise ValueError("All symbols start after START. Choose an earlier START or different universe.")

    # 3) Determine common window across the kept symbols
    kept_starts = [starts[sym] for sym in keep]
    kept_ends   = [ends[sym]   for sym in keep]

    # Start at requested START (or later if any kept symbol begins even later—shouldn't happen after filter)
    common_start = max([min_start] + kept_starts)
    # End at earliest last-available timestamp across kept symbols
    common_end   = min(kept_ends)

    print(f"\nCommon window (kept symbols): {common_start} → {common_end}")

    # 4) Concatenate and align on a full time grid within common window
    px_raw = pd.concat([series[sym] for sym in keep], axis=1).sort_index()

    pd_freq = PD_FREQ_MAP.get(interval, None)
    if pd_freq is None:
        raise ValueError(f"No pandas frequency mapping for interval={interval}. Add it to PD_FREQ_MAP.")

    full_idx = pd.date_range(common_start, common_end, freq=pd_freq, tz="UTC")
    px = px_raw.reindex(full_idx).ffill().dropna(how="any").copy()

    # 5) Log returns
    ret = np.log(px / px.shift(1)).dropna()

    print(f"\nFinal shapes — px: {px.shape}, ret: {ret.shape}")
    return px, ret


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Building symbol universe...")
    candidates = get_spot_symbols(client, ALLOWED_QUOTES)
    univ = top_by_quote_volume(client, candidates, TOP_N)
    print("Universe:", univ)

    print("\nFetching price history...")
    px, ret = build_px_ret(client, univ, INTERVAL, START, END, pause=PAUSE, limit=LIMIT)

    # Final sanity
    print(f"\nPrices shape: {px.shape} | Returns shape: {ret.shape}")
    print(f"Date range: {px.index.min()} → {px.index.max()}  (~{len(px)} rows)")

    px_path = os.path.join(DATA_DIR, f"px_{INTERVAL}.csv")
    ret_path = os.path.join(DATA_DIR, f"ret_{INTERVAL}.csv")
    px.to_csv(px_path)
    ret.to_csv(ret_path)
    print(f"Saved px to {px_path}")
    print(f"Saved ret to {ret_path}")

if __name__ == "__main__":
    main()
