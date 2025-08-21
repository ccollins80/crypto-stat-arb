from binance.client import Client as bnb_client
import pandas as pd, numpy as np, time, os
from typing import Optional, Union

# ---------- Config ----------
USE_BINANCE_US = True
ALLOWED_QUOTES = {"USDT"}          # Keep only USDT-quoted pairs for consistency
TOP_N = 20
INTERVAL = "1h"                  
START = "2023-01-01"
END   = None
PAUSE = 0.25
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")  # ../data folder
# -----------------------------

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
    stats = client.get_ticker()
    qv = {}
    for row in stats:
        sym = row["symbol"]
        if sym in candidates:
            try:
                qv[sym] = float(row["quoteVolume"])
            except:
                pass
    ranked = sorted(qv.items(), key=lambda kv: kv[1], reverse=True)
    return [s for s, _ in ranked[:top_n]]

def get_klines_df(client, symbol, interval, start_ts, end_ts=None, pause=0.25):
    start_ms = to_ms(start_ts)
    end_ms   = to_ms(end_ts)
    data = client.get_historical_klines(symbol, interval, start_ms, end_ms)
    cols = ['open_time','open','high','low','close','volume','close_time',
            'quote_volume','num_trades','taker_base_volume','taker_quote_volume','ignore']
    df = pd.DataFrame(data, columns=cols)
    if df.empty:
        return df
    df["open_time"]  = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    time.sleep(pause)
    return df.set_index("open_time")

def build_px_ret(client, symbols, interval, start_ts, end_ts=None):
    series = {}
    for sym in symbols:
        df = get_klines_df(client, sym, interval, start_ts, end_ts, pause=PAUSE)
        if df.empty:
            continue
        series[sym] = df["close"].astype(float).rename(sym)
    if not series:
        raise ValueError("No data returned. Check interval/range/symbols.")
    px = pd.concat(series.values(), axis=1).sort_index()

    full_idx = pd.date_range(px.index.min(), px.index.max(), freq=interval, tz="UTC")
    px = px.reindex(full_idx).ffill()
    px = px.dropna(how="any").copy()

    ret = np.log(px / px.shift(1)).dropna()
    return px, ret

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    candidates = get_spot_symbols(client, ALLOWED_QUOTES)
    univ = top_by_quote_volume(client, candidates, TOP_N)
    print("Universe:", univ)

    px, ret = build_px_ret(client, univ, INTERVAL, START, END)


    px_path = os.path.join(DATA_DIR, f"px_{INTERVAL}.csv")
    ret_path = os.path.join(DATA_DIR, f"ret_{INTERVAL}.csv")
    px.to_csv(px_path)
    ret.to_csv(ret_path)
    print(f"Saved px to {px_path}")
    print(f"Saved ret to {ret_path}") 