import os, io, json, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf

BASE = "https://contract.mexc.com/api/v1/contract"
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

TFS = {"1d": 1, "2d": 2, "3d": 3, "5d": 5, "7d": 7}
MIN_SCORE = int(os.environ.get("MIN_SCORE", "70"))
MIN_DRAWDOWN = float(os.environ.get("MIN_DRAWDOWN", "0.55"))   # 55% crash from peak
MIN_AVG_TURNOVER = float(os.environ.get("MIN_TURNOVER", "20000"))  # USDT/day
INCLUDE_STOCKS = os.environ.get("INCLUDE_STOCKS", "0") == "1"
STATE_FILE = "sent.json"

# stocks / ETFs / metals / indices that must be skipped for crypto-only mode
NON_CRYPTO = {
    "XAU", "XAG", "GOLD", "SILVER", "OIL", "USOIL", "UKOIL", "SPY", "QQQ", "IWM",
    "TSLA", "NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NFLX", "AMD",
    "INTC", "COIN", "MSTR", "HOOD", "PLTR", "CRCL", "CRWV", "BABA", "UBER", "DIS",
    "PYPL", "SHOP", "SQ", "ORCL", "AVGO", "TSM", "MU", "SMCI", "ARM", "GME", "AMC",
    "NKE", "KO", "PEP", "JPM", "BAC", "V", "MA", "WMT", "COST", "BA", "F", "GM",
    "CBRS", "SPX", "NDX", "DJI", "EURUSD", "GBPUSD", "USDJPY",
}

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0"})


def get_json(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 + i * 2)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1 + i)
    return None


def get_symbols():
    j = get_json(f"{BASE}/detail")
    out = []
    if not j or not j.get("data"):
        return out
    for d in j["data"]:
        sym = d.get("symbol", "")
        if not sym.endswith("_USDT"):
            continue
        if d.get("state", 0) != 0:
            continue
        base = d.get("baseCoin", sym.split("_")[0]).upper()
        if not INCLUDE_STOCKS and base in NON_CRYPTO:
            continue
        out.append(sym)
    return sorted(set(out))


def fetch_daily(sym):
    now = int(time.time())
    j = get_json(f"{BASE}/kline/{sym}",
                 {"interval": "Day1", "start": now - 1100 * 86400, "end": now})
    try:
        d = j["data"]
        n = len(d["time"])
        if n < 40:
            return None
        df = pd.DataFrame({
            "ts": np.array(d["time"], dtype="int64"),
            "open": np.array(d["open"], dtype=float),
            "high": np.array(d["high"], dtype=float),
            "low": np.array(d["low"], dtype=float),
            "close": np.array(d["close"], dtype=float),
            "vol": np.array(d["vol"], dtype=float),
        })
        df["turn"] = df["vol"] * df["close"]
        return df
    except Exception:
        return None


def resample(d, n):
    now = time.time()
    day = d["ts"] // 86400
    off = 3 if n == 7 else 0          # 7d candles start on Monday
    b = (day + off) // n
    g = d.groupby(b)
    out = pd.DataFrame({
        "ts": g["ts"].first(),
        "Open": g["open"].first(),
        "High": g["high"].max(),
        "Low": g["low"].min(),
        "Close": g["close"].last(),
        "Volume": g["vol"].sum(),
    })
    start_day = out.index.values * n - off
    out["closed"] = ((start_day + n) * 86400) <= now
    out.index = pd.to_datetime(out["ts"], unit="s")
    return out


def add_emas(df):
    for n in (6, 12, 20):
        df[f"e{n}"] = df["Close"].ewm(span=n, adjust=False).mean()
    return df


def analyze(c):
    """c = closed candles only (with EMAs). Returns dict or None."""
    if len(c) < 45:
        return None
    look = c.iloc[-150:]
    peak = look["High"].max()
    peak_pos = int(np.argmax(look["High"].values))
    bars_since_peak = len(look) - 1 - peak_pos
    low_all = look["Low"].min()
    if peak <= 0 or low_all <= 0:
        return None
    drawdown = 1 - low_all / peak
    if drawdown < MIN_DRAWDOWN or bars_since_peak < 15:
        return None

    t = c.iloc[-1]       # trigger = last closed candle
    p = c.iloc[-2]
    green = t["Close"] > t["Open"]
    above_all = t["Close"] > max(t["e6"], t["e12"], t["e20"])
    fresh = p["Close"] <= max(p["e6"], p["e12"], p["e20"])
    if not (green and above_all and fresh):
        return None

    base = c.iloc[-21:-1]
    base_low = base["Low"].min()
    near_lows = base_low <= low_all * 1.25 and p["Close"] <= low_all * 1.8
    q80, q20 = base["Close"].quantile(0.8), base["Close"].quantile(0.2)
    flat = q20 > 0 and (q80 / q20) <= 1.45
    spread = (max(p["e6"], p["e12"], p["e20"]) - min(p["e6"], p["e12"], p["e20"])) / p["Close"]
    converged = spread <= 0.08
    body = abs(t["Close"] - t["Open"])
    avg_body = (base["Close"] - base["Open"]).abs().mean()
    strong = (body / t["Close"] >= 0.04) or (avg_body > 0 and body >= 1.5 * avg_body)
    not_pumped = t["Close"] <= base_low * 1.8

    score = 20 + 10 + 20  # crash + green above EMAs + fresh cross
    score += 15 if flat else 0
    score += 15 if converged else 0
    score += 10 if near_lows else 0
    score += 10 if strong else 0
    if not not_pumped:
        score -= 20
    if score < MIN_SCORE:
        return None
    return {
        "score": int(score),
        "drawdown": round(drawdown * 100, 1),
        "gain": round((t["Close"] / t["Open"] - 1) * 100, 1),
        "price": float(t["Close"]),
        "trigger_ts": int(t["ts"]),
        "trigger_idx": c.index[-1],
        "flat": bool(flat), "converged": bool(converged),
    }


def make_chart(all_df, trig_idx, sym, tf):
    df = add_emas(all_df.copy()).iloc[-90:]
    mk = pd.Series(np.nan, index=df.index)
    if trig_idx in mk.index:
        mk.loc[trig_idx] = df.loc[trig_idx, "Low"] * 0.93
    aps = [
        mpf.make_addplot(df["e6"], color="#00e5ff", width=1),
        mpf.make_addplot(df["e12"], color="#ffd600", width=1),
        mpf.make_addplot(df["e20"], color="#ff4081", width=1.2),
        mpf.make_addplot(mk, type="scatter", marker="^", markersize=140, color="lime"),
    ]
    style = mpf.make_mpf_style(base_mpf_style="nightclouds", gridstyle=":")
    buf = io.BytesIO()
    mpf.plot(df[["Open", "High", "Low", "Close", "Volume"]], type="candle", style=style,
             addplot=aps, volume=True, title=f"{sym}  {tf}  (live)",
             savefig=dict(fname=buf, dpi=110, format="png"), figsize=(9, 6))
    buf.seek(0)
    return buf.getvalue()


def send_photo(img, caption):
    if not TG_TOKEN or not TG_CHAT:
        print("Telegram secrets missing:\n", caption)
        return
    r = requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                      data={"chat_id": TG_CHAT, "caption": caption, "parse_mode": "HTML"},
                      files={"photo": ("chart.png", img)}, timeout=60)
    if not r.ok:
        print("Telegram error", r.text)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st):
    cutoff = time.time() - 30 * 86400
    st = {k: v for k, v in st.items() if v > cutoff}
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


def scan_symbol(sym):
    d = fetch_daily(sym)
    if d is None:
        return None
    if d["turn"].tail(10).mean() < MIN_AVG_TURNOVER:
        return None
    hits = []
    frames = {}
    for tf, n in TFS.items():
        r = resample(d, n)
        frames[tf] = r
        closed = add_emas(r[r["closed"]].copy())
        res = analyze(closed)
        if res:
            res["tf"] = tf
            hits.append(res)
    if not hits:
        return None
    return sym, hits, frames


def main():
    state = load_state()
    syms = get_symbols()
    print("pairs:", len(syms))
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(scan_symbol, syms):
            if r:
                results.append(r)
            time.sleep(0.05)
    print("matches:", len(results))

    sent = 0
    for sym, hits, frames in results:
        new = [h for h in hits if f"{sym}|{h['tf']}|{h['trigger_ts']}" not in state]
        if not new:
            continue
        best = max(new, key=lambda h: h["score"])
        try:
            img = make_chart(frames[best["tf"]][["Open", "High", "Low", "Close", "Volume"]],
                             best["trigger_idx"], sym, best["tf"])
        except Exception as e:
            print("chart error", sym, e)
            continue
        tfs = ", ".join(h["tf"] for h in new)
        base = sym.replace("_USDT", "")
        cap = (f"🟢 <b>{sym}</b> — bottom reversal\n"
               f"Timeframes: {tfs}\n"
               f"Score: {best['score']}/100 ({best['tf']})\n"
               f"Crash from peak: -{best['drawdown']}%\n"
               f"Trigger candle: +{best['gain']}%\n"
               f"Price: {best['price']:.8g}\n"
               f"https://futures.mexc.com/exchange/{sym}")
        send_photo(img, cap)
        now = time.time()
        for h in new:
            state[f"{sym}|{h['tf']}|{h['trigger_ts']}"] = now
        sent += 1
        time.sleep(1.5)
    save_state(state)
    print("alerts sent:", sent)


if __name__ == "__main__":
    main()
