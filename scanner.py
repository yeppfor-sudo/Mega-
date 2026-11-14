import os, io, json, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

BASE = "https://contract.mexc.com/api/v1/contract"
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

TFS = {"1d": 1, "2d": 2, "3d": 3, "5d": 5, "7d": 7}
MIN_SCORE = int(os.environ.get("MIN_SCORE", "70"))
MIN_DRAWDOWN = float(os.environ.get("MIN_DRAWDOWN", "0.55"))
MIN_AVG_TURNOVER = float(os.environ.get("MIN_TURNOVER", "20000"))
INCLUDE_STOCKS = os.environ.get("INCLUDE_STOCKS", "0") == "1"
WATERMARK = os.environ.get("WATERMARK", "MEXC")
STATE_FILE = "sent.json"

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

# ---------- MEXC style colors ----------
UP = "#14B375"
DN = "#F2495D"
C_E6 = "#F0B90B"
C_E12 = "#B05CF0"
C_E20 = "#3B82F6"


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

    t = c.iloc[-1]
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

    score = 20 + 10 + 20
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


# ---------- MEXC style chart ----------
def fmt_price(x):
    ax_ = abs(x)
    if ax_ >= 1000:
        return f"{x:,.2f}"
    if ax_ >= 1:
        return f"{x:.4f}"
    if ax_ >= 0.01:
        return f"{x:.5f}"
    if ax_ >= 0.0001:
        return f"{x:.6f}"
    return f"{x:.8f}"


def make_chart(all_df, trig_idx, sym, tf):
    df = add_emas(all_df.copy()).iloc[-70:]
    n = len(df)
    x = np.arange(n)
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    v = df["Volume"].values
    col = np.where(c >= o, UP, DN)

    fig = plt.figure(figsize=(7.5, 9), dpi=120, facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[4.4, 1], hspace=0.03,
                          left=0.02, right=0.86, top=0.855, bottom=0.05)
    ax = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, axv):
        a.set_facecolor("white")
        for s in a.spines.values():
            s.set_visible(False)
        a.grid(True, color="#EEF0F3", lw=0.8)
        a.set_axisbelow(True)

    # ----- header: timeframe tabs like MEXC -----
    tabs = ["1D", "2D", "3D", "5D", "7D"]
    active = tf.upper()
    for i, t in enumerate(tabs):
        xx = 0.09 + i * 0.15
        is_act = (t == active)
        fig.text(xx, 0.955, t, ha="center", va="center", fontsize=15,
                 color="#111111" if is_act else "#8A919E",
                 weight="bold" if is_act else "normal")
        if is_act:
            fig.add_artist(Line2D([xx - 0.035, xx + 0.035], [0.932, 0.932],
                                  color="#1F6FEB", lw=2.5, transform=fig.transFigure))
    fig.add_artist(Line2D([0, 1], [0.928, 0.928], color="#E6E8EB", lw=1,
                          transform=fig.transFigure))

    fig.text(0.03, 0.902, sym.replace("_", "") + "  Perpetual", fontsize=14,
             weight="bold", color="#111111", va="center")
    chg = (c[-1] / o[-1] - 1) * 100
    fig.text(0.97, 0.902, f"{fmt_price(c[-1])}  {chg:+.2f}%", fontsize=11,
             weight="bold", color=UP if chg >= 0 else DN, va="center", ha="right")
    fig.text(0.03, 0.874, f"EMA6: {fmt_price(df['e6'].iloc[-1])}", fontsize=8.5,
             color=C_E6, va="center")
    fig.text(0.31, 0.874, f"EMA12: {fmt_price(df['e12'].iloc[-1])}", fontsize=8.5,
             color=C_E12, va="center")
    fig.text(0.60, 0.874, f"EMA20: {fmt_price(df['e20'].iloc[-1])}", fontsize=8.5,
             color=C_E20, va="center")

    # ----- candles -----
    rng = max(h.max() - l.min(), 1e-12)
    ax.vlines(x, l, h, colors=col, linewidth=1.1, zorder=2)
    body_h = np.maximum(np.abs(c - o), rng * 0.002)
    ax.bar(x, body_h, bottom=np.minimum(o, c), width=0.72, color=col, zorder=3)

    ax.plot(x, df["e6"].values, color=C_E6, lw=1.1, zorder=4)
    ax.plot(x, df["e12"].values, color=C_E12, lw=1.1, zorder=4)
    ax.plot(x, df["e20"].values, color=C_E20, lw=1.2, zorder=4)

    pad = rng * 0.10
    ax.set_ylim(l.min() - pad, h.max() + pad)
    ax.set_xlim(-1, n + 4)

    # watermark
    ax.text(0.55, 0.52, WATERMARK, transform=ax.transAxes, fontsize=58, weight="bold",
            color="#9AA0A6", alpha=0.18, ha="center", va="center", zorder=0)

    # ----- high / low labels (like MEXC) -----
    hi_i, lo_i = int(np.argmax(h)), int(np.argmin(l))
    for i, val in ((hi_i, h[hi_i]), (lo_i, l[lo_i])):
        dx = 5 if i < n - 14 else -5
        ax.annotate(fmt_price(val), xy=(i, val), xytext=(i + dx, val),
                    ha="left" if dx > 0 else "right", va="center",
                    fontsize=9, color="#222222", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#8A919E", lw=0.9,
                                    shrinkA=2, shrinkB=0))

    # ----- signal marker on trigger candle -----
    if trig_idx in df.index:
        pos = df.index.get_loc(trig_idx)
        ax.scatter([pos], [l[pos] - pad * 0.55], marker="^", s=110, color=UP,
                   edgecolors="white", linewidths=0.8, zorder=7)
        ax.text(pos, l[pos] - pad * 0.95, "SIGNAL", ha="center", va="top",
                fontsize=7.5, color=UP, weight="bold", zorder=7)

    # ----- last price line + tag -----
    last_col = UP if c[-1] >= o[-1] else DN
    ax.axhline(c[-1], color=last_col, ls="--", lw=0.9, zorder=1)
    ax.text(1.005, c[-1], fmt_price(c[-1]), transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8.5, color="white", zorder=9,
            clip_on=False,
            bbox=dict(boxstyle="square,pad=0.28", fc=last_col, ec="none"))

    # ----- axes style -----
    ax.yaxis.tick_right()
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, _: fmt_price(val)))
    ax.tick_params(axis="y", colors="#8A919E", labelsize=8, length=0)
    ax.tick_params(axis="x", labelbottom=False, length=0)

    # ----- volume -----
    axv.bar(x, v, color=col, width=0.72, alpha=0.9)
    axv.set_ylim(0, max(v.max(), 1e-12) * 1.25)
    axv.set_yticks([])
    axv.text(0.01, 0.88, "Vol", transform=axv.transAxes, fontsize=8, color="#8A919E",
             va="top")
    step = max(n // 5, 1)
    ticks = list(range(0, n, step))
    axv.set_xticks(ticks)
    axv.set_xticklabels([df.index[i].strftime("%m-%d") for i in ticks])
    axv.tick_params(axis="x", colors="#8A919E", labelsize=8, length=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
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
