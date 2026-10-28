import ccxt, pandas as pd, requests, os, io, time
import matplotlib.pyplot as plt
from datetime import datetime

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_text(msg):
    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                  data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})

def send_photo(fig, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    requests.post(url, data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
                  files={"photo": buf})

def ema(s,n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def is_doji(o,h,l,c):
    rng = h-l
    return rng>0 and abs(c-o)/rng < 0.25

def is_hammer(o,h,l,c):
    body = abs(c-o)
    if body==0: return False
    lower = min(o,c)-l
    upper = h-max(o,c)
    return lower > body*1.8 and upper < body*0.8

def is_star(o,h,l,c):
    body = abs(c-o)
    rng = h-l
    if rng==0: return False
    upper = h-max(o,c)
    lower = min(o,c)-l
    return (upper > body*1.5 or lower > body*1.5) and body/rng < 0.4

def is_bullish_candle(o,h,l,c):
    return is_doji(o,h,l,c) or is_hammer(o,h,l,c) or is_star(o,h,l,c)

def scan_reversal(df):
    if len(df) < 80: return None
    df = df.iloc[:-1].copy()  # previous closed only
    df['ema6'] = ema(df['c'],6)
    df['ema12'] = ema(df['c'],12)
    df['ema20'] = ema(df['c'],20)
    df['rsi'] = rsi(df['c'],14)
    df['vol_ma20'] = df['v'].rolling(20).mean()
    
    for end in range(len(df)-1, len(df)-8, -1):
        for win_size in [2,3,4,5,6,7,8]:
            start = end - win_size +1
            if start < 10: continue
            window = df.iloc[start:end+1]
            
            low_window = window['l'].min()
            low_60 = df.iloc[max(0,end-60):end]['l'].min()
            if low_window > low_60 * 1.08: continue  # must be at bottom/support/sweep
            
            ema_down = df['ema6'].iloc[end] < df['ema12'].iloc[end] and df['ema12'].iloc[end] < df['ema20'].iloc[end]*1.03
            below_ema = (df['c'].iloc[end-15:end] < df['ema20'].iloc[end-15:end]).sum() >= 10
            if not (ema_down or below_ema): continue
            
            green_cnt = (window['c'] > window['o']).sum()
            if green_cnt < 1: continue
            
            bullish_cnt = sum(is_bullish_candle(r['o'],r['h'],r['l'],r['c']) for _,r in window.iterrows())
            if bullish_cnt < 1: continue
            
            # === VOLUME FILTER ===
            vol_window_avg = window['v'].mean()
            vol_20_avg = df['vol_ma20'].iloc[end]
            green_vols = [r['v'] for _,r in window.iterrows() if r['c']>r['o']]
            max_green_vol = max(green_vols) if green_vols else 0
            vol_exhaustion = vol_window_avg < vol_20_avg * 0.9
            vol_spike_green = max_green_vol > vol_20_avg * 1.2
            if not (vol_exhaustion or vol_spike_green): continue
            
            # === NEW RSI OVERSOLD FILTER ===
            rsi_window = window['rsi'].min()
            rsi_curr = df['rsi'].iloc[end]
            # RSI must be oversold <35 at bottom like SAGA, VELVET, BLESS all at oversold
            if rsi_window > 38 and rsi_curr > 40: continue  # not oversold enough
            
            curr_price = df.iloc[end]['c']
            sl = low_window * 0.965
            risk = curr_price - sl
            if risk <=0: continue
            tp1 = curr_price + risk*1.5
            tp2 = curr_price + risk*2.8
            tp3 = df['ema20'].iloc[end]
            
            return {
                'end_idx': end, 'start_idx': start, 'win_size': win_size,
                'bottom': low_window, 'curr': curr_price, 'green': int(green_cnt),
                'bullish': int(bullish_cnt), 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
                'ema6': df['ema6'].iloc[end], 'ema12': df['ema12'].iloc[end], 'ema20': df['ema20'].iloc[end],
                'low_60': low_60, 'vol_avg': vol_window_avg, 'vol_ma20': vol_20_avg,
                'vol_spike': vol_spike_green, 'rsi': rsi_curr, 'rsi_low': rsi_window, 'df': df
            }
    return None

def draw_chart(res, symbol, exchange):
    df = res['df'].tail(90)
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(11,7), gridspec_kw={'height_ratios':[3,1,1]}, sharex=True)
    fig.patch.set_facecolor('white')
    
    for idx, row in df.iterrows():
        color = '#00b26a' if row['c']>=row['o'] else '#ff3b30'
        ax1.plot([idx, idx], [row['l'], row['h']], color=color, lw=1)
        ax1.plot([idx, idx], [row['o'], row['c']], color=color, lw=3)
    
    ax1.plot(df.index, df['ema6'], color='#e38bd7', lw=1.1, label=f"EMA6:{res['ema6']:.5f}")
    ax1.plot(df.index, df['ema12'], color='#f0a500', lw=1.1, label=f"EMA12:{res['ema12']:.5f}")
    ax1.plot(df.index, df['ema20'], color='#5b4cff', lw=1.2, label=f"EMA20:{res['ema20']:.5f}")
    ax1.axvspan(res['start_idx'], res['end_idx'], color='lightgreen', alpha=0.35)
    ax1.axhline(res['sl'], color='red', ls='--', lw=1.2, label=f"SL {res['sl']:.6f}")
    ax1.axhline(res['tp1'], color='#00b26a', ls=':', lw=1, label=f"TP1")
    ax1.axhline(res['tp2'], color='#00b26a', ls='-', lw=1.2, label=f"TP2 {res['tp2']:.6f}")
    
    ax1.set_title(f"{symbol} {exchange} 1D REVERSAL {res['win_size']}c RSI{res['rsi']:.1f} Green{res['green']}", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(alpha=0.2)
    
    colors = ['#00b26a' if c>=o else '#ff3b30' for c,o in zip(df['c'], df['o'])]
    ax2.bar(df.index, df['v'], color=colors, alpha=0.7)
    ax2.plot(df.index, df['vol_ma20'], color='#e38bd7', lw=1)
    ax2.set_ylabel("VOL"); ax2.grid(alpha=0.2)
    
    ax3.plot(df.index, df['rsi'], color='#5b4cff', lw=1.2)
    ax3.axhline(35, color='red', ls='--', lw=0.8, label='Oversold 35')
    ax3.axhline(70, color='green', ls='--', lw=0.8, label='Overbought 70')
    ax3.fill_between(df.index, 0, 35, color='lightgreen', alpha=0.2)
    ax3.set_ylabel("RSI14"); ax3.legend(fontsize=6); ax3.grid(alpha=0.2)
    plt.tight_layout()
    return fig

def main():
    exchanges = [
        ('MEXC', ccxt.mexc({'enableRateLimit':True, 'options':{'defaultType':'swap'}})),
        ('BYBIT', ccxt.bybit({'enableRateLimit':True, 'options':{'defaultType':'swap'}})),
        ('OKX', ccxt.okx({'enableRateLimit':True, 'options':{'defaultType':'swap'}})),
        ('BITGET', ccxt.bitget({'enableRateLimit':True, 'options':{'defaultType':'swap'}})),
        ('GATE', ccxt.gate({'enableRateLimit':True, 'options':{'defaultType':'swap'}})),
    ]
    total=0
    for ex_name, ex in exchanges:
        try:
            ex.load_markets()
            syms = [s for s in ex.markets if s.endswith('/USDT:USDT') and 'STOCK' not in s and 'ETF' not in s][:700]
            send_text(f"🔍 {ex_name} {len(syms)} perps 1D reversal + Vol + RSI<35")
            for sym in syms:
                try:
                    ohlcv = ex.fetch_ohlcv(sym, '1D', limit=120)
                    df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                    res = scan_reversal(df)
                    if res:
                        clean = sym.replace('/USDT:USDT','')
                        fig = draw_chart(res, clean, ex_name)
                        rr = (res['tp2']-res['curr'])/(res['curr']-res['sl'])
                        vol_text = "Spike" if res['vol_spike'] else "Exhaustion"
                        caption = f"🔥 <b>{clean} {ex_name} 1D BOTTOM</b>\n{res['win_size']} candles | Green:{res['green']} Bullish:{res['bullish']} Doji/Hammer/Star\nPrice: {res['curr']:.6f} Bottom: {res['bottom']:.6f}\nEMA6 {res['ema6']:.5f} EMA12 {res['ema12']:.5f} EMA20 {res['ema20']:.5f}\n📊 Vol {vol_text} RSI {res['rsi']:.1f} (Low {res['rsi_low']:.1f}) Oversold <35 ✅\n<b>SL:</b> {res['sl']:.6f} <b>TP1:</b> {res['tp1']:.6f} (1.5R) <b>TP2:</b> {res['tp2']:.6f} (2.8R)\nRR 1:{rr:.1f} | {datetime.utcnow().strftime('%Y-%m-%d')}"
                        send_photo(fig, caption)
                        total+=1
                        time.sleep(1.3)
                    time.sleep(0.07)
                except: time.sleep(0.3); continue
        except Exception as e:
            send_text(f"⚠️ {ex_name} {e}"); continue
    send_text(f"✅ Done - Found {total} reversal setups Vol + RSI<35 like your photos")

if __name__ == "__main__": main()
