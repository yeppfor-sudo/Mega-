import ccxt, pandas as pd, requests, os, time, io
import matplotlib.pyplot as plt
from datetime import datetime

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_text(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode":"HTML"})

def send_photo(fig, caption):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    requests.post(url, data={"chat_id": CHAT_ID, "caption": caption, "parse_mode":"HTML"}, files={"photo": buf})

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def is_ain_squeeze(df):
    if len(df) < 100: return None
    i = len(df)-1
    close = df['c']
    df['ema6'] = ema(close, 6)
    df['ema12'] = ema(close, 12)
    df['ema20'] = ema(close, 20)
    
    curr = close.iloc[i]
    high_recent = df['h'].iloc[i-30:i].max()
    low_96 = df['l'].iloc[i-96:i].min()
    pump = (high_recent / low_96 -1)*100 if low_96 else 0
    
    if pump < 60:  # AIN pumped 0.06112 -> 0.16818 = 175%, need 60%+ 
        return None
    
    last15 = df.iloc[i-15:i+1]
    squeeze = (last15['h'].max() - last15['l'].min()) / last15['l'].min() * 100
    if squeeze > 5:  # your requirement within 5% flat
        return None
    
    ema6 = df['ema6'].iloc[i]
    ema12 = df['ema12'].iloc[i]
    ema20 = df['ema20'].iloc[i]
    if curr < ema20: return None  # hold EMA
    if abs(ema6 - ema20)/ema20*100 > 4: return None  # EMA cluster squeeze
    if curr < high_recent * 0.85: return None
    
    vol_pump = df['v'].iloc[i-96:i-20].max()
    vol_now = df['v'].iloc[i-10:i].mean()
    if vol_now > vol_pump * 0.9: return None
    
    return {"pump": round(pump,1), "squeeze": round(squeeze,2), "price": curr, "high": high_recent, "low": low_96, "ema6": ema6, "ema12": ema12, "ema20": ema20}

def draw_chart(df, symbol, tf, info):
    d = df.tail(100).copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), gridspec_kw={'height_ratios': [3,1]}, sharex=True)
    fig.patch.set_facecolor('white')
    
    for idx, row in d.iterrows():
        color = '#00b26a' if row['c'] >= row['o'] else '#ff3b30'
        ax1.plot([idx, idx], [row['l'], row['h']], color=color, linewidth=1)
        ax1.plot([idx, idx], [row['o'], row['c']], color=color, linewidth=3)
    
    ax1.plot(d.index, d['ema6'], color='#e38bd7', linewidth=1.2, label=f"EMA6:{info['ema6']:.5f}")
    ax1.plot(d.index, d['ema12'], color='#f0a500', linewidth=1.2, label=f"EMA12:{info['ema12']:.5f}")
    ax1.plot(d.index, d['ema20'], color='#5b4cff', linewidth=1.2, label=f"EMA20:{info['ema20']:.5f}")
    
    ax1.set_title(f"{symbol} {tf} - Pump {info['pump']}% Squeeze {info['squeeze']}% Ready to Break", fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(alpha=0.2)
    
    colors = ['#00b26a' if c>=o else '#ff3b30' for c,o in zip(d['c'], d['o'])]
    ax2.bar(d.index, d['v'], color=colors, alpha=0.6)
    ax2.plot(d.index, d['v'].rolling(5).mean(), color='#e38bd7', linewidth=1, label='MA5:VOL')
    ax2.plot(d.index, d['v'].rolling(10).mean(), color='#6bb6ff', linewidth=1, label='MA10:VOL')
    ax2.legend(fontsize=7)
    plt.tight_layout()
    return fig

def main():
    ex = ccxt.mexc({'enableRateLimit': True})
    ex.load_markets()
    tickers = ex.fetch_tickers()
    perps = {k:v for k,v in tickers.items() if k.endswith('/USDT:USDT') and v.get('percentage') is not None and 'STOCK' not in k}
    sorted_gainers = sorted(perps.items(), key=lambda x: x[1]['percentage'], reverse=True)
    top_symbols = [s for s,_ in sorted_gainers[:60]]
    
    send_text(f"🚀 Scanning {len(top_symbols)} MEXC 24h top gainers - AIN type 10m/15m")
    
    for tf in ['10m', '15m']:
        for sym in top_symbols:
            try:
                ohlcv = ex.fetch_ohlcv(sym, tf, limit=150)
                df = pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
                res = is_ain_squeeze(df)
                if res:
                    clean = sym.replace('/USDT:USDT','')
                    df['ema6'] = ema(df['c'],6); df['ema12']=ema(df['c'],12); df['ema20']=ema(df['c'],20)
                    fig = draw_chart(df, clean, tf, res)
                    caption = f"🔥 <b>{clean} {tf}</b>\nPump +{res['pump']}% Low {res['low']:.5f} -> High {res['high']:.5f}\nNow {res['price']:.5f} Squeeze {res['squeeze']}% (5% flat)\nEMA6 {res['ema6']:.5f} EMA12 {res['ema12']:.5f} EMA20 {res['ema20']:.5f}\n<b>Hold EMA + Ready to Break like AIN</b>"
                    send_photo(fig, caption)
                    time.sleep(1)
                time.sleep(0.2)
            except:
                time.sleep(0.3)
                continue

if __name__ == "__main__":
    main()
