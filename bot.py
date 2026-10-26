import os, ccxt, pandas as pd, requests, json, time, pytz
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("CHAT_ID")
PKT = pytz.timezone('Asia/Karachi')
HIST_FILE = "alerts_history.json"
TIMEFRAMES = ['2h','4h','6h','8h','12h','1d','3d','5d','7d','30d']
COOLDOWN = 24 # once per pair per 24h

def send_msg(t):
    requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", json={"chat_id":CHAT,"text":t,"parse_mode":"Markdown"}, timeout=20)
def send_photo(p,c):
    with open(p,'rb') as f:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto", data={"chat_id":CHAT,"caption":c,"parse_mode":"Markdown"}, files={"photo":f}, timeout=30)
def load_hist():
    try:
        with open(HIST_FILE,'r') as f: return json.load(f)
    except: return {}
def save_hist(d):
    with open(HIST_FILE,'w') as f: json.dump(d,f)

exchange = ccxt.mexc({'enableRateLimit': True, 'options':{'defaultType':'swap'}})
exchange.load_markets()

def fetch_tf(sym, tf):
    try:
        if tf in ['2h','4h','6h','8h','12h']:
            h=int(tf.replace('h',''))
            ohlcv=exchange.fetch_ohlcv(sym,'1h',limit=500)
            df=pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            df['time']=pd.to_datetime(df['t'], unit='ms', utc=True)
            df.set_index('time', inplace=True)
            df=df.resample(f'{h}h').agg({'o':'first','h':'max','l':'min','c':'last','v':'sum'}).dropna().reset_index()
            return df
        elif tf=='1d':
            ohlcv=exchange.fetch_ohlcv(sym,'1d',limit=200)
            df=pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            df['time']=pd.to_datetime(df['t'], unit='ms', utc=True)
            return df
        else:
            rule={'3d':'3D','5d':'5D','7d':'7D','30d':'30D'}[tf]
            need=900 if tf=='30d' else 400
            ohlcv=exchange.fetch_ohlcv(sym,'1d',limit=need)
            df=pd.DataFrame(ohlcv, columns=['t','o','h','l','c','v'])
            df['time']=pd.to_datetime(df['t'], unit='ms', utc=True)
            df.set_index('time', inplace=True)
            df=df.resample(rule).agg({'o':'first','h':'max','l':'min','c':'last','v':'sum'}).dropna().reset_index()
            return df
    except: return None

def check(df):
    if len(df)<60: return None
    df['ema5']=df['c'].ewm(span=5).mean()
    df['ema10']=df['c'].ewm(span=10).mean()
    df['ema20']=df['c'].ewm(span=20).mean()
    df['vol_sma']=df['v'].rolling(20).mean()
    tr1=df['h']-df['l']
    tr2=(df['h']-df['c'].shift()).abs()
    tr3=(df['l']-df['c'].shift()).abs()
    tr=pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
    df['atr']=tr.ewm(span=14).mean()
    df['body']=(df['c']-df['o']).abs()
    df['body_avg']=df['body'].rolling(10).mean()

    last=df.iloc[-1]
    prev=df.iloc[-20:-1]

    # SETUP: Exhaustion
    if (prev['c'] < prev['ema20']).sum() < 14: return None
    if df['ema20'].iloc[-1] >= df['ema20'].iloc[-15]: return None
    if df['v'].iloc[-11:-1].mean() > df['v'].iloc[-30:-11].mean()*0.9: return None
    if abs(last['ema5']-last['ema10'])/last['ema20'] > 0.015: return None

    # TRIGGER: CHOCH Volume
    if pd.isna(last['vol_sma']) or last['vol_sma']==0: return None
    vol_mult = last['v']/last['vol_sma']
    if vol_mult < 3.5: return None
    if not (last['c'] > last['o']): return None
    if last['c']-last['o'] < last['atr']*1.0: return None
    if last['c']-last['o'] < last['body_avg']*1.5: return None
    # For your 6 photos - must break ALL EMAs
    if not (last['c'] > last['ema5'] and last['c'] > last['ema10'] and last['c'] > last['ema20']): return None
    swing_high=df.iloc[-6:-1]['h'].max()
    if not (last['c'] > swing_high): return None
    upper_wick=last['h']-max(last['o'],last['c'])
    if (upper_wick/(last['h']-last['l'])) > 0.30: return None

    return {'price':float(last['c']), 'vol_mult':round(float(vol_mult),2), 'atr':float(last['atr']),
            'swing':float(swing_high), 'ema5':float(last['ema5']), 'ema20':float(last['ema20']),
            'entry':float(last['c']), 'sl':float(last['l']*0.985)}

def draw(df,sym,tf,s):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[3,1]}, facecolor='black')
    ax1.set_facecolor('black'); ax2.set_facecolor('black')
    view=df.tail(50).copy()
    x=range(len(view))
    ax1.plot(x,view['ema5'],color='#ff69f5',lw=1,label='EMA5')
    ax1.plot(x,view['ema10'],color='#ffab00',lw=1,label='EMA10')
    ax1.plot(x,view['ema20'],color='#7c4dff',lw=1.2,label='EMA20')
    for i,(_,r) in enumerate(view.iterrows()):
        col='#00ff88' if r['c']>=r['o'] else '#ff4444'
        ax1.plot([i,i],[r['l'],r['h']],color=col,lw=1)
        ax1.plot([i,i],[r['o'],r['c']],color=col,lw=4)
    ax1.axhline(s['swing'],color='cyan',ls='--',lw=1)
    ax1.set_title(f"{sym} {tf} | CHOCH BREAK {s['price']:.6f} Vol {s['vol_mult']}x", color='white', fontsize=11)
    colors=['#00ff88' if c>=o else '#ff4444' for c,o in zip(view['c'],view['o'])]
    ax2.bar(x,view['v'],color=colors,alpha=0.7)
    ax2.plot(x,view['vol_sma'],color='white',lw=1)
    plt.tight_layout()
    path=f"chart_{sym.replace('/','')}_{tf}.png"
    plt.savefig(path,dpi=150,facecolor='black')
    plt.close()
    return path

hist=load_hist()
syms=[s for s in exchange.markets if s.endswith('/USDT:USDT')]
try:
    tickers=exchange.fetch_tickers()
    syms=[s for s in syms if tickers.get(s,{}).get('quoteVolume',0)>500_000]
except: pass
syms=syms[:120]
found=0
now=datetime.now(PKT)

for sym in syms:
    if sym in hist and time.time()-hist[sym] < COOLDOWN*3600: continue
    for tf in TIMEFRAMES:
        df=fetch_tf(sym,tf)
        if df is None or len(df)<60: continue
        setup=check(df)
        if not setup: continue
        risk=setup['entry']-setup['sl']
        tp1=setup['entry']+risk*2
        tp2=setup['entry']+risk*3.5
        chart=draw(df,sym,tf,setup)
        cap=f"🟢 *EXHAUSTION CHOCH*\n\n*Pair:* {sym}\n*TF:* {tf} Closed\n*Price:* `{setup['price']:.6f}`\n*Vol:* `{setup['vol_mult']}x` >=3.5x\n*Break:* > EMA5/10/20 + Swing {setup['swing']:.6f}\n*Entry:* `{setup['entry']:.6f}`\n*SL:* `{setup['sl']:.6f}`\n*TP1:* `{tp1:.6f}` 2R\n*TP2:* `{tp2:.6f}` 3.5R\n*Time PKT:* {now.strftime('%Y-%m-%d %I:%M %p')}"
        send_photo(chart,cap)
        hist[sym]=time.time()
        save_hist(hist)
        found+=1
        time.sleep(1)
        break

if found==0:
    send_msg("*please patience*")
