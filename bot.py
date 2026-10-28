import ccxt, pandas as pd, requests, os, io, time
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_text(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=15)
        print(f"TG send {r.status_code}: {r.text[:300]}")
        return r.status_code == 200
    except Exception as e:
        print(f"TG FAIL {e}")
        return False

print("=== BOT START ===")
print(f"BOT_TOKEN exists: {bool(BOT_TOKEN)} len {len(BOT_TOKEN) if BOT_TOKEN else 0}")
print(f"CHAT_ID: {CHAT_ID}")

# TEST TELEGRAM FIRST - YOU WILL RECEIVE THIS
if not BOT_TOKEN or not CHAT_ID:
    print("ERROR: Secrets missing! Go to Repo > Settings > Secrets and variables > Actions > New secret")
    print("Need BOT_TOKEN and CHAT_ID")
    exit(1)

ok = send_text(f"🚀 <b>Test #36 - Bot Started</b>\nTime {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\nIf you see this, secrets are OK. Now scanning...")
if not ok:
    print("Telegram failed - check BOT_TOKEN/CHAT_ID")
    exit(1)

# If you get test message, continue scanning with simple version
def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, p=14):
    d=s.diff(); g=(d.where(d>0,0)).ewm(alpha=1/p, adjust=False).mean(); l=(-d.where(d<0,0)).ewm(alpha=1/p, adjust=False).mean()
    return 100-(100/(1+g/l))

def is_doji(o,h,l,c): return (h-l)>0 and abs(c-o)/(h-l) < 0.3
def is_hammer(o,h,l,c):
    b=abs(c-o); lo=min(o,c)-l; up=h-max(o,c)
    return b>0 and lo>b*1.5 and up<b

def is_bull(o,h,l,c): return is_doji(o,h,l,c) or is_hammer(o,h,l,c)

def scan(df):
    if len(df)<70: return None
    df=df.iloc[:-1].copy()
    df['ema6']=ema(df['c'],6); df['ema12']=ema(df['c'],12); df['ema20']=ema(df['c'],20)
    df['rsi']=rsi(df['c'],14); df['vm']=df['v'].rolling(20).mean()
    for end in range(len(df)-1, len(df)-10, -1):
        for ws in [2,3,4,5,6]:
            s=end-ws+1; w=df.iloc[s:end+1]
            low=w['l'].min(); low60=df.iloc[max(0,end-60):end]['l'].min()
            if low>low60*1.15: continue
            if (w['c']>w['o']).sum()<1: continue
            if sum(is_bull(r['o'],r['h'],r['l'],r['c']) for _,r in w.iterrows())<1: continue
            if w['rsi'].min()>50: continue
            curr=df.iloc[end]['c']; sl=low*0.965; risk=curr-sl
            if risk<=0: continue
            return {'s':s,'e':end,'ws':ws,'bottom':low,'curr':curr,'sl':sl,'tp1':curr+risk*1.5,'tp2':curr+risk*2.8,'df':df,'rsi':df['rsi'].iloc[end]}
    return None

def chart(res, sym, ex):
    df=res['df'].tail(80); fig,ax=plt.subplots(figsize=(11,5))
    for idx,row in df.iterrows():
        col='#00b26a' if row['c']>=row['o'] else '#ff3b30'
        ax.plot([idx,idx],[row['l'],row['h']],color=col,lw=1); ax.plot([idx,idx],[row['o'],row['c']],color=col,lw=3)
    ax.plot(df.index,df['ema6'],color='#e38bd7',lw=1); ax.plot(df.index,df['ema12'],color='#f0a500',lw=1); ax.plot(df.index,df['ema20'],color='#5b4cff',lw=1.2)
    ax.axvspan(res['s'],res['e'],color='lightgreen',alpha=0.3); ax.axhline(res['sl'],color='red',ls='--'); ax.axhline(res['tp2'],color='green',ls='-')
    ax.set_title(f"{sym} {ex} 1D REVERSAL RSI{res['rsi']:.1f}"); ax.grid(alpha=0.2)
    return fig

def send_photo(fig, cap):
    try:
        buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=150,bbox_inches='tight'); buf.seek(0); plt.close(fig)
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto", data={"chat_id":CHAT_ID,"caption":cap,"parse_mode":"HTML"}, files={"photo":buf}, timeout=20)
    except Exception as e: print(f"photo fail {e}")

exchanges=[('MEXC',ccxt.mexc({'enableRateLimit':True,'options':{'defaultType':'swap'}})),('BYBIT',ccxt.bybit({'enableRateLimit':True,'options':{'defaultType':'swap'}}))]
total=0; scanned=0
for exn,ex in exchanges:
    try:
        ex.load_markets(); syms=[s for s in ex.markets if s.endswith('/USDT:USDT')][:400]
        for sym in syms:
            try:
                ohlcv=ex.fetch_ohlcv(sym,'1D',limit=90); df=pd.DataFrame(ohlcv,columns=['t','o','h','l','c','v']); scanned+=1
                res=scan(df)
                if res:
                    clean=sym.replace('/USDT:USDT',''); fig=chart(res,clean,exn)
                    rr=(res['tp2']-res['curr'])/(res['curr']-res['sl']); cap=f"🔥 <b>{clean} {exn} 1D BOTTOM</b>\nRSI {res['rsi']:.1f} Price {res['curr']:.6f} SL {res['sl']:.6f} TP2 {res['tp2']:.6f} RR 1:{rr:.1f}"
                    send_photo(fig,cap); total+=1; time.sleep(1)
            except: time.sleep(0.2); continue
    except Exception as e: print(f"{exn} err {e}"); continue

send_text(f"✅ Done - Scanned {scanned} pairs - Found {total} reversal")
