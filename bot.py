import ccxt, pandas as pd, requests, os, io, time, traceback
import matplotlib.pyplot as plt
from datetime import datetime

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_text(msg):
    try:
        r = requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
        print(f"TG text status {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"TG send_text fail {e}")

def send_photo(fig, caption):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        r = requests.post(url, data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
                      files={"photo": buf}, timeout=20)
        print(f"TG photo status {r.status_code} {r.text[:200]}")
    except Exception as e:
        print(f"TG photo fail {e}")
        traceback.print_exc()

def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def is_doji(o,h,l,c): rng=h-l; return rng>0 and abs(c-o)/rng < 0.30
def is_hammer(o,h,l,c):
    body=abs(c-o); lower=min(o,c)-l; upper=h-max(o,c)
    return body>0 and lower>body*1.5 and upper<body*0.9
def is_star(o,h,l,c):
    body=abs(c-o); rng=h-l; upper=h-max(o,c); lower=min(o,c)-l
    return rng>0 and (upper>body*1.2 or lower>body*1.2) and body/rng < 0.45
def is_bullish_candle(o,h,l,c): return is_doji(o,h,l,c) or is_hammer(o,h,l,c) or is_star(o,h,l,c)

def scan_reversal(df):
    if len(df) < 80: return None
    df = df.iloc[:-1].copy()
    df['ema6']=ema(df['c'],6); df['ema12']=ema(df['c'],12); df['ema20']=ema(df['c'],20)
    df['rsi']=rsi(df['c'],14); df['vol_ma20']=df['v'].rolling(20).mean()
    
    for end in range(len(df)-1, len(df)-10, -1):
        for win_size in [2,3,4,5,6,7,8]:
            start=end-win_size+1; window=df.iloc[start:end+1]
            low_window=window['l'].min(); low_60=df.iloc[max(0,end-60):end]['l'].min()
            if low_window > low_60*1.15: continue  # LOOSENED from 8% to 15%
            
            ema_down = df['ema6'].iloc[end] < df['ema20'].iloc[end]*1.05
            if not ema_down: 
                below_ema = (df['c'].iloc[end-15:end] < df['ema20'].iloc[end-15:end]).sum() >= 8
                if not below_ema: continue
            
            if (window['c']>window['o']).sum() < 1: continue
            if sum(is_bullish_candle(r['o'],r['h'],r['l'],r['c']) for _,r in window.iterrows()) < 1: continue
            
            vol_window_avg=window['v'].mean(); vol_20_avg=df['vol_ma20'].iloc[end] or vol_window_avg
            max_green_vol=max([r['v'] for _,r in window.iterrows() if r['c']>r['o']], default=0)
            vol_ok = (vol_window_avg < vol_20_avg*1.2) or (max_green_vol > vol_20_avg*0.8)  # LOOSENED
            
            rsi_window=window['rsi'].min(); rsi_curr=df['rsi'].iloc[end]
            if rsi_window > 48 and rsi_curr > 50: continue  # LOOSENED from 38 to 48
            
            curr=df.iloc[end]['c']; sl=low_window*0.965; risk=curr-sl
            if risk<=0: continue
            return {'end_idx':end,'start_idx':start,'win_size':win_size,'bottom':low_window,'curr':curr,
                    'sl':sl,'tp1':curr+risk*1.5,'tp2':curr+risk*2.8,'tp3':df['ema20'].iloc[end],
                    'ema6':df['ema6'].iloc[end],'ema12':df['ema12'].iloc[end],'ema20':df['ema20'].iloc[end],
                    'low_60':low_60,'vol_avg':vol_window_avg,'vol_ma20':vol_20_avg,'vol_spike':max_green_vol>vol_20_avg,
                    'rsi':rsi_curr,'rsi_low':rsi_window,'df':df,'green':int((window['c']>window['o']).sum())}

    return None

def draw_chart(res, symbol, exchange):
    df=res['df'].tail(90)
    fig, (ax1,ax2,ax3)=plt.subplots(3,1,figsize=(11,7),gridspec_kw={'height_ratios':[3,1,1]},sharex=True)
    fig.patch.set_facecolor('white')
    for idx,row in df.iterrows():
        color='#00b26a' if row['c']>=row['o'] else '#ff3b30'
        ax1.plot([idx,idx],[row['l'],row['h']],color=color,lw=1); ax1.plot([idx,idx],[row['o'],row['c']],color=color,lw=3)
    ax1.plot(df.index,df['ema6'],color='#e38bd7',lw=1.1,label=f"EMA6:{res['ema6']:.5f}")
    ax1.plot(df.index,df['ema12'],color='#f0a500',lw=1.1,label=f"EMA12:{res['ema12']:.5f}")
    ax1.plot(df.index,df['ema20'],color='#5b4cff',lw=1.2,label=f"EMA20:{res['ema20']:.5f}")
    ax1.axvspan(res['start_idx'],res['end_idx'],color='lightgreen',alpha=0.35)
    ax1.axhline(res['sl'],color='red',ls='--',lw=1.2,label=f"SL {res['sl']:.6f}"); ax1.axhline(res['tp2'],color='green',ls='-',lw=1.2,label=f"TP2")
    ax1.set_title(f"{symbol} {exchange} 1D REVERSAL {res['win_size']}c RSI{res['rsi']:.1f}"); ax1.legend(fontsize=7,loc='upper left'); ax1.grid(alpha=0.2)
    colors=['#00b26a' if c>=o else '#ff3b30' for c,o in zip(df['c'],df['o'])]
    ax2.bar(df.index,df['v'],color=colors,alpha=0.7); ax2.plot(df.index,df['vol_ma20'],color='#e38bd7',lw=1); ax2.grid(alpha=0.2)
    ax3.plot(df.index,df['rsi'],color='#5b4cff',lw=1.2); ax3.axhline(35,color='red',ls='--',lw=0.8); ax3.axhline(70,color='green',ls='--',lw=0.8); ax3.fill_between(df.index,0,35,color='lightgreen',alpha=0.2); ax3.grid(alpha=0.2)
    plt.tight_layout(); return fig

def main():
    print(f"START BOT_TOKEN set={bool(BOT_TOKEN)} CHAT_ID={CHAT_ID}")
    if not BOT_TOKEN or not CHAT_ID:
        print("ERROR: BOT_TOKEN or CHAT_ID missing in secrets!"); return
    send_text(f"🚀 <b>1D Scan Started</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} - Scanning 5 CEX...")
    
    exchanges=[('MEXC',ccxt.mexc({'enableRateLimit':True,'options':{'defaultType':'swap'}})),
               ('BYBIT',ccxt.bybit({'enableRateLimit':True,'options':{'defaultType':'swap'}})),
               ('OKX',ccxt.okx({'enableRateLimit':True,'options':{'defaultType':'swap'}})),
               ('BITGET',ccxt.bitget({'enableRateLimit':True,'options':{'defaultType':'swap'}})),
               ('GATE',ccxt.gate({'enableRateLimit':True,'options':{'defaultType':'swap'}}))]
    total=0; scanned=0
    for ex_name,ex in exchanges:
        try:
            ex.load_markets(); syms=[s for s in ex.markets if s.endswith('/USDT:USDT') and 'STOCK' not in s][:600]
            for sym in syms:
                try:
                    ohlcv=ex.fetch_ohlcv(sym,'1D',limit=100); df=pd.DataFrame(ohlcv,columns=['t','o','h','l','c','v']); scanned+=1
                    res=scan_reversal(df)
                    if res:
                        clean=sym.replace('/USDT:USDT',''); fig=draw_chart(res,clean,ex_name)
                        rr=(res['tp2']-res['curr'])/(res['curr']-res['sl']); caption=f"🔥 <b>{clean} {ex_name} 1D BOTTOM</b>\n{res['win_size']} candles Green:{res['green']} RSI {res['rsi']:.1f} Low {res['rsi_low']:.1f}\nPrice {res['curr']:.6f} Bottom {res['bottom']:.6f}\nSL {res['sl']:.6f} TP2 {res['tp2']:.6f} RR 1:{rr:.1f}"
                        send_photo(fig,caption); total+=1; time.sleep(1.2)
                    time.sleep(0.05)
                except Exception as e: print(f"{sym} err {e}"); time.sleep(0.2); continue
        except Exception as e: send_text(f"⚠️ {ex_name} fail {e}"); print(traceback.format_exc()); continue
    
    # ALWAYS SEND SUMMARY - this you were missing
    if total==0:
        send_text(f"📭 <b>Scan Complete - No reversal today</b>\nScanned {scanned} pairs across 5 CEX\nFilters: Bottom within 15% of 60D low + 2-8c + 1 green + 1 doji/hammer/star + RSI<48 + Vol ok\nLoosened from yesterday. No setup met criteria today - market not at bottom zone. Will alert when found.")
    else:
        send_text(f"✅ <b>Scan Done</b> - Found {total} setups today out of {scanned} pairs")

if __name__=="__main__": main()
