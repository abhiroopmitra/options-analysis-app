# app.py - FIXED IV 0% bug - works for ANY ticker
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Options App", layout="wide")
try:
    from curl_cffi import requests as cRequests
    SESSION = cRequests.Session(impersonate="chrome")
except:
    SESSION = None

def bs_delta_prob(S,K,T,r,q,sigma):
    if T<=0 or sigma<=0: return 0.5, 0.5
    try:
        sqrtT=math.sqrt(T)
        d1=(math.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrtT)
        d2=d1-sigma*sqrtT
        delta=float(norm.cdf(d1)*math.exp(-q*T))
        prob_itm=float(norm.cdf(d2))
        return delta, prob_itm
    except:
        return 0.5, 0.5

def prob_be(S,BE,T,r,q,sigma):
    if T<=0 or BE<=0 or sigma<=0: return 0
    try:
        sqrtT=math.sqrt(T)
        d1=(math.log(S/BE)+(r-q+0.5*sigma**2)*T)/(sigma*sqrtT)
        d2=d1-sigma*sqrtT
        return float(norm.cdf(d2))
    except:
        return 0

def rsi(s,p=14):
    d=s.diff(); g=d.where(d>0,0); l=-d.where(d<0,0)
    ag=g.ewm(alpha=1/p,min_periods=p).mean(); al=l.ewm(alpha=1/p,min_periods=p).mean()
    return 100-(100/(1+ag/al))
def atr(df,p=14):
    hl=df['High']-df['Low']; hc=(df['High']-df['Close'].shift()).abs(); lc=(df['Low']-df['Close'].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1); return tr.rolling(p).mean()

@st.cache_data(ttl=300)
def analyze_stock(ticker):
    tkr=yf.Ticker(ticker, session=SESSION) if SESSION else yf.Ticker(ticker)
    daily=tkr.history(period="1y", interval="1d", auto_adjust=True)
    if daily.empty: return None
    daily=daily.rename(columns=str.title); c=daily['Close']
    daily['SMA50']=c.rolling(50).mean(); daily['SMA200']=c.rolling(200).mean()
    daily['RSI']=rsi(c); daily['ATR']=atr(daily)
    hist_vol=float(c.pct_change().std()*np.sqrt(252))
    if np.isnan(hist_vol) or hist_vol<0.15: hist_vol=0.45
    if hist_vol>1.2: hist_vol=0.8
    return {'daily':daily,'price':float(c.iloc[-1]),'rsi':float(daily['RSI'].iloc[-1]),'atr':float(daily['ATR'].iloc[-1]),'hist_vol':hist_vol}

@st.cache_data(ttl=300, show_spinner=False)
def get_chain(ticker, dte_min, dte_max):
    tkr=yf.Ticker(ticker, session=SESSION) if SESSION else yf.Ticker(ticker)
    try: exps=tkr.options
    except: return pd.DataFrame()
    today=datetime.now().date(); dated=[]
    for e in exps:
        try: dte=(datetime.strptime(e,"%Y-%m-%d").date()-today).days; dated.append((e,dte))
        except: continue
    dated=sorted(dated, key=lambda x:x[1])
    filt=[x for x in dated if dte_min<=x[1]<=dte_max]
    if not filt: filt=dated[:6]
    calls=[]
    for exp_str,dte in filt[:6]:
        try:
            ch=tkr.option_chain(exp_str); c=ch.calls.copy(); c['expiration']=exp_str; c['dte']=dte; calls.append(c)
        except: continue
    return pd.concat(calls) if calls else pd.DataFrame()

st.sidebar.title("Inputs")
ticker=st.sidebar.text_input("Ticker", "INTC").upper()
max_risk=st.sidebar.number_input("Max Risk $", 50, 20000, 5000, 50)
dte_min=st.sidebar.slider("Min DTE", 0, 30, 7, help="Set 7 to avoid 1-day options - 1 day needs huge move")
dte_max=st.sidebar.slider("Max DTE", 7, 365, 45)

if st.sidebar.button("Analyze", type="primary"):
    sa=analyze_stock(ticker)
    if not sa: st.error("No data"); st.stop()
    price=sa['price']; hist_vol=sa['hist_vol']
    c1,c2,c3=st.columns(3); c1.metric(f"{ticker} Price", f"{price:.2f}"); c2.metric("Hist Vol", f"{hist_vol*100:.0f}%"); c3.metric("RSI / ATR", f"{sa['rsi']:.0f} / {sa['atr']:.2f}")
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=sa['daily'].index, open=sa['daily']['Open'], high=sa['daily']['High'], low=sa['daily']['Low'], close=sa['daily']['Close']))
    fig.add_trace(go.Scatter(x=sa['daily'].index, y=sa['daily']['SMA50'], name="SMA50")); fig.update_layout(height=350, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)

    df=get_chain(ticker, dte_min, dte_max)
    if df.empty: st.error("Yahoo blocked, wait 30s"); st.stop()

    df['ask']=df['ask'].fillna(0); df['bid']=df['bid'].fillna(0); df['lastPrice']=df['lastPrice'].fillna(0)
    df['real_price']=df.apply(lambda r: r['ask'] if r['ask']>0 else (r['lastPrice'] if r['lastPrice']>0 else 0), axis=1)
    df=df[df['real_price']>0].copy()
    df['max_loss']=df['real_price']*100
    
    # FIX IV 0% BUG HERE
    def clean_iv(iv):
        if pd.isna(iv) or iv==0 or iv<0.05 or iv>3.0:
            return hist_vol
        return iv
    df['iv_raw']=df['impliedVolatility']
    df['iv']=df['iv_raw'].apply(clean_iv)
    df['iv_is_est']=df['iv_raw'].apply(lambda x: True if pd.isna(x) or x==0 or x<0.05 or x>3.0 else False)

    dlist=[]; plist=[]; belist=[]; pbelist=[]
    for _,r in df.iterrows():
        T=max(r['dte'],1)/365
        delta, prob = bs_delta_prob(price, r['strike'], T, 0.045, 0, r['iv'])
        be=r['strike']+r['real_price']
        pbe=prob_be(price, be, T, 0.045, 0, r['iv'])
        dlist.append(delta); plist.append(prob); belist.append(be); pbelist.append(pbe)
    df['delta']=dlist; df['prob_itm']=plist; df['breakeven']=belist; df['prob_profit']=pbelist
    df['dist']=(df['strike']-price).abs()

    affordable=df[df['max_loss']<=max_risk].copy()
    near=affordable[affordable['dist']<=price*0.3].copy()
    good=near[(near['real_price']>=0.10) & (near['delta']>=0.20) & (near['delta']<=0.80)].copy()
    if good.empty: good=near.sort_values('dist').head(15).copy()
    good=good.sort_values('prob_profit', ascending=False)

    st.subheader(f"Top Calls for {ticker} within ${max_risk}")
    for _,r in good.head(5).iterrows():
        with st.container(border=True):
            iv_text = f"{r['iv']*100:.0f}% (est from hist vol)" if r['iv_is_est'] else f"{r['iv']*100:.0f}%"
            st.write(f"**BUY {ticker} {r['expiration']} Call {r['strike']} @ ${r['real_price']:.2f} = ${r['max_loss']:.0f} | Delta {r['delta']:.2f}**")
            st.write(f"BE ${r['breakeven']:.2f} ({(r['breakeven']/price-1)*100:+.2f}%) | Prob ITM {r['prob_itm']*100:.0f}% | Prob Profit {r['prob_profit']*100:.0f}% | Vol {r['volume']} | IV {iv_text}")
            if r['dte']<=2:
                st.warning(f"This expires in {r['dte']} days - needs {(r['breakeven']/price-1)*100:+.2f}% in {r['dte']} days. Prob is low because time is short. Try Min DTE 7.")
            else:
                st.success(f"Simple: Pay ${r['max_loss']:.0f}. Need {ticker} > ${r['breakeven']:.2f} to profit. ~{r['prob_profit']*100:.0f}% chance with {r['iv']*100:.0f}% vol.")
else:
    st.info("Set INTC Max Risk 5000 DTE 7-45, or SPY 1000 DTE 7-45. Set Min DTE to 7 not 1 to avoid 0% prob 1-day options.")
