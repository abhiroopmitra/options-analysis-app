# app.py - FIXED for INTC/AAPL/SPY any ticker - no lottery tickets
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

def bs_delta(S,K,T,r,q,sigma):
    if T<=0 or sigma<=0: return 0.5
    try:
        sqrtT=math.sqrt(T)
        d1=(math.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrtT)
        return float(norm.cdf(d1)*math.exp(-q*T))
    except: return 0.5

def calc_prob_be(S,BE,T,r,q,sigma):
    if T<=0 or BE<=0: return 0
    try:
        sqrtT=math.sqrt(T)
        d1=(math.log(S/BE)+(r-q+0.5*sigma**2)*T)/(sigma*sqrtT)
        d2=d1-sigma*sqrtT
        return float(norm.cdf(d2))
    except: return 0

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
    return {'daily':daily,'price':float(c.iloc[-1]),'rsi':float(daily['RSI'].iloc[-1]),'atr':float(daily['ATR'].iloc[-1])}

@st.cache_data(ttl=300, show_spinner=False)
def get_chain(ticker, dte_min, dte_max):
    tkr=yf.Ticker(ticker, session=SESSION) if SESSION else yf.Ticker(ticker)
    try: exps=tkr.options
    except: return pd.DataFrame(), []
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
    df=pd.concat(calls) if calls else pd.DataFrame()
    return df, dated

st.sidebar.title("Inputs")
ticker=st.sidebar.text_input("Ticker", "INTC").upper()
max_risk=st.sidebar.number_input("Max Risk $ per trade", 50, 20000, 5000, 50)
dte_min=st.sidebar.slider("Min DTE", 0, 30, 7)
dte_max=st.sidebar.slider("Max DTE", 7, 365, 45)

if st.sidebar.button("Analyze", type="primary"):
    sa=analyze_stock(ticker)
    if not sa: st.error("No data"); st.stop()
    price=sa['price']
    c1,c2=st.columns(2); c1.metric(f"{ticker} Price", f"{price:.2f}"); c2.metric("RSI / ATR", f"{sa['rsi']:.0f} / {sa['atr']:.2f}")
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=sa['daily'].index, open=sa['daily']['Open'], high=sa['daily']['High'], low=sa['daily']['Low'], close=sa['daily']['Close']))
    fig.add_trace(go.Scatter(x=sa['daily'].index, y=sa['daily']['SMA50'], name="SMA50")); fig.update_layout(height=350, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)

    df, dated = get_chain(ticker, dte_min, dte_max)
    if df.empty: st.error("Yahoo blocked, wait 30 sec and click Analyze again"); st.stop()
    st.caption(f"Found {len(df)} calls from {len(dated)} expiries. Example expiries: {dated[:3]}")

    # Fix market closed - use lastPrice when ask=0
    df['ask'] = df['ask'].fillna(0); df['bid'] = df['bid'].fillna(0); df['lastPrice'] = df['lastPrice'].fillna(0)
    df['real_price'] = df.apply(lambda r: r['ask'] if r['ask']>0 else (r['lastPrice'] if r['lastPrice']>0 else r['bid']), axis=1)
    df = df[df['real_price']>0].copy()
    df['max_loss'] = df['real_price']*100
    df['iv'] = df['impliedVolatility'].fillna(0.5).replace(0,0.5)
    
    # Greeks
    df['delta'] = df.apply(lambda r: bs_delta(price, r['strike'], max(r['dte'],1)/365, 0.045, 0, r['iv']), axis=1)
    df['breakeven'] = df['strike'] + df['real_price']
    df['prob_profit'] = df.apply(lambda r: calc_prob_be(price, r['breakeven'], max(r['dte'],1)/365, 0.045, 0, r['iv']), axis=1)
    df['dist'] = (df['strike']-price).abs()

    affordable = df[df['max_loss']<=max_risk].copy()
    if affordable.empty:
        cheapest = df.sort_values('max_loss').iloc[0]
        st.error(f"Even cheapest call costs ${cheapest['max_loss']:.0f} (${cheapest['real_price']:.2f}) for {ticker} {cheapest['strike']} strike. That's > your Max Risk ${max_risk}. Increase Max Risk.")
        st.dataframe(df.sort_values('max_loss').head(10)[['expiration','dte','strike','real_price','max_loss','delta']])
        st.stop()

    # Near price = within 30% of price
    near = affordable[affordable['dist'] <= price*0.3].copy()
    # Good = Delta 0.2 to 0.8 and not lottery $0.01
    good = near[(near['delta']>=0.15) & (near['delta']<=0.85) & (near['real_price']>=0.10)].copy()
    if good.empty:
        # Fallback: show closest to ATM regardless of delta
        good = near.sort_values('dist').head(20).copy()
    
    good['score'] = (1 - (good['dist']/price)) * 50 + good['prob_profit']*50
    good = good.sort_values('score', ascending=False)

    st.subheader(f"Top Calls for {ticker} within ${max_risk} - closest to ATM")
    for _,r in good.head(5).iterrows():
        with st.container(border=True):
            st.write(f"**BUY {ticker} {r['expiration']} Call {r['strike']} @ ${r['real_price']:.2f} = ${r['max_loss']:.0f} max loss | Delta {r['delta']:.2f}**")
            st.write(f"BE ${r['breakeven']:.2f} ({(r['breakeven']/price-1)*100:+.2f}%) | Prob Profit {r['prob_profit']*100:.0f}% | Vol {r['volume']} | IV {r['iv']*100:.0f}%")
            if r['delta']<0.15:
                st.warning(f"Delta {r['delta']:.3f} = lottery ticket, needs {ticker} to jump {(r['strike']/price-1)*100:.1f}%")
            else:
                st.success(f"Simple: Pay ${r['max_loss']:.0f}. Need {ticker} > ${r['breakeven']:.2f} to profit. ~{r['prob_profit']*100:.0f}% chance.")

    with st.expander("See all affordable near price"):
        st.dataframe(good[['expiration','dte','strike','real_price','max_loss','delta','breakeven','prob_profit','volume']].head(20))
else:
    st.info("Set ticker INTC, Max Risk 5000, DTE 7-45, click Analyze. For AAPL/SPY use Max Risk 1000.")
