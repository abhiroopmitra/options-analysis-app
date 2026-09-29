# app.py - FIXED for closed market + any ticker
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

def bs_greeks(S,K,T,r,q,sigma, otype='call'):
    if T<=0 or sigma<=0: return {'delta':0,'theta':0,'prob_itm':0}
    try:
        sqrtT=math.sqrt(T)
        d1=(math.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*sqrtT)
        d2=d1-sigma*sqrtT
        Nd1=norm.cdf(d1); Nd2=norm.cdf(d2); nd1=norm.pdf(d1)
        exp_qt=math.exp(-q*T); exp_rt=math.exp(-r*T)
        if otype=='call':
            delta=exp_qt*Nd1; prob=Nd2
            theta_y=-(S*exp_qt*nd1*sigma)/(2*sqrtT)-r*K*exp_rt*Nd2+q*S*exp_qt*Nd1
        else:
            delta=exp_qt*(Nd1-1); prob=norm.cdf(-d2)
            theta_y=-(S*exp_qt*nd1*sigma)/(2*sqrtT)+r*K*exp_rt*norm.cdf(-d2)-q*S*exp_qt*norm.cdf(-d1)
        return {'delta':delta,'theta':theta_y/365,'prob_itm':prob}
    except: return {'delta':0,'theta':0,'prob_itm':0}

def calc_prob_be(S,BE,T,r,q,sigma, otype='call'):
    if T<=0 or sigma<=0 or BE<=0: return 0
    try:
        sqrtT=math.sqrt(T)
        d1=(math.log(S/BE)+(r-q+0.5*sigma**2)*T)/(sigma*sqrtT)
        d2=d1-sigma*sqrtT
        return norm.cdf(d2) if otype=='call' else norm.cdf(-d2)
    except: return 0

def rsi(s,p=14):
    d=s.diff(); g=d.where(d>0,0); l=-d.where(d<0,0)
    ag=g.ewm(alpha=1/p,min_periods=p).mean(); al=l.ewm(alpha=1/p,min_periods=p).mean()
    rs=ag/al; return 100-(100/(1+rs))

def atr(df,p=14):
    hl=df['High']-df['Low']; hc=(df['High']-df['Close'].shift()).abs(); lc=(df['Low']-df['Close'].shift()).abs()
    tr=pd.concat([hl,hc,lc],axis=1).max(axis=1); return tr.rolling(p).mean()

@st.cache_data(ttl=300)
def analyze_stock(ticker):
    tkr = yf.Ticker(ticker, session=SESSION) if SESSION else yf.Ticker(ticker)
    daily = tkr.history(period="1y", interval="1d", auto_adjust=True)
    if daily.empty: return None
    daily=daily.rename(columns=str.title)
    c=daily['Close']
    daily['SMA50']=c.rolling(50).mean(); daily['SMA200']=c.rolling(200).mean(); daily['SMA20']=c.rolling(20).mean()
    daily['RSI']=rsi(c); daily['ATR']=atr(daily)
    e12=c.ewm(span=12).mean(); e26=c.ewm(span=26).mean(); daily['MACD']=e12-e26; daily['MACD_Sig']=daily['MACD'].ewm(span=9).mean()
    price=float(c.iloc[-1]); rsi_v=float(daily['RSI'].iloc[-1]); atr_v=float(daily['ATR'].iloc[-1])
    roof=float(daily['High'].rolling(20).max().iloc[-1]); floor=float(daily['Low'].rolling(20).min().iloc[-1])
    return {'daily':daily,'price':price,'rsi':rsi_v,'atr':atr_v,'roof':roof,'floor':floor,'verdict':"BULLISH" if price>daily['SMA50'].iloc[-1] else "BEARISH"}

@st.cache_data(ttl=300)
def get_rates(ticker):
    try: r=float(yf.Ticker("^IRX", session=SESSION).history(period="5d")['Close'].iloc[-1]/100) if SESSION else float(yf.Ticker("^IRX").history(period="5d")['Close'].iloc[-1]/100)
    except: r=0.045
    try: 
        info = (yf.Ticker(ticker, session=SESSION).info if SESSION else yf.Ticker(ticker).info)
        q=info.get('dividendYield',0) or 0
    except: q=0
    return r,float(q)

@st.cache_data(ttl=300, show_spinner=False)
def get_expiries(ticker):
    try:
        tkr = yf.Ticker(ticker, session=SESSION) if SESSION else yf.Ticker(ticker)
        return tkr.options
    except: return []

def get_chain(ticker, dte_min, dte_max):
    tkr = yf.Ticker(ticker, session=SESSION) if SESSION else yf.Ticker(ticker)
    exps = get_expiries(ticker)
    if not exps: return pd.DataFrame(), []
    today=datetime.now().date()
    dated=[]
    for e in exps:
        try: dte=(datetime.strptime(e,"%Y-%m-%d").date()-today).days; dated.append((e,dte))
        except: continue
    dated=sorted(dated, key=lambda x:x[1])
    filt=[x for x in dated if dte_min <= x[1] <= dte_max]
    if not filt: filt=dated[:4]
    calls=[]
    for exp_str,dte in filt[:4]:
        try:
            ch=tkr.option_chain(exp_str)
            c=ch.calls.copy(); c['expiration']=exp_str; c['dte']=dte; calls.append(c)
        except: continue
    df = pd.concat(calls) if calls else pd.DataFrame()
    return df, dated

st.sidebar.title("Inputs")
ticker = st.sidebar.text_input("Enter ANY Ticker", "AAPL").upper()
max_risk = st.sidebar.number_input("Max Risk $ per trade", 50, 10000, 500, 50)
dte_min = st.sidebar.slider("Min DTE", 0, 30, 7)
dte_max = st.sidebar.slider("Max DTE", 7, 365, 45)

if st.sidebar.button("Analyze", type="primary"):
    sa=analyze_stock(ticker)
    if not sa: st.error(f"No daily data for {ticker}"); st.stop()
    r_rate,q = get_rates(ticker)
    c1,c2,c3=st.columns(3); c1.metric(f"{ticker} Price", f"{sa['price']:.2f}"); c2.metric("Verdict", sa['verdict']); c3.metric("RSI / ATR", f"{sa['rsi']:.0f} / {sa['atr']:.2f}")
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=sa['daily'].index, open=sa['daily']['Open'], high=sa['daily']['High'], low=sa['daily']['Low'], close=sa['daily']['Close'], name=ticker))
    fig.add_trace(go.Scatter(x=sa['daily'].index, y=sa['daily']['SMA50'], name="SMA50")); fig.add_trace(go.Scatter(x=sa['daily'].index, y=sa['daily']['SMA200'], name="SMA200"))
    fig.update_layout(height=400, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)

    with st.spinner(f"Pulling {ticker} options..."):
        calls_df, dated = get_chain(ticker, dte_min, dte_max)

    if calls_df.empty:
        st.error("Yahoo blocked or no chain. Wait 30 sec and try DTE 0-365"); st.stop()

    # FIX: Use lastPrice when bid/ask is 0 (market closed)
    calls_df['ask'] = calls_df['ask'].fillna(0)
    calls_df['bid'] = calls_df['bid'].fillna(0)
    calls_df['lastPrice'] = calls_df['lastPrice'].fillna(0)
    calls_df['real_price'] = calls_df.apply(lambda r: r['ask'] if r['ask']>0 else (r['lastPrice'] if r['lastPrice']>0 else 0), axis=1)
    calls_df['real_bid'] = calls_df.apply(lambda r: r['bid'] if r['bid']>0 else r['real_price']*0.9, axis=1)
    calls_df['iv'] = calls_df['impliedVolatility'].fillna(0.4).replace(0,0.4)
    calls_df['mid'] = (calls_df['real_price']+calls_df['real_bid'])/2

    # Calculate Greeks
    dlist=[]; tlist=[]; belist=[]; pbelist=[]
    for _,row in calls_df.iterrows():
        T=row['dte']/365 if row['dte']>0 else 0.0027
        g=bs_greeks(sa['price'], row['strike'], T, r_rate, q, row['iv'], 'call')
        dlist.append(g['delta']); tlist.append(g['theta'])
        be=row['strike']+row['real_price']
        belist.append(be)
        pbelist.append(calc_prob_be(sa['price'], be, T, r_rate, q, row['iv'], 'call'))
    calls_df['delta']=dlist; calls_df['theta']=tlist; calls_df['breakeven']=belist; calls_df['prob_profit']=pbelist
    calls_df['max_loss']=calls_df['real_price']*100
    calls_df['dist_from_price'] = (calls_df['strike'] - sa['price']).abs()

    # Filter: Only options within 20% of current price and within max risk
    near = calls_df[calls_df['dist_from_price'] <= sa['price']*0.2].copy()
    good = near[(near['max_loss']<=max_risk) & (near['max_loss']>0) & (near['delta']>0.15) & (near['delta']<0.85)].copy()
    good = good.sort_values(['dte','dist_from_price'])

    st.subheader(f"Top Calls for {ticker} within ${max_risk}")
    if good.empty:
        st.warning(f"No calls under ${max_risk} near price. {ticker} is ${sa['price']:.2f}. Example: A 250 call costs $8905 > $300. Try Max Risk $500 or $1000. Showing cheapest near price:")
        cheapest = near[near['max_loss']>0].sort_values('max_loss').head(10)
        st.dataframe(cheapest[['expiration','dte','strike','real_bid','real_price','lastPrice','volume','delta','breakeven','max_loss']])
    else:
        for _,r in good.head(5).iterrows():
            with st.container(border=True):
                st.write(f"**{ticker} {r['expiration']} Call {r['strike']} - ${r['real_price']:.2f} = ${r['max_loss']:.0f} max loss**")
                st.write(f"BE ${r['breakeven']:.2f} ({(r['breakeven']/sa['price']-1)*100:+.2f}%) | Delta {r['delta']:.2f} | Theta ${r['theta']*100:.1f}/day | Prob Profit {r['prob_profit']*100:.0f}% | Vol {r['volume']}")
                st.info(f"Simple: Pay ${r['max_loss']:.0f} to buy 100 shares at ${r['strike']}. Need {ticker} > ${r['breakeven']:.2f} to profit. ~{r['prob_profit']*100:.0f}% chance. Lose ${abs(r['theta']*100):.0f}/day if stalls.")
else:
    st.info("Set ticker like AAPL, NVDA, SPY. Set Max Risk 500, Min DTE 7, Max DTE 45, click Analyze.")
