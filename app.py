# app.py - Works for ANY ticker + Max Risk filter
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Options Analysis App", layout="wide")

def bs_greeks(S,K,T,r,q,sigma, option_type='call'):
    if T<=0 or sigma<=0: return {'delta':0,'theta':0,'prob_itm':0}
    try:
        sqrtT = math.sqrt(T)
        d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrtT)
        d2 = d1 - sigma*sqrtT
        Nd1 = norm.cdf(d1); Nd2 = norm.cdf(d2); nd1 = norm.pdf(d1)
        exp_qt = math.exp(-q*T); exp_rt = math.exp(-r*T)
        if option_type=='call':
            delta = exp_qt*Nd1; prob_itm = Nd2
            theta_y = - (S*exp_qt*nd1*sigma)/(2*sqrtT) - r*K*exp_rt*Nd2 + q*S*exp_qt*Nd1
        else:
            delta = exp_qt*(Nd1-1); prob_itm = norm.cdf(-d2)
            theta_y = - (S*exp_qt*nd1*sigma)/(2*sqrtT) + r*K*exp_rt*norm.cdf(-d2) - q*S*exp_qt*norm.cdf(-d1)
        return {'delta':delta,'theta':theta_y/365,'prob_itm':prob_itm}
    except: return {'delta':0,'theta':0,'prob_itm':0}

def prob_profit_be(S, BE, T, r, q, sigma, otype='call'):
    if T<=0 or sigma<=0: return 0
    try:
        sqrtT = math.sqrt(T)
        d1 = (math.log(S/BE) + (r - q + 0.5*sigma**2)*T) / (sigma*sqrtT)
        d2 = d1 - sigma*sqrtT
        return norm.cdf(d2) if otype=='call' else norm.cdf(-d2)
    except: return 0

def rsi(series, period=14):
    delta = series.diff(); gain = delta.where(delta>0,0); loss = -delta.where(delta<0,0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain/avg_loss; return 100 - (100/(1+rs))

def atr(df, period=14):
    hl = df['High']-df['Low']; hc = (df['High']-df['Close'].shift()).abs(); lc = (df['Low']-df['Close'].shift()).abs()
    tr = pd.concat([hl,hc,lc], axis=1).max(axis=1); return tr.rolling(period).mean()

@st.cache_data(ttl=300)
def analyze_stock(ticker):
    tkr = yf.Ticker(ticker); daily = tkr.history(period="1y", interval="1d", auto_adjust=True)
    if daily.empty: return None
    daily = daily.rename(columns=str.title)
    close = daily['Close']
    daily['SMA50']=close.rolling(50).mean(); daily['SMA200']=close.rolling(200).mean(); daily['SMA20']=close.rolling(20).mean()
    daily['RSI']=rsi(close); daily['ATR']=atr(daily)
    ema12=close.ewm(span=12).mean(); ema26=close.ewm(span=26).mean()
    daily['MACD']=ema12-ema26; daily['MACD_Sig']=daily['MACD'].ewm(span=9).mean()
    price=float(close.iloc[-1]); rsi_val=float(daily['RSI'].iloc[-1]); atr_val=float(daily['ATR'].iloc[-1])
    sma50=float(daily['SMA50'].iloc[-1]); sma200=float(daily['SMA200'].iloc[-1]); sma20=float(daily['SMA20'].iloc[-1])
    macd=float(daily['MACD'].iloc[-1]); macd_sig=float(daily['MACD_Sig'].iloc[-1])
    roof=float(daily['High'].rolling(20).max().iloc[-1]); floor=float(daily['Low'].rolling(20).min().iloc[-1])
    highs = daily['High'].rolling(10, center=True).max(); lows = daily['Low'].rolling(10, center=True).min()
    res = sorted(daily[daily['High']==highs]['High'].tail(20).tolist(), reverse=True)[:3]
    sup = sorted(daily[daily['Low']==lows]['Low'].tail(20).tolist())[:3]
    supports=[s for s in sup if s<price][:2]; resistances=[r for r in res if r>price][:2]
    bull_pts = (1 if price>sma50 else 0) + (2 if price>sma200 else 0) + (1 if sma20>sma50 else 0) + (1 if macd>macd_sig else 0)
    if price>sma200 and rsi_val<50: verdict="BULLISH PULLBACK"
    elif bull_pts>=4: verdict="BULLISH"
    elif bull_pts<=1: verdict="BEARISH"
    else: verdict="NEUTRAL"
    return {'daily':daily,'price':price,'rsi':rsi_val,'atr':atr_val,'roof':roof,'floor':floor,'supports':supports,'resistances':resistances,'verdict':verdict,'target_up':resistances[0] if resistances else roof,'target_down':supports[0] if supports else floor}

@st.cache_data(ttl=300)
def get_rates(ticker):
    try: r=yf.Ticker("^IRX").history(period="5d")['Close'].iloc[-1]/100
    except: r=0.045
    try: q=yf.Ticker(ticker).info.get('dividendYield',0) or 0
    except: q=0
    return float(r), float(q)

@st.cache_data(ttl=300)
def get_options_chain(ticker, dte_min, dte_max):
    tkr=yf.Ticker(ticker)
    try: exps=tkr.options
    except: return pd.DataFrame()
    today=datetime.now().date(); filt=[]
    for e in exps:
        dte=(datetime.strptime(e,"%Y-%m-%d").date()-today).days
        if dte_min<=dte<=dte_max: filt.append((e,dte))
    filt=sorted(filt, key=lambda x:x[1])[:4]
    calls_list=[]
    for exp_str,dte in filt:
        try:
            ch=tkr.option_chain(exp_str); c=ch.calls.copy(); c['expiration']=exp_str; c['dte']=dte
            calls_list.append(c)
        except: continue
    return pd.concat(calls_list) if calls_list else pd.DataFrame()

st.sidebar.title("Inputs")
ticker = st.sidebar.text_input("Enter ANY Ticker", "SPY").upper()
max_risk = st.sidebar.number_input("Max Risk $ per trade", 50, 10000, 500, 50)
dte_min = st.sidebar.slider("Min DTE", 1, 30, 7)
dte_max = st.sidebar.slider("Max DTE", 10, 90, 45)

if st.sidebar.button("Analyze", type="primary"):
    sa=analyze_stock(ticker)
    if not sa: st.error(f"No data for {ticker}"); st.stop()
    r_rate, div_y = get_rates(ticker)
    c1,c2,c3=st.columns(3); c1.metric(f"{ticker} Price", f"{sa['price']:.2f}"); c2.metric("Verdict", sa['verdict']); c3.metric("RSI / ATR", f"{sa['rsi']:.0f} / {sa['atr']:.2f}")
    st.write(f"Roof 20d: {sa['roof']:.2f} | Floor 20d: {sa['floor']:.2f} | Supports: {sa['supports']} | Resistances: {sa['resistances']} | Target Up: {sa['target_up']:.2f}")
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=sa['daily'].index, open=sa['daily']['Open'], high=sa['daily']['High'], low=sa['daily']['Low'], close=sa['daily']['Close'], name=ticker)); fig.add_trace(go.Scatter(x=sa['daily'].index, y=sa['daily']['SMA50'], name="SMA50")); fig.add_trace(go.Scatter(x=sa['daily'].index, y=sa['daily']['SMA200'], name="SMA200")); fig.update_layout(height=400, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)
    with st.spinner(f"Pulling {ticker} options from Yahoo..."):
        calls_df = get_options_chain(ticker, dte_min, dte_max)
        if calls_df.empty: st.error("No options found"); st.stop()
        hist_vol=sa['daily']['Close'].pct_change().std()*np.sqrt(252)
        if np.isnan(hist_vol): hist_vol=0.2
        calls_df['mid']=(calls_df['bid'].fillna(0)+calls_df['ask'].fillna(0))/2; calls_df['mid']=calls_df['mid'].where(calls_df['mid']>0, calls_df['lastPrice'])
        calls_df['iv']=calls_df['impliedVolatility'].fillna(hist_vol).replace(0,hist_vol)
        deltas=[]; thetas=[]; prob=[]; be=[]; prob_be=[]
        for _,row in calls_df.iterrows():
            T=row['dte']/365; g=bs_greeks(sa['price'], row['strike'], T, r_rate, div_y, row['iv'], 'call')
            deltas.append(g['delta']); thetas.append(g['theta']); prob.append(g['prob_itm'])
            b=row['strike']+row['ask']; be.append(b); prob_be.append(prob_profit_be(sa['price'], b, T, r_rate, div_y, row['iv'], 'call'))
        calls_df['delta']=deltas; calls_df['theta']=thetas; calls_df['prob_itm']=prob; calls_df['breakeven']=be; calls_df['prob_profit']=prob_be
        calls_df['max_loss']=calls_df['ask']*100; calls_df['spread_pct']=(calls_df['ask']-calls_df['bid'])/calls_df['mid'].replace(0,1)
        good_calls=calls_df[(calls_df['max_loss']<=max_risk) & (calls_df['volume']>20) & (calls_df['spread_pct']<0.25) & (calls_df['delta']>0.15) & (calls_df['delta']<0.8)].copy()
        good_calls['score']=good_calls['prob_profit']*50 + (1-good_calls['spread_pct'].clip(0,1))*20 + good_calls['delta']*30
        good_calls=good_calls.sort_values('score', ascending=False)
    st.subheader(f"Top Calls for {ticker} within ${max_risk}")
    if good_calls.empty: st.warning(f"No calls under ${max_risk}. For SPY your 760 call needed $1090 and 775 call needed $279. Increase Max Risk or try AAPL/MSFT which are cheaper.")
    else:
        for _,r in good_calls.head(3).iterrows():
            with st.container(border=True):
                st.write(f"**{ticker} {r['expiration']} Call {r['strike']} - ${r['ask']:.2f} = ${r['max_loss']:.0f} max loss | Score {r['score']:.0f}**")
                st.write(f"BE ${r['breakeven']:.2f} ({(r['breakeven']/sa['price']-1)*100:+.2f}%) | Delta {r['delta']:.2f} | Theta ${r['theta']*100:.1f}/day | Prob Profit {r['prob_profit']*100:.0f}%")
                st.info(f"Simple: Pay ${r['max_loss']:.0f} to buy 100 shares at ${r['strike']}. Need {ticker} > ${r['breakeven']:.2f} to profit. ~{r['prob_profit']*100:.0f}% chance. Lose ${abs(r['theta']*100):.0f}/day if stalls.")
else:
    st.info("On left, type ANY ticker - SPY, AAPL, NVDA, TSLA - set Max Risk and click Analyze.")
