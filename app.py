# app.py - Fixed for Streamlit Cloud - Works for ANY ticker
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Options App - Any Ticker", layout="wide")

# Try to create a browser-like session to avoid Yahoo block
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

def prob_be(S,BE,T,r,q,sigma, otype='call'):
    if T<=0 or sigma<=0: return 0
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
    # supports near price only
    sup = daily.nsmallest(30,'Low'); res = daily.nlargest(30,'High')
    supports = sorted([float(x) for x in sup['Low'].unique() if x < price])[-2:]
    resistances = sorted([float(x) for x in res['High'].unique() if x > price])[:2]
    bull = (1 if price>daily['SMA50'].iloc[-1] else 0)+(2 if price>daily['SMA200'].iloc[-1] else 0)+(1 if daily['SMA20'].iloc[-1]>daily['SMA50'].iloc[-1] else 0)
    verdict = "BULLISH PULLBACK" if price>daily['SMA200'].iloc[-1] and rsi_v<55 else "BULLISH" if bull>=3 else "BEARISH" if bull<=1 else "NEUTRAL"
    return {'daily':daily,'price':price,'rsi':rsi_v,'atr':atr_v,'roof':roof,'floor':floor,'supports':supports,'resistances':resistances,'verdict':verdict,'target_up':resistances[0] if resistances else roof}

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
def get_options_debug(ticker):
    try:
        tkr = yf.Ticker(ticker, session=SESSION) if SESSION else yf.Ticker(ticker)
        exps = tkr.options
        return exps, None
    except Exception as e:
        return [], str(e)

def get_chain(ticker, dte_min, dte_max):
    tkr = yf.Ticker(ticker, session=SESSION) if SESSION else yf.Ticker(ticker)
    exps, err = get_options_debug(ticker)
    if not exps: return pd.DataFrame(), exps, err
    today=datetime.now().date()
    dated=[]
    for e in exps:
        try: dte=(datetime.strptime(e,"%Y-%m-%d").date()-today).days; dated.append((e,dte))
        except: continue
    dated=sorted(dated, key=lambda x:x[1])
    # filter but if filter kills all, fallback to first 4
    filt=[x for x in dated if dte_min <= x[1] <= dte_max]
    if not filt: filt=dated[:4]
    calls=[]
    for exp_str,dte in filt[:4]:
        try:
            ch=tkr.option_chain(exp_str)
            c=ch.calls.copy(); c['expiration']=exp_str; c['dte']=dte; calls.append(c)
        except Exception as e:
            continue
    df = pd.concat(calls) if calls else pd.DataFrame()
    return df, dated, None

# UI
st.sidebar.title("Inputs")
ticker = st.sidebar.text_input("Enter ANY Ticker", "SPY").upper()
max_risk = st.sidebar.number_input("Max Risk $ per trade", 50, 10000, 1000, 50)
dte_min = st.sidebar.slider("Min DTE", 0, 30, 0)
dte_max = st.sidebar.slider("Max DTE", 7, 365, 180)
st.sidebar.caption("Tip: Set Max DTE to 180 to see Oct 2026 chain")

if st.sidebar.button("Analyze", type="primary"):
    st.cache_data.clear()
    sa=analyze_stock(ticker)
    if not sa: st.error(f"No daily data for {ticker}"); st.stop()
    r_rate,q = get_rates(ticker)
    c1,c2,c3=st.columns(3); c1.metric(f"{ticker} Price", f"{sa['price']:.2f}"); c2.metric("Verdict", sa['verdict']); c3.metric("RSI / ATR", f"{sa['rsi']:.0f} / {sa['atr']:.2f}")
    st.write(f"Roof 20d: {sa['roof']:.2f} | Floor 20d: {sa['floor']:.2f} | Supports: {sa['supports']} | Resist: {sa['resistances']} | Target: {sa['target_up']:.2f}")
    fig=go.Figure(); fig.add_trace(go.Candlestick(x=sa['daily'].index, open=sa['daily']['Open'], high=sa['daily']['High'], low=sa['daily']['Low'], close=sa['daily']['Close'], name=ticker))
    fig.add_trace(go.Scatter(x=sa['daily'].index, y=sa['daily']['SMA50'], name="SMA50")); fig.add_trace(go.Scatter(x=sa['daily'].index, y=sa['daily']['SMA200'], name="SMA200"))
    fig.update_layout(height=400, xaxis_rangeslider_visible=False); st.plotly_chart(fig, use_container_width=True)

    with st.spinner(f"Pulling {ticker} options..."):
        calls_df, dated, err = get_chain(ticker, dte_min, dte_max)
        exps,_ = get_options_debug(ticker)

    with st.expander("Debug - See available expiries"):
        st.write(f"Yahoo returned {len(exps)} expiries:", exps[:15])
        st.write("With DTEs:", dated[:10])
        if err: st.error(f"Yahoo error: {err}")
        if SESSION: st.success("Using browser session - anti-block enabled")
        else: st.warning("No browser session - may get blocked")

    if calls_df.empty:
        st.error(f"No options found for {ticker}. Yahoo is blocking or no chain in {dte_min}-{dte_max} DTE. Try DTE 0-365 and click Analyze again after 30 sec.")
        st.stop()

    hist_vol=sa['daily']['Close'].pct_change().std()*np.sqrt(252); hist_vol=0.2 if np.isnan(hist_vol) else hist_vol
    calls_df['mid']=(calls_df['bid'].fillna(0)+calls_df['ask'].fillna(0))/2; calls_df['mid']=calls_df['mid'].where(calls_df['mid']>0, calls_df['lastPrice'])
    calls_df['iv']=calls_df['impliedVolatility'].fillna(hist_vol).replace(0,hist_vol)
    deltas=[]; thetas=[]; prob=[]; be=[]; prob_be=[]
    for _,row in calls_df.iterrows():
        T=row['dte']/365; g=bs_greeks(sa['price'], row['strike'], T, r_rate, q, row['iv'], 'call')
        deltas.append(g['delta']); thetas.append(g['theta']); prob.append(g['prob_itm'])
        b=row['strike']+row['ask']; be.append(b); prob_be.append(prob_be(sa['price'], b, T, r_rate, q, row['iv'], 'call'))
    calls_df['delta']=deltas; calls_df['theta']=thetas; calls_df['prob_itm']=prob; calls_df['breakeven']=be; calls_df['prob_profit']=prob_be
    calls_df['max_loss']=calls_df['ask']*100; calls_df['spread_pct']=(calls_df['ask']-calls_df['bid'])/calls_df['mid'].replace(0,1)

    good=calls_df[(calls_df['max_loss']<=max_risk) & (calls_df['volume']>=10) & (calls_df['spread_pct']<0.4) & (calls_df['delta']>0.15) & (calls_df['delta']<0.85)].copy()
    good['score']=good['prob_profit']*50 + (1-good['spread_pct'].clip(0,1))*20 + good['delta']*30
    good=good.sort_values('score', ascending=False)

    st.subheader(f"Top Calls for {ticker} within ${max_risk}")
    if good.empty:
        st.warning(f"All calls > ${max_risk} or too illiquid. Your SPY 760 call was $1090, 775 call $279. Increase Max Risk to 1000 or set DTE to 0-365.")
        st.dataframe(calls_df[['expiration','dte','strike','bid','ask','lastPrice','volume','delta','breakeven']].head(10))
    else:
        for _,r in good.head(5).iterrows():
            with st.container(border=True):
                st.write(f"**{ticker} {r['expiration']} Call {r['strike']} - Ask ${r['ask']:.2f} = ${r['max_loss']:.0f} max loss | Score {r['score']:.0f}**")
                st.write(f"BE ${r['breakeven']:.2f} ({(r['breakeven']/sa['price']-1)*100:+.2f}%) | Delta {r['delta']:.2f} | Theta ${r['theta']*100:.1f}/day | Prob Profit {r['prob_profit']*100:.0f}% | Vol {r['volume']}")
                st.info(f"Simple: Pay ${r['max_loss']:.0f} to buy 100 shares at ${r['strike']}. Need {ticker} > ${r['breakeven']:.2f} to profit. ~{r['prob_profit']*100:.0f}% chance. Lose ${abs(r['theta']*100):.0f}/day if stalls.")
else:
    st.info("Left side: Type ANY ticker like SPY, AAPL, NVDA, TSLA. Set Max Risk $1000, DTE 0-180, then click Analyze.")
