#!/usr/bin/env python3
"""
P-TRACK DASHBOARD  (Streamlit web app)
======================================
A point-and-click browser front-end for ptrack_screener.py. No coding needed
to use it -- pick a universe, click Run, and read the ranked setups with charts.

RUN IT:
    pip install -r requirements.txt
    streamlit run app.py
Your browser opens automatically at http://localhost:8501
"""
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import ptrack_screener as P

st.set_page_config(page_title="P-TRACK Screener", layout="wide", page_icon="📈")

# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 P-TRACK")
st.sidebar.caption("Bullish technical setup screener")

mode = st.sidebar.radio("Mode", ["Screen for setups", "Backtest the rules"])

universe = st.sidebar.selectbox(
    "Universe",
    ["sp500", "nasdaq100", "sp1500", "all"],
    help="sp500 is fastest. 'all' scans every US-listed stock (~7000) and is slow.",
)
top_n = st.sidebar.slider("How many setups to show", 5, 50, 25, step=5)
balance = st.sidebar.checkbox("Sector-balance the list", value=False)
max_per_sector = st.sidebar.slider("Max picks per sector", 1, 8, 3,
                                   disabled=not balance)

st.sidebar.markdown("---")
st.sidebar.caption("Not financial advice. Rule-based pattern detection on "
                   "historical data; breakouts can fail. Verify on your own "
                   "charts and manage risk.")

# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False, ttl=3600)
def load_spy():
    import yfinance as yf
    spy = yf.download("SPY", period="2y", auto_adjust=False, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    return spy

@st.cache_data(show_spinner=False, ttl=1800)
def get_universe(spec):
    return P.load_universe(spec)

def run_scan(universe, top_n, balance, max_per_sector, status, bar):
    spy = load_spy()
    tickers = get_universe(universe)
    status.write(f"Universe **{universe}** — {len(tickers)} symbols. Downloading…")
    results = []
    batch = 200
    nb = (len(tickers) + batch - 1) // batch
    for bi, i in enumerate(range(0, len(tickers), batch)):
        chunk = tickers[i:i + batch]
        data = P.download(chunk)
        single = len(chunk) == 1
        for tkr in chunk:
            try:
                df = P.get_one(data, tkr, single)
                if df is None:
                    continue
                m, why = P.evaluate(df, spy, tkr, tkr)
                if m is not None:
                    m["_df"] = P.enrich(df)
                    results.append(m)
            except Exception:
                pass
        bar.progress((bi + 1) / nb)
        time.sleep(0.3)
    results.sort(key=lambda x: x["score"], reverse=True)
    if balance:
        for m in results:
            m["sector"] = P.fetch_sector(m["ticker"])
        top = P.sector_balanced_select(results, top_n, max_per_sector)
    else:
        for m in results:
            m.setdefault("sector", "")
        top = results[:top_n]
    return results, top

def run_backtest(universe, status, bar):
    spy = load_spy()
    tickers = get_universe(universe)
    status.write(f"Backtesting **{universe}** — {len(tickers)} symbols (slow)…")
    trades = []
    batch = 200
    nb = (len(tickers) + batch - 1) // batch
    for bi, i in enumerate(range(0, len(tickers), batch)):
        chunk = tickers[i:i + batch]
        data = P.download(chunk)
        single = len(chunk) == 1
        for tkr in chunk:
            try:
                df = P.get_one(data, tkr, single)
                if df is None:
                    continue
                trades += P.backtest_ticker(P.enrich(df), spy)
            except Exception:
                pass
        bar.progress((bi + 1) / nb)
        time.sleep(0.3)
    return trades

# ---------------------------------------------------------------- main UI
st.title("P-TRACK — Bullish Setup Screener")

if mode == "Screen for setups":
    st.write("Pick a universe in the sidebar, then run the scan. Every number is "
             "computed from real price/volume data.")
    if st.button("▶ Run screen", type="primary"):
        status = st.empty()
        bar = st.progress(0.0)
        with st.spinner("Scanning…"):
            results, top = run_scan(universe, top_n, balance, max_per_sector, status, bar)
        bar.empty()
        status.success(f"{len(results)} stocks passed all filters. Showing top {len(top)}.")

        if not top:
            st.warning("No setups passed today. Try a broader universe or check back later.")
        else:
            rows = []
            for r, m in enumerate(top, 1):
                row = {"Rank": r}
                row.update(P.to_row(m))
                if m.get("sector"):
                    row["Sector"] = m["sector"]
                rows.append(row)
            df_table = pd.DataFrame(rows)
            st.subheader("Ranked setups")
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download CSV", df_table.to_csv(index=False),
                               "ptrack_results.csv", "text/csv")

            st.subheader("Charts & rationale")
            for r, m in enumerate(top, 1):
                sec = f" · {m['sector']}" if m.get("sector") else ""
                with st.expander(f"#{r}  {m['ticker']} — {m['pattern']} — "
                                 f"${m['price']:.2f} — score {m['score']}/100{sec}",
                                 expanded=(r <= 3)):
                    components.html(P.chart_svg(m["_df"], m), height=470)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Breakout", f"{m['breakout']:.2f}",
                              f"{m['dist_to_breakout']*100:+.1f}% away")
                    c2.metric("Stop", f"{m['stop']:.2f}")
                    c3.metric("Target / R-R", f"{m['target']:.2f}", f"{m['rr']:.2f} R")
                    st.caption("**Why this rank:** " + P.rank_reason(m, r))
                    st.caption("**Volume:** " + P.vol_trend_text(m))

else:  # Backtest
    st.write("Measure how these breakout setups performed historically "
             "(point-in-time detection, enter on breakout, exit at target/stop/time-stop).")
    step = st.slider("Thoroughness (lower = more trades, slower)", 1, 10, 5)
    P.CFG["bt_step"] = step
    if st.button("▶ Run backtest", type="primary"):
        status = st.empty()
        bar = st.progress(0.0)
        with st.spinner("Backtesting…"):
            trades = run_backtest(universe, status, bar)
        bar.empty()
        if not trades:
            st.warning("No trades generated. Try a broader universe or lower thoroughness.")
        else:
            report = P.run_backtest(trades)
            status.success(f"{len(trades)} simulated trades.")
            import numpy as np
            rs = np.array([t[0] for t in trades])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Trades", len(rs))
            c2.metric("Win rate", f"{(rs>0).mean()*100:.0f}%")
            c3.metric("Expectancy", f"{rs.mean():+.2f} R")
            wins, losses = rs[rs>0], rs[rs<=0]
            pf = wins.sum()/abs(losses.sum()) if losses.sum() else float('inf')
            c4.metric("Profit factor", f"{pf:.2f}")
            st.code(report)
            st.download_button("⬇ Download report", report, "ptrack_backtest.txt")
