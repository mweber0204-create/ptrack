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
import os
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import ptrack_screener as P

st.set_page_config(page_title="P-TRACK Screener", layout="centered",
                   page_icon="📈", initial_sidebar_state="expanded")

# Phone-friendly styling: full-width buttons, comfortable text, tighter margins.
st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 860px;}
  .stButton > button, .stDownloadButton > button {
      width: 100%; padding: 0.7rem 1rem; font-size: 1.05rem; border-radius: 10px;}
  html, body, [class*="css"] {font-size: 1.02rem;}
  div[data-testid="stDataFrame"] {overflow-x: auto;}
  @media (max-width: 640px) {
      h1 {font-size: 1.5rem;}
      .block-container {padding-left: 0.8rem; padding-right: 0.8rem;}
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar
st.sidebar.title("📈 P-TRACK")
st.sidebar.caption("Bullish technical setup screener")

mode = st.sidebar.radio("Mode", ["Screen for stocks", "Backtest the rules"])

universe = st.sidebar.selectbox(
    "Universe",
    ["sp500", "nasdaq100", "sp1500", "all"],
    help="sp500 is fastest. 'all' scans every US-listed stock (~7000) and is slow.",
)
top_n = st.sidebar.slider("How many stocks to show (strongest first)", 5, 50, 25, step=5)

# ---- Profitability levers (apply to BOTH screening and backtest) ----
with st.sidebar.expander("⚙️ Edge filters (advanced)"):
    st.caption("Tilt toward what the backtest showed works. Defaults = original rules.")
    pattern_choice = st.radio(
        "Patterns to trade",
        ["All patterns", "Proven only (VCP + Flat Base + Cup & Handle)", "VCP only"],
        help="VCP was your highest win-rate pattern. 'VCP only' judges by it alone.")
    market_filter = st.checkbox("Only when market (SPY) is above its 200-day",
                                help="Skips long setups during weak markets.")
    min_score = st.slider("Minimum setup score", 0, 90, 0, step=5,
                          help="Only take setups scoring at least this. 0 = no cutoff.")
    st.caption("(Each stock is graded 0–100 on overall strength — this hides "
               "anything below your number. 0 = show all.)")
    min_rr = st.slider("Minimum reward-to-risk", 0.0, 3.0, 0.0, step=0.25,
                       help="Hide setups whose target is too close to the stop. "
                            "0 = show all. Try 1.5. Note: with the standard stop, "
                            "most setups are under 1, so a high value may show few names.")
    tight_stop = st.checkbox("Use tighter stop (just under support)",
                             help="Smaller risk = higher reward-to-risk, but the stop "
                                  "is easier to hit. Test it in Backtest mode before trusting it.")

# push the chosen levers into the engine config
if pattern_choice == "VCP only":
    P.CFG["allowed_patterns"] = {"Volatility Contraction Pattern (VCP)"}
elif pattern_choice.startswith("Proven"):
    P.CFG["allowed_patterns"] = {"Volatility Contraction Pattern (VCP)",
                                 "Flat Base", "Cup and Handle"}
else:
    P.CFG["allowed_patterns"] = None
P.CFG["require_market_uptrend"] = market_filter
P.CFG["min_score"] = float(min_score)
P.CFG["min_rr"] = float(min_rr)
P.CFG["tight_stop"] = tight_stop

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

def run_scan(universe, top_n, status, bar):
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

# ---------------------------------------------------------------- hero logo
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")

def show_logo():
    if os.path.exists(LOGO_PATH):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(LOGO_PATH, use_container_width=True)

# ---------------------------------------------------------------- main UI
st.title("P-TRACK — Bullish Setup Screener")

if mode == "Screen for stocks":
    st.write("Pick a universe in the sidebar, then run the scan. Every number is "
             "computed from real price/volume data.")
    if st.button("▶ Run screen", type="primary"):
        status = st.empty()
        bar = st.progress(0.0)
        with st.spinner("Scanning…"):
            results, top = run_scan(universe, top_n, status, bar)
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
                    components.html(P.chart_svg(m["_df"], m), height=430, scrolling=False)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Breakout", f"{m['breakout']:.2f}",
                              f"{m['dist_to_breakout']*100:+.1f}% away")
                    c2.metric("Stop", f"{m['stop']:.2f}")
                    c3.metric("Target / R-R", f"{m['target']:.2f}", f"{m['rr']:.2f} R")
                    st.caption("**Why this rank:** " + P.rank_reason(m, r))
                    st.caption("**Volume:** " + P.vol_trend_text(m))

    show_logo()

else:  # Backtest
    st.write("Measure how these breakout setups performed historically "
             "(point-in-time detection, enter on breakout, exit at target/stop/time-stop).")
    step = st.slider("Thoroughness (lower = more trades, slower)", 1, 10, 5)
    P.CFG["bt_step"] = step
    with st.expander("💵 Realistic costs & exit (optional)"):
        slp = st.slider("Slippage on entry (%)", 0.0, 1.0, 0.0, step=0.05,
                        help="Real breakout fills come in a bit high. 0.2–0.3% is typical.")
        costr = st.slider("Round-trip cost (in R)", 0.0, 0.20, 0.0, step=0.01,
                          help="Commissions/spread expressed as a fraction of your risk.")
        be = st.checkbox("Move stop to breakeven after +1R",
                         help="Locks in no-loss once a trade is up 1R. Fewer losers, "
                              "but also caps some winners that pull back first.")
    P.CFG["bt_slippage_pct"] = slp / 100.0
    P.CFG["bt_cost_R"] = costr
    P.CFG["bt_breakeven_after_1R"] = be
    st.caption("The **Edge filters** in the sidebar (pattern/market/score) also apply here.")
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
