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
import squeeze_screener as SQ
import squeeze_history as HIST

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

_adv = st.sidebar.checkbox("Show advanced tools (breakout + backtest)", value=False)
if _adv:
    mode = st.sidebar.radio("Mode", ["🔥 Squeeze Radar", "Screen for stocks", "Backtest the rules"])
else:
    mode = "🔥 Squeeze Radar"

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
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mainpic.jpg")

def show_logo():
    if os.path.exists(LOGO_PATH):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(LOGO_PATH, use_container_width=True)

def render_squeeze_candidate(c, header, expanded=True):
    """Shared display for one squeeze candidate (used by scan + single analyzer)."""
    cov = c.get("coverage")
    with st.expander(header, expanded=expanded):
        m1, m2, m3 = st.columns(3)
        m1.metric("Squeeze score", f"{c.get('score',0)}/100", c.get("tier","").split(" — ")[-1])
        m2.metric("Data coverage", f"{cov*100:.0f}%" if cov is not None else "—")
        m3.metric("5-day move", f"{c['ret_5d']*100:+.0f}%" if c.get("ret_5d") is not None else "—",
                  "LATE" if c.get("late") else "")
        if c.get("required_fails"):
            st.markdown("**❌ Fails a required condition:**")
            for rf in c["required_fails"]:
                st.markdown(f"- {rf}")
        if c.get("required_unknown"):
            st.caption("Unknown (free-data gaps): " + "; ".join(c["required_unknown"]))
        if c.get("history_flags"):
            st.markdown("**📈 What CHANGED since last scan (the early edge):**")
            for hf in c["history_flags"]:
                st.markdown(f"- {hf}")
        if c.get("flags"):
            st.markdown("**Leading signals firing:**")
            for fl in c["flags"]:
                st.markdown(f"- {fl}")
        if c.get("breakdown"):
            st.markdown("**Weighted model (your 8 indicators):**")
            bd = pd.DataFrame([
                {"Indicator": b["indicator"], "Weight": b["weight"],
                 "Band": ({0.25:"Low",0.5:"Med",0.75:"High",1.0:"Extreme"}.get(b["frac"], "—")
                          if b["frac"] is not None else "not measured"),
                 "Points": (b["contribution"] if b["contribution"] is not None else "—")}
                for b in c["breakdown"]])
            st.dataframe(bd, use_container_width=True, hide_index=True)
            if cov is not None and cov < 0.99:
                st.caption(f"⚠ Score normalized over the {cov*100:.0f}% of model weight measurable "
                           "for free. Cost-to-Borrow + Utilization (38%) need IBKR/Fintel/Ortex.")
        if c.get("disq"):
            st.markdown("**Checks before acting:**")
            for d in c["disq"]:
                st.markdown(f"- {d}")

# ---------------------------------------------------------------- main UI
st.title("P-TRACK — Bullish Setup Screener")

if mode == "🔥 Squeeze Radar":
    st.subheader("Squeeze Radar — early-detection (v1)")
    try:
        _cloud = HIST.using_cloud()
        _rows, _days = HIST.snapshot_count()
        st.caption(("🧠 Memory: **cloud (Supabase)** — persists across reboots. "
                    if _cloud else
                    "🧠 Memory: **local file** — on the hosted app this resets on reboot; "
                    "run on your Mac (or add Supabase) to keep history. ")
                   + f"Logged: {_rows} rows over {_days} day(s). The more you run it, "
                     "the more 'what changed' signals appear.")
    except Exception:
        pass
    st.write("Finds stocks **building** short-squeeze pressure using free, *leading* "
             "signals — not the 2-week-old short-interest lists everyone sees. It seeds "
             "candidates from recent **SEC 13D/13G filings**, **insider buys**, and "
             "**WallStreetBets mention velocity**, then scores each on shorts-adding-into-"
             "strength, relative-volume accumulation, and short-interest fuel.")
    # ---------- analyze one specific stock on demand ----------
    with st.container(border=True):
        st.markdown("#### 🔎 Analyze a specific stock")
        ac1, ac2 = st.columns([3, 1])
        sym = ac1.text_input("Ticker symbol", placeholder="e.g. GME", label_visibility="collapsed")
        go = ac2.button("Analyze", use_container_width=True)
        with st.expander("Have Cost-to-Borrow / Utilization? (optional, lifts coverage)"):
            mc1, mc2 = st.columns(2)
            man_ctb = mc1.number_input("Cost to Borrow %", min_value=0.0, value=0.0, step=1.0)
            man_util = mc2.number_input("Utilization %", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
        incl_feeds = st.checkbox("Also check 13D / insider / WSB feeds (slower)", value=False)
        if go and sym.strip():
            with st.spinner(f"Analyzing {sym.upper().strip()}…"):
                one = SQ.analyze_ticker(sym, ctb=(man_ctb or None), util=(man_util or None),
                                        include_feeds=incl_feeds)
            if one.get("error"):
                st.error(f"Couldn't analyze: {one['error']}")
            elif one.get("ok"):
                render_squeeze_candidate(
                    one, f"{one['ticker']} — Tier {one.get('tier','?')} — "
                         f"score {one.get('score',0)}/100", expanded=True)
            else:
                st.warning(f"No price data for '{sym.upper().strip()}'. Double-check the symbol "
                           "(US-listed), then try again — Yahoo occasionally rate-limits.")

    st.markdown("#### 📡 Scan the market for setups")
    wl_text = st.text_input("Optional: add your own tickers (comma-separated) to always include",
                            "")
    runup = st.slider("Exclude anything already up more than this in the last 5 days (anti-late)",
                      10, 100, 30, step=5,
                      help="Stocks past this 5-day gain are treated as 'already running' and moved "
                           "to a separate late bucket — the goal is to catch the build, not chase the candle.")
    SQ.CFG["max_recent_runup"] = runup / 100.0
    if st.button("🔥 Run Squeeze Radar", type="primary"):
        watch = [t.strip().upper() for t in wl_text.replace(" ", ",").split(",") if t.strip()]
        status = st.empty(); bar = st.progress(0.0)
        status.write("Pulling EDGAR filings, insider buys, WSB velocity, and short-interest…")
        with st.spinner("Scanning leading signals… (first run can take a minute)"):
            try:
                res = SQ.run_squeeze(watch or None, progress=lambda f: bar.progress(min(f, 1.0)))
            except Exception as e:
                res = []
                st.error(f"Data fetch issue: {e}. Try again — free sources sometimes rate-limit.")
        bar.empty()
        if not res:
            status.warning("No candidates surfaced. Try again shortly, or add tickers above.")
        else:
            late  = [c for c in res if c.get("late")]
            disq  = [c for c in res if c.get("disqualified") and not c.get("late")]
            prime = [c for c in res if not c.get("late") and not c.get("disqualified")]
            status.success(f"{len(prime)} qualified PRE-move candidates · {len(disq)} fail required "
                           f"conditions · {len(late)} already-running (late).")

            def render(c, r):
                render_squeeze_candidate(
                    c, f"#{r}  {c['ticker']} — Tier {c.get('tier','?')} — "
                       f"score {c.get('score',0)}/100"
                       + (f" — ▲{c['ret_5d']*100:.0f}% 5d" if c.get('ret_5d') else ""),
                    expanded=(r <= 5))

            st.subheader(f"🎯 Pre-move setups ({len(prime)})")
            if prime:
                pf = pd.DataFrame([SQ.to_row(c) for c in prime])
                pf.insert(0, "Rank", range(1, len(pf) + 1))
                st.dataframe(pf, use_container_width=True, hide_index=True)
                st.download_button("⬇ Download pre-move CSV", pf.to_csv(index=False),
                                   "squeeze_premove.csv", "text/csv")
                for r, c in enumerate(prime[:20], 1):
                    render(c, r)
            else:
                st.info("No early setups cleared the filter right now. That's normal — "
                        "true pre-squeeze conditions are rare. Lower the run-up cutoff or check back.")

            if disq:
                with st.expander(f"❌ Fail required conditions ({len(disq)}) — "
                                 "not real squeeze candidates (low SI, fast exit, thin volume, or stale).",
                                 expanded=False):
                    dq = pd.DataFrame([SQ.to_row(c) for c in disq])
                    st.dataframe(dq, use_container_width=True, hide_index=True)
            if late:
                with st.expander(f"⛔ Already running — too late ({len(late)}). "
                                 "Shown for awareness; you'd be chasing.", expanded=False):
                    lf = pd.DataFrame([SQ.to_row(c) for c in late])
                    st.dataframe(lf, use_container_width=True, hide_index=True)
    st.caption("Free/live sources: SEC EDGAR, OpenInsider, Apewisdom (WSB), Yahoo Finance. "
               "Short interest is bi-monthly (fuel, not trigger). Borrow-fee / utilization / "
               "options-sweep / dark-pool are not in v1 (need IBKR or a paid feed). "
               "**Not financial advice — squeeze speculation is extremely high risk.**")
    show_logo()

elif mode == "Screen for stocks":
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
