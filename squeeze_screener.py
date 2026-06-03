#!/usr/bin/env python3
"""
SQUEEZE RADAR  --  Early-Detection Score (EDS) engine, lean v1
==============================================================
Implements a focused, FREE/LIVE subset of the short-squeeze Early-Detection
framework. The whole point of the framework is to beat the 2-week FINRA lag by
leaning on LEADING signals, so v1 scores the four highest-value free signals:

  1. Smart money    -> EDGAR 13D/13G activist filings + OpenInsider insider buys
  2. Doubling down  -> short interest RISING while price RISES (yfinance)
  3. Accumulation   -> relative-volume trend before a price move (yfinance)
  4. Social velocity-> first/accelerating WSB mentions (Apewisdom, free API)

Short interest % of float (free from yfinance, ~2wk lagged) is used as the
"fuel" filter exactly as the framework prescribes -- a baseline, not a trigger.

DESIGN: instead of scanning all ~7000 tickers (slow, and most have no signal),
we SEED candidates from the leading-signal feeds (Apewisdom / OpenInsider /
EDGAR) plus an optional watchlist, then enrich + score only those. That is both
faster and more faithful to the framework (start from leading signals).

DATA NOTES
  * All live fetches run on YOUR machine / Streamlit Cloud (open network).
  * Free SI is bi-monthly; treat the SI level as fuel, the RATE-OF-CHANGE and
    the leading signals as the edge.
  * Borrow fee / utilization / real-time options sweeps / dark pool are NOT in
    v1 (need IBKR or a paid feed). Slots are left for them later.

NOT INVESTMENT ADVICE. Squeeze speculation is extremely high risk.
"""
import os, re, io, sys, time, json, datetime as dt
import urllib.request, urllib.parse
import numpy as np
import pandas as pd
try:
    import squeeze_history as HIST
except Exception:
    HIST = None

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SEC_UA = "ptrack-squeeze-radar research contact@example.com"   # SEC requires a UA w/ contact

CFG = dict(
    min_si_pct        = 0.15,   # framework "required": SI >= ~15% of float (fuel)
    require_si_fuel   = False,   # if True, drop candidates below min_si_pct (when SI known)
    min_avg_volume    = 300_000, # tradeability floor
    edgar_lookback_d  = 21,      # how many days of 13D/13G filings to scan
    max_candidates    = 400,     # cap enrichment work
    max_recent_runup  = 0.30,    # >this 5-day gain = "already ran" -> late bucket (anti-lag)
    flat_band         = 0.10,    # |5-day move| <= this counts as "price still flat"
)

# ==========================================================================
# LEADING-SIGNAL FEEDS  (each returns a dict ticker -> signal info)
# ==========================================================================
def _get(url, headers=None, timeout=25):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout).read()

def fetch_apewisdom(pages=2):
    """Free WSB mention tracker. Returns {ticker: {mentions, prev, growth, rank}}."""
    out = {}
    for p in range(1, pages + 1):
        try:
            raw = _get(f"https://apewisdom.io/api/v1.0/filter/wallstreetbets/page/{p}")
            data = json.loads(raw)
            for r in data.get("results", []):
                tk = str(r.get("ticker", "")).upper().strip()
                if not tk:
                    continue
                m = float(r.get("mentions") or 0)
                prev = r.get("mentions_24h_ago")
                prev = float(prev) if prev not in (None, "", "null") else None
                growth = (m / prev) if prev else None       # None => first appearance
                out[tk] = dict(mentions=m, prev=prev, growth=growth,
                               rank=r.get("rank"))
        except Exception as e:
            print(f"  ! apewisdom page {p}: {e}", file=sys.stderr)
    return out

def fetch_openinsider_buys():
    """Free recent insider open-market purchases. Returns {ticker: total_$value}."""
    out = {}
    try:
        raw = _get("http://openinsider.com/latest-insider-purchases-25k").decode("latin-1")
        tables = pd.read_html(io.StringIO(raw))
        df = max(tables, key=len)
        # find the columns we need (OpenInsider layout)
        cols = {str(c).strip(): c for c in df.columns}
        tcol = next((cols[c] for c in cols if c.lower() == "ticker"), None)
        vcol = next((cols[c] for c in cols if "value" in c.lower()), None)
        ttcol = next((cols[c] for c in cols if "trade type" in c.lower()), None)
        if tcol is None:
            return out
        for _, row in df.iterrows():
            tt = str(row.get(ttcol, "")) if ttcol else "P"
            if "P" not in tt:                  # P = purchase
                continue
            tk = str(row[tcol]).upper().strip()
            if not tk or not tk.isalpha():
                continue
            val = 0.0
            if vcol is not None:
                val = abs(float(re.sub(r"[^0-9.\-]", "", str(row[vcol])) or 0))
            out[tk] = out.get(tk, 0.0) + val
    except Exception as e:
        print(f"  ! openinsider: {e}", file=sys.stderr)
    return out

def fetch_edgar_activist(days=21):
    """Free SEC EDGAR full-text search for recent 13D/13G activist filings.
    Returns {ticker: form} (form in {'SC 13D','SC 13G',...})."""
    out = {}
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    for form in ("SC 13D", "SC 13D/A", "SC 13G"):
        try:
            url = ("https://efts.sec.gov/LATEST/search-index?q=%22%22"
                   f"&forms={urllib.parse.quote(form)}"
                   f"&startdt={start}&enddt={end}")
            raw = _get(url, headers={"User-Agent": SEC_UA})
            data = json.loads(raw)
            for hit in data.get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                for name in src.get("display_names", []):
                    m = re.search(r"\(([A-Z]{1,6})\)", name)
                    if m:
                        tk = m.group(1)
                        # 13D outranks 13G if both seen
                        if tk not in out or form.startswith("SC 13D"):
                            out[tk] = form
        except Exception as e:
            print(f"  ! edgar {form}: {e}", file=sys.stderr)
    return out

# ==========================================================================
# ENRICHMENT  (per-candidate fundamentals + price)
# ==========================================================================
def get_si_info(tkr):
    """yfinance short-interest + float fundamentals (free, bi-monthly)."""
    import yfinance as yf
    info = {}
    try:
        info = yf.Ticker(tkr).get_info()
    except Exception:
        return {}
    return dict(
        short_pct   = info.get("shortPercentOfFloat"),
        shares_short= info.get("sharesShort"),
        short_prior = info.get("sharesShortPriorMonth"),
        days_cover  = info.get("shortRatio"),
        float_shares= info.get("floatShares"),
        name        = info.get("shortName") or info.get("longName") or tkr,
    )

def get_options_pc(tkr):
    """Approximate options 'call flow' via put/call OPEN-INTEREST ratio across the
    nearest 1-2 expiries (free from yfinance). Lower ratio = more call-heavy.
    Not real-time sweep data, but a free directional proxy."""
    import yfinance as yf
    try:
        t = yf.Ticker(tkr)
        exps = list(t.options or [])[:2]
        calls_oi = puts_oi = 0
        for e in exps:
            ch = t.option_chain(e)
            calls_oi += float(ch.calls["openInterest"].fillna(0).sum())
            puts_oi  += float(ch.puts["openInterest"].fillna(0).sum())
        if calls_oi <= 0:
            return None
        return round(puts_oi / calls_oi, 3)
    except Exception:
        return None

def price_features(df):
    """relative volume, recent return, price-holding from a daily OHLCV frame."""
    if df is None or len(df) < 40:
        return None
    close = df["Close"]; vol = df["Volume"]
    rvol = float(vol.iloc[-3:].mean()) / float(vol.iloc[-23:-3].mean() + 1e-9)
    ret_1m = float(close.iloc[-1] / close.iloc[-21] - 1)
    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0
    ret_10d = float(close.iloc[-1] / close.iloc[-11] - 1) if len(close) >= 11 else 0.0
    near_high = float(close.iloc[-1] / close.iloc[-252:].max()) if len(close) >= 60 else 0.0
    recent_dd = float(close.iloc[-5:].min() / close.iloc[-6] - 1)  # worst recent dip
    adv = float(vol.iloc[-30:].mean())
    return dict(rvol=rvol, ret_1m=ret_1m, ret_5d=ret_5d, ret_10d=ret_10d,
                near_high=near_high, recent_dd=recent_dd, adv=adv,
                price=float(close.iloc[-1]))

# ==========================================================================
# WEIGHTED MODEL  (user's 8-indicator framework, exact weights & thresholds)
# ==========================================================================
WEIGHTS = {  # must sum to 100
    "SI%": 22, "CTB": 20, "UTIL": 18, "DTC": 12,
    "CALLS": 12, "SOCIAL": 8, "FLOAT": 5, "CHART": 3,
}
# band -> fraction of that indicator's weight earned
_BAND = {"low": 0.25, "med": 0.50, "high": 0.75, "extreme": 1.0}

def _band_frac(value, lo, med, hi):
    """Return fraction for ascending thresholds: <lo=low, <med=med, <hi=high, else extreme."""
    if value is None:
        return None
    if value < lo:   return _BAND["low"]
    if value < med:  return _BAND["med"]
    if value < hi:   return _BAND["high"]
    return _BAND["extreme"]

def _indicator_fracs(c):
    """Return {indicator: fraction or None} per the framework thresholds."""
    f = {}
    # SI % of float (normalize 0-1 vs 0-100)
    sp = c.get("short_pct")
    spv = (sp*100 if (sp is not None and sp <= 1.5) else sp) if sp is not None else None
    f["SI%"]  = _band_frac(spv, 15, 30, 50)
    # Cost to borrow (annualized %) -- only if supplied (IBKR/manual)
    f["CTB"]  = _band_frac(c.get("ctb"), 10, 50, 150)
    # Utilization (%) -- only if supplied
    f["UTIL"] = _band_frac(c.get("util"), 50, 80, 95)
    # Days to cover
    f["DTC"]  = _band_frac(c.get("days_cover"), 3, 5, 10)
    # Options: lower put/call ratio = more call-heavy = higher band (descending)
    pc = c.get("pc_ratio")
    if pc is None:
        f["CALLS"] = None
    else:
        f["CALLS"] = (_BAND["extreme"] if pc < 0.4 else _BAND["high"] if pc < 0.6
                      else _BAND["med"] if pc < 0.8 else _BAND["low"])
    # Social velocity (we always check the WSB feed -> absence = baseline 'low')
    if c.get("soc_prev") in (None,) and c.get("soc_mentions"):
        f["SOCIAL"] = _BAND["high"]                      # first appearance from zero
    elif c.get("soc_growth"):
        g = c["soc_growth"]
        f["SOCIAL"] = (_BAND["extreme"] if g >= 20 else _BAND["high"] if g >= 5
                       else _BAND["med"] if g >= 2 else _BAND["low"])
    else:
        f["SOCIAL"] = _BAND["low"]                        # baseline, not trending
    # Float size (smaller = higher band) -- descending in shares
    fl = c.get("float_shares")
    if fl is None:
        f["FLOAT"] = None
    else:
        m = fl/1e6
        f["FLOAT"] = (_BAND["extreme"] if m < 10 else _BAND["high"] if m < 50
                      else _BAND["med"] if m < 200 else _BAND["low"])
    # Chart break (from near-high + relative volume)
    nh, rv = c.get("near_high"), c.get("rvol")
    if nh is None:
        f["CHART"] = None
    elif nh >= 1.0 and (rv or 0) >= 1.5: f["CHART"] = _BAND["extreme"]
    elif nh >= 0.98:                     f["CHART"] = _BAND["high"]
    elif nh >= 0.90:                     f["CHART"] = _BAND["med"]
    else:                                f["CHART"] = _BAND["low"]
    return f

def score_weighted(c):
    """0-100 weighted score, re-normalized over the indicators we could measure.
    Returns (score, breakdown_list, coverage_fraction)."""
    fr = _indicator_fracs(c)
    measured_w = sum(WEIGHTS[k] for k, v in fr.items() if v is not None)
    earned = sum(WEIGHTS[k] * v for k, v in fr.items() if v is not None)
    score = round((earned / measured_w) * 100, 1) if measured_w else 0.0
    coverage = measured_w / 100.0
    breakdown = []
    for k in WEIGHTS:
        v = fr[k]
        breakdown.append(dict(indicator=k, weight=WEIGHTS[k], frac=v,
                              contribution=(round(WEIGHTS[k]*v, 1) if v is not None else None)))
    return score, breakdown, coverage

# ==========================================================================
# DESCRIPTIVE FLAGS  (human-readable; not the ranking score)
# ==========================================================================
def score_candidate(c):
    """c is a merged dict of signals. Returns (score, parts, flags)."""
    parts, flags = {}, []

    # ---- FUEL: short interest % of float (0-4) ----
    sp = c.get("short_pct")
    if sp is not None:
        spv = sp * 100 if sp <= 1.5 else sp        # normalize 0-1 vs 0-100
        parts["si_pct"] = (4 if spv >= 40 else 3 if spv >= 30 else
                           2 if spv >= 20 else 1 if spv >= 10 else 0)
        if spv >= 15: flags.append(f"SI {spv:.0f}% of float")
    # ---- days to cover (0-4) ----
    dtc = c.get("days_cover")
    if dtc:
        parts["dtc"] = (4 if dtc >= 12 else 3 if dtc >= 8 else
                        2 if dtc >= 5 else 1 if dtc >= 3 else 0)
        if dtc >= 3: flags.append(f"{dtc:.1f} days to cover")

    # ---- DOUBLING DOWN: SI rising while price rising (0-7) ----
    ss, sprior = c.get("shares_short"), c.get("short_prior")
    ret_1m = c.get("ret_1m")
    if ss and sprior and ret_1m is not None and sprior > 0:
        si_chg = ss / sprior - 1
        if ret_1m > 0 and si_chg > 0:
            if   ret_1m >= 0.15 and si_chg >= 0.05: parts["doubling_down"] = 7
            elif ret_1m >= 0.10 and si_chg >= 0.05: parts["doubling_down"] = 5
            elif ret_1m >= 0.05 and si_chg >= 0.03: parts["doubling_down"] = 3
            if parts.get("doubling_down"):
                flags.append(f"shorts adding (+{si_chg*100:.0f}%) into a +{ret_1m*100:.0f}% move")
        # also reward pure SI build (rate of change) up to +4
        if si_chg >= 0.25:   parts["si_roc"] = 4
        elif si_chg >= 0.10: parts["si_roc"] = 3
        elif si_chg >= 0.05: parts["si_roc"] = 2
        if parts.get("si_roc"): flags.append(f"SI up {si_chg*100:.0f}% vs prior month")

    # ---- SMART MONEY: 13D / 13G / insider buys (0-8) ----
    form = c.get("edgar_form")
    if form:
        parts["smart_money"] = 8 if form.startswith("SC 13D") else 4
        flags.append(f"recent {form} filing")
    if c.get("insider_buy_$"):
        parts["insider"] = 5
        flags.append(f"insider buying (${c['insider_buy_$']:,.0f})")

    # ---- ACCUMULATION: relative volume trend (0-5) ----
    rvol = c.get("rvol")
    ret_5d = c.get("ret_5d")
    flat = (ret_5d is not None and abs(ret_5d) <= CFG["flat_band"])
    if rvol:
        base = (5 if rvol >= 2 else 3 if rvol >= 1.5 else 1 if rvol >= 1.2 else 0)
        # the pre-move fingerprint: volume building while price is STILL FLAT.
        # if price already ran, volume is just the move's exhaust -> heavily discount.
        if flat:
            parts["rvol"] = base
            if rvol >= 1.5:
                flags.append(f"volume building {rvol:.1f}x while price still flat (accumulation)")
        else:
            parts["rvol"] = round(base * 0.3, 1)   # discount late volume

    # ---- EARLINESS / resilience (0-5), NOT 'already at highs' ----
    dd = c.get("recent_dd")
    if flat and dd is not None and dd <= -0.04 and (ret_5d or 0) >= -0.02:
        # dipped intraday but closed flat/up = buyers absorbing supply, pre-move
        parts["resilience"] = 5; flags.append("dips bought / holding support under pressure")
    elif flat and (ret_1m is not None and -0.05 <= ret_1m <= 0.15):
        parts["resilience"] = 2; flags.append("coiling — quiet, not extended")

    # ---- SOCIAL VELOCITY (0-6) ----
    g, mentions, prev = c.get("soc_growth"), c.get("soc_mentions"), c.get("soc_prev")
    if mentions:
        if prev in (None, 0):
            parts["social"] = 6; flags.append("FIRST WSB mention (day-1 social)")
        elif g and g >= 2:
            parts["social"] = 5; flags.append(f"WSB mentions {g:.1f}x in 24h")
        elif g and g >= 1.3:
            parts["social"] = 3; flags.append("WSB mentions rising")
        else:
            parts["social"] = 1

    score = float(sum(parts.values()))
    return round(score, 1), parts, flags

def tier(score):
    # weighted model is a true 0-100 scale
    return ("A — PRIME" if score >= 75 else
            "B — BUILDING" if score >= 55 else
            "C — WATCHLIST" if score >= 35 else
            "D — SKIP")

def disqualifier_notes(c):
    """Cheap, free disqualifier flags (full checks need filings -> manual)."""
    notes = []
    if (c.get("ret_5d") or 0) >= CFG["max_recent_runup"]:
        notes.insert(0, f"⛔ already +{c['ret_5d']*100:.0f}% in 5 days — this IS the move, "
                        "not the setup. You're late.")
    if (c.get("ret_1m") or 0) >= 0.40:
        notes.append("⚠ already +40% in a month — may be late / chasing")
    sp = c.get("short_pct")
    if sp is not None:
        spv = sp*100 if sp <= 1.5 else sp
        if spv < 10:
            notes.append("⚠ low short interest — weak squeeze fuel")
    if c.get("adv") and c["adv"] < CFG["min_avg_volume"]:
        notes.append("⚠ thin volume — hard to trade")
    notes.append("Manual check: ATM shelf (S-3), convertible debt, cash runway, options market")
    return notes

# ==========================================================================
# ORCHESTRATION
# ==========================================================================
def seed_candidates(watchlist=None):
    """Union of leading-signal feeds -> dict ticker -> partial signal dict."""
    ape  = fetch_apewisdom()
    ins  = fetch_openinsider_buys()
    act  = fetch_edgar_activist(CFG["edgar_lookback_d"])
    cands = {}
    def slot(tk):
        return cands.setdefault(tk, {"ticker": tk})
    for tk, d in ape.items():
        s = slot(tk); s["soc_mentions"]=d["mentions"]; s["soc_prev"]=d["prev"]; s["soc_growth"]=d["growth"]
    for tk, v in ins.items():
        slot(tk)["insider_buy_$"] = v
    for tk, f in act.items():
        slot(tk)["edgar_form"] = f
    for tk in (watchlist or []):
        slot(str(tk).upper().strip())
    # cap
    return dict(list(cands.items())[:CFG["max_candidates"]])

def run_squeeze(watchlist=None, progress=None):
    import yfinance as yf
    cands = seed_candidates(watchlist)
    tickers = list(cands.keys())
    if not tickers:
        return []
    # bulk price download
    try:
        data = yf.download(tickers, period="1y", group_by="ticker",
                           auto_adjust=False, threads=True, progress=False)
    except Exception:
        data = None
    results = []
    n = len(tickers)
    for i, tk in enumerate(tickers):
        c = cands[tk]
        # price features (pull this ticker's frame out of the bulk download)
        df = None
        try:
            if data is not None and isinstance(data.columns, pd.MultiIndex):
                if tk in data.columns.get_level_values(0):
                    df = data[tk]
            elif data is not None and len(tickers) == 1:
                df = data
        except Exception:
            df = None
        if df is None:
            try: df = yf.download(tk, period="1y", auto_adjust=False, progress=False)
            except Exception: df = None
        pf = price_features(df.dropna() if df is not None else None)
        if pf: c.update(pf)
        # SI fundamentals + options put/call proxy
        c.update({k: v for k, v in get_si_info(tk).items() if v is not None})
        pc = get_options_pc(tk)
        if pc is not None: c["pc_ratio"] = pc
        # filters
        if CFG["require_si_fuel"] and c.get("short_pct") is not None:
            spv = c["short_pct"]*100 if c["short_pct"] <= 1.5 else c["short_pct"]
            if spv < CFG["min_si_pct"]*100:
                if progress: progress((i+1)/n)
                continue
        # WEIGHTED model score (authoritative) + descriptive flags
        score, breakdown, coverage = score_weighted(c)
        _, parts, flags = score_candidate(c)
        c["score"], c["breakdown"], c["coverage"] = score, breakdown, coverage
        c["parts"], c["flags"] = parts, flags
        c["late"] = (c.get("ret_5d") or 0) >= CFG["max_recent_runup"]
        c["tier"] = tier(score)
        c["disq"] = disqualifier_notes(c)
        results.append(c)
        if progress: progress((i+1)/n)
        time.sleep(0.05)
    # ---- MEMORY: log today's snapshot + attach rate-of-change signals ----
    if HIST is not None:
        try:
            HIST.record(results)
            HIST.attach_changes(results)
        except Exception as e:
            print(f"  ! history: {e}", file=sys.stderr)
    # pre-move setups first (not late), then by score; late movers sink to the bottom
    results.sort(key=lambda x: (x.get("late", False), -x["score"]))
    return results

def analyze_ticker(tkr, ctb=None, util=None, include_feeds=False):
    """Run the full weighted model on ONE symbol on demand. Fast & defensive:
    uses only the quick Yahoo data by default. The three broad feeds (EDGAR /
    OpenInsider / WSB) can be slow or rate-limited on a hosted app, so they are
    OFF by default for single-stock analysis and never allowed to break it.
    ctb/util can be passed manually to lift coverage toward 100%. Always returns
    a dict (never raises); c['error'] is set if something went wrong."""
    tk = str(tkr).upper().strip()
    c = {"ticker": tk, "ok": False}
    try:
        import yfinance as yf
        if ctb not in (None, "", 0):  c["ctb"] = float(ctb)
        if util not in (None, "", 0): c["util"] = float(util)
        # price history (quick)
        df = None
        try:
            df = yf.download(tk, period="1y", auto_adjust=False, progress=False)
            if df is not None and isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except Exception:
            df = None
        pf = price_features(df.dropna() if df is not None else None)
        if pf: c.update(pf)
        # short interest / float + options proxy (each wrapped internally)
        try: c.update({k: v for k, v in get_si_info(tk).items() if v is not None})
        except Exception: pass
        try:
            pc = get_options_pc(tk)
            if pc is not None: c["pc_ratio"] = pc
        except Exception: pass
        # optional, never-blocking leading-signal feeds
        if include_feeds:
            try:
                a = fetch_apewisdom().get(tk)
                if a: c["soc_mentions"]=a["mentions"]; c["soc_prev"]=a["prev"]; c["soc_growth"]=a["growth"]
            except Exception: pass
            try:
                v = fetch_openinsider_buys().get(tk)
                if v: c["insider_buy_$"] = v
            except Exception: pass
            try:
                f = fetch_edgar_activist(CFG["edgar_lookback_d"]).get(tk)
                if f: c["edgar_form"] = f
            except Exception: pass
        # score
        score, breakdown, coverage = score_weighted(c)
        _, parts, flags = score_candidate(c)
        c["score"], c["breakdown"], c["coverage"] = score, breakdown, coverage
        c["parts"], c["flags"] = parts, flags
        c["late"] = (c.get("ret_5d") or 0) >= CFG["max_recent_runup"]
        c["tier"] = tier(score)
        c["disq"] = disqualifier_notes(c)
        c["ok"] = bool(pf)            # did we get price data?
        # MEMORY: log + attach change-signals for this ticker
        if HIST is not None and c["ok"]:
            try:
                HIST.record([c]); HIST.attach_changes([c])
            except Exception:
                pass
    except Exception as e:
        c["error"] = f"{type(e).__name__}: {e}"
    return c

def to_row(c):
    sp = c.get("short_pct")
    spv = (sp*100 if (sp is not None and sp <= 1.5) else sp) if sp is not None else None
    return dict(
        Ticker=c["ticker"], Name=(c.get("name") or "")[:28],
        Score=c.get("score"), Tier=c.get("tier"),
        Coverage_pct=round(c["coverage"]*100) if c.get("coverage") is not None else None,
        Price=round(c["price"],2) if c.get("price") else None,
        ShortPctFloat=round(spv,1) if spv is not None else None,
        DaysToCover=c.get("days_cover"),
        PutCall=c.get("pc_ratio"),
        RelVol=round(c["rvol"],2) if c.get("rvol") else None,
        Ret5D_pct=round(c["ret_5d"]*100,1) if c.get("ret_5d") is not None else None,
        Ret1M_pct=round(c["ret_1m"]*100,1) if c.get("ret_1m") is not None else None,
        Late=("LATE" if c.get("late") else ""),
        Signals="; ".join(c.get("flags", [])),
    )

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Squeeze Radar (EDS v1)")
    ap.add_argument("--watchlist", default=None, help="optional file, one ticker/line")
    ap.add_argument("--csv", default="squeeze_results.csv")
    a = ap.parse_args()
    wl = None
    if a.watchlist:
        wl = [l.strip() for l in open(a.watchlist) if l.strip() and not l.startswith("#")]
    print("[*] seeding candidates from EDGAR + OpenInsider + Apewisdom ...")
    res = run_squeeze(wl, progress=lambda f: None)
    print(f"[*] scored {len(res)} candidates")
    rows = [to_row(c) for c in res[:50]]
    df = pd.DataFrame(rows); df.insert(0, "Rank", range(1, len(df)+1))
    df.to_csv(a.csv, index=False)
    print(df.head(25).to_string(index=False))
    print(f"\nSaved -> {a.csv}")
