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

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
SEC_UA = "ptrack-squeeze-radar research contact@example.com"   # SEC requires a UA w/ contact

CFG = dict(
    min_si_pct        = 0.15,   # framework "required": SI >= ~15% of float (fuel)
    require_si_fuel   = False,   # if True, drop candidates below min_si_pct (when SI known)
    min_avg_volume    = 300_000, # tradeability floor
    edgar_lookback_d  = 21,      # how many days of 13D/13G filings to scan
    max_candidates    = 400,     # cap enrichment work
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

def price_features(df):
    """relative volume, recent return, price-holding from a daily OHLCV frame."""
    if df is None or len(df) < 40:
        return None
    close = df["Close"]; vol = df["Volume"]
    rvol = float(vol.iloc[-3:].mean()) / float(vol.iloc[-23:-3].mean() + 1e-9)
    ret_1m = float(close.iloc[-1] / close.iloc[-21] - 1)
    near_high = float(close.iloc[-1] / close.iloc[-252:].max()) if len(close) >= 60 else 0.0
    recent_dd = float(close.iloc[-5:].min() / close.iloc[-6] - 1)  # worst recent dip
    adv = float(vol.iloc[-30:].mean())
    return dict(rvol=rvol, ret_1m=ret_1m, near_high=near_high,
                recent_dd=recent_dd, adv=adv, price=float(close.iloc[-1]))

# ==========================================================================
# SCORING  (lean EDS v1)
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
    if rvol:
        parts["rvol"] = (5 if rvol >= 2 else 3 if rvol >= 1.5 else 1 if rvol >= 1.2 else 0)
        if rvol >= 1.5: flags.append(f"relative volume {rvol:.1f}x")

    # ---- PRICE HOLDING / resilience (0-6) ----
    nh, dd = c.get("near_high"), c.get("recent_dd")
    if nh is not None:
        if nh >= 0.98:        parts["price_hold"] = 6; flags.append("at/near 52wk high")
        elif dd is not None and dd > -0.02 and (ret_1m or 0) >= 0:
            parts["price_hold"] = 3; flags.append("holding up under pressure")

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
    # thresholds calibrated for the v1 signal subset (max ~46 in practice)
    return ("A — PRIME" if score >= 28 else
            "B — BUILDING" if score >= 18 else
            "C — WATCHLIST" if score >= 9 else
            "D — SKIP")

def disqualifier_notes(c):
    """Cheap, free disqualifier flags (full checks need filings -> manual)."""
    notes = []
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
        # SI fundamentals
        c.update({k: v for k, v in get_si_info(tk).items() if v is not None})
        # filters
        if CFG["require_si_fuel"] and c.get("short_pct") is not None:
            spv = c["short_pct"]*100 if c["short_pct"] <= 1.5 else c["short_pct"]
            if spv < CFG["min_si_pct"]*100:
                if progress: progress((i+1)/n)
                continue
        sc, parts, flags = score_candidate(c)
        c["score"], c["parts"], c["flags"] = sc, parts, flags
        c["tier"] = tier(sc)
        c["disq"] = disqualifier_notes(c)
        results.append(c)
        if progress: progress((i+1)/n)
        time.sleep(0.05)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

def to_row(c):
    sp = c.get("short_pct")
    spv = (sp*100 if (sp is not None and sp <= 1.5) else sp) if sp is not None else None
    return dict(
        Ticker=c["ticker"], Name=(c.get("name") or "")[:28],
        Score=c.get("score"), Tier=c.get("tier"),
        Price=round(c["price"],2) if c.get("price") else None,
        ShortPctFloat=round(spv,1) if spv is not None else None,
        DaysToCover=c.get("days_cover"),
        RelVol=round(c["rvol"],2) if c.get("rvol") else None,
        Ret1M_pct=round(c["ret_1m"]*100,1) if c.get("ret_1m") is not None else None,
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
