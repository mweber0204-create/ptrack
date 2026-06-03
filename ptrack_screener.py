#!/usr/bin/env python3
"""
P-TRACK  --  Bullish Technical Setup Screener
=============================================
Scans a universe of U.S.-listed stocks and surfaces only the highest-quality
bullish continuation setups (Flat Base, Bull Flag, Ascending Triangle,
Cup-and-Handle, VCP, High-Tight Flag) using a fully transparent, rule-based
methodology. Every number it prints is COMPUTED from real OHLCV data -- nothing
is guessed.

--------------------------------------------------------------------------
WHAT IT CHECKS  (all "hard" filters must pass for a stock to be eligible)
--------------------------------------------------------------------------
TREND
  * Price > 50-day SMA
  * 50-day SMA > 200-day SMA
  * Price within 10% of 52-week high
  * 3-month return beats SPY (positive relative strength)
STRUCTURE
  * Tight consolidation >= 10 trading days
  * Consolidation range <= 15% (high-to-low)
  * Multiple touches of a defined resistance / breakout level
  * No close that breaks decisively below consolidation support
VOLUME
  * Average daily volume >= 500,000 shares
  * Volume contracts inside the consolidation vs. the prior leg
  * Accumulation: more up-volume than down-volume in the base
EXCLUSIONS
  * Below 50-day SMA, lower highs, declining relative strength,
    < 500k ADV, erratic/over-volatile action, no clean breakout level.

For every passing stock it reports ticker, name, price, pattern, breakout
level, support, distance-to-breakout, 20/50/200 SMAs, 52-wk high, ADV,
volume-trend read, pattern duration, breakout R/R, stop, measured-move
target, and a 1-100 setup score, then ranks the top 25 with a written
rationale for each rank.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
  pip install yfinance pandas numpy
  python ptrack_screener.py --universe sp500 --top 25
  python ptrack_screener.py --universe all   --top 25      # ~7000 names, slow
  python ptrack_screener.py --universe my_tickers.txt       # one ticker/line
  python ptrack_screener.py --universe sp500 --csv out.csv --report report.txt

  # NEW: self-contained HTML report with a chart per setup
  python ptrack_screener.py --universe sp500 --html setups.html

  # NEW: sector-balanced ranking (no more than N picks per GICS sector)
  python ptrack_screener.py --universe sp1500 --max-per-sector 3

  # NEW: historical backtest of the setup edge (win rate, expectancy, etc.)
  python ptrack_screener.py --universe sp500 --backtest --backtest-step 5

NOTE: yfinance pulls free Yahoo Finance data. For 'all', expect a long run and
occasional throttling; re-run if some tickers fail to download.
"""

import argparse
import sys
import io
import time
import math
import datetime as dt

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# macOS SSL fix: the python.org build doesn't trust web certs until you run
# "Install Certificates.command". Point Python at the certifi CA bundle that
# ships with pandas/yfinance so urllib + requests verify correctly. Harmless
# on systems that are already fine.
# --------------------------------------------------------------------------
try:
    import os as _os, ssl as _ssl, certifi as _certifi
    _ca = _certifi.where()
    _os.environ.setdefault("SSL_CERT_FILE", _ca)
    _os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca)
    _ssl._create_default_https_context = (
        lambda *a, **k: _ssl.create_default_context(cafile=_ca))
except Exception:
    pass

# --------------------------------------------------------------------------
# CONFIG -- every threshold from the spec lives here so it's easy to tune.
# --------------------------------------------------------------------------
CFG = dict(
    # Trend
    near_high_pct          = 0.10,   # price must be within 10% of 52wk high
    rs_lookback_days       = 63,     # ~3 months for relative strength
    # Structure / consolidation
    min_base_days          = 10,     # tight consolidation >= 10 trading days
    max_base_days          = 65,     # cap so we don't call a 1-yr drift a "base"
    max_base_range_pct     = 0.15,   # consolidation range <= 15%
    min_resistance_touches = 2,      # multiple touches of resistance
    breakdown_buffer       = 0.02,   # close > support*(1-2%) = "no breakdown"
    resistance_zone_pct    = 0.03,   # within 3% of base high counts as a "touch"
    # Volume
    min_adv                = 500_000,
    adv_window             = 50,
    vol_contraction_max    = 0.90,   # base avg vol must be <= 90% of prior-leg avg
    # Volatility sanity (exclude erratic action)
    max_atr_pct            = 0.07,   # 14-day ATR > 7% of price = too erratic
    # Entry / risk model
    breakout_entry_buffer  = 0.005,  # assume fill 0.5% above breakout level
    stop_buffer            = 0.03,   # stop set 3% below chosen support
    min_history_days       = 260,    # need ~1y+ of data
    # Sector balancing
    max_per_sector         = 4,      # default cap of picks per GICS sector
    # Backtest model
    bt_entry_window        = 20,     # bars after detection to allow a breakout trigger
    bt_hold_days           = 40,     # max bars to hold a triggered trade
    bt_step                = 5,      # evaluate the tape every N bars (speed/coverage)
    bt_min_history         = 260,    # bars required before first evaluation
    # --- PROFITABILITY LEVERS (all OFF by default = original spec) ----------
    allowed_patterns       = None,   # None = trade all; or a set() of names to keep
    vcp_score_bonus        = 0.0,    # add this to the score for VCP setups
    require_market_uptrend = False,  # only take setups when SPY > its own 200d SMA
    min_score              = 0.0,    # screen/backtest: ignore setups below this score
    bt_slippage_pct        = 0.0,    # extra % paid above entry to model slippage
    bt_cost_R              = 0.0,    # fixed round-trip cost per trade, in R
    bt_breakeven_after_1R  = False,  # once +1R, move stop to entry (lock breakeven)
)

# Canonical pattern names (use these in allowed_patterns)
PATTERNS = ["Flat Base", "Bull Flag", "Ascending Triangle", "Cup and Handle",
            "Volatility Contraction Pattern (VCP)", "High-Tight Flag"]

# --------------------------------------------------------------------------
# UNIVERSE BUILDERS
# --------------------------------------------------------------------------
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

def _read_html(url):
    """Fetch a page with a browser-like User-Agent (Wikipedia 403s bare urllib)
    and hand the HTML to pandas. Falls back to a stooq mirror if Wikipedia is
    unreachable for the S&P 500 list."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    return pd.read_html(io.StringIO(html))

def get_sp500():
    try:
        t = _read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return [str(s).replace(".", "-").strip() for s in t["Symbol"].tolist()]
    except Exception:
        # fallback mirror that lists S&P 500 components
        t = _read_html("https://www.slickcharts.com/sp500")[0]
        col = "Symbol" if "Symbol" in t.columns else t.columns[2]
        return [str(s).replace(".", "-").strip() for s in t[col].tolist()]

def get_nasdaq100():
    tables = _read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    for tb in tables:
        cols = [c.lower() for c in tb.columns.astype(str)]
        if any("ticker" in c or "symbol" in c for c in cols):
            col = [c for c in tb.columns if str(c).lower() in ("ticker", "symbol")][0]
            return [str(s).replace(".", "-").strip() for s in tb[col].tolist()]
    raise RuntimeError("Could not parse Nasdaq-100 constituents")

def get_sp1500():
    syms = set(get_sp500())
    for url in ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"):
        try:
            for tb in _read_html(url):
                for c in tb.columns:
                    if str(c).lower() in ("symbol", "ticker"):
                        syms.update(str(s).replace(".", "-").strip() for s in tb[c].tolist())
        except Exception:
            pass
    return sorted(syms)

def get_all_us():
    """Full U.S. listed universe from the Nasdaq Trader symbol directory."""
    import urllib.request
    out = set()
    for f in ("nasdaqlisted.txt", "otherlisted.txt"):
        try:
            req = urllib.request.Request(
                f"https://www.nasdaqtrader.com/dynamic/SymDir/{f}",
                headers={"User-Agent": _UA})
            raw = urllib.request.urlopen(req, timeout=30).read().decode("latin-1")
            df = pd.read_csv(io.StringIO(raw), sep="|")
            col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
            for s in df[col].astype(str):
                s = s.strip()
                if s and "$" not in s and "." not in s and s.isalpha():
                    out.add(s)
        except Exception as e:
            print(f"  ! could not load {f}: {e}", file=sys.stderr)
    return sorted(out)

def load_universe(spec):
    if spec == "sp500":       return get_sp500()
    if spec == "nasdaq100":   return get_nasdaq100()
    if spec == "sp1500":      return get_sp1500()
    if spec == "all":         return get_all_us()
    # else: treat as a file path, one ticker per line
    with open(spec) as fh:
        return [ln.strip().upper() for ln in fh if ln.strip() and not ln.startswith("#")]

# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def download(tickers, period="2y"):
    import yfinance as yf
    data = yf.download(tickers, period=period, group_by="ticker",
                       auto_adjust=False, threads=True, progress=True)
    return data

def get_one(data, tkr, single):
    if single:
        df = data.copy()
    else:
        if tkr not in data.columns.get_level_values(0):
            return None
        df = data[tkr].copy()
    df = df.dropna(subset=["Close"])
    return df if len(df) >= CFG["min_history_days"] else None

# --------------------------------------------------------------------------
# INDICATORS
# --------------------------------------------------------------------------
def atr_pct(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float((tr.rolling(n).mean().iloc[-1]) / c.iloc[-1])

def enrich(df):
    df = df.copy()
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    return df

# --------------------------------------------------------------------------
# CONSOLIDATION / BASE DETECTION
# --------------------------------------------------------------------------
def find_base(df):
    """
    Walk back from the most recent bar to find the longest valid tight
    consolidation (between min_base_days and max_base_days) whose high-to-low
    range is <= max_base_range_pct and that has not closed decisively below
    its own support. Returns a dict of base metrics or None.
    """
    close = df["Close"].values
    high  = df["High"].values
    low   = df["Low"].values
    vol   = df["Volume"].values
    n = len(df)

    best = None
    for length in range(CFG["max_base_days"], CFG["min_base_days"] - 1, -1):
        if length >= n - 5:
            continue
        seg_hi = df["High"].iloc[-length:]
        seg_lo = df["Low"].iloc[-length:]
        seg_cl = df["Close"].iloc[-length:]
        seg_vl = df["Volume"].iloc[-length:]

        base_high = float(seg_hi.max())
        base_low  = float(seg_lo.min())
        if base_high <= 0:
            continue
        rng = (base_high - base_low) / base_high
        if rng > CFG["max_base_range_pct"]:
            continue

        # No decisive breakdown: no CLOSE below support*(1-buffer)
        support = base_low
        if (seg_cl < support * (1 - CFG["breakdown_buffer"])).sum() > 0:
            continue

        # Resistance touches near the base high
        zone = base_high * (1 - CFG["resistance_zone_pct"])
        touches = int((seg_hi >= zone).sum())
        if touches < CFG["min_resistance_touches"]:
            continue

        # Volume contraction vs the prior leg of equal length
        prior_vol = df["Volume"].iloc[-2 * length:-length]
        base_vol  = seg_vl
        if len(prior_vol) < length // 2:
            continue
        contraction = float(base_vol.mean()) / float(prior_vol.mean() + 1e-9)

        # Accumulation: up-volume vs down-volume inside the base
        ch = seg_cl.diff()
        up_vol   = float(seg_vl[ch > 0].sum())
        down_vol = float(seg_vl[ch < 0].sum())
        accumulation = up_vol / (down_vol + 1e-9)

        best = dict(
            length=length, base_high=base_high, base_low=base_low,
            range_pct=rng, support=support, touches=touches,
            vol_contraction=contraction, accumulation=accumulation,
            base_vol=float(base_vol.mean()), prior_vol=float(prior_vol.mean()),
        )
        break  # longest valid base wins
    return best

# --------------------------------------------------------------------------
# PATTERN CLASSIFICATION  (heuristic, transparent)
# --------------------------------------------------------------------------
def classify_pattern(df, base):
    close = df["Close"]
    seg = df.iloc[-base["length"]:]
    hi, lo, cl = seg["High"], seg["Low"], seg["Close"]
    x = np.arange(len(seg))

    # Prior run-up ("flagpole") = % gain in the ~30 bars before the base
    pole_lb = min(30, len(df) - base["length"] - 1)
    pole = 0.0
    if pole_lb > 5:
        p0 = float(close.iloc[-base["length"] - pole_lb])
        p1 = float(close.iloc[-base["length"]])
        pole = (p1 - p0) / p0 if p0 > 0 else 0.0

    # Slope of the lows (rising lows -> ascending triangle / flag)
    low_slope = np.polyfit(x, lo.values, 1)[0] / (float(lo.mean()) + 1e-9)
    high_slope = np.polyfit(x, hi.values, 1)[0] / (float(hi.mean()) + 1e-9)
    rng = base["range_pct"]
    length = base["length"]

    # VCP: progressively tightening range across thirds of the base
    third = max(1, length // 3)
    r1 = (seg["High"][:third].max() - seg["Low"][:third].min())
    r2 = (seg["High"][third:2*third].max() - seg["Low"][third:2*third].min())
    r3 = (seg["High"][-third:].max() - seg["Low"][-third:].min())
    vcp = (r1 > r2 > r3) and (r3 > 0)

    # Cup-and-handle: U-shape (mid of base notably lower than both ends),
    # then a small handle drift near the end.
    mid = seg["Close"].iloc[length//3:2*length//3].min()
    ends = (float(seg["Close"].iloc[0]) + float(seg["Close"].iloc[-1])) / 2
    cup = mid < ends * 0.95 and length >= 25

    scores = {}
    scores["High-Tight Flag"]       = (pole >= 0.50 and rng <= 0.10 and length <= 25)
    scores["Bull Flag"]             = (pole >= 0.15 and rng <= 0.12 and high_slope <= 0.001 and length <= 25)
    scores["Volatility Contraction Pattern (VCP)"] = vcp
    scores["Ascending Triangle"]    = (low_slope > 0.0008 and abs(high_slope) < 0.0008)
    scores["Cup and Handle"]        = cup
    scores["Flat Base"]             = (rng <= 0.12 and abs(high_slope) < 0.0008 and abs(low_slope) < 0.0010)

    # priority order (most specific / strongest first)
    for name in ["High-Tight Flag", "Volatility Contraction Pattern (VCP)",
                 "Cup and Handle", "Ascending Triangle", "Bull Flag", "Flat Base"]:
        if scores.get(name):
            return name, pole
    return "Flat Base", pole  # default if it cleared the base filter but matched none strongly

# --------------------------------------------------------------------------
# SCORING
# --------------------------------------------------------------------------
def score_setup(m):
    """Weighted 0-100 score from normalized sub-components (all already passed
    hard filters; this ranks QUALITY among survivors)."""
    s = 0.0
    parts = {}

    # Trend strength (20): how far price sits above the 200d, MA stack health
    above200 = (m["price"] / m["sma200"] - 1) if m["sma200"] else 0
    t = np.clip(above200 / 0.40, 0, 1) * 12        # up to 12 for being well above LT trend
    t += np.clip((m["sma50"]/m["sma200"]-1)/0.15, 0, 1) * 8  # MA stack spread up to 8
    parts["trend"] = t

    # Proximity to 52wk high (10)
    p = np.clip(1 - (m["dist_to_high"] / CFG["near_high_pct"]), 0, 1) * 10
    parts["near_high"] = p

    # Relative strength vs SPY 3mo (15)
    r = np.clip(m["rs_3m"] / 0.20, 0, 1) * 15
    parts["rel_strength"] = r

    # Consolidation tightness (15): tighter is better
    c = np.clip(1 - (m["range_pct"] / CFG["max_base_range_pct"]), 0, 1) * 15
    parts["tightness"] = c

    # Volume contraction (10): lower base vol vs prior leg is better
    v = np.clip((1 - m["vol_contraction"]) / 0.5, 0, 1) * 10
    parts["vol_contraction"] = v

    # Accumulation (10): up/down volume ratio
    a = np.clip((m["accumulation"] - 1) / 1.0, 0, 1) * 10
    parts["accumulation"] = a

    # Pattern duration adequacy (5): reward a real, mature base
    d = np.clip((m["length"] - CFG["min_base_days"]) / 20, 0, 1) * 5
    parts["duration"] = d

    # Risk/reward at breakout (10)
    rr = np.clip(m["rr"] / 3.0, 0, 1) * 10
    parts["risk_reward"] = rr

    # Clean structure / resistance touches (5)
    st = np.clip((m["touches"] - 2) / 3, 0, 1) * 5
    parts["structure"] = st

    # Optional bonus for the historically strongest pattern (VCP)
    if m.get("pattern") == "Volatility Contraction Pattern (VCP)" and CFG["vcp_score_bonus"]:
        parts["vcp_bonus"] = float(CFG["vcp_score_bonus"])

    s = sum(parts.values())
    return round(min(100, s), 1), parts

# --------------------------------------------------------------------------
# CORE EVALUATION OF ONE TICKER
# --------------------------------------------------------------------------
def evaluate(df, spy, tkr, name):
    df = enrich(df)
    last = df.iloc[-1]
    price = float(last["Close"])
    sma20, sma50, sma200 = float(last["SMA20"]), float(last["SMA50"]), float(last["SMA200"])
    if any(map(lambda v: v != v, [sma20, sma50, sma200])):
        return None, "insufficient MA history"

    # ---- MARKET-REGIME FILTER (optional) ----
    if CFG["require_market_uptrend"]:
        spy_sma200 = float(spy["Close"].rolling(200).mean().iloc[-1])
        if spy_sma200 == spy_sma200 and float(spy["Close"].iloc[-1]) <= spy_sma200:
            return None, "market below its 200d (regime filter)"

    # ---- HARD TREND FILTERS ----
    if price <= sma50:                       return None, "below 50d SMA"
    if sma50 <= sma200:                      return None, "50d not above 200d"
    high_52w = float(df["High"].iloc[-252:].max())
    dist_to_high = (high_52w - price) / high_52w
    if dist_to_high > CFG["near_high_pct"]:  return None, "more than 10% off 52wk high"

    # relative strength vs SPY over ~3 months
    lb = CFG["rs_lookback_days"]
    stock_ret = price / float(df["Close"].iloc[-lb]) - 1
    spy_ret   = float(spy["Close"].iloc[-1]) / float(spy["Close"].iloc[-lb]) - 1
    rs_3m = stock_ret - spy_ret
    if rs_3m <= 0:                           return None, "not outperforming SPY (3m)"

    # lower-highs / declining RS exclusion: 1-month RS line must not be falling
    rs_line = (df["Close"] / spy["Close"].reindex(df.index).ffill())
    if float(rs_line.iloc[-1]) <= float(rs_line.iloc[-21]):
        return None, "relative strength rolling over"

    # ---- VOLUME FILTER ----
    adv = float(df["Volume"].iloc[-CFG["adv_window"]:].mean())
    if adv < CFG["min_adv"]:                 return None, "ADV below 500k"

    # ---- VOLATILITY SANITY ----
    if atr_pct(df) > CFG["max_atr_pct"]:     return None, "erratic / too volatile (ATR%)"

    # ---- STRUCTURE / BASE ----
    base = find_base(df)
    if base is None:                         return None, "no valid tight base / breakout level"
    if base["vol_contraction"] > CFG["vol_contraction_max"]:
        return None, "volume not contracting in base"
    if base["accumulation"] < 1.0:           return None, "distribution (down-vol > up-vol) in base"

    pattern, pole = classify_pattern(df, base)

    # ---- PATTERN ALLOW-LIST (optional) ----
    if CFG["allowed_patterns"] and pattern not in CFG["allowed_patterns"]:
        return None, f"pattern '{pattern}' excluded by filter"

    # ---- ENTRY / RISK MODEL ----
    breakout = base["base_high"]
    entry    = breakout * (1 + CFG["breakout_entry_buffer"])
    support  = base["support"]
    stop     = support * (1 - CFG["stop_buffer"])
    base_depth = base["base_high"] - base["base_low"]
    # measured move: flagpole for flags, base depth otherwise
    if pattern in ("Bull Flag", "High-Tight Flag"):
        target = breakout + max(pole, 0) * df["Close"].iloc[-base["length"]]
        if target <= breakout:                # fallback
            target = breakout + base_depth
    else:
        target = breakout + base_depth
    risk = entry - stop
    reward = target - entry
    rr = reward / risk if risk > 0 else 0
    dist_to_breakout = (breakout - price) / price

    m = dict(
        ticker=tkr, name=name, price=price,
        sma20=sma20, sma50=sma50, sma200=sma200,
        high_52w=high_52w, dist_to_high=dist_to_high,
        rs_3m=rs_3m, adv=adv,
        length=base["length"], range_pct=base["range_pct"],
        touches=base["touches"], vol_contraction=base["vol_contraction"],
        accumulation=base["accumulation"],
        pattern=pattern, breakout=breakout, support=support,
        stop=stop, target=target, rr=rr,
        dist_to_breakout=dist_to_breakout,
        base_vol=base["base_vol"], prior_vol=base["prior_vol"],
    )
    m["score"], m["parts"] = score_setup(m)

    # ---- MIN-SCORE SELECTIVITY (optional) ----
    if CFG["min_score"] and m["score"] < CFG["min_score"]:
        return None, f"score {m['score']} below min {CFG['min_score']}"

    return m, "ok"

# --------------------------------------------------------------------------
# REPORTING
# --------------------------------------------------------------------------
def vol_trend_text(m):
    contr = (1 - m["vol_contraction"]) * 100
    return (f"base avg vol {m['base_vol']:,.0f} vs prior leg {m['prior_vol']:,.0f} "
            f"({contr:+.0f}% contraction); up/down vol ratio "
            f"{m['accumulation']:.2f} ({'accumulation' if m['accumulation']>=1.2 else 'mild accumulation'})")

def rank_reason(m, rank):
    p = m["parts"]
    drivers = sorted(p.items(), key=lambda kv: kv[1], reverse=True)
    top = ", ".join(f"{k} ({v:.1f})" for k, v in drivers[:3])
    weak = ", ".join(f"{k} ({v:.1f})" for k, v in drivers[-2:])
    return (f"#{rank}  score {m['score']}/100. Strongest contributors: {top}. "
            f"Held back by: {weak}. RS +{m['rs_3m']*100:.1f}% vs SPY (3m), "
            f"{m['dist_to_high']*100:.1f}% off 52wk high, base {m['length']}d / "
            f"{m['range_pct']*100:.1f}% wide, breakout R/R {m['rr']:.2f}.")

def to_row(m):
    return dict(
        Ticker=m["ticker"], Company=m["name"], Price=round(m["price"],2),
        Pattern=m["pattern"], Breakout=round(m["breakout"],2),
        Support=round(m["support"],2),
        DistToBreakout_pct=round(m["dist_to_breakout"]*100,2),
        SMA20=round(m["sma20"],2), SMA50=round(m["sma50"],2), SMA200=round(m["sma200"],2),
        High52w=round(m["high_52w"],2), ADV=int(m["adv"]),
        BaseDays=m["length"], BaseRange_pct=round(m["range_pct"]*100,2),
        Stop=round(m["stop"],2), Target=round(m["target"],2),
        RiskReward=round(m["rr"],2), Score=m["score"],
        VolumeTrend=vol_trend_text(m),
    )

# --------------------------------------------------------------------------
# SECTOR BALANCING
# --------------------------------------------------------------------------
_SECTOR_CACHE = {}

def fetch_sector(tkr):
    """Best-effort GICS sector via yfinance. Cached. Falls back to 'Unknown'."""
    if tkr in _SECTOR_CACHE:
        return _SECTOR_CACHE[tkr]
    sec = "Unknown"
    try:
        import yfinance as yf
        info = yf.Ticker(tkr).get_info()
        sec = info.get("sector") or "Unknown"
    except Exception:
        sec = "Unknown"
    _SECTOR_CACHE[tkr] = sec
    return sec

def sector_balanced_select(ranked, top_n, max_per_sector):
    """Take the highest-scoring setups subject to a per-sector cap. If the cap
    leaves us short of top_n, backfill with the next-best regardless of sector."""
    chosen, chosen_ids, counts = [], set(), {}
    for m in ranked:
        sec = m.get("sector", "Unknown")
        if counts.get(sec, 0) < max_per_sector:
            chosen.append(m); chosen_ids.add(id(m))
            counts[sec] = counts.get(sec, 0) + 1
        if len(chosen) >= top_n:
            return chosen
    for m in ranked:                       # backfill if caps left us short
        if id(m) not in chosen_ids:
            chosen.append(m)
            if len(chosen) >= top_n:
                break
    return chosen

# --------------------------------------------------------------------------
# SVG CHART  (no external libraries -- self-contained)
# --------------------------------------------------------------------------
def chart_svg(df, m, bars=170, W=900, Hp=300, Hv=90):
    """Render a price chart (close + 20/50/200 SMA), the shaded consolidation
    base, and breakout/support/stop/target reference lines, plus a volume
    sub-panel -- all as a standalone <svg> string."""
    d = df.iloc[-bars:].copy()
    n = len(d)
    pad_l, pad_r, pad_t, pad_b = 58, 92, 10, 18
    plot_w = W - pad_l - pad_r
    price_vals = pd.concat([d["High"], d["Low"], d["SMA200"].fillna(d["Low"])]).values
    refs = [m["breakout"], m["support"], m["stop"], m["target"]]
    pmin = float(np.nanmin([np.nanmin(price_vals)] + refs))
    pmax = float(np.nanmax([np.nanmax(price_vals)] + refs))
    span = (pmax - pmin) or 1.0
    pmin -= span * 0.05; pmax += span * 0.05; span = pmax - pmin

    def X(i):  return pad_l + (i / max(n - 1, 1)) * plot_w
    def Yp(p): return pad_t + (pmax - p) / span * Hp

    def polyline(series, color, wdt=1.5, dash=""):
        pts = []
        for i, v in enumerate(series):
            if v == v:  # not NaN
                pts.append(f"{X(i):.1f},{Yp(float(v)):.1f}")
        if not pts:
            return ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="{wdt}"{da} points="{" ".join(pts)}"/>'

    svg = [f'<svg viewBox="0 0 {W} {Hp+Hv+pad_t+pad_b+24}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="Arial,Helvetica,sans-serif" font-size="14">']
    svg.append(f'<rect x="0" y="0" width="{W}" height="{Hp+Hv+pad_t+pad_b+24}" fill="#ffffff"/>')
    # price gridlines + labels
    for k in range(5):
        p = pmin + span * k / 4
        y = Yp(p)
        svg.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+plot_w}" y2="{y:.1f}" stroke="#eef1f5"/>')
        svg.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" fill="#888">{p:.2f}</text>')

    # shaded consolidation base over the last m['length'] bars
    base_len = min(m["length"], n - 1)
    bx0 = X(n - base_len - 1); bx1 = X(n - 1)
    by0 = Yp(m["breakout"]); by1 = Yp(m["support"])
    svg.append(f'<rect x="{bx0:.1f}" y="{by0:.1f}" width="{(bx1-bx0):.1f}" height="{(by1-by0):.1f}" '
               f'fill="#f4d35e" fill-opacity="0.18" stroke="#e0b500" stroke-dasharray="3 3"/>')

    # reference lines
    for p, color, label in [(m["breakout"], "#1a9850", "Breakout"),
                            (m["support"], "#888888", "Support"),
                            (m["stop"], "#d73027", "Stop"),
                            (m["target"], "#3b6fb6", "Target")]:
        y = Yp(p)
        svg.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+plot_w}" y2="{y:.1f}" '
                   f'stroke="{color}" stroke-width="1.2" stroke-dasharray="6 4"/>')
        svg.append(f'<text x="{pad_l+plot_w+4}" y="{y+3:.1f}" fill="{color}">{label} {p:.2f}</text>')

    # SMAs + close
    svg.append(polyline(d["SMA200"].tolist(), "#b07aa1", 1.3))
    svg.append(polyline(d["SMA50"].tolist(),  "#f28e2b", 1.3))
    svg.append(polyline(d["SMA20"].tolist(),  "#76b7b2", 1.3))
    svg.append(polyline(d["Close"].tolist(),  "#222222", 1.8))

    # volume panel
    vtop = pad_t + Hp + 14
    vmax = float(d["Volume"].max()) or 1.0
    bw = plot_w / max(n, 1) * 0.8
    chg = d["Close"].diff().fillna(0).values
    for i, v in enumerate(d["Volume"].values):
        h = (v / vmax) * Hv
        col = "#9bd0a0" if chg[i] >= 0 else "#e3a6a1"
        svg.append(f'<rect x="{X(i)-bw/2:.1f}" y="{vtop+Hv-h:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{col}"/>')
    svg.append(f'<text x="{pad_l-6}" y="{vtop+10:.1f}" text-anchor="end" fill="#888">Vol</text>')

    # legend
    lg = [("Close","#222222"),("20d","#76b7b2"),("50d","#f28e2b"),("200d","#b07aa1")]
    lx = pad_l
    ly = Hp+Hv+pad_t+pad_b+18
    for lab,col in lg:
        svg.append(f'<line x1="{lx}" y1="{ly-3}" x2="{lx+16}" y2="{ly-3}" stroke="{col}" stroke-width="2"/>')
        svg.append(f'<text x="{lx+20}" y="{ly}" fill="#555">{lab}</text>')
        lx += 70
    svg.append("</svg>")
    return "".join(svg)

def build_html(top, universe, path):
    css = """
    body{font-family:Arial,Helvetica,sans-serif;margin:24px;color:#1d2733;background:#fafbfc}
    h1{margin:0 0 4px}.sub{color:#667;margin-bottom:18px}
    .card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:22px;
          box-shadow:0 1px 3px rgba(0,0,0,.05)}
    .hd{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap}
    .tk{font-size:20px;font-weight:700}.pat{color:#3b6fb6;font-weight:600}
    .score{font-size:20px;font-weight:700;color:#1a9850}
    table.m{border-collapse:collapse;margin-top:8px;font-size:12.5px}
    table.m td{padding:2px 14px 2px 0;color:#33404d}
    table.m td b{color:#1d2733}
    .why{margin-top:8px;font-size:12.5px;color:#445;background:#f4f7fb;border-left:3px solid #3b6fb6;padding:8px 10px;border-radius:4px}
    .disc{font-size:11.5px;color:#7a3b00;background:#fff7ec;border:1px solid #e0a96d;border-radius:8px;padding:10px 12px;margin-bottom:18px}
    .sumtbl{border-collapse:collapse;width:100%;font-size:12.5px;margin-bottom:24px;background:#fff}
    .sumtbl th{background:#1a3c6e;color:#fff;padding:6px 8px;text-align:left}
    .sumtbl td{padding:5px 8px;border-bottom:1px solid #eef1f5}
    """
    rows = []
    for i, m in enumerate(top, 1):
        rows.append(f"<tr><td>{i}</td><td><b>{m['ticker']}</b></td><td>{m.get('sector','')}</td>"
                    f"<td>{m['pattern']}</td><td>${m['price']:.2f}</td><td>{m['breakout']:.2f}</td>"
                    f"<td>{m['target']:.2f}</td><td>{m['rr']:.2f}</td><td><b>{m['score']}</b></td></tr>")
    summary = ("<table class='sumtbl'><tr><th>#</th><th>Ticker</th><th>Sector</th><th>Pattern</th>"
               "<th>Price</th><th>Breakout</th><th>Target</th><th>R/R</th><th>Score</th></tr>"
               + "".join(rows) + "</table>")

    cards = []
    for i, m in enumerate(top, 1):
        svg = chart_svg(m["_df"], m)
        mt = (f"<table class='m'><tr>"
              f"<td><b>Price</b> ${m['price']:.2f}</td><td><b>Breakout</b> {m['breakout']:.2f}</td>"
              f"<td><b>Support</b> {m['support']:.2f}</td><td><b>Stop</b> {m['stop']:.2f}</td>"
              f"<td><b>Target</b> {m['target']:.2f}</td><td><b>R/R</b> {m['rr']:.2f}</td></tr><tr>"
              f"<td><b>SMA20</b> {m['sma20']:.2f}</td><td><b>SMA50</b> {m['sma50']:.2f}</td>"
              f"<td><b>SMA200</b> {m['sma200']:.2f}</td><td><b>52w hi</b> {m['high_52w']:.2f}</td>"
              f"<td><b>ADV</b> {m['adv']:,.0f}</td><td><b>To breakout</b> {m['dist_to_breakout']*100:+.2f}%</td></tr><tr>"
              f"<td><b>Base</b> {m['length']}d</td><td><b>Range</b> {m['range_pct']*100:.1f}%</td>"
              f"<td><b>Touches</b> {m['touches']}</td><td colspan='3'><b>Sector</b> {m.get('sector','')}</td></tr></table>")
        cards.append(
            f"<div class='card'><div class='hd'><div><span class='tk'>#{i} {m['ticker']}</span> "
            f"&nbsp;<span class='pat'>{m['pattern']}</span></div><div class='score'>{m['score']}/100</div></div>"
            f"{svg}{mt}<div class='why'>{rank_reason(m, i)}<br>Volume: {vol_trend_text(m)}</div></div>")

    disc = ("<div class='disc'><b>Not financial advice.</b> Rule-based pattern detection on historical "
            "data. Breakouts fail and measured-move targets are geometry, not forecasts. Verify on your "
            "own charts and manage risk.</div>")
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>P-TRACK setups</title>"
            f"<style>{css}</style></head><body>"
            f"<h1>P-TRACK — Top Bullish Setups</h1>"
            f"<div class='sub'>universe={universe} &nbsp;|&nbsp; generated {dt.datetime.now():%Y-%m-%d %H:%M} "
            f"&nbsp;|&nbsp; {len(top)} setups</div>{disc}{summary}{''.join(cards)}</body></html>")
    with open(path, "w") as fh:
        fh.write(html)

# --------------------------------------------------------------------------
# BACKTEST  -- measures the historical edge of the setup rules
# --------------------------------------------------------------------------
def backtest_ticker(df, spy):
    """Walk the tape; whenever a valid setup is detected, simulate entering on a
    breakout trigger and exiting at target/stop/time-stop. Returns a list of
    realized R-multiples and hold lengths."""
    trades = []
    n = len(df)
    t = CFG["bt_min_history"]
    ew, hold = CFG["bt_entry_window"], CFG["bt_hold_days"]
    while t < n - 2:
        sl = df.iloc[:t + 1]
        spy_sl = spy.loc[:sl.index[-1]]
        if len(spy_sl) < CFG["rs_lookback_days"] + 1:
            t += CFG["bt_step"]; continue
        m, why = evaluate(sl, spy_sl, "BT", "BT")
        if m is None or m["rr"] <= 0:
            t += CFG["bt_step"]; continue

        # entry includes the breakout buffer plus optional slippage
        entry = m["breakout"] * (1 + CFG["breakout_entry_buffer"] + CFG["bt_slippage_pct"])
        stop, target = m["stop"], m["target"]
        risk = entry - stop
        if risk <= 0:
            t += CFG["bt_step"]; continue

        # look for a breakout trigger within the entry window
        trig = None
        for k in range(t + 1, min(t + 1 + ew, n)):
            if float(df["High"].iloc[k]) >= entry:
                trig = k; break
        if trig is None:
            t += CFG["bt_step"]; continue

        # manage the trade forward
        cur_stop = stop
        r, exit_k = None, None
        for j in range(trig, min(trig + hold, n)):
            lo, hi = float(df["Low"].iloc[j]), float(df["High"].iloc[j])
            if lo <= cur_stop:                     # stop first (conservative)
                r = (cur_stop - entry) / risk; exit_k = j; break
            if hi >= target:
                r = (target - entry) / risk; exit_k = j; break
            # optional: once the trade is up +1R, lift the stop to breakeven
            if CFG["bt_breakeven_after_1R"] and hi >= entry + risk:
                cur_stop = max(cur_stop, entry)
        if r is None:                              # time stop at last close
            exit_k = min(trig + hold, n) - 1
            r = (float(df["Close"].iloc[exit_k]) - entry) / risk
        r -= CFG["bt_cost_R"]                       # subtract round-trip cost
        trades.append((r, exit_k - trig, m["pattern"], m["score"]))
        t = exit_k + CFG["bt_step"]                # resume after the trade closes
    return trades

def run_backtest(all_trades):
    rs = np.array([t[0] for t in all_trades], float)
    if len(rs) == 0:
        return "No trades generated — loosen filters or widen the date range."
    wins = rs[rs > 0]; losses = rs[rs <= 0]
    win_rate = len(wins) / len(rs)
    expectancy = rs.mean()
    pf = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    holds = np.array([t[1] for t in all_trades], float)
    # per-pattern breakdown
    pats = {}
    for tr in all_trades:
        pats.setdefault(tr[2], []).append(tr[0])
    # per-score-bucket breakdown (score is the 4th item when present)
    buckets = {"<50": [], "50-59": [], "60-69": [], "70-79": [], "80+": []}
    have_scores = all(len(tr) >= 4 for tr in all_trades)
    if have_scores:
        for tr in all_trades:
            sc = tr[3]
            key = ("80+" if sc >= 80 else "70-79" if sc >= 70 else
                   "60-69" if sc >= 60 else "50-59" if sc >= 50 else "<50")
            buckets[key].append(tr[0])

    # which optional levers were active (for the header)
    flags = []
    if CFG["allowed_patterns"]:        flags.append(f"patterns={sorted(CFG['allowed_patterns'])}")
    if CFG["require_market_uptrend"]:  flags.append("market>200d")
    if CFG["min_score"]:               flags.append(f"min_score={CFG['min_score']}")
    if CFG["bt_slippage_pct"]:         flags.append(f"slippage={CFG['bt_slippage_pct']*100:.2f}%")
    if CFG["bt_cost_R"]:               flags.append(f"cost={CFG['bt_cost_R']}R")
    if CFG["bt_breakeven_after_1R"]:   flags.append("breakeven@1R")

    lines = []
    lines.append("P-TRACK BACKTEST — historical edge of the breakout rules")
    lines.append("=" * 64)
    if flags:
        lines.append("Active levers       : " + "; ".join(flags))
    lines.append(f"Trades simulated     : {len(rs)}")
    lines.append(f"Win rate             : {win_rate*100:.1f}%")
    lines.append(f"Expectancy / trade   : {expectancy:+.2f} R")
    lines.append(f"Avg win / avg loss   : {wins.mean() if len(wins) else 0:+.2f}R / "
                 f"{losses.mean() if len(losses) else 0:+.2f}R")
    lines.append(f"Profit factor        : {pf:.2f}")
    lines.append(f"Median R             : {np.median(rs):+.2f}R")
    lines.append(f"Avg hold (bars)      : {holds.mean():.1f}")
    lines.append(f"Best / worst trade   : {rs.max():+.2f}R / {rs.min():+.2f}R")
    lines.append("")
    lines.append("By pattern:")
    for p, arr in sorted(pats.items(), key=lambda kv: -len(kv[1])):
        a = np.array(arr)
        lines.append(f"  {p:42s} n={len(a):4d}  win={100*np.mean(a>0):4.0f}%  exp={a.mean():+.2f}R")
    if have_scores:
        lines.append("")
        lines.append("By setup score:")
        for k in ["<50", "50-59", "60-69", "70-79", "80+"]:
            a = np.array(buckets[k])
            if len(a):
                lines.append(f"  {k:7s} n={len(a):4d}  win={100*np.mean(a>0):4.0f}%  exp={a.mean():+.2f}R")
    lines.append("")
    lines.append("Interpretation: expectancy > 0 and profit factor > 1 imply a positive")
    lines.append("historical edge for these setups. Results now reflect any active levers")
    lines.append("above. Past behavior does not guarantee future results.")
    return "\n".join(lines)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="P-TRACK bullish setup screener")
    ap.add_argument("--universe", default="sp500",
                    help="sp500 | nasdaq100 | sp1500 | all | <path to tickers.txt>")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--csv", default="ptrack_results.csv")
    ap.add_argument("--report", default="ptrack_report.txt")
    ap.add_argument("--html", default=None, help="write a chart-per-setup HTML report to this path")
    ap.add_argument("--max-per-sector", type=int, default=None,
                    help="cap picks per GICS sector (sector-balanced ranking)")
    ap.add_argument("--backtest", action="store_true",
                    help="run a historical backtest of the setup edge instead of a live scan")
    ap.add_argument("--backtest-step", type=int, default=None, help="evaluate the tape every N bars")
    ap.add_argument("--bt-report", default="ptrack_backtest.txt")
    ap.add_argument("--batch", type=int, default=200, help="download batch size")
    # --- profitability levers (optional) ---
    ap.add_argument("--patterns", default=None,
                    help="comma-separated patterns to trade, e.g. 'VCP,Flat Base,Cup and Handle'")
    ap.add_argument("--vcp-bonus", type=float, default=0.0, help="add to score for VCP setups")
    ap.add_argument("--market-filter", action="store_true",
                    help="only take setups when SPY is above its 200d SMA")
    ap.add_argument("--min-score", type=float, default=0.0, help="ignore setups below this score")
    ap.add_argument("--slippage", type=float, default=0.0,
                    help="backtest slippage as a fraction, e.g. 0.003 = 0.3%%")
    ap.add_argument("--cost-r", type=float, default=0.0, help="backtest round-trip cost in R")
    ap.add_argument("--breakeven", action="store_true",
                    help="backtest: move stop to breakeven once +1R")
    args = ap.parse_args()
    if args.backtest_step:
        CFG["bt_step"] = args.backtest_step
    # apply levers (accepts 'VCP' as a shorthand for the full VCP name)
    if args.patterns:
        names = []
        for p in args.patterns.split(","):
            p = p.strip()
            if p.upper() == "VCP":
                p = "Volatility Contraction Pattern (VCP)"
            names.append(p)
        CFG["allowed_patterns"] = set(names)
    CFG["vcp_score_bonus"]        = args.vcp_bonus
    CFG["require_market_uptrend"] = args.market_filter
    CFG["min_score"]              = args.min_score
    CFG["bt_slippage_pct"]        = args.slippage
    CFG["bt_cost_R"]              = args.cost_r
    CFG["bt_breakeven_after_1R"]  = args.breakeven

    print(f"[1/4] Building universe: {args.universe}")
    tickers = load_universe(args.universe)
    print(f"      {len(tickers)} symbols")

    print("[2/4] Downloading SPY benchmark + price history (this can take a while)")
    import yfinance as yf
    spy = yf.download("SPY", period="2y", auto_adjust=False, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # ---------------- BACKTEST MODE ----------------
    if args.backtest:
        print("[3/4] Backtesting setup rules across history (this is the slow path)")
        all_trades = []
        for i in range(0, len(tickers), args.batch):
            chunk = tickers[i:i+args.batch]
            print(f"      batch {i//args.batch+1}: {len(chunk)} tickers")
            data = download(chunk)
            single = len(chunk) == 1
            for tkr in chunk:
                try:
                    df = get_one(data, tkr, single)
                    if df is None:
                        continue
                    all_trades += backtest_ticker(enrich(df), spy)
                except Exception:
                    pass
            time.sleep(1)
        report = run_backtest(all_trades)
        with open(args.bt_report, "w") as fh:
            fh.write(report)
        print("[4/4] Done.")
        print(f"      Backtest -> {args.bt_report}\n")
        print(report)
        return

    # ---------------- LIVE SCAN ----------------
    results, reasons = [], {}
    for i in range(0, len(tickers), args.batch):
        chunk = tickers[i:i+args.batch]
        print(f"      batch {i//args.batch+1}: {len(chunk)} tickers")
        data = download(chunk)
        single = len(chunk) == 1
        for tkr in chunk:
            try:
                df = get_one(data, tkr, single)
                if df is None:
                    reasons[tkr] = "insufficient history/data"; continue
                m, why = evaluate(df, spy, tkr, tkr)
                if m is None:
                    reasons[tkr] = why
                else:
                    m["_df"] = enrich(df)          # keep enriched data for charts
                    results.append(m)
            except Exception as e:
                reasons[tkr] = f"error: {e}"
        time.sleep(1)  # be polite to the data source

    print(f"[3/4] {len(results)} stocks passed ALL hard filters")
    results.sort(key=lambda x: x["score"], reverse=True)

    # sector-balanced selection (optional)
    if args.max_per_sector is not None:
        cap = args.max_per_sector
        print(f"      Fetching sectors for {len(results)} qualifiers (cap {cap}/sector)")
        for m in results:
            m["sector"] = fetch_sector(m["ticker"])
        top = sector_balanced_select(results, args.top, cap)
    else:
        for m in results:                          # sector still shown if cheap to get
            m.setdefault("sector", "")
        top = results[:args.top]

    rows = [to_row(m) for m in top]
    out = pd.DataFrame(rows)
    out.insert(0, "Rank", range(1, len(out)+1))
    if any(m.get("sector") for m in top):
        out.insert(3, "Sector", [m.get("sector", "") for m in top])
    out.to_csv(args.csv, index=False)

    # text report with rationale
    lines = []
    lines.append("P-TRACK  --  TOP BULLISH SETUPS")
    lines.append(f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}  |  universe={args.universe}  "
                 f"|  {len(results)} qualifiers, showing top {len(top)}"
                 + (f"  |  <= {args.max_per_sector}/sector" if args.max_per_sector else ""))
    lines.append("=" * 78)
    for rank, m in enumerate(top, 1):
        sec = f"  [{m['sector']}]" if m.get("sector") else ""
        lines.append(f"\n{rank}. {m['ticker']}  ({m['pattern']})  ${m['price']:.2f}   score {m['score']}/100{sec}")
        lines.append(f"    Breakout {m['breakout']:.2f} | Support {m['support']:.2f} | "
                     f"Stop {m['stop']:.2f} | Target {m['target']:.2f} | R/R {m['rr']:.2f}")
        lines.append(f"    SMA20 {m['sma20']:.2f} | SMA50 {m['sma50']:.2f} | SMA200 {m['sma200']:.2f} | "
                     f"52wk-hi {m['high_52w']:.2f} | ADV {m['adv']:,.0f}")
        lines.append(f"    Base {m['length']}d, {m['range_pct']*100:.1f}% wide, {m['touches']} resistance touches | "
                     f"{m['dist_to_breakout']*100:+.2f}% to breakout")
        lines.append(f"    Volume: {vol_trend_text(m)}")
        lines.append(f"    WHY THIS RANK: {rank_reason(m, rank)}")
    report = "\n".join(lines)
    with open(args.report, "w") as fh:
        fh.write(report)

    if args.html:
        build_html(top, args.universe, args.html)

    print("[4/4] Done.")
    print(f"      CSV    -> {args.csv}")
    print(f"      Report -> {args.report}")
    if args.html:
        print(f"      HTML   -> {args.html}")
    print("\n" + report[:2000])

if __name__ == "__main__":
    main()
