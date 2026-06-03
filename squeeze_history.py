#!/usr/bin/env python3
"""
SQUEEZE HISTORY  --  the app's "memory" (Phase 1)
=================================================
The framework's earliest signals are RATE OF CHANGE — short interest rising
across cycles, borrow fee accelerating week-over-week, utilization climbing,
mention velocity. A stateless screener can't see any of that. This module logs
a daily snapshot per ticker and computes the change-signals from history.

Storage = a simple CSV next to this file (zero dependencies, human-readable,
one row per ticker per day, upserted). When you run the app/logger on a machine
where the file persists (your Mac, or a scheduled job), history accumulates and
the radar gets smarter over time.

NOTE on hosting: Streamlit Cloud's disk is ephemeral (wiped on reboot), so to
build durable history either (a) run the daily logger locally, or (b) upgrade
the STORE to a cloud DB later (Supabase/Google Sheets) — this module is written
so only save()/load() need to change for that.
"""
import os, datetime as dt
import pandas as pd

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "squeeze_history.csv")

# columns we track over time
FIELDS = ["date", "ticker", "score", "tier", "short_pct", "ctb", "util",
          "days_cover", "mentions", "rvol", "price", "near_high"]

def _normalize_si(c):
    sp = c.get("short_pct")
    if sp is None:
        return None
    return sp * 100 if sp <= 1.5 else sp        # store SI as a 0-100 percent

def _row(c, today):
    return {
        "date": today, "ticker": c.get("ticker"),
        "score": c.get("score"), "tier": c.get("tier"),
        "short_pct": _normalize_si(c), "ctb": c.get("ctb"), "util": c.get("util"),
        "days_cover": c.get("days_cover"), "mentions": c.get("soc_mentions"),
        "rvol": c.get("rvol"), "price": c.get("price"), "near_high": c.get("near_high"),
    }

# ---- storage primitives (swap these two for a cloud DB later) -------------
def load(path=DEFAULT_PATH):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=FIELDS)
    return pd.DataFrame(columns=FIELDS)

def save(df, path=DEFAULT_PATH):
    df.to_csv(path, index=False)

# ---- public API -----------------------------------------------------------
def record(candidates, path=DEFAULT_PATH, today=None):
    """Append/replace today's snapshot for each candidate (upsert per ticker/day)."""
    today = today or dt.date.today().isoformat()
    df = load(path)
    for col in FIELDS:
        if col not in df.columns:
            df[col] = None
    new_rows = [_row(c, today) for c in candidates if c.get("ticker")]
    if not new_rows:
        return df
    add = pd.DataFrame(new_rows)
    # drop any existing rows for the same ticker+date, then append fresh
    if len(df):
        mask = ~((df["date"] == today) & (df["ticker"].isin(add["ticker"])))
        kept = df[mask]
        df = add if kept.empty else pd.concat([kept, add], ignore_index=True)
    else:
        df = add
    save(df, path)
    return df

def changes(ticker, path=DEFAULT_PATH, df=None):
    """Compute rate-of-change signals for a ticker from its prior snapshots.
    Returns (flags:list[str], deltas:dict)."""
    if df is None:
        df = load(path)
    h = df[df["ticker"] == ticker].copy()
    if len(h) < 2:
        return [], {}
    h = h.sort_values("date")
    # last two DISTINCT days
    days = list(dict.fromkeys(h["date"].tolist()))
    if len(days) < 2:
        return [], {}
    prev = h[h["date"] == days[-2]].iloc[-1]
    cur  = h[h["date"] == days[-1]].iloc[-1]
    flags, d = [], {}

    def num(x):
        try: return float(x)
        except (TypeError, ValueError): return None

    # short interest change
    si0, si1 = num(prev["short_pct"]), num(cur["short_pct"])
    if si0 is not None and si1 is not None:
        d["si_change"] = si1 - si0
        if si1 - si0 >= 1.0:
            flags.append(f"📈 Short interest rising: {si0:.0f}% → {si1:.0f}% of float")
        # rising streak across all snapshots
        sis = [num(x) for x in h["short_pct"].tolist() if num(x) is not None]
        streak = 0
        for i in range(len(sis) - 1, 0, -1):
            if sis[i] > sis[i-1]: streak += 1
            else: break
        if streak >= 2:
            flags.append(f"📈 Short interest up {streak} readings in a row (pressure building)")
            d["si_streak"] = streak

    # borrow fee acceleration (the #1 leading tell, when CTB data is present)
    c0, c1 = num(prev["ctb"]), num(cur["ctb"])
    if c0 is not None and c1 is not None and c1 - c0 >= 5:
        d["ctb_change"] = c1 - c0
        flags.append(f"🔥 Borrow fee ACCELERATING: {c0:.0f}% → {c1:.0f}% (supply tightening)")

    # utilization climbing
    u0, u1 = num(prev["util"]), num(cur["util"])
    if u0 is not None and u1 is not None and u1 - u0 >= 3:
        d["util_change"] = u1 - u0
        flags.append(f"🔒 Utilization climbing: {u0:.0f}% → {u1:.0f}% (toward supply ceiling)")

    # social mention velocity
    m0, m1 = num(prev["mentions"]), num(cur["mentions"])
    if m1 is not None:
        if (m0 is None or m0 == 0) and m1 > 0:
            flags.append(f"🗣️ NEW social mentions appeared ({m1:.0f})")
        elif m0 and m1 >= 2 * m0:
            flags.append(f"🗣️ Social mentions {m1/m0:.1f}× since last scan (velocity)")
        d["mention_change"] = (m1 - (m0 or 0))

    # price-vs-SI divergence over time (doubling down, confirmed across days)
    p0, p1 = num(prev["price"]), num(cur["price"])
    if p0 and p1 and si0 is not None and si1 is not None:
        if p1 > p0 and si1 > si0:
            flags.append("⚡ Shorts ADDED while price ROSE since last scan (doubling down)")

    # score trend
    s0, s1 = num(prev["score"]), num(cur["score"])
    if s0 is not None and s1 is not None:
        d["score_change"] = s1 - s0
        if s1 - s0 >= 8:
            flags.append(f"⬆️ Squeeze score rising: {s0:.0f} → {s1:.0f}")

    return flags, d

def attach_changes(candidates, path=DEFAULT_PATH):
    """Add c['history_flags'] / c['history_deltas'] to each candidate."""
    df = load(path)
    for c in candidates:
        fl, d = changes(c.get("ticker", ""), df=df)
        c["history_flags"] = fl
        c["history_deltas"] = d
    return candidates

def snapshot_count(path=DEFAULT_PATH):
    df = load(path)
    return (len(df), df["date"].nunique() if len(df) else 0)
