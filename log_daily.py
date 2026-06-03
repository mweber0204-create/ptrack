#!/usr/bin/env python3
"""
DAILY LOGGER  --  build the Squeeze Radar's memory over time.

Run this once a day (manually, or scheduled) on a machine where the history
file persists (e.g. your Mac). Each run scans the market, scores candidates,
and appends today's snapshot to squeeze_history.csv. The more days you log,
the more rate-of-change signals the radar can detect (rising SI, accelerating
borrow fee, mention velocity, etc.).

USAGE
    python log_daily.py                 # scan + log today's snapshot
    python log_daily.py --watchlist my_tickers.txt   # also include your tickers

SCHEDULE IT (Mac, every weekday at 8am) with cron:
    crontab -e
    0 8 * * 1-5  cd "/Users/<you>/Stock screener" && /usr/bin/python3 log_daily.py >> log_daily.out 2>&1
"""
import argparse, datetime as dt
import squeeze_screener as SQ
import squeeze_history as HIST

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watchlist", default=None, help="optional file, one ticker/line")
    args = ap.parse_args()
    wl = None
    if args.watchlist:
        wl = [l.strip() for l in open(args.watchlist) if l.strip() and not l.startswith("#")]

    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M}] scanning + logging snapshot...")
    res = SQ.run_squeeze(wl, progress=lambda f: None)   # run_squeeze records history internally
    n_rows, n_days = HIST.snapshot_count()
    print(f"  scored {len(res)} candidates. History now holds {n_rows} rows across {n_days} day(s).")
    # show any change-signals detected today (needs >= 2 days of history)
    movers = [c for c in res if c.get("history_flags")]
    if movers:
        print(f"  {len(movers)} names show CHANGE since last scan:")
        for c in movers[:15]:
            print(f"   - {c['ticker']} (score {c.get('score')}): " + " | ".join(c["history_flags"]))
    else:
        print("  No change-signals yet (need at least 2 logged days to compare).")

if __name__ == "__main__":
    main()
