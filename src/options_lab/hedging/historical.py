"""Historical delta-hedging study: sell a 30-day ATM SPY call each month at VIX-implied
vol, hedge daily on actual SPY closes, and record the hedged P&L at expiry.

Information discipline (no lookahead)
-------------------------------------
Every input to a decision on date t comes from rows of the market-data frame
dated <= t:
  * entry price, strike, VIX, rate, trailing dividend yield: entry-date row and earlier;
  * daily hedge ratio: that day's close, remaining time, and entry-date VIX / q;
  * the expiry date comes from the exchange calendar, which is public in advance.
`hedge_trade` only ever reads md.loc[date] for dates it has reached, so running it
on data truncated at any date d reproduces the full-data decisions up to d
(tests/test_no_lookahead.py checks this).

Proxy caveat: VIX is computed from S&P 500 index (SPX) options across many strikes
(a model-free 30-day variance-swap rate), not from SPY at-the-money options.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from options_lab.iv.chain import tbill_discount_to_cc
from options_lab.pricing.black_scholes import bs_delta, bs_gamma, bs_price

DATA = Path(__file__).resolve().parents[3] / "data"
CACHE = DATA / "spy_vix_irx_daily.csv"


@dataclass(frozen=True)
class HistConfig:
    start: str = "2007-01-01"
    end: str = "2025-12-31"
    days_to_expiry: int = 30
    cost_bps: float = 0.0
    vol_shift: float = 0.0     # vol points added to VIX (sensitivity analysis only)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def download_market_data(start="2006-01-01", end="2025-12-31", path: Path = CACHE) -> pd.DataFrame:
    """SPY (unadjusted close, adjusted close, dividends), ^VIX close, ^IRX close.
    Starts a year before the study so the first trailing dividend yield has history."""
    import datetime as dt

    import yfinance as yf

    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def get(tk, **kw):
        df = yf.download(tk, start=start, end=end_excl, auto_adjust=False, progress=False, **kw)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(tk, axis=1, level=1)
        return df

    spy = get("SPY", actions=True)
    vix = get("^VIX")
    irx = get("^IRX")
    md = pd.DataFrame({
        "close": spy["Close"],
        "adj_close": spy["Adj Close"],
        "dividend": spy["Dividends"].fillna(0.0),
        "vix": vix["Close"].reindex(spy.index),
        "irx_pct": irx["Close"].reindex(spy.index),
    })
    md.index = pd.DatetimeIndex(md.index).tz_localize(None).normalize()
    md.index.name = "date"
    missing = {c: int(md[c].isna().sum()) for c in ("vix", "irx_pct")}
    # Forward-fill only: uses the last value known on or before each date.
    md[["vix", "irx_pct"]] = md[["vix", "irx_pct"]].ffill()
    path.parent.mkdir(exist_ok=True)
    md.to_csv(path)
    meta = {"downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "start": str(md.index.min().date()), "end": str(md.index.max().date()),
            "rows": int(len(md)), "forward_filled_rows": missing,
            "sources": {"SPY": "yfinance, auto_adjust=False", "^VIX": "yfinance", "^IRX": "yfinance"}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return md


def load_market_data(path: Path = CACHE, refresh=False) -> pd.DataFrame:
    if refresh or not path.exists():
        return download_market_data(path=path)
    return pd.read_csv(path, index_col="date", parse_dates=True)


def trailing_div_yield(md: pd.DataFrame, date) -> float:
    """Dividends with ex-date in (date - 365d, date], divided by the close on `date`."""
    window = md.loc[(md.index > date - pd.Timedelta(days=365)) & (md.index <= date), "dividend"]
    return float(window.sum() / md.at[date, "close"])


# --------------------------------------------------------------------------
# Trade schedule
# --------------------------------------------------------------------------
def trade_schedule(md: pd.DataFrame, cfg: HistConfig) -> pd.DataFrame:
    idx = md.index[(md.index >= cfg.start) & (md.index <= cfg.end)]
    entries = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first().values
    rows = []
    for e in pd.DatetimeIndex(entries):
        target = e + pd.Timedelta(days=cfg.days_to_expiry)
        if target > idx[-1]:
            break
        expiry = idx[idx <= target][-1]
        rows.append({"entry": e, "expiry": expiry})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# One trade
# --------------------------------------------------------------------------
def hedge_trade(md: pd.DataFrame, entry, expiry, cost_bps: float = 0.0, vol_shift: float = 0.0):
    """Short one call (on 1 share) at entry, delta-hedge at each close until expiry.
    Implied vol = (VIX + vol_shift) / 100, fixed for the life of the trade.

    Returns (summary dict, daily DataFrame of decisions). If md ends before expiry
    the daily frame stops at the last available date and the summary is None.
    """
    entry, expiry = pd.Timestamp(entry), pd.Timestamp(expiry)
    c = cost_bps * 1e-4
    row0 = md.loc[entry]
    S0, K = float(row0.close), float(row0.close)
    sigma = (float(row0.vix) + vol_shift) / 100.0
    r0 = tbill_discount_to_cc(float(row0.irx_pct))
    q = trailing_div_yield(md, entry)
    T0 = (expiry - entry).days / 365.0
    premium = float(bs_price(S0, K, T0, r0, sigma, q, "call"))

    dates = md.index[(md.index >= entry) & (md.index <= expiry)]
    delta, cash, costs, gamma_pnl = 0.0, premium, 0.0, 0.0
    prev_date, prev_r, prev_S, prev_gamma, prev_tau = None, r0, S0, None, None
    log = []
    for d in dates:
        row = md.loc[d]
        S = float(row.close)
        if prev_date is not None:
            dt_years = (d - prev_date).days / 365.0
            cash *= math.exp(prev_r * dt_years)                  # financing
            cash += delta * float(row.dividend)                  # dividend on shares held
            ret = math.log(float(row.adj_close) / float(md.at[prev_date, "adj_close"]))
            gamma_pnl += 0.5 * prev_gamma * prev_S**2 * (sigma**2 * dt_years - ret**2)
        r = tbill_discount_to_cc(float(row.irx_pct))
        tau = (expiry - d).days / 365.0
        if d < expiry:
            new_delta = float(bs_delta(S, K, tau, r, sigma, q, "call"))
            g = float(bs_gamma(S, K, tau, r, sigma, q))
        else:
            new_delta, g = 0.0, 0.0                              # unwind at expiry
        traded = new_delta - delta
        cash -= traded * S + c * abs(traded) * S
        costs += c * abs(traded) * S
        log.append({"date": d, "S": S, "tau": tau, "r": r, "delta": new_delta, "cash": cash})
        delta, prev_date, prev_r, prev_S, prev_gamma = new_delta, d, r, S, g

    daily = pd.DataFrame(log).set_index("date")
    if dates[-1] < expiry:
        return None, daily

    ST = float(md.at[expiry, "close"])
    payoff = max(ST - K, 0.0)
    pnl = cash - payoff
    rets = np.diff(np.log(md.loc[dates, "adj_close"].values))
    realised = math.sqrt(np.sum(rets**2) / T0)
    summary = {
        "entry": entry, "expiry": expiry, "days": (expiry - entry).days, "n_hedges": len(dates) - 1,
        "S0": S0, "K": K, "ST": ST, "vix": float(row0.vix), "r": r0, "q": q, "premium": premium,
        "payoff": payoff, "costs": costs, "pnl": pnl, "pnl_pct": 100 * pnl / S0,
        "gamma_approx_pct": 100 * gamma_pnl / S0, "realised_vol": 100 * realised,
        "spy_return_pct": 100 * (ST / S0 - 1),
    }
    return summary, daily


def run_study(md: pd.DataFrame, cfg: HistConfig = HistConfig()) -> pd.DataFrame:
    sched = trade_schedule(md, cfg)
    rows = []
    for e, x in zip(sched.entry, sched.expiry):
        s, _ = hedge_trade(md, e, x, cfg.cost_bps, cfg.vol_shift)
        rows.append(s)
    df = pd.DataFrame(rows)
    df["implied_vol_used"] = df.vix + cfg.vol_shift
    df["vrp_vol"] = df.vix - df.realised_vol
    df["vrp_var"] = (df.vix**2 - df.realised_vol**2) / 100**2
    return df
