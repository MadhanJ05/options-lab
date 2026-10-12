"""No lookahead in the historical study: every hedge decision on date t must be
reproducible from data dated <= t only.

Runs on a synthetic market-data frame so the test needs no network access, and
again on the cached SPY/VIX data when it is present.
"""
import numpy as np
import pandas as pd
import pytest

from options_lab.hedging.historical import CACHE, HistConfig, hedge_trade, load_market_data, trade_schedule, trailing_div_yield


def synthetic_market(seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2006-01-02", "2008-12-31")
    S = 100 * np.exp(np.cumsum(0.012 * rng.standard_normal(len(idx))))
    div = np.zeros(len(idx))
    div[::63] = 0.5
    adj = S * np.cumprod(1 + div / S)
    return pd.DataFrame({"close": S, "adj_close": adj, "dividend": div,
                         "vix": 20 + 5 * rng.standard_normal(len(idx)).cumsum() * 0.05 + 0.0,
                         "irx_pct": np.full(len(idx), 3.0)}, index=idx).rename_axis("date")


def market_frames():
    frames = [("synthetic", synthetic_market())]
    if CACHE.exists():
        frames.append(("spy", load_market_data()))
    return frames


@pytest.mark.parametrize("name,md", market_frames())
def test_truncated_data_reproduces_decisions(name, md):
    sched = trade_schedule(md, HistConfig(start="2007-01-01", end=str(md.index[-1].date())))
    for e, x in list(zip(sched.entry, sched.expiry))[::5][:6]:
        _, full = hedge_trade(md, e, x)
        for d in full.index[:-1]:
            _, partial = hedge_trade(md.loc[:d], e, x)
            pd.testing.assert_frame_equal(partial, full.loc[:d])


@pytest.mark.parametrize("name,md", market_frames())
def test_scrambling_the_future_does_not_change_the_past(name, md):
    sched = trade_schedule(md, HistConfig(start="2007-01-01", end=str(md.index[-1].date())))
    e, x = sched.entry.iloc[3], sched.expiry.iloc[3]
    _, full = hedge_trade(md, e, x)
    cut = full.index[len(full) // 2]
    poisoned = md.copy()
    future = poisoned.index > cut
    rng = np.random.default_rng(0)
    for col in ("close", "adj_close", "vix", "irx_pct"):
        poisoned.loc[future, col] *= rng.uniform(0.5, 1.5, future.sum())
    poisoned.loc[future, "dividend"] = 99.0
    _, after = hedge_trade(poisoned, e, x)
    pd.testing.assert_frame_equal(after.loc[:cut], full.loc[:cut])


def test_trailing_dividend_yield_ignores_future_dividends():
    md = synthetic_market()
    d = md.index[300]
    base = trailing_div_yield(md, d)
    md2 = md.copy()
    md2.loc[md2.index > d, "dividend"] = 50.0
    assert trailing_div_yield(md2, d) == base


def test_expiry_within_30_calendar_days():
    md = synthetic_market()
    sched = trade_schedule(md, HistConfig())
    gap = (sched.expiry - sched.entry).dt.days
    assert gap.max() <= 30 and gap.min() >= 26
