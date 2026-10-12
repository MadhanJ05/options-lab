"""SPY option chain: download, clean, implied vols, smile and ATM term structure.

Pipeline
--------
1. fetch_raw_chain()  - yfinance download of every listed expiry, saved verbatim
                        to data/ with a JSON metadata file (download time, spot, rate).
2. clean_chain()      - pure function of the raw snapshot: mid quotes, liquidity
                        filters, parity-implied forward per expiry, OTM selection.
3. add_implied_vols() - Newton/Brent IV on each cleaned quote.
4. atm_term_structure() - ATM IV per expiry by interpolation in log-moneyness.

Because steps 2-4 only read the saved snapshot, rerunning them is reproducible.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .solver import implied_vol_array

DATA = Path(__file__).resolve().parents[3] / "data"


@dataclass(frozen=True)
class CleanConfig:
    min_days: float = 5.0           # drop expiries closer than this (calendar days)
    max_days: float = 760.0
    max_rel_spread: float = 0.25    # (ask - bid) / mid
    min_mid: float = 0.05           # dollars
    min_volume: int = 10
    min_open_interest: int = 100
    n_parity_strikes: int = 10      # strikes nearest spot used to infer the forward


def tbill_discount_to_cc(discount_yield_pct: float, days: int = 91) -> float:
    """^IRX is the 13-week T-bill discount yield d (in %). Price = 1 - d * days/360;
    continuously compounded rate r = -ln(price) / (days/365)."""
    price = 1.0 - discount_yield_pct / 100.0 * days / 360.0
    return -math.log(price) / (days / 365.0)


# --------------------------------------------------------------------------
# 1. Download
# --------------------------------------------------------------------------
def fetch_raw_chain(ticker: str = "SPY", data_dir: Path = DATA) -> tuple[pd.DataFrame, dict]:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    downloaded_at = dt.datetime.now(dt.timezone.utc)
    hist = tk.history(period="10d", auto_adjust=False)
    spot = float(hist["Close"].iloc[-1])
    quote_date = hist.index[-1].date()   # last completed session = date quotes refer to
    irx_hist = yf.Ticker("^IRX").history(period="10d")
    irx = float(irx_hist["Close"].iloc[-1])
    vix_hist = yf.Ticker("^VIX").history(period="10d")
    vix_hist.index = vix_hist.index.date

    frames = []
    for exp in tk.options:
        chain = tk.option_chain(exp)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            df = df.copy()
            df["expiry"] = exp
            df["kind"] = kind
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"impliedVolatility": "yahoo_iv"})

    meta = {
        "ticker": ticker,
        "downloaded_at_utc": downloaded_at.isoformat(timespec="seconds"),
        "quote_date": quote_date.isoformat(),
        "spot_close": spot,
        "irx_discount_yield_pct": irx,
        "irx_date": irx_hist.index[-1].date().isoformat(),
        "risk_free_cc": tbill_discount_to_cc(irx),
        "vix_close_on_quote_date": float(vix_hist["Close"].get(quote_date, float("nan"))),
        "n_raw_rows": int(len(raw)),
        "n_expiries": int(raw.expiry.nunique()),
    }
    data_dir.mkdir(exist_ok=True)
    stem = f"spy_chain_raw_{meta['quote_date']}"
    raw.to_csv(data_dir / f"{stem}.csv", index=False)
    (data_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return raw, meta


def load_raw_chain(data_dir: Path = DATA, quote_date: str | None = None):
    files = sorted(data_dir.glob("spy_chain_raw_*.csv"))
    if quote_date:
        files = [f for f in files if quote_date in f.name]
    if not files:
        raise FileNotFoundError("no cached chain in data/; run with --refresh")
    path = files[-1]
    meta = json.loads(path.with_suffix(".json").read_text())
    return pd.read_csv(path), meta


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def implied_forward(df_exp: pd.DataFrame, spot: float, r: float, T: float, n: int) -> float:
    """Put-call parity C - P = e^{-rT}(F - K)  =>  F = K + e^{rT}(C - P).
    Median over the n strikes nearest spot that have both a call and a put mid."""
    c = df_exp[df_exp.kind == "call"].set_index("strike").mid
    p = df_exp[df_exp.kind == "put"].set_index("strike").mid
    both = pd.concat([c, p], axis=1, keys=["c", "p"]).dropna()
    if both.empty:
        return math.nan
    near = both.iloc[np.argsort(np.abs(both.index.values - spot))[:n]]
    return float(np.median(near.index.values + math.exp(r * T) * (near.c - near.p)))


def _slope_ok(slope: float, kind: str, disc: float, tol: float = 1e-9) -> bool:
    """Vertical-spread bounds: puts 0 <= dP/dK <= e^{-rT}, calls -e^{-rT} <= dC/dK <= 0."""
    if kind == "put":
        return -tol <= slope <= disc + tol
    return -disc - tol <= slope <= tol


def drop_strike_arbitrage(g: pd.DataFrame, r: float) -> pd.DataFrame:
    """Keep the largest subset of quotes (sorted by strike) with no vertical-spread
    arbitrage between consecutive kept strikes, i.e. delete the fewest quotes.

    If consecutive slopes satisfy the bounds, so does every pair (a slope over a wider
    interval is a weighted average of the consecutive ones), so an O(n^2) longest-chain
    dynamic programme over "last kept quote" finds the exact minimum deletion.
    """
    g = g.sort_values("strike")
    kind, disc = g.kind.iloc[0], math.exp(-r * g["T"].iloc[0])
    K, P = g.strike.values, g.mid.values
    n = len(g)
    best = np.ones(n, dtype=int)
    prev = np.full(n, -1)
    for j in range(n):
        for i in range(j):
            if best[i] + 1 > best[j] and _slope_ok((P[j] - P[i]) / (K[j] - K[i]), kind, disc):
                best[j], prev[j] = best[i] + 1, i
    j = int(np.argmax(best))
    keep = []
    while j >= 0:
        keep.append(j)
        j = prev[j]
    return g.iloc[sorted(keep)]


def clean_chain(raw: pd.DataFrame, meta: dict, cfg: CleanConfig = CleanConfig()):
    """Return (clean OTM quotes, per-expiry forward table, filter funnel,
    quotes removed by the static-arbitrage check)."""
    df = raw.copy()
    spot, r = meta["spot_close"], meta["risk_free_cc"]
    # Quotes are from the close of quote_date (16:00 New York); options expire at the
    # close of their expiry date, so T is a whole number of calendar days / 365.
    quote_date = pd.Timestamp(meta["quote_date"])
    df["days"] = (pd.to_datetime(df.expiry) - quote_date).dt.days.astype(float)
    df["T"] = df.days / 365.0
    df["mid"] = 0.5 * (df.bid + df.ask)
    df["rel_spread"] = (df.ask - df.bid) / df.mid

    funnel = [("raw rows", len(df))]

    def keep(mask, label):
        nonlocal df
        df = df[mask.fillna(False)]
        funnel.append((label, len(df)))

    keep((df.days >= cfg.min_days) & (df.days <= cfg.max_days),
         f"{cfg.min_days:g} <= days to expiry <= {cfg.max_days:g}")
    keep((df.bid > 0) & (df.ask > df.bid), "bid > 0 and ask > bid")

    # Forward per expiry uses all two-sided quotes (before liquidity filters,
    # which would remove some ATM pairs and make the forward noisier).
    fwd_rows = []
    for exp, g in df.groupby("expiry"):
        T = g["T"].iloc[0]
        F = implied_forward(g, spot, r, T, cfg.n_parity_strikes)
        q = r - math.log(F / spot) / T if np.isfinite(F) else math.nan
        fwd_rows.append({"expiry": exp, "days": g.days.iloc[0], "T": T, "forward": F,
                         "implied_div_yield": q})
    fwd = pd.DataFrame(fwd_rows)
    df = df.merge(fwd[["expiry", "forward", "implied_div_yield"]], on="expiry")

    keep(df.rel_spread <= cfg.max_rel_spread, f"relative spread <= {cfg.max_rel_spread:.0%}")
    keep(df.mid >= cfg.min_mid, f"mid >= ${cfg.min_mid:.2f}")
    keep(df.volume >= cfg.min_volume, f"volume >= {cfg.min_volume}")
    keep(df.openInterest >= cfg.min_open_interest, f"open interest >= {cfg.min_open_interest}")
    otm = ((df.kind == "put") & (df.strike < df.forward)) | ((df.kind == "call") & (df.strike >= df.forward))
    keep(otm, "out-of-the-money side only")

    before = df
    df = pd.concat([drop_strike_arbitrage(g, r) for _, g in df.groupby(["expiry", "kind"])])
    funnel.append(("no vertical-spread arbitrage across strikes", len(df)))
    dropped_arb = before.loc[before.index.difference(df.index)]

    df = df.assign(log_moneyness=np.log(df.strike / df.forward)).sort_values(["expiry", "strike"])
    return df.reset_index(drop=True), fwd, funnel, dropped_arb


# --------------------------------------------------------------------------
# 3. Implied vols
# --------------------------------------------------------------------------
def add_implied_vols(df: pd.DataFrame, meta: dict):
    iv, status = implied_vol_array(df.mid.values, meta["spot_close"], df.strike.values,
                                   df["T"].values, meta["risk_free_cc"],
                                   df.implied_div_yield.values, df.kind.values)
    out = df.assign(iv=iv, iv_status=status)
    return out


# --------------------------------------------------------------------------
# 4. ATM term structure
# --------------------------------------------------------------------------
def atm_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """ATM (k = 0) IV per expiry: linear interpolation in log-moneyness between the
    nearest quote on each side of the forward. Expiries without quotes on both
    sides within |k| < 0.05 are skipped rather than extrapolated."""
    rows = []
    for exp, g in df.dropna(subset=["iv"]).groupby("expiry"):
        below = g[g.log_moneyness < 0].sort_values("log_moneyness").tail(1)
        above = g[g.log_moneyness >= 0].sort_values("log_moneyness").head(1)
        if below.empty or above.empty:
            continue
        k0, k1 = below.log_moneyness.iloc[0], above.log_moneyness.iloc[0]
        if k0 < -0.05 or k1 > 0.05:
            continue
        v0, v1 = below.iv.iloc[0], above.iv.iloc[0]
        w = (0 - k0) / (k1 - k0) if k1 > k0 else 0.5
        rows.append({"expiry": exp, "days": g.days.iloc[0], "T": g["T"].iloc[0],
                     "atm_iv": v0 + w * (v1 - v0), "k_below": k0, "k_above": k1})
    return pd.DataFrame(rows)
