"""Part 2: SPY implied volatility smile and ATM term structure.

Usage:
  python scripts/run_iv.py            # use cached snapshot in data/
  python scripts/run_iv.py --refresh  # download a new snapshot first
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from options_lab.iv.chain import (
    DATA, CleanConfig, add_implied_vols, atm_term_structure, clean_chain,
    fetch_raw_chain, load_raw_chain,
)
from options_lab.plotting import RESULTS, SERIES, TEXT_2, apply_style, save

SMILE_TARGET_DAYS = [7, 30, 60, 90, 180, 365]
# Sequential blue ramp (light -> dark = short -> long maturity).
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def pick_expiries(chain: pd.DataFrame, targets, min_quotes=30) -> list:
    """Nearest expiry to each target maturity among expiries with enough clean quotes
    to draw a smile (display choice only; all expiries feed the term structure)."""
    counts = chain.groupby("expiry").size()
    available_days = chain.groupby("expiry").days.first()[counts >= min_quotes]
    chosen = []
    for t in targets:
        exp = (available_days - t).abs().idxmin()
        if exp not in chosen:
            chosen.append(exp)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    raw, meta = fetch_raw_chain() if args.refresh else load_raw_chain()
    cfg = CleanConfig()
    clean, fwd, funnel, dropped_arb = clean_chain(raw, meta, cfg)
    chain = add_implied_vols(clean, meta)
    failed = chain[chain.iv.isna()]
    chain = chain.dropna(subset=["iv"])
    funnel.append(("implied vol solved", len(chain)))

    stem = f"spy_chain_clean_{meta['quote_date']}"
    chain.to_csv(DATA / f"{stem}.csv", index=False)
    term = atm_term_structure(chain)

    apply_style()
    # Smile per selected expiry
    exps = pick_expiries(chain, SMILE_TARGET_DAYS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for color, exp in zip(RAMP, exps):
        g = chain[chain.expiry == exp].sort_values("log_moneyness")
        label = f"{exp} ({int(g.days.iloc[0])}d)"
        axes[0].plot(g.log_moneyness, 100 * g.iv, color=color, marker="o", markersize=3,
                     linewidth=1.5, label=label)
        axes[1].plot(g.log_moneyness / np.sqrt(g["T"]), 100 * g.iv, color=color, marker="o",
                     markersize=3, linewidth=1.5, label=label)
    for ax in axes:
        ax.axvline(0, color=TEXT_2, linewidth=0.8)
        ax.set_ylabel("Implied volatility (%)")
    axes[0].set(title=f"SPY volatility smile, quotes at close {meta['quote_date']}",
                xlabel="Log-moneyness  ln(K / F)")
    axes[1].set(title="Same smiles against maturity-scaled moneyness",
                xlabel="ln(K / F) / sqrt(T)", xlim=(-2.0, 0.8))
    axes[1].legend(title="Expiry", loc="lower left")
    save(fig, "iv_smile.png")

    # ATM term structure
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(term.days, 100 * term.atm_iv, color=SERIES[0], marker="o")
    ax.set(title=f"SPY ATM implied volatility term structure, {meta['quote_date']}",
           xlabel="Days to expiry", ylabel="ATM implied volatility (%)")
    save(fig, "iv_term_structure.png")

    # Tables
    funnel_df = pd.DataFrame(funnel, columns=["filter", "rows_remaining"])
    funnel_df.to_csv(RESULTS / "iv_cleaning_funnel.csv", index=False)
    fwd.to_csv(RESULTS / "iv_forwards.csv", index=False)
    dropped_arb[["expiry", "kind", "strike", "bid", "ask", "mid", "volume", "openInterest",
                 "lastTradeDate"]].to_csv(RESULTS / "iv_dropped_arbitrage_quotes.csv", index=False)
    term.to_csv(RESULTS / "iv_term_structure.csv", index=False)
    yahoo_diff = (chain.iv - chain.yahoo_iv).abs()
    # One-day check of the Part 3 proxy: VIX vs SPY 30-day ATM IV on the quote date.
    atm30 = float(np.interp(30, term.days, term.atm_iv))
    vix_close = meta.get("vix_close_on_quote_date")
    summary = {
        **meta,
        "clean_config": cfg.__dict__,
        "n_clean_quotes": int(len(chain)),
        "n_expiries_clean": int(chain.expiry.nunique()),
        "iv_solver_status_counts": chain.iv_status.value_counts().to_dict(),
        "iv_failures": failed.iv_status.value_counts().to_dict(),
        "n_dropped_static_arbitrage": int(len(dropped_arb)),
        "smile_expiries": exps,
        "median_abs_diff_vs_yahoo_iv": float(yahoo_diff.median()),
        "spy_atm_iv_30d_interpolated": atm30,
        "vix_minus_spy_atm_30d_vol_pts": (vix_close - 100 * atm30) if vix_close else None,
    }
    (RESULTS / "iv_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    pd.set_option("display.width", 160)
    print(json.dumps(summary, indent=2, default=str))
    print(funnel_df.to_string(index=False))
    print(fwd.round(5).to_string(index=False))
    print(term.round(4).to_string(index=False))
    for exp in exps:
        g = chain[chain.expiry == exp]
        print(exp, len(g), "quotes, k range", g.log_moneyness.min().round(3), g.log_moneyness.max().round(3))


if __name__ == "__main__":
    main()
