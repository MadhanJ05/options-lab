"""Part 3.2: historical variance risk premium study on SPY with VIX as implied vol.

Usage:
  python scripts/run_historical.py            # use cached data/spy_vix_irx_daily.csv
  python scripts/run_historical.py --refresh  # re-download from yfinance

Outputs (results/): hist_trades.csv, hist_summary.json, hist_worst_months.csv,
hist_yearly.csv, hist_vol_sensitivity.csv, hist_cost_comparison.csv and figures.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from options_lab.hedging.historical import HistConfig, load_market_data, run_study
from options_lab.plotting import NEG, POS, RESULTS, SERIES, TEXT_2, apply_style, save


def summary_stats(df: pd.DataFrame) -> dict:
    p = df.pnl_pct
    n = len(p)
    cum = p.cumsum()
    return {
        "n_months": n,
        "first_entry": str(df.entry.min().date()),
        "last_expiry": str(df.expiry.max().date()),
        "mean_vix": float(df.vix.mean()),
        "mean_realised_vol": float(df.realised_vol.mean()),
        "mean_vrp_vol_pts": float(df.vrp_vol.mean()),
        "median_vrp_vol_pts": float(df.vrp_vol.median()),
        "t_stat_vrp_vol": float(df.vrp_vol.mean() / (df.vrp_vol.std(ddof=1) / np.sqrt(n))),
        "mean_vrp_variance": float(df.vrp_var.mean()),
        "share_months_vix_above_realised": float((df.vrp_vol > 0).mean()),
        "mean_pnl_pct_of_spot": float(p.mean()),
        "median_pnl_pct_of_spot": float(p.median()),
        "std_pnl_pct_of_spot": float(p.std(ddof=1)),
        "t_stat_mean_pnl": float(p.mean() / (p.std(ddof=1) / np.sqrt(n))),
        "hit_rate": float((p > 0).mean()),
        "skewness": float(stats.skew(p)),
        "excess_kurtosis": float(stats.kurtosis(p)),
        "lag1_autocorr": float(p.autocorr(1)),
        "worst_month_pct": float(p.min()),
        "best_month_pct": float(p.max()),
        "cumulative_pnl_pct": float(cum.iloc[-1]),
        "max_drawdown_pct": float((cum - cum.cummax()).min()),
        "annualised_mean_over_std": float(p.mean() / p.std(ddof=1) * np.sqrt(12)),
        "mean_premium_pct_of_spot": float((100 * df.premium / df.S0).mean()),
        "corr_pnl_vs_gamma_approx": float(df.pnl_pct.corr(df.gamma_approx_pct)),
        "corr_pnl_vs_vrp_var": float(df.pnl_pct.corr(df.vrp_var)),
    }


def plot_vix_vs_realised(df):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    ax.plot(df.entry, df.vix, color=SERIES[0], linewidth=1.5, label="VIX at entry (implied)")
    ax.plot(df.entry, df.realised_vol, color=SERIES[1], linewidth=1.5, label="Realised vol over the next 30 days")
    ax.set(title="Implied (VIX) vs subsequently realised 30-day SPY volatility", ylabel="Annualised vol (%)")
    ax.legend(loc="upper right")
    ax = axes[1]
    colors = np.where(df.vrp_vol >= 0, POS, NEG)
    ax.bar(df.entry, df.vrp_vol, width=25, color=colors)
    ax.axhline(0, color=TEXT_2, linewidth=0.8)
    ax.set(title=f"VIX minus realised vol (mean {df.vrp_vol.mean():+.1f} vol pts)", ylabel="Vol points")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "hist_vix_vs_realised.png")


def plot_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df.pnl_pct, bins=40, color=SERIES[0], alpha=0.9)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.axvline(df.pnl_pct.mean(), color=SERIES[1], linewidth=1.5, label=f"mean {df.pnl_pct.mean():+.2f}%")
    for height, (_, w) in zip((6, 11, 6), df.nsmallest(3, "pnl_pct").iterrows()):
        ax.annotate(w.entry.strftime("%b %Y"), (w.pnl_pct, 1), xytext=(w.pnl_pct, height),
                    ha="center", fontsize=8, color=TEXT_2,
                    arrowprops=dict(arrowstyle="-", color=TEXT_2, linewidth=0.6))
    ax.set(title="Monthly delta-hedged P&L of a short 30-day ATM SPY call, 2007-2025",
           xlabel="Hedged P&L at expiry (% of SPY price at entry)", ylabel="Months")
    ax.legend()
    save(fig, "hist_pnl_distribution.png")


def plot_cumulative(df, df_cost):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df.expiry, df.pnl_pct.cumsum(), color=SERIES[0], label="no transaction costs")
    ax.plot(df_cost.expiry, df_cost.pnl_pct.cumsum(), color=SERIES[1], label="2 bps on stock trades")
    for label, (a, b) in {"GFC": ("2008-09-01", "2008-12-31"), "COVID": ("2020-02-15", "2020-04-30"),
                          "Apr 2025": ("2025-03-25", "2025-05-05")}.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=NEG, alpha=0.10, linewidth=0)
        ax.text(pd.Timestamp(a), ax.get_ylim()[1] * 0.97, label, fontsize=8, color=TEXT_2, va="top")
    ax.set(title="Cumulative hedged P&L: selling one 30-day ATM call per $100 of SPY each month",
           ylabel="Cumulative P&L (% of entry spot, summed)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save(fig, "hist_cumulative_pnl.png")


def plot_gamma_approx(df):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(df.gamma_approx_pct, df.pnl_pct, s=18, color=SERIES[0], alpha=0.7, linewidths=0)
    lim = [min(df.gamma_approx_pct.min(), df.pnl_pct.min()) - 0.2, max(df.gamma_approx_pct.max(), df.pnl_pct.max()) + 0.2]
    ax.plot(lim, lim, color=TEXT_2, linewidth=1, label="45° line")
    ax.set(title=f"Actual vs gamma-theta P&L (corr {df.pnl_pct.corr(df.gamma_approx_pct):.2f})",
           xlabel="Σ ½ Γ S² (σ_VIX² dt − r_t²)  (% of spot)", ylabel="Actual hedged P&L (% of spot)")
    ax.legend(loc="upper left")
    save(fig, "hist_pnl_vs_gamma_approx.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    md = load_market_data(refresh=args.refresh)

    base = HistConfig()
    df = run_study(md, base)
    df_cost = run_study(md, replace(base, cost_bps=2.0))
    df.to_csv(RESULTS / "hist_trades.csv", index=False)

    summ = summary_stats(df)
    (RESULTS / "hist_summary.json").write_text(json.dumps(summ, indent=2))

    cols = ["entry", "expiry", "vix", "realised_vol", "vrp_vol", "spy_return_pct", "premium", "pnl_pct"]
    worst = df.nsmallest(10, "pnl_pct")[cols]
    worst[["entry", "expiry"]] = worst[["entry", "expiry"]].apply(lambda c: c.dt.strftime("%Y-%m-%d"))
    worst = worst.round(3)
    worst.to_csv(RESULTS / "hist_worst_months.csv", index=False)

    yearly = df.groupby(df.entry.dt.year).agg(
        months=("pnl_pct", "size"), total_pnl_pct=("pnl_pct", "sum"), hit_rate=("pnl_pct", lambda s: (s > 0).mean()),
        mean_vix=("vix", "mean"), mean_realised=("realised_vol", "mean"), worst_month_pct=("pnl_pct", "min"))
    yearly.index.name = "year"
    yearly.to_csv(RESULTS / "hist_yearly.csv")

    costs = pd.DataFrame([
        {"cost_bps": 0.0, **{k: summ[k] for k in ("mean_pnl_pct_of_spot", "hit_rate", "cumulative_pnl_pct")}},
        {"cost_bps": 2.0, **{k: v for k, v in summary_stats(df_cost).items()
                             if k in ("mean_pnl_pct_of_spot", "hit_rate", "cumulative_pnl_pct")}},
    ])
    costs.to_csv(RESULTS / "hist_cost_comparison.csv", index=False)

    sens = []
    for shift in (0.0, -1.0, -2.0, -3.0):
        s = summary_stats(run_study(md, replace(base, vol_shift=shift)))
        sens.append({"implied_vol": f"VIX {shift:+.0f}" if shift else "VIX", "vol_shift": shift,
                     **{k: s[k] for k in ("mean_pnl_pct_of_spot", "t_stat_mean_pnl", "hit_rate",
                                          "cumulative_pnl_pct", "worst_month_pct")}})
    sens = pd.DataFrame(sens)
    sens.to_csv(RESULTS / "hist_vol_sensitivity.csv", index=False)

    apply_style()
    plot_vix_vs_realised(df)
    plot_distribution(df)
    plot_cumulative(df, df_cost)
    plot_gamma_approx(df)

    pd.set_option("display.width", 180)
    print(json.dumps(summ, indent=2))
    print(worst.to_string(index=False))
    print(yearly.round(3).to_string())
    print(costs.round(4).to_string(index=False))
    print(sens.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
