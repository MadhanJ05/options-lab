"""Part 1: price options three ways, show convergence, variance reduction, benchmarks.

Outputs (results/):
  pricing_three_ways.csv, binomial_convergence.png, american_premium.csv,
  american_premium.png, mc_variance_reduction.csv, mc_convergence.png,
  benchmarks.csv
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from options_lab.plotting import RESULTS, SERIES, TEXT_2, apply_style, save
from options_lab.pricing.binomial import crr_price, crr_price_naive
from options_lab.pricing.black_scholes import bs_price
from options_lab.pricing.monte_carlo import (
    METHODS,
    bgk_discrete_barrier,
    geometric_asian_call,
    mc_asian_call,
    mc_asian_call_naive,
    mc_european,
    mc_up_and_out_call,
)

SEED = 20240101
S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.015


def timed(fn, repeats=3):
    """Best-of-n wall time in seconds and the last result."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def three_ways():
    rows = []
    for kind in ("call", "put"):
        for strike in (80.0, 100.0, 120.0):
            bs = float(bs_price(S, strike, T, r, sigma, q, kind))
            tree = crr_price(S, strike, T, r, sigma, q, kind, steps=2000)
            mc = mc_european(S, strike, T, r, sigma, q, kind, n_paths=1_000_000,
                             rng=np.random.default_rng(SEED), method="antithetic+control")
            rows.append({
                "kind": kind, "K": strike, "black_scholes": bs, "crr_2000": tree,
                "mc_price": mc.price, "mc_se": mc.se,
                "crr_minus_bs": tree - bs, "mc_z_score": (mc.price - bs) / mc.se,
            })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "pricing_three_ways.csv", index=False)
    return df


def binomial_convergence():
    Ns = np.arange(5, 501)
    bs = float(bs_price(S, K, T, r, sigma, q, "call"))
    err = np.array([crr_price(S, K, T, r, sigma, q, "call", steps=int(n)) - bs for n in Ns])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(Ns[Ns % 2 == 0], bs + err[Ns % 2 == 0], color=SERIES[0], label="CRR, even N")
    ax.plot(Ns[Ns % 2 == 1], bs + err[Ns % 2 == 1], color=SERIES[1], label="CRR, odd N")
    ax.axhline(bs, color=TEXT_2, linewidth=1, label=f"Black-Scholes {bs:.4f}")
    ax.set(title="CRR price oscillates around Black-Scholes",
           xlabel="Tree steps N", ylabel="ATM call price", ylim=(bs - 0.12, bs + 0.12))
    ax.legend()

    ax = axes[1]
    ax.loglog(Ns, np.abs(err), color=SERIES[0], linewidth=1, label="|CRR - BS|")
    ax.loglog(Ns, 1.0 / Ns, color=TEXT_2, linewidth=1, linestyle="--", label="reference 1/N")
    ax.set(title="Absolute error falls at rate O(1/N)", xlabel="Tree steps N (log)",
           ylabel="Absolute pricing error (log)")
    ax.legend()
    save(fig, "binomial_convergence.png")
    return err


def american_premium():
    strikes = np.arange(70, 135, 5, dtype=float)
    rows = []
    for k in strikes:
        amer = crr_price(S, k, T, r, sigma, q, "put", steps=2000, american=True)
        euro_tree = crr_price(S, k, T, r, sigma, q, "put", steps=2000)
        rows.append({"K": k, "american_put": amer, "european_put_crr": euro_tree,
                     "european_put_bs": float(bs_price(S, k, T, r, sigma, q, "put")),
                     "early_exercise_premium": amer - euro_tree})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "american_premium.csv", index=False)

    # Premium vs N for the ATM put, to show it is not a tree artefact.
    Ns = np.array([25, 50, 100, 200, 400, 800, 1600])
    prem_n = [crr_price(S, K, T, r, sigma, q, "put", steps=int(n), american=True)
              - crr_price(S, K, T, r, sigma, q, "put", steps=int(n)) for n in Ns]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(df.K, df.early_exercise_premium, color=SERIES[0], marker="o")
    ax.set(title="Early-exercise premium of the American put",
           xlabel="Strike K (spot = 100)", ylabel="American - European put")
    ax = axes[1]
    ax.semilogx(Ns, prem_n, color=SERIES[0], marker="o")
    ax.set(title="ATM premium converges as the tree is refined",
           xlabel="Tree steps N (log)", ylabel="American - European put")
    save(fig, "american_premium.png")
    return df


def variance_reduction():
    n = 200_000
    rows = []
    specs = {
        "European call": lambda m: mc_european(S, K, T, r, sigma, q, "call", n_paths=n,
                                               rng=np.random.default_rng(SEED), method=m),
        "Arithmetic Asian call": lambda m: mc_asian_call(S, K, T, r, sigma, q, n_fixings=252,
                                                         n_paths=n, rng=np.random.default_rng(SEED),
                                                         method=m),
        "Up-and-out call (H=130)": lambda m: mc_up_and_out_call(S, K, 130.0, T, r, sigma, q,
                                                                n_monitor=252, n_paths=n,
                                                                rng=np.random.default_rng(SEED),
                                                                method=m),
    }
    reference = {
        "European call": ("Black-Scholes", float(bs_price(S, K, T, r, sigma, q, "call"))),
        "Arithmetic Asian call": ("none (geometric Asian = %.4f)"
                                  % geometric_asian_call(S, K, T, r, sigma, q, 252), float("nan")),
        "Up-and-out call (H=130)": ("BGK approx.", bgk_discrete_barrier(S, K, 130.0, T, r, sigma, q, 252)),
    }
    for product, fn in specs.items():
        base_se = None
        for m in METHODS:
            secs, res = timed(lambda: fn(m), repeats=1)
            base_se = res.se if m == "plain" else base_se
            vrf = (base_se / res.se) ** 2
            # Efficiency also charges for runtime: VRF * t_plain / t_method.
            rows.append({"product": product, "method": m, "price": res.price, "std_error": res.se,
                         "variance_reduction": vrf, "seconds": secs,
                         "reference": reference[product][0],
                         "reference_value": reference[product][1]})
    df = pd.DataFrame(rows)
    t_plain = df[df.method == "plain"].set_index("product").seconds
    df["efficiency_gain"] = df.variance_reduction * df["product"].map(t_plain) / df.seconds
    df.to_csv(RESULTS / "mc_variance_reduction.csv", index=False)
    return df


def mc_convergence():
    """Standard error vs number of paths for the arithmetic Asian: 1/sqrt(n) for all
    methods, but at very different levels."""
    ns = np.array([1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, m in enumerate(METHODS):
        se = [mc_asian_call(S, K, T, r, sigma, q, 252, n_paths=int(n),
                            rng=np.random.default_rng(SEED), method=m).se for n in ns]
        ax.loglog(ns, se, color=SERIES[i], marker="o", label=m)
    ax.set(title="Arithmetic Asian: standard error vs paths",
           xlabel="Simulated paths (log)", ylabel="Standard error (log)")
    ax.legend()
    save(fig, "mc_convergence.png")


def benchmarks():
    rows = []
    # 1. Black-Scholes on 100k strikes: numpy broadcast vs Python loop over scalars.
    strikes = np.linspace(50, 150, 100_000)
    t_vec, _ = timed(lambda: bs_price(S, strikes, T, r, sigma, q))

    def bs_loop():
        out = []
        for k in strikes:
            d1 = (math.log(S / k) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
            nd1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
            nd2 = 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
            out.append(S * math.exp(-q * T) * nd1 - k * math.exp(-r * T) * nd2)
        return out
    t_loop, _ = timed(bs_loop)
    rows.append(("Black-Scholes, 100k strikes", t_vec, t_loop))

    # 2. CRR American put, N = 2000 steps.
    t_vec, _ = timed(lambda: crr_price(S, K, T, r, sigma, q, "put", 2000, american=True))
    t_loop, _ = timed(lambda: crr_price_naive(S, K, T, r, sigma, q, "put", 2000, american=True), 1)
    rows.append(("CRR American put, N=2000", t_vec, t_loop))

    # 3. Arithmetic Asian MC, 20k paths x 252 fixings.
    t_vec, _ = timed(lambda: mc_asian_call(S, K, T, r, sigma, q, 252, n_paths=20_000,
                                           rng=np.random.default_rng(SEED)))
    t_loop, _ = timed(lambda: mc_asian_call_naive(S, K, T, r, sigma, q, 252, n_paths=20_000), 1)
    rows.append(("Asian MC, 20k paths x 252 steps", t_vec, t_loop))

    df = pd.DataFrame(rows, columns=["task", "vectorised_s", "naive_loop_s"])
    df["speedup"] = df.naive_loop_s / df.vectorised_s
    df.to_csv(RESULTS / "benchmarks.csv", index=False)
    return df


def main():
    apply_style()
    RESULTS.mkdir(exist_ok=True)
    pd.set_option("display.width", 160)
    print("== Three pricing methods ==\n", three_ways().round(5).to_string(index=False))
    err = binomial_convergence()
    print(f"\n== CRR error at N=500: {err[-1]:+.5f}, N=499: {err[-2]:+.5f}")
    print("\n== American put premium ==\n", american_premium().round(4).to_string(index=False))
    print("\n== Variance reduction ==\n", variance_reduction().round(5).to_string(index=False))
    mc_convergence()
    print("\n== Benchmarks ==\n", benchmarks().round(4).to_string(index=False))


if __name__ == "__main__":
    main()
