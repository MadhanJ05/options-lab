"""Monte Carlo pricing under geometric Brownian motion.

Variance reduction:
  * antithetic variates (Boyle 1977): pair each normal draw Z with -Z
  * control variates: Y_cv = Y - b (X - E[X]), b = Cov(Y, X) / Var(X)
    - European options: discounted terminal stock price, E = S e^{-qT}
    - arithmetic Asian: geometric Asian with the same fixings (Kemna & Vorst 1990)
    - up-and-out barrier call: vanilla European call (Black-Scholes)

Paths are simulated with the exact log-normal transition, so there is no
time-discretisation bias at the monitoring dates. Simulation runs in batches
to bound memory; all arithmetic inside a batch is vectorised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black_scholes import bs_price

METHODS = ("plain", "antithetic", "control", "antithetic+control")


@dataclass
class MCResult:
    method: str
    price: float
    se: float
    n_paths: int

    def ci(self, z=1.96):
        return self.price - z * self.se, self.price + z * self.se


# --------------------------------------------------------------------------
# Path generation
# --------------------------------------------------------------------------
def gbm_paths(S0, T, r, q, sigma, n_steps, n_paths, rng, antithetic=False):
    """Return array (n_paths, n_steps + 1) of GBM prices, column 0 = S0.

    With antithetic=True, rows [0, n/2) use Z and rows [n/2, n) use -Z,
    so row i and row i + n/2 form an antithetic pair.
    """
    dt = T / n_steps
    if antithetic:
        if n_paths % 2:
            raise ValueError("n_paths must be even for antithetic sampling")
        Z = rng.standard_normal((n_paths // 2, n_steps))
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal((n_paths, n_steps))
    log_inc = (r - q - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * Z
    log_paths = np.cumsum(log_inc, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return paths


def _batches(n_paths, batch_size):
    done = 0
    while done < n_paths:
        b = min(batch_size, n_paths - done)
        yield b
        done += b


# --------------------------------------------------------------------------
# Generic estimator
# --------------------------------------------------------------------------
def _estimate(Y, X, EX, method):
    """Turn discounted payoffs Y (and control X with mean EX) into price and SE.

    For antithetic methods, Y and X are already pair-averaged (one value per
    independent pair), so the plain sample SE formula applies to them.
    """
    use_cv = "control" in method
    if use_cv:
        Xc = X - EX
        b = np.dot(Y - Y.mean(), Xc) / np.dot(Xc - Xc.mean(), Xc - Xc.mean())
        Y = Y - b * Xc
    n = Y.size
    return float(Y.mean()), float(Y.std(ddof=1) / math.sqrt(n))


def _run(payoff_fn, S0, T, r, q, sigma, n_steps, n_paths, rng, method, batch_size):
    """Simulate in batches; payoff_fn(paths) -> (discounted payoff, discounted control)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    anti = "antithetic" in method
    Ys, Xs = [], []
    for b in _batches(n_paths, batch_size):
        paths = gbm_paths(S0, T, r, q, sigma, n_steps, b, rng, antithetic=anti)
        Y, X = payoff_fn(paths)
        if anti:
            h = b // 2
            Y = 0.5 * (Y[:h] + Y[h:])
            X = 0.5 * (X[:h] + X[h:])
        Ys.append(Y)
        Xs.append(X)
    return np.concatenate(Ys), np.concatenate(Xs)


# --------------------------------------------------------------------------
# European options
# --------------------------------------------------------------------------
def mc_european(S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000,
                rng=None, method="plain", batch_size=500_000):
    """European option; one exact step to maturity is enough (payoff depends on S_T only)."""
    rng = np.random.default_rng() if rng is None else rng
    phi = 1.0 if kind == "call" else -1.0
    disc = math.exp(-r * T)

    def payoff(paths):
        ST = paths[:, -1]
        return disc * np.maximum(phi * (ST - K), 0.0), disc * ST

    Y, X = _run(payoff, S, T, r, q, sigma, 1, n_paths, rng, method, batch_size)
    price, se = _estimate(Y, X, S * math.exp(-q * T), method)
    return MCResult(method, price, se, n_paths)


# --------------------------------------------------------------------------
# Asian options
# --------------------------------------------------------------------------
def geometric_asian_call(S, K, T, r, sigma, q=0.0, n_fixings=252):
    """Closed form for a discretely monitored geometric-average Asian call.

    Fixings at t_i = i T / n, i = 1..n. ln G is normal with
      mean  m = ln S + (r - q - sigma^2/2) T (n+1) / (2n)
      var   v = sigma^2 T (n+1)(2n+1) / (6 n^2)
    and the price is e^{-rT} [e^{m + v/2} N(d1) - K N(d2)],
      d1 = (m - ln K + v) / sqrt(v), d2 = d1 - sqrt(v).
    As n -> infinity this tends to the continuous Kemna-Vorst (1990) formula.
    """
    n = n_fixings
    m = math.log(S) + (r - q - 0.5 * sigma**2) * T * (n + 1) / (2 * n)
    v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6 * n**2)
    d1 = (m - math.log(K) + v) / math.sqrt(v)
    d2 = d1 - math.sqrt(v)
    return math.exp(-r * T) * (math.exp(m + 0.5 * v) * norm.cdf(d1) - K * norm.cdf(d2))


def mc_asian_call(S, K, T, r, sigma, q=0.0, n_fixings=252, n_paths=100_000,
                  rng=None, method="plain", batch_size=20_000):
    """Arithmetic-average Asian call, averaging over fixings 1..n (S0 excluded).
    Control variate: geometric-average Asian call on the same path."""
    rng = np.random.default_rng() if rng is None else rng
    disc = math.exp(-r * T)

    def payoff(paths):
        fix = paths[:, 1:]
        arith = fix.mean(axis=1)
        geo = np.exp(np.log(fix).mean(axis=1))
        return disc * np.maximum(arith - K, 0.0), disc * np.maximum(geo - K, 0.0)

    Y, X = _run(payoff, S, T, r, q, sigma, n_fixings, n_paths, rng, method, batch_size)
    EX = geometric_asian_call(S, K, T, r, sigma, q, n_fixings)
    price, se = _estimate(Y, X, EX, method)
    return MCResult(method, price, se, n_paths)


# --------------------------------------------------------------------------
# Barrier options
# --------------------------------------------------------------------------
def mc_up_and_out_call(S, K, H, T, r, sigma, q=0.0, n_monitor=252, n_paths=100_000,
                       rng=None, method="plain", batch_size=20_000):
    """Discretely monitored up-and-out call (knocked out if any S_{t_i} >= H).
    Control variate: the vanilla European call with the same strike."""
    rng = np.random.default_rng() if rng is None else rng
    disc = math.exp(-r * T)

    def payoff(paths):
        ST = paths[:, -1]
        alive = paths[:, 1:].max(axis=1) < H
        vanilla = np.maximum(ST - K, 0.0)
        return disc * vanilla * alive, disc * vanilla

    Y, X = _run(payoff, S, T, r, q, sigma, n_monitor, n_paths, rng, method, batch_size)
    EX = bs_price(S, K, T, r, sigma, q, "call")
    price, se = _estimate(Y, X, EX, method)
    return MCResult(method, price, se, n_paths)


def up_and_out_call_continuous(S, K, H, T, r, sigma, q=0.0):
    """Closed form for a continuously monitored up-and-out call, H > K, no rebate
    (Merton 1973; Reiner & Rubinstein 1991; Hull's textbook notation)."""
    if S >= H:
        return 0.0
    if H <= K:
        return 0.0
    st = sigma * math.sqrt(T)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    x1 = math.log(S / H) / st + lam * st
    y = math.log(H**2 / (S * K)) / st + lam * st
    y1 = math.log(H / S) / st + lam * st
    c_ui = (S * math.exp(-q * T) * norm.cdf(x1)
            - K * math.exp(-r * T) * norm.cdf(x1 - st)
            - S * math.exp(-q * T) * (H / S) ** (2 * lam) * (norm.cdf(-y) - norm.cdf(-y1))
            + K * math.exp(-r * T) * (H / S) ** (2 * lam - 2)
            * (norm.cdf(-y + st) - norm.cdf(-y1 + st)))
    return float(bs_price(S, K, T, r, sigma, q, "call") - c_ui)


def bgk_discrete_barrier(S, K, H, T, r, sigma, q=0.0, n_monitor=252):
    """Broadie-Glasserman-Kou (1997) approximation for a discrete up barrier:
    price the continuous option with H shifted up by exp(0.5826 sigma sqrt(dt))."""
    beta = 0.5826  # -zeta(1/2) / sqrt(2 pi)
    H_adj = H * math.exp(beta * sigma * math.sqrt(T / n_monitor))
    return up_and_out_call_continuous(S, K, H_adj, T, r, sigma, q)


# --------------------------------------------------------------------------
# Naive reference implementation (benchmarking only)
# --------------------------------------------------------------------------
def mc_asian_call_naive(S, K, T, r, sigma, q=0.0, n_fixings=252, n_paths=10_000, seed=0):
    """Pure-Python double loop over paths and time steps. Same model, no numpy."""
    import random

    rnd = random.Random(seed)
    dt = T / n_fixings
    drift = (r - q - 0.5 * sigma**2) * dt
    vol = sigma * math.sqrt(dt)
    total = 0.0
    for _ in range(n_paths):
        s, acc = S, 0.0
        for _ in range(n_fixings):
            s *= math.exp(drift + vol * rnd.gauss(0.0, 1.0))
            acc += s
        total += max(acc / n_fixings - K, 0.0)
    return math.exp(-r * T) * total / n_paths
