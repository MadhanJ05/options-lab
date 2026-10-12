"""Implied volatility: Newton-Raphson on vega with a Brent's-method fallback.

Edge cases handled
------------------
* Prices outside the no-arbitrage bounds -> NaN, status "below_lower_bound"/"above_upper_bound".
* Prices at (numerically) intrinsic value -> NaN, status "at_intrinsic": any sigma near
  zero fits, so the vol is not identified.
* Deep ITM options are converted to the equivalent OTM option by put-call parity
  (same implied vol, much better conditioned).
* Tiny vega / Newton stepping out of range -> Brent on [SIGMA_LO, SIGMA_HI].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from options_lab.pricing.black_scholes import bs_price, bs_vega

SIGMA_LO, SIGMA_HI = 1e-6, 10.0


@dataclass
class IVResult:
    sigma: float
    status: str      # "newton", "brent", or a failure reason
    iterations: int


def no_arbitrage_bounds(S, K, T, r, q=0.0, kind="call"):
    fwd_s = S * math.exp(-q * T)     # PV of the stock net of dividends
    pv_k = K * math.exp(-r * T)
    if kind == "call":
        return max(fwd_s - pv_k, 0.0), fwd_s
    return max(pv_k - fwd_s, 0.0), pv_k


def implied_vol_detail(price, S, K, T, r, q=0.0, kind="call", tol=1e-12, max_iter=50) -> IVResult:
    """tol is on the Newton step in sigma. A price-residual stop at machine
    precision (1e-15 * S) is also used, because tiny OTM prices make any fixed
    absolute price tolerance too loose."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if not (np.isfinite(price) and T > 0 and S > 0 and K > 0):
        return IVResult(math.nan, "invalid_input", 0)

    lo, hi = no_arbitrage_bounds(S, K, T, r, q, kind)
    abs_eps = 1e-12 * max(S, K)
    if price < lo - abs_eps:
        return IVResult(math.nan, "below_lower_bound", 0)
    if price > hi + abs_eps:
        return IVResult(math.nan, "above_upper_bound", 0)

    # Put-call parity: C - P = S e^{-qT} - K e^{-rT}. Work with the OTM side.
    parity = S * math.exp(-q * T) - K * math.exp(-r * T)
    if kind == "call" and parity > 0:
        price, kind = price - parity, "put"
    elif kind == "put" and parity < 0:
        price, kind = price + parity, "call"
    # Now the option is OTM (or ATM-forward); its lower bound is 0.
    _, hi = no_arbitrage_bounds(S, K, T, r, q, kind)
    if price <= abs_eps:
        return IVResult(math.nan, "at_intrinsic", 0)
    if price >= hi - abs_eps:
        return IVResult(math.nan, "at_upper_bound", 0)

    def f(sig):
        return float(bs_price(S, K, T, r, sig, q, kind)) - price

    # Initial guess: Manaster-Koehler point of maximum vega, floored by the
    # Brenner-Subrahmanyam ATM approximation.
    F = S * math.exp((r - q) * T)
    mk = math.sqrt(2.0 * abs(math.log(F / K)) / T)
    bs_atm = math.sqrt(2.0 * math.pi / T) * price / (S * math.exp(-q * T))
    sig = min(max(mk, bs_atm, 0.05), 5.0)

    for i in range(1, max_iter + 1):
        diff = f(sig)
        if abs(diff) < 1e-15 * S:
            return IVResult(sig, "newton", i)
        vega = float(bs_vega(S, K, T, r, sig, q))
        if vega < 1e-12 * S:
            break
        step = diff / vega
        new = sig - step
        if not (SIGMA_LO < new < SIGMA_HI):
            break
        if abs(step) < tol * max(1.0, sig):
            return IVResult(new, "newton", i)
        sig = new

    # Fallback: f is increasing in sigma, f(LO) < 0 < f(HI) for interior prices.
    f_lo, f_hi = f(SIGMA_LO), f(SIGMA_HI)
    if f_lo > 0 or f_hi < 0:
        return IVResult(math.nan, "no_bracket", max_iter)
    root, info = brentq(f, SIGMA_LO, SIGMA_HI, xtol=1e-14, rtol=1e-12, full_output=True)
    return IVResult(float(root), "brent", info.iterations)


def implied_vol(price, S, K, T, r, q=0.0, kind="call", **kw) -> float:
    return implied_vol_detail(price, S, K, T, r, q, kind, **kw).sigma


def implied_vol_array(prices, S, K, T, r, q, kinds):
    """Element-wise IV for arrays (e.g. an option chain). Inputs broadcast."""
    prices, S, K, T, r, q, kinds = np.broadcast_arrays(prices, S, K, T, r, q, kinds)
    out = np.empty(prices.shape)
    status = np.empty(prices.shape, dtype=object)
    for idx in np.ndindex(prices.shape):
        res = implied_vol_detail(float(prices[idx]), float(S[idx]), float(K[idx]),
                                 float(T[idx]), float(r[idx]), float(q[idx]), str(kinds[idx]))
        out[idx], status[idx] = res.sigma, res.status
    return out, status
