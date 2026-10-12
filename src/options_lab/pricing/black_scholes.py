"""Black-Scholes-Merton pricing and analytic Greeks with continuous dividend yield.

Conventions
-----------
S      spot price
K      strike
T      time to expiry in years
r      continuously compounded risk-free rate
q      continuous dividend yield
sigma  volatility (annualised)
kind   "call" or "put"

Greeks are partial derivatives in natural units:
  delta = dV/dS, gamma = d2V/dS2, vega = dV/dsigma (per 1.00 of vol),
  theta = dV/dt (per year, i.e. -dV/dT), rho = dV/dr (per 1.00 of rate).
All functions accept numpy arrays and broadcast.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, q, sigma):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _sign(kind: str) -> int:
    if kind == "call":
        return 1
    if kind == "put":
        return -1
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """Black-Scholes-Merton price. phi = +1 for calls, -1 for puts:
    V = phi * [S e^{-qT} N(phi d1) - K e^{-rT} N(phi d2)]."""
    phi = _sign(kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return phi * (S * np.exp(-q * T) * norm.cdf(phi * d1)
                  - K * np.exp(-r * T) * norm.cdf(phi * d2))


def bs_delta(S, K, T, r, sigma, q=0.0, kind="call"):
    phi = _sign(kind)
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return phi * np.exp(-q * T) * norm.cdf(phi * d1)


def bs_gamma(S, K, T, r, sigma, q=0.0, kind="call"):
    # Same for calls and puts (put-call parity is linear in S).
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma, q=0.0, kind="call"):
    # Same for calls and puts.
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S, K, T, r, sigma, q=0.0, kind="call"):
    """dV/dt with calendar time t moving forward (= -dV/dT), per year."""
    phi = _sign(kind)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    decay = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    return (decay
            + phi * q * S * np.exp(-q * T) * norm.cdf(phi * d1)
            - phi * r * K * np.exp(-r * T) * norm.cdf(phi * d2))


def bs_rho(S, K, T, r, sigma, q=0.0, kind="call"):
    phi = _sign(kind)
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    return phi * K * T * np.exp(-r * T) * norm.cdf(phi * d2)


def bs_greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict:
    args = (S, K, T, r, sigma, q, kind)
    return {
        "price": bs_price(*args),
        "delta": bs_delta(*args),
        "gamma": bs_gamma(*args),
        "vega": bs_vega(*args),
        "theta": bs_theta(*args),
        "rho": bs_rho(*args),
    }
