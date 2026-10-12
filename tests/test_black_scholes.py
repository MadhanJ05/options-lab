import numpy as np
import pytest

from options_lab.pricing.black_scholes import bs_greeks, bs_price

CASES = [
    # S, K, T, r, sigma, q
    (100, 100, 1.0, 0.05, 0.20, 0.00),
    (100, 80, 0.5, 0.03, 0.35, 0.02),
    (100, 130, 2.0, 0.01, 0.15, 0.015),
    (50, 55, 0.1, 0.04, 0.60, 0.00),
]


@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_put_call_parity(S, K, T, r, sigma, q):
    # C - P = S e^{-qT} - K e^{-rT}
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_known_value():
    # Hull textbook-style benchmark: S=K=100, T=1, r=5%, sigma=20% -> 10.4506
    assert bs_price(100, 100, 1.0, 0.05, 0.2) == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,sigma,q", CASES)
def test_greeks_match_finite_differences(S, K, T, r, sigma, q, kind):
    g = bs_greeks(S, K, T, r, sigma, q, kind)

    def V(S=S, T=T, r=r, sigma=sigma):
        return bs_price(S, K, T, r, sigma, q, kind)

    hS, hv, hT, hr = 1e-3 * S, 1e-4, 1e-5, 1e-5
    fd = {
        "delta": (V(S=S + hS) - V(S=S - hS)) / (2 * hS),
        "gamma": (V(S=S + hS) - 2 * V() + V(S=S - hS)) / hS**2,
        "vega": (V(sigma=sigma + hv) - V(sigma=sigma - hv)) / (2 * hv),
        "theta": -(V(T=T + hT) - V(T=T - hT)) / (2 * hT),
        "rho": (V(r=r + hr) - V(r=r - hr)) / (2 * hr),
    }
    for name, approx in fd.items():
        assert g[name] == pytest.approx(approx, rel=1e-4, abs=1e-6), name
