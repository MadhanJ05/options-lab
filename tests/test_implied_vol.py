import math

import numpy as np
import pytest

from options_lab.iv.solver import implied_vol, implied_vol_array, implied_vol_detail
from options_lab.pricing.black_scholes import bs_price, bs_vega

GRID = [
    (S, K, T, sigma, kind)
    for S in (100.0,)
    for K in (50.0, 80.0, 95.0, 100.0, 105.0, 120.0, 160.0)
    for T in (7 / 365, 0.25, 1.0, 3.0)
    for sigma in (0.05, 0.15, 0.3, 0.8, 2.0)
    for kind in ("call", "put")
]


@pytest.mark.parametrize("S,K,T,sigma,kind", GRID)
def test_round_trip(S, K, T, sigma, kind):
    r, q = 0.04, 0.015
    price = float(bs_price(S, K, T, r, sigma, q, kind))
    # Identifiability: if the time value is below double precision relative to
    # the price, no solver can recover sigma. Skip those (documented limitation).
    otm_value = min(price, float(bs_price(S, K, T, r, sigma, q, "put" if kind == "call" else "call")))
    if otm_value < 1e-10 or float(bs_vega(S, K, T, r, sigma, q)) < 1e-8:
        pytest.skip("vol not identifiable at double precision")
    res = implied_vol_detail(price, S, K, T, r, q, kind)
    # Achievable accuracy = price rounding error / vega (the condition number).
    # BS prices carry ~1e-13 * S absolute rounding error from cancellation.
    vega = float(bs_vega(S, K, T, r, sigma, q))
    tol = max(1e-6 * sigma, 1e-13 * S / vega)
    assert abs(res.sigma - sigma) <= tol, res


def test_outside_bounds_returns_nan():
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.05, 0.0
    assert implied_vol_detail(S + 1, S, K, T, r, q, "call").status == "above_upper_bound"
    intrinsic_lb = S - K * math.exp(-r * T)
    assert implied_vol_detail(intrinsic_lb - 0.5, S, K, T, r, q, "call").status == "below_lower_bound"
    assert implied_vol_detail(-0.01, S, 120.0, T, r, q, "put").status == "below_lower_bound"


def test_at_intrinsic_is_not_identified():
    S, K, T, r = 100.0, 60.0, 0.25, 0.05
    lb = S - K * math.exp(-r * T)
    res = implied_vol_detail(lb, S, K, T, r, 0.0, "call")
    assert math.isnan(res.sigma) and res.status == "at_intrinsic"


def test_deep_itm_put_uses_parity_and_recovers_sigma():
    S, K, T, r, sigma = 100.0, 200.0, 0.5, 0.05, 0.25
    price = float(bs_price(S, K, T, r, sigma, 0.0, "put"))
    assert implied_vol(price, S, K, T, r, 0.0, "put") == pytest.approx(sigma, rel=1e-6)


def test_brent_fallback_when_newton_does_not_converge():
    # Starve Newton of iterations so the fallback path must produce the answer.
    S, K, T, r, sigma = 100.0, 130.0, 0.5, 0.03, 0.35
    price = float(bs_price(S, K, T, r, sigma))
    res = implied_vol_detail(price, S, K, T, r, max_iter=1)
    assert res.status == "brent"
    assert res.sigma == pytest.approx(sigma, rel=1e-8)


def test_vectorised_wrapper():
    K = np.array([90.0, 100.0, 110.0])
    kinds = np.array(["put", "call", "call"])
    sig = np.array([0.25, 0.2, 0.18])
    prices = np.array([bs_price(100, k, 0.5, 0.03, s, 0.01, kd) for k, s, kd in zip(K, sig, kinds)])
    iv, status = implied_vol_array(prices, 100.0, K, 0.5, 0.03, 0.01, kinds)
    np.testing.assert_allclose(iv, sig, rtol=1e-7)


def test_static_arbitrage_filter_removes_spike():
    import pandas as pd

    from options_lab.iv.chain import drop_strike_arbitrage
    # Mirrors the real 602 put in the 2026-10-09 snapshot.
    g = pd.DataFrame({"strike": [595.0, 600, 602, 605, 610], "mid": [3.80, 3.975, 10.555, 4.16, 4.35],
                      "kind": "put", "T": 0.47})
    out = drop_strike_arbitrage(g, 0.04)
    assert list(out.strike) == [595.0, 600, 605, 610]


def test_static_arbitrage_filter_removes_endpoint_outlier():
    import pandas as pd

    from options_lab.iv.chain import drop_strike_arbitrage
    # Mirrors the real 1270 call (Dec 2027) priced above the 1200/1210 calls.
    g = pd.DataFrame({"strike": [1120.0, 1125, 1200, 1210, 1270], "mid": [0.89, 0.86, 0.42, 0.39, 0.615],
                      "kind": "call", "T": 1.19})
    out = drop_strike_arbitrage(g, 0.04)
    assert list(out.strike) == [1120.0, 1125, 1200, 1210]
