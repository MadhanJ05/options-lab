import math

import numpy as np
import pytest

from options_lab.pricing.black_scholes import bs_price
from options_lab.pricing.monte_carlo import (
    METHODS,
    bgk_discrete_barrier,
    geometric_asian_call,
    gbm_paths,
    mc_asian_call,
    mc_european,
    mc_up_and_out_call,
)

SEED = 42

EURO_CASES = [
    (100, 100, 1.0, 0.05, 0.20, 0.00, "call"),
    (100, 100, 1.0, 0.05, 0.20, 0.00, "put"),
    (100, 85, 0.5, 0.03, 0.30, 0.02, "call"),
    (100, 120, 2.0, 0.02, 0.25, 0.01, "put"),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("S,K,T,r,sigma,q,kind", EURO_CASES)
def test_mc_european_within_3se_of_bs(S, K, T, r, sigma, q, kind, method):
    res = mc_european(S, K, T, r, sigma, q, kind, n_paths=200_000,
                      rng=np.random.default_rng(SEED), method=method)
    bs = bs_price(S, K, T, r, sigma, q, kind)
    assert abs(res.price - bs) < 3 * res.se, (res, bs)


def test_geometric_asian_closed_form_matches_simulation():
    # Validates the control variate's expectation, which the CV estimator relies on.
    S, K, T, r, sigma, q, n = 100, 100, 1.0, 0.05, 0.2, 0.01, 52
    paths = gbm_paths(S, T, r, q, sigma, n, 200_000, np.random.default_rng(SEED))
    geo = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    Y = math.exp(-r * T) * np.maximum(geo - K, 0.0)
    se = Y.std(ddof=1) / math.sqrt(Y.size)
    assert abs(Y.mean() - geometric_asian_call(S, K, T, r, sigma, q, n)) < 3 * se


def test_asian_estimators_agree_and_cv_reduces_variance():
    kw = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0, n_fixings=52, n_paths=50_000)
    plain = mc_asian_call(**kw, rng=np.random.default_rng(SEED), method="plain")
    cv = mc_asian_call(**kw, rng=np.random.default_rng(SEED), method="control")
    assert abs(plain.price - cv.price) < 3 * math.hypot(plain.se, cv.se)
    assert cv.se < plain.se / 10


def test_barrier_matches_bgk_and_is_below_vanilla():
    S, K, H, T, r, sigma = 100, 100, 130, 1.0, 0.05, 0.2
    res = mc_up_and_out_call(S, K, H, T, r, sigma, n_monitor=252, n_paths=100_000,
                             rng=np.random.default_rng(SEED), method="antithetic")
    # BGK is an approximation, so allow its own small error on top of MC noise.
    assert abs(res.price - bgk_discrete_barrier(S, K, H, T, r, sigma, 0.0, 252)) < 3 * res.se + 0.02
    assert res.price < bs_price(S, K, T, r, sigma)


def test_antithetic_paths_are_mirrored():
    p = gbm_paths(100, 1.0, 0.05, 0.0, 0.2, 10, 8, np.random.default_rng(0), antithetic=True)
    log_inc = np.diff(np.log(p), axis=1)
    drift = (0.05 - 0.5 * 0.04) * 0.1
    np.testing.assert_allclose(log_inc[:4] - drift, -(log_inc[4:] - drift), atol=1e-12)
