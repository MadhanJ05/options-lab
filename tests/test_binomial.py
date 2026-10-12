import numpy as np
import pytest

from options_lab.pricing.binomial import crr_price, crr_price_naive
from options_lab.pricing.black_scholes import bs_price

PARAMS = [
    (100, 100, 1.0, 0.05, 0.20, 0.00),
    (100, 90, 0.5, 0.03, 0.30, 0.02),
    (100, 115, 2.0, 0.02, 0.25, 0.01),
]


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S,K,T,r,sigma,q", PARAMS)
def test_european_converges_to_bs(S, K, T, r, sigma, q, kind):
    # CRR error is O(1/N); with N=2000 the error is a few tenths of a cent.
    tree = crr_price(S, K, T, r, sigma, q, kind, steps=2000)
    assert tree == pytest.approx(bs_price(S, K, T, r, sigma, q, kind), abs=5e-3)


@pytest.mark.parametrize("K", [70, 90, 100, 110, 130])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_american_put_never_below_european(K, q):
    for sigma in (0.1, 0.3, 0.6):
        amer = crr_price(100, K, 1.0, 0.05, sigma, q, "put", steps=400, american=True)
        euro = crr_price(100, K, 1.0, 0.05, sigma, q, "put", steps=400, american=False)
        assert amer >= euro - 1e-12


def test_american_call_no_dividend_equals_european():
    # Merton (1973): never optimal to exercise a call early on a non-dividend stock.
    amer = crr_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", steps=500, american=True)
    euro = crr_price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", steps=500, american=False)
    assert amer == pytest.approx(euro, abs=1e-10)


def test_vectorised_matches_naive():
    for american in (False, True):
        a = crr_price(100, 105, 1.0, 0.05, 0.25, 0.01, "put", steps=200, american=american)
        b = crr_price_naive(100, 105, 1.0, 0.05, 0.25, 0.01, "put", steps=200, american=american)
        assert a == pytest.approx(b, abs=1e-10)
