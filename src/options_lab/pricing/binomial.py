"""Cox-Ross-Rubinstein (1979) binomial tree for European and American options."""
from __future__ import annotations

import numpy as np


def crr_price(S, K, T, r, sigma, q=0.0, kind="call", steps=500, american=False):
    """Price an option on a recombining CRR tree.

    u = exp(sigma sqrt(dt)), d = 1/u, p = (exp((r-q)dt) - d) / (u - d).
    Backward induction is vectorised across nodes within each time slice.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability {p:.4f} outside (0,1); use more steps")
    disc = np.exp(-r * dt)
    phi = 1.0 if kind == "call" else -1.0

    # Terminal stock prices: S u^j d^(N-j) for j = 0..N (j = number of up moves)
    j = np.arange(steps + 1)
    ST = S * u**j * d ** (steps - j)
    V = np.maximum(phi * (ST - K), 0.0)

    for n in range(steps - 1, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            Sn = S * u ** j[: n + 1] * d ** (n - j[: n + 1])
            V = np.maximum(V, phi * (Sn - K))
    return float(V[0])


def crr_price_naive(S, K, T, r, sigma, q=0.0, kind="call", steps=500, american=False):
    """Same tree with explicit double loop. Used only for runtime benchmarking."""
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    phi = 1.0 if kind == "call" else -1.0
    V = [max(phi * (S * u**j * d ** (steps - j) - K), 0.0) for j in range(steps + 1)]
    for n in range(steps - 1, -1, -1):
        for j in range(n + 1):
            cont = disc * (p * V[j + 1] + (1.0 - p) * V[j])
            if american:
                cont = max(cont, phi * (S * u**j * d ** (n - j) - K))
            V[j] = cont
    return V[0]
