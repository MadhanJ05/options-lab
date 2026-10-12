"""Monte Carlo study of delta-hedging a short call under GBM.

A dealer sells one call at the Black-Scholes price using implied vol sigma_imp, then
holds delta(sigma_imp) shares, rebalancing every `hedge_every` simulation steps. The
stock follows GBM with real-world drift mu and realised vol sigma_real. Cash earns r.
Transaction costs are cost_bps of traded notional on every share trade, including
the initial hedge and the final unwind.

Hedged P&L at expiry (per option, in currency):
    PnL = cash_T + delta_{last} * S_T - (cost of unwind) - max(S_T - K, 0)

Gamma-theta approximation (continuous-time limit, here evaluated per step):
    PnL ~= sum_t 0.5 * Gamma_t * S_t^2 * (sigma_imp^2 * dt - r_t^2) * e^{r (T - t)}
where r_t = ln(S_{t+1} / S_t) is the realised log-return over step t.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from options_lab.pricing.black_scholes import bs_delta, bs_gamma, bs_price


@dataclass(frozen=True)
class HedgeSimConfig:
    S0: float = 100.0
    K: float = 100.0
    days: int = 30               # calendar days to expiry
    trading_days: int = 21       # trading days in that window
    steps_per_day: int = 4       # simulation grid finer than daily
    r: float = 0.04
    mu: float = 0.08
    sigma_imp: float = 0.20
    sigma_real: float = 0.20
    n_paths: int = 20_000
    seed: int = 7

    @property
    def T(self) -> float:
        return self.days / 365.0

    @property
    def n_steps(self) -> int:
        return self.trading_days * self.steps_per_day


def simulate_paths(cfg: HedgeSimConfig, rng=None) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    dt = cfg.T / cfg.n_steps
    Z = rng.standard_normal((cfg.n_paths, cfg.n_steps))
    inc = (cfg.mu - 0.5 * cfg.sigma_real**2) * dt + cfg.sigma_real * math.sqrt(dt) * Z
    S = np.empty((cfg.n_paths, cfg.n_steps + 1))
    S[:, 0] = cfg.S0
    S[:, 1:] = cfg.S0 * np.exp(np.cumsum(inc, axis=1))
    return S


def hedge_pnl(S: np.ndarray, cfg: HedgeSimConfig, hedge_every: int | None, cost_bps: float = 0.0):
    """Hedged P&L for each path.

    hedge_every: rebalance every this many simulation steps; None = hedge once at
                 inception and never rebalance; 0 = no hedge at all (naked short).
    Returns dict with arrays 'pnl', 'cost', 'n_trades' and the premium.
    """
    n_paths, n_cols = S.shape
    n = n_cols - 1
    dt = cfg.T / n
    cost_rate = cost_bps * 1e-4
    premium = float(bs_price(cfg.S0, cfg.K, cfg.T, cfg.r, cfg.sigma_imp))

    delta = np.zeros(n_paths)
    cash = np.full(n_paths, premium)
    costs = np.zeros(n_paths)
    trades = 0
    for i in range(n):
        tau = cfg.T - i * dt
        rebalance = (i == 0 and hedge_every != 0) or (hedge_every and i % hedge_every == 0)
        if rebalance:
            new_delta = bs_delta(S[:, i], cfg.K, tau, cfg.r, cfg.sigma_imp)
            traded = new_delta - delta
            cash -= traded * S[:, i]
            c = cost_rate * np.abs(traded) * S[:, i]
            cash -= c
            costs += c
            delta = new_delta
            trades += 1
        cash *= math.exp(cfg.r * dt)

    ST = S[:, -1]
    unwind_cost = cost_rate * np.abs(delta) * ST
    costs += unwind_cost
    pnl = cash + delta * ST - unwind_cost - np.maximum(ST - cfg.K, 0.0)
    return {"pnl": pnl, "cost": costs, "n_rebalances": trades, "premium": premium}


def gamma_theta_approx(S: np.ndarray, cfg: HedgeSimConfig, hedge_every: int = 1) -> np.ndarray:
    """sum over hedge intervals of 0.5 Gamma S^2 (sigma_imp^2 dt - r_t^2), grown to T."""
    n = S.shape[1] - 1
    dt = cfg.T / n
    idx = np.arange(0, n, hedge_every)
    total = np.zeros(S.shape[0])
    for i in idx:
        j = min(i + hedge_every, n)
        h = (j - i) * dt
        tau = cfg.T - i * dt
        g = bs_gamma(S[:, i], cfg.K, tau, cfg.r, cfg.sigma_imp)
        ret = np.log(S[:, j] / S[:, i])
        total += 0.5 * g * S[:, i] ** 2 * (cfg.sigma_imp**2 * h - ret**2) * math.exp(cfg.r * (cfg.T - i * dt))
    return total


def summarize(pnl: np.ndarray) -> dict:
    n = pnl.size
    m, s = float(pnl.mean()), float(pnl.std(ddof=1))
    return {"mean": m, "std": s, "t_stat": m / (s / math.sqrt(n)),
            "q05": float(np.quantile(pnl, 0.05)), "q95": float(np.quantile(pnl, 0.95))}
