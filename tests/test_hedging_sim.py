from dataclasses import replace

import numpy as np

from options_lab.hedging.simulation import (
    HedgeSimConfig, gamma_theta_approx, hedge_pnl, simulate_paths, summarize,
)

CFG = HedgeSimConfig(n_paths=20_000, seed=11)
DAILY = CFG.steps_per_day


def test_mean_pnl_indistinguishable_from_zero_when_implied_equals_realised():
    S = simulate_paths(CFG)
    stats = summarize(hedge_pnl(S, CFG, DAILY, cost_bps=0.0)["pnl"])
    # Two-sided test at roughly the 1% level.
    assert abs(stats["t_stat"]) < 2.58, stats


def test_pnl_variance_grows_as_rebalancing_gets_less_frequent():
    S = simulate_paths(CFG)
    stds = [hedge_pnl(S, CFG, he)["pnl"].std() for he in (1, DAILY, 5 * DAILY, None)]
    assert all(a < b for a, b in zip(stds, stds[1:])), stds
    # Error scales roughly like 1/sqrt(number of hedges): 4x the hedges -> about half the std.
    assert 1.8 < stds[1] / stds[0] < 2.2


def test_transaction_costs_lower_mean_pnl_more_for_daily_than_weekly():
    S = simulate_paths(CFG)
    drag = {}
    for he in (DAILY, 5 * DAILY):
        drag[he] = hedge_pnl(S, CFG, he, 0.0)["pnl"].mean() - hedge_pnl(S, CFG, he, 2.0)["pnl"].mean()
    assert drag[DAILY] > drag[5 * DAILY] > 0


def test_sign_of_pnl_follows_implied_minus_realised():
    for sr, sign in ((0.15, 1), (0.25, -1)):
        cfg = replace(CFG, sigma_real=sr)
        S = simulate_paths(cfg)
        assert sign * hedge_pnl(S, cfg, DAILY)["pnl"].mean() > 0


def test_gamma_theta_approximation_tracks_actual_pnl():
    S = simulate_paths(CFG)
    pnl = hedge_pnl(S, CFG, DAILY)["pnl"]
    approx = gamma_theta_approx(S, CFG, DAILY)
    assert np.corrcoef(pnl, approx)[0, 1] > 0.95
    slope = np.polyfit(approx, pnl, 1)[0]
    assert 0.9 < slope < 1.1


def test_risk_neutral_mean_is_zero_even_without_rebalancing():
    # Under Q the discounted option price is a martingale, so E[PnL] = 0 at any frequency.
    cfg = replace(CFG, mu=CFG.r, n_paths=50_000)
    S = simulate_paths(cfg)
    for he in (DAILY, None):
        assert abs(summarize(hedge_pnl(S, cfg, he)["pnl"])["t_stat"]) < 3
