# options-lab

An options pricing and volatility risk engine in Python. It does three things:

1. **Prices options three ways** (Black-Scholes-Merton, a Cox-Ross-Rubinstein tree, and Monte Carlo) and shows that they agree.
2. **Extracts implied volatility** from a real SPY option chain (snapshot of quotes at the close of **2026-10-09**) and plots the smile and the ATM term structure.
3. **Measures the variance risk premium (VRP).** A simulation shows why a delta-hedged short option earns implied minus realised variance. A historical study then sells a 30-day ATM SPY call every month from January 2007 to December 2025, prices it at VIX, and hedges it daily.

All results are in [`report/report.md`](report/report.md). Every number and figure there is produced by the scripts in `scripts/` from cached data in `data/`, with fixed random seeds.

---

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # or: uv venv --python 3.12
pip install -r requirements.txt
pip install -e .

pytest                        # 310 tests, ~3 s
python scripts/run_all.py     # regenerates results/ from cached data, ~20 s, no network
```

To download fresh market data, run `python scripts/run_all.py --refresh`. This writes a new dated chain snapshot to `data/`, so the results will change.

| Script | What it produces |
|---|---|
| `scripts/run_pricing.py` | three-way pricing table, CRR convergence, American premium, MC variance reduction, benchmarks |
| `scripts/run_iv.py [--refresh]` | cleaned SPY chain, IV smile, ATM term structure, cleaning funnel |
| `scripts/run_hedging_sim.py` | GBM delta-hedging study |
| `scripts/run_historical.py [--refresh]` | SPY/VIX monthly hedged-call study, VRP statistics |

## Repository layout

```
src/options_lab/
  pricing/black_scholes.py   BSM price + analytic Greeks (continuous dividend yield)
  pricing/binomial.py        CRR tree, European & American, vectorised + naive reference
  pricing/monte_carlo.py     GBM paths; plain / antithetic / control-variate estimators;
                             arithmetic Asian, up-and-out barrier, geometric Asian closed form,
                             continuous barrier closed form + BGK discrete correction
  iv/solver.py               Newton-Raphson IV with Brent fallback and edge-case handling
  iv/chain.py                yfinance download, cleaning, parity-implied forwards, IVs, term structure
  hedging/simulation.py      Monte Carlo delta-hedging study
  hedging/historical.py      SPY/VIX/IRX data, trade schedule, lookahead-free daily hedge
  plotting.py                shared figure style
scripts/                     one script per part + run_all.py
tests/                       pytest suite
data/                        cached market data (dated) + JSON metadata
results/                     figures (.png) and tables (.csv / .json)
report/report.md             write-up
```

---

## Methodology and formulas

Notation: S is spot, K is strike, T is years to expiry, r is the continuously compounded risk-free rate, q is the continuous dividend yield, σ is volatility, and N(·) is the standard normal CDF.

### Part 1: Pricing engine

**Black-Scholes-Merton** (Black & Scholes 1973; Merton 1973). Let φ = +1 for a call and −1 for a put. Then

```
V  = φ [ S e^{-qT} N(φ d1) − K e^{-rT} N(φ d2) ]
d1 = [ln(S/K) + (r − q + σ²/2) T] / (σ √T),   d2 = d1 − σ √T
```

The Greeks are given in natural units. Vega and rho are per 1.00 change in σ or r (divide by 100 to get "per 1 point"). Theta is per year of calendar time (divide by 365 to get per day).

```
delta = φ e^{-qT} N(φ d1)              gamma = e^{-qT} n(d1) / (S σ √T)
vega  = S e^{-qT} n(d1) √T             rho   = φ K T e^{-rT} N(φ d2)
theta = −S e^{-qT} n(d1) σ / (2√T) + φ q S e^{-qT} N(φ d1) − φ r K e^{-rT} N(φ d2)
```

**CRR binomial tree** (Cox, Ross & Rubinstein 1979). Each step moves the price up by u = e^{σ√Δt} or down by d = 1/u. The risk-neutral up-probability is p = (e^{(r−q)Δt} − d)/(u − d). Prices are found by backward induction, V = e^{−rΔt}[p V_up + (1−p) V_down]. For an American option, each node takes the larger of that value and the exercise value. Each time slice is a single numpy operation. The error falls like O(1/N) and alternates sign between even and odd N.

**Monte Carlo** (Boyle 1977). Paths are simulated with the exact lognormal step, ln S_{t+Δt} = ln S_t + (r − q − σ²/2)Δt + σ√Δt Z, so there is no discretisation bias at the monitoring dates. The paths are vectorised and generated in batches to limit memory use.

* *Antithetic variates*: Z and −Z form a pair. The standard error is computed from the **pair averages**, because the two halves of a pair are not independent.
* *Control variates*: Ŷ = Y − b(X − E[X]), with b = Cov(Y,X)/Var(X) estimated from the same sample. Estimating b in-sample adds a bias of order 1/n, which is negligible at these path counts.
  * European options use the discounted terminal stock as the control, with E = S e^{−qT}.
  * The arithmetic Asian uses the **geometric Asian** call as the control (Kemna & Vorst 1990). The closed form below is for discrete fixings t_i = iT/n. ln G is normal with mean m = ln S + (r − q − σ²/2)T(n+1)/(2n) and variance v = σ²T(n+1)(2n+1)/(6n²), so the price is e^{−rT}[e^{m+v/2}N(d1) − K N(d2)], where d1 = (m − ln K + v)/√v and d2 = d1 − √v.
  * The up-and-out call uses the vanilla call as the control. This control is weak, and the report explains why.
* *Variance reduction factor* = (SE_plain / SE_method)² at the **same number of paths**. It tells you how many times more plain paths would be needed to reach the same accuracy.
* The barrier price is checked against the continuous-barrier closed form (Merton 1973; Reiner & Rubinstein 1991) after shifting the barrier by e^{0.5826 σ √Δt}. This is the Broadie-Glasserman-Kou (1997) continuity correction for discrete monitoring.

### Part 2: Implied volatility

* **Solver.** Newton-Raphson iterates σ ← σ − (BS(σ) − price)/vega. It starts from the larger of the Manaster-Koehler (1982) point √(2|ln(F/K)|/T) and the Brenner-Subrahmanyam (1988) ATM approximation. If vega is tiny or a step leaves [1e-6, 10], the solver falls back to Brent's method on that bracket. Brent cannot fail to converge there, because BS is strictly increasing in σ.
* **Edge cases.** Prices outside the no-arbitrage bounds return NaN with a reason code (`below_lower_bound`, `above_upper_bound`). The bounds are max(S e^{−qT} − K e^{−rT}, 0) ≤ C ≤ S e^{−qT} for calls and max(K e^{−rT} − S e^{−qT}, 0) ≤ P ≤ K e^{−rT} for puts. Deep ITM quotes are converted to the OTM option with the same implied vol by put-call parity, which keeps the solve well conditioned. Prices at intrinsic value return `at_intrinsic`, because no unique σ exists there.
* **Precision limit.** IV accuracy is roughly price accuracy divided by vega. The round-trip tests use exactly this tolerance.
* **Chain cleaning.** All thresholds were fixed before looking at any results:
  * bid > 0 and ask > bid; quotes are valued at the bid/ask mid
  * relative spread ≤ 25%, mid ≥ $0.05, volume ≥ 10, open interest ≥ 100, 5 to 760 days to expiry
  * OTM side only: puts for K < F, calls for K ≥ F
  * a static no-arbitrage check across strikes, requiring 0 ≤ ∂P/∂K ≤ e^{−rT} and −e^{−rT} ≤ ∂C/∂K ≤ 0. It removes the **fewest** quotes needed, using an exact dynamic program, so that no vertical-spread arbitrage remains.
* **Forward and dividends.** Each expiry's forward comes from put-call parity, F = K + e^{rT}(C − P), taking the median over the 10 strikes nearest spot. The implied dividend yield q = r − ln(F/S)/T is then used in BS. Log-moneyness is k = ln(K/F).
* **Rate.** The rate comes from ^IRX, the 13-week T-bill discount yield d. It is converted with r = −ln(1 − d·91/360)/(91/365).
* **ATM IV.** For each expiry, ATM IV is interpolated linearly in k between the nearest quotes on either side of the forward. Expiries without quotes within |k| < 0.05 on both sides are skipped rather than extrapolated.

### Part 3: Delta hedging and the variance risk premium

**Gamma-theta decomposition.** Suppose you are short an option priced at implied vol σ_i and delta-hedged with BS delta at σ_i. Over each interval the hedged P&L is approximately

```
dPnL ≈ ½ Γ S² (σ_i² dt − (dS/S)²)
```

The short position collects theta at the implied variance and pays out gamma on the realised squared return. Summed over the path, the P&L is the gamma-weighted difference between implied and realised variance. This is the variance risk premium seen through one option. Because the weight is ½ΓS², *where* the variance occurs relative to the strike matters as well as how much there is.

**Simulation.** The setup is a 30-calendar-day ATM call with S = K = 100, σ_i = 20%, r = 4%, and real-world drift μ = 8%. There are 21 trading days, simulated at 4 points per day, over 20,000 paths. Accounting uses a self-financing cash account. Transaction costs are 2 bps of traded notional, charged on every trade including the first hedge and the final unwind. Rebalancing is daily, weekly, or never after inception.

**Historical study.**

| Item | Choice |
|---|---|
| Entry | first trading day of each month, Jan 2007 to Dec 2025 (228 trades) |
| Expiry | last trading day ≤ entry + 30 calendar days |
| Strike | SPY close at entry (ATM spot) |
| Implied vol | VIX close at entry / 100, used for both pricing and hedge deltas for the whole trade |
| Rate | ^IRX at each date, converted as above, forward-filled on bond-market holidays |
| Dividends | q = trailing 365-day dividends / spot at entry; the hedge earns Δ × dividend on ex-dates |
| Hedge | BS delta at each daily close, unwound at expiry; the option settles at the expiry close |
| P&L units | % of entry spot, i.e. $ per $100 of SPY notional. Cumulative P&L is a simple sum, not compounded |
| Realised vol | √(Σ r_t² / T), with daily log returns of dividend-adjusted closes over the option's life and T in calendar years. This is the quantity in the gamma-theta formula |

**No lookahead.** Every decision on date t reads only rows dated ≤ t. The trading calendar used to set expiry is published in advance. `tests/test_no_lookahead.py` checks this in two ways. First, it truncates the data at every date inside a trade and confirms the decisions up to that date are unchanged. Second, it scrambles all prices, VIX, and rates after a cut-off and confirms nothing before the cut-off changes. I also planted a deliberate bug (a delta computed from the next day's close) and confirmed both tests fail on it.

---

## Limitations

* **VIX is a proxy.** VIX is computed from S&P 500 index (SPX) options, not SPY. It is also a model-free 30-day *variance-swap* rate built from a strip of OTM options, so it includes the put skew. It is **not** an ATM implied vol, and it usually sits above ATM IV. On the snapshot date, VIX closed at 14.84 while SPY's own 30-day ATM IV was 12.05%, a gap of 2.8 vol points. This matters for the two historical results in different ways:
  * *VIX − realised vol* is a legitimate measure of the VRP. It is the variance-swap definition used by Carr & Wu (2009).
  * *Hedged ATM call P&L priced at VIX* is likely **overstated**. The report gives a sensitivity table at VIX − 1, − 2, and − 3. At VIX − 3 the mean P&L is no longer statistically significant. Only one day of the VIX/ATM gap is measured here. A proper fix would use a historical SPY ATM IV series (for example OptionMetrics), which is not freely available.
* **The historical option premium is theoretical.** The option is assumed to trade at the BS-at-VIX price, with no option bid/ask. A real seller would receive less.
* **American exercise.** SPY options are American, while the IVs are European BS IVs. Using OTM options only keeps the early-exercise premium small but not zero. The parity-implied forward is also approximate for American options.
* **Flat rate.** A single 13-week T-bill rate is used for every expiry, with no term structure of rates.
* **Data quality.** yfinance is unofficial. The chain was downloaded at about 00:10 New York time on Monday 2026-10-12, so the quotes are from Friday's close. Yahoo's `volume` column is the volume on the contract's *last traded day*, which can be months old. The static-arbitrage filter removed 3 such stale quotes. Adjusted closes may be revised by the vendor, which is one reason the data are cached.
* **The simulation assumes GBM**: constant volatility, no jumps, and no stochastic volatility. Real hedging error has fatter tails, as the historical excess kurtosis of 6.9 shows.
* **Statistics.** t-stats assume independent monthly trades. The lag-1 autocorrelation is 0.04, but 228 months include only a few crises, and those dominate the tails. P&L is per unit of notional, not a return on margin capital.
* **Benchmarks** depend on the machine. These were measured on Apple Silicon with Python 3.12.

---

## References

These citations were assembled from standard bibliographic records. **They have not yet been verified through LibKey.**

Required:

1. Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654. doi:10.1086/260062
2. Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal of Economics and Management Science*, 4(1), 141–183. doi:10.2307/3003143
3. Cox, J. C., Ross, S. A. and Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *Journal of Financial Economics*, 7(3), 229–263. doi:10.1016/0304-405X(79)90015-1
4. Boyle, P. P. (1977). Options: A Monte Carlo Approach. *Journal of Financial Economics*, 4(3), 323–338. doi:10.1016/0304-405X(77)90005-8
5. Kemna, A. G. Z. and Vorst, A. C. F. (1990). A Pricing Method for Options Based on Average Asset Values. *Journal of Banking & Finance*, 14(1), 113–129. doi:10.1016/0378-4266(90)90039-5
6. Carr, P. and Wu, L. (2009). Variance Risk Premiums. *Review of Financial Studies*, 22(3), 1311–1341. doi:10.1093/rfs/hhn038

Methods used in the code:

7. Broadie, M., Glasserman, P. and Kou, S. (1997). A Continuity Correction for Discrete Barrier Options. *Mathematical Finance*, 7(4), 325–349. doi:10.1111/1467-9965.00035
8. Manaster, S. and Koehler, G. (1982). The Calculation of Implied Variances from the Black-Scholes Model: A Note. *Journal of Finance*, 37(1), 227–230. doi:10.1111/j.1540-6261.1982.tb01105.x
9. Brenner, M. and Subrahmanyam, M. G. (1988). A Simple Formula to Compute the Implied Standard Deviation. *Financial Analysts Journal*, 44(5), 80–83. doi:10.2469/faj.v44.n5.80
10. Reiner, E. and Rubinstein, M. (1991). Breaking Down the Barriers. *Risk*, 4(8), 28–35.
