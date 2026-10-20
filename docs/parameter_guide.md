# Parameter Calibration Guide — Delta Hedging P&L

---

## The Running Example

Default: **S₀ = K = 100, T = 1 year, r = 2%, σ = 20%** — a one-year at-the-money call, held long and delta-hedged.

At-the-money keeps gamma near its peak so every effect in the app (the breakeven move, the jump floor, the vega leak) is visible at reasonable sample sizes. Moving the strike away from spot shrinks gamma and mutes every downstream effect; that's expected, not a bug — try it to build intuition for why desks care most about hedging near-the-money risk.

---

## Monte Carlo Paths

Default: **5,000**

| Paths | Use case |
|-------|----------|
| 1,000–2,000 | Fast iteration; wider Monte Carlo noise on tail statistics (VaR/CVaR) |
| 5,000 | Default — stable means and stds, usable tail quantiles |
| 10,000 | Tighter VaR95/CVaR95, slower on the Model Risk and Risk Report pages |

The 1/√N convergence sweep and the transaction-cost frontier each re-run the hedge at 6–9 different rebalancing frequencies, so path count multiplies runtime there more than on single-frequency pages.

---

## Rebalancing Frequency (`N`)

The number of hedge rebalances per year. The 1/√N law page sweeps `N` from 8 to 512 automatically; other pages that need a single frequency (P&L Attribution, Transaction Costs' band comparison) default to **252** (daily).

| `N` | Interpretation |
|-----|-----------------|
| 12 | Monthly — visibly noisy replication P&L |
| 52 | Weekly |
| 252 | Daily — the industry-standard reference frequency |

---

## Transaction Cost Rate (`kc`, in bps)

Default: **10 basis points**, one-way

A reasonable all-in figure for a liquid large-cap. Raising it lowers the optimal rebalancing frequency $N^*=\lambda b/a$ (dearer trading means rebalance less); a name with wider spreads or more market impact should use a higher rate here.

---

## Band Half-Width (`h`)

Default: **0.05** (delta units)

The no-transaction band's half-width: rebalance only when $|\Delta_{\text{target}} - \Delta_{\text{held}}| > h$. Smaller `h` behaves like clock hedging (frequent small trades); larger `h` trades rarely but lets replication risk drift further between trades. Whalley and Wilmott (1997) show the theoretically optimal band scales as the cube root of the cost rate — use the Transaction Costs page's frontier plot to see where a given `h` sits relative to the clock family at the same cost.

---

## Jump Parameters (Merton)

Defaults: **λ = 0.5/year, μⱼ = −10%, σⱼ = 15%** (a stylized crash scenario)

These are deliberately generic; the **SPY Calibration** page estimates the same three parameters from real data via robust threshold jump detection, typically returning a higher intensity and smaller individual jump size than the stylized default — real markets tend to move via more frequent, smaller dislocations rather than rare, large ones.

---

## Heston Parameters

Defaults: **κ = 3.0, ξ = 0.4, ρ = −0.7**, with $v_0=\theta=\sigma_{\text{imp}}^2$ so the expected variance budget matches the hedger's implied vol.

Check the **Feller condition** $2\kappa\theta \ge \xi^2$ shown on the page — when violated, the discrete Euler scheme's `max(v,0)` truncation does more work, and the simulated variance process spends more time pinned near zero than the continuous-time model implies.

---

## Random Seed

Every simulation is seeded from the sidebar's **Random seed** control (default 42), so results are reproducible at fixed parameters. Distinct pages offset the seed slightly (e.g. `seed+1`) so that, say, the P&L Attribution page's single illustrative path isn't identical to the paths used in the convergence sweep.
