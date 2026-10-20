# Mathematical Derivations — Delta Hedging P&L

---

## 1. The Replication Argument

The underlying follows geometric Brownian motion under the physical measure, $dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$, and we consider a European call $V(S,t)$. By Ito's lemma,

```math
dV = \Big(\partial_t V + \mu S\,\partial_S V + \tfrac12\sigma^2 S^2\,\partial_{SS}V\Big)dt + \sigma S\,\partial_S V\,dW
```

Form the hedged book $\Pi = V - \Delta S$ with $\Delta = \partial_S V$. This choice eliminates the $dW$ term — **and with it the drift $\mu$**:

```math
d\Pi = \Big(\partial_t V + \tfrac12\sigma^2 S^2\,\partial_{SS}V\Big)dt
```

A riskless book must earn the risk-free rate, $d\Pi = r(V-\Delta S)\,dt$; equating the two gives the Black-Scholes PDE and its closed-form solution

```math
C = S\,N(d_1) - Ke^{-rT}N(d_2), \qquad d_1 = \frac{\ln(S/K)+(r+\sigma^2/2)T}{\sigma\sqrt T}, \qquad d_2 = d_1 - \sigma\sqrt T
```

with Greeks $\Delta = N(d_1)$, $\Gamma = N'(d_1)/(S\sigma\sqrt T)$, $\Theta = -SN'(d_1)\sigma/(2\sqrt T) - rKe^{-rT}N(d_2)$, $\mathcal V = SN'(d_1)\sqrt T$.

---

## 2. The Gamma-Theta Identity

Consider one rebalancing interval $[t,t+\delta t]$ with the hedge frozen at $\Delta=\partial_S V(S,t)$. The book holds the option, the short stock, and a cash account earning $r$. Expanding $V$ to second order and substituting $\Theta$ from the PDE, the four financing terms cancel exactly, leaving

```math
\boxed{\text{P\&L over one step} = \tfrac12\Gamma\big(\delta S^2 - \sigma^2S^2\delta t\big) + O(\delta t^{3/2})}
```

A delta-hedged long option is a pure bet on **realized versus implied variance**, paid through gamma. The two forces balance at the breakeven move $|\delta S| = \sigma S\sqrt{\delta t}$.

---

## 3. Discrete Hedging and the 1/√N Law

Summing the step identity along a path with $N$ rebalances, $\delta t = T/N$: conditionally on step $k$'s information, $\delta S_k^2 = \sigma^2 S_k^2\delta t\cdot Z_k^2$ with $Z_k\sim N(0,1)$, so each term is a centered chi-squared variable scaled by local gamma (Boyle and Emanuel, 1980). Terms are asymptotically uncorrelated across steps, so

```math
\mathrm{Var}(\text{P\&L}) \approx \sum_k 2\Big(\tfrac12\Gamma_k\sigma^2S_k^2\delta t\Big)^2 = O(1/N) \quad\Longrightarrow\quad \mathrm{std}\sim b/\sqrt N
```

Equivalently: all randomness lives in the realized quadratic variation $\sum_k\delta S_k^2$ against the smooth budget the option was priced at — the hedging error *is* the discretization error of quadratic variation, seen through gamma.

---

## 4. Exact P&L Attribution

Splitting step $k$ into a space move at frozen time then a time move at the new stock level:

```math
\text{gamma}_k = \big[V(S_k,\tau_{k-1}) - V(S_{k-1},\tau_{k-1})\big] - \Delta_{k-1}(S_k-S_{k-1}) \ge 0 \quad\text{(convexity)}
```

```math
\text{theta}_k = V(S_k,\tau_k) - V(S_k,\tau_{k-1}) \le 0 \quad\text{(time decay)}
```

Their sum telescopes exactly to the hedged P&L (at $r=0$, to machine precision). Whether the net difference of two large, nearly-mirrored streams ends positive is decided by one number: realized volatility versus the implied volatility the option was priced at.

---

## 5. Transaction Costs and the Optimal Frequency

Each rebalance pays a proportional cost $k_c|\Delta_k-\Delta_{k-1}|S_k$. Since $\mathbb{E}|\delta S| = \sigma S\sqrt{\delta t}\sqrt{2/\pi}$,

```math
\mathbb{E}[\text{total rebalancing cost}] \approx k_c\sigma\sqrt{2/\pi}\,\overline{\Gamma S^2}\sqrt{NT} \propto \sqrt N
```

while risk falls like $b/\sqrt N$. Charging risk at a one-sided multiplier $\lambda$ and minimizing $J(N)=c_0+a\sqrt N+\lambda b/\sqrt N$ gives

```math
N^* = \frac{\lambda b}{a}
```

**Leland's approximation.** The expected cost flow can be absorbed as a widened pricing volatility, $\hat\sigma^2 = \sigma^2 + \dfrac{2k_c\sigma}{\sqrt{\delta t}}\sqrt{2/\pi}$ — a leading-order approximation (Kabanov and Safarian, 1997, show the exact limiting claim has a nonzero error), verified numerically in the app to within a few percent.

**Band hedging.** Trading only when $|\Delta_{\text{target}}-\Delta_{\text{held}}| > h$ dominates any fixed-clock rule on the cost-risk plane: the same replication risk for less money (Whalley and Wilmott, 1997, show the optimal band scales as the cube root of the cost rate).

---

## 6. Model Risk I — Jump-Diffusion (Merton)

Jumps arrive as a Poisson process with intensity $\lambda$; each multiplies price by $J=e^Y$, $Y\sim N(\mu_J,\sigma_J^2)$. The compensated dynamics keep expected growth at $\mu$:

```math
\frac{dS}{S} = (\mu-\lambda\bar\kappa)dt + \sigma\,dW + (J-1)\,dN, \qquad \bar\kappa = \mathbb{E}[J]-1 = e^{\mu_J+\sigma_J^2/2}-1
```

Jumps contribute $\lambda(\mu_J^2+\sigma_J^2)$ per year of quadratic variation, so the app **variance-matches** the diffusion volatility, $\sigma_{\text{diff}}^2 = \sigma_{\text{imp}}^2 - \lambda(\mu_J^2+\sigma_J^2)$, isolating model *shape* from mispriced level. A jump teleports the stock with the hedge frozen, locking in a convexity gap that **no rebalancing frequency reduces** — the $1/\sqrt N$ law dies onto a jump floor, and because gamma is local, a large jump collects less curvature than the theta rent paid, leaving a small systematic negative drift for the long hedger.

---

## 7. Model Risk II — Stochastic Volatility (Heston) and Drift Invariance

```math
dS_t = \mu S_t dt + \sqrt{v_t}S_t\,dW_t^S, \qquad dv_t = \kappa(\theta-v_t)dt + \xi\sqrt{v_t}\,dW_t^v, \qquad d\langle W^S,W^v\rangle_t = \rho\,dt
```

simulated via the full-truncation Euler scheme (Lord, Koekkoek and van Dijk, 2010): $v^+=\max(v,0)$ in both drift and diffusion. The book now carries two sources of randomness; a stock position spans only one, leaving an unhedged $\partial_vV\,dv$ **vega** term that no choice of $\Delta$ removes — a risk floor smaller than the jump floor (since variance moves continuously and partially averages out) but equally immune to rebalancing frequency.

**Drift invariance.** Hedging the same call under wildly different real-world drifts $\mu$ leaves the hedged P&L distribution essentially unchanged, while the unhedged option's P&L swings enormously — the experimental face of Girsanov's theorem, and exactly why option prices can be computed in the risk-neutral world.

---

## 8. Calibration and Risk Reporting

**Robust jump detection.** $\hat\sigma_{\text{daily}} = 1.4826\cdot\mathrm{MAD}(r)$ estimates diffusion volatility without contamination from the jumps being hunted; returns with $|r_k| > 4\hat\sigma_{\text{daily}}$ are flagged, giving $`\hat\lambda = \#\text{flags}/\text{years}`$ and moment estimates of $(\hat\mu_J,\hat\sigma_J)$ from the flagged returns.

**VaR and CVaR.** With losses as positive numbers,

```math
\text{VaR}_{95} = -q_{5\%}(\text{P\&L}), \qquad \text{CVaR}_{95} = -\mathbb{E}[\text{P\&L}\mid\text{P\&L}\le q_{5\%}]
```

CVaR is a coherent risk measure (Artzner, Delbaen, Eber and Heath, 1999) and dominates VaR for reporting, since VaR is blind beyond its own threshold.
