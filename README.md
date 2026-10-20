# Delta Hedging P&L

**A Monte Carlo study of what it costs to replicate an option, and where the Black-Scholes hedge breaks**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This app implements a complete delta-hedging research framework: the Black-Scholes replication argument from first principles, the error of discrete rebalancing, exact P&L attribution, cost-aware optimal rebalancing, and how the hedge fails under jump-diffusion and stochastic volatility. It closes by calibrating the jump model to real S&P 500 ETF data and producing a desk-style morning risk report.

**What the app covers:**

- Black-Scholes pricing engine with closed-form Greeks, validated against finite differences and put-call parity
- The gamma-theta P&L identity, derived exactly and verified against full repricing
- A self-financing discrete hedging engine and the $1/\sqrt{N}$ law of replication error
- Exact P&L attribution into convexity (gamma) and time-decay (theta) streams that tie out to the cent
- Cost-aware hedging: the cost-risk frontier, the optimal rebalancing frequency, and move-based (band) hedging
- Model risk: the jump floor under Merton dynamics, the vega leak under Heston dynamics, and drift invariance
- Calibration of the jump-diffusion model to real SPY data via robust jump detection
- A morning risk report: mean, volatility, VaR, and CVaR of the hedged P&L across calibrated worlds

---

## Mathematical Foundation — [Full Derivations](docs/math_derivations.md)

Delta-hedge the book $\Pi = V - \Delta S$ with $\Delta = \partial_S V$. The $dW$ term — and with it the real-world drift $\mu$ — cancels, leaving a riskless book that must earn $r$. This gives the Black-Scholes PDE and, after a second-order expansion of one hedged step, the **gamma-theta identity**:

```math
\text{P\&L over one step} = \tfrac12\,\Gamma\,\big(\delta S^2 - \sigma^2 S^2\,\delta t\big) + O(\delta t^{3/2})
```

A delta-hedged long option is a pure bet on **realized versus implied variance**, paid through gamma. Discretely rebalanced at $N$ dates per year, the replication error is mean-zero noise whose standard deviation falls like $1/\sqrt{N}$ — exactly the discretization error of quadratic variation, seen through the option's gamma.

---

## Project Structure

```
delta-hedging-pnl/
│
├── app.py                  # Streamlit interactive app
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
└── docs/
    ├── math_derivations.md   # Replication argument, gamma-theta identity, model risk proofs
    └── parameter_guide.md    # How to calibrate the option, cost rate, band width, and simulation size
```

---

## Quick Start

```bash
git clone https://github.com/vij-akshat/delta-hedging-pnl.git
cd delta-hedging-pnl
pip install -r requirements.txt
streamlit run app.py
```

---

## App Features

- **Pricing engine & validation** — closed-form Greeks checked against finite differences and put-call parity
- **Gamma-theta identity** — one-day hedged P&L, full reprice vs. the second-order identity
- **The 1/√N law** — convergence sweep and distributional shrinkage as rebalancing frequency increases
- **Exact P&L attribution** — convexity (gamma) and time-decay (theta) streams that tie out to machine precision
- **Transaction costs** — the cost-risk frontier, optimal rebalancing frequency, and band vs. clock hedging
- **Model risk** — the jump floor (Merton), the vega leak (Heston), and drift invariance (Girsanov, made visible)
- **Live SPY calibration** — robust threshold jump detection via `yfinance`, then hedge in the calibrated world
- **Morning risk report** — VaR95/CVaR95 across the Black-Scholes, Merton, Heston, and SPY-calibrated worlds

---

## Key Results — [Parameter Calibration Guide](docs/parameter_guide.md)

- Quadrupling the rebalancing frequency halves the replication risk (the $1/\sqrt N$ law)
- Proportional costs grow like $\sqrt N$ against risk falling like $1/\sqrt N$, giving an interior optimal frequency $N^* = \lambda b/a$
- Move-based band hedging dominates clock hedging: the same replication risk for less money
- Under variance-matched jumps, the $1/\sqrt N$ law breaks down entirely — a jump floor appears that no rebalancing frequency reduces
- The hedged P&L distribution is invariant to the real-world drift — the experimental content of risk-neutral pricing
- Model risk is a tail phenomenon, not a cost phenomenon: CVaR roughly doubles in worlds the book was not priced in, at essentially unchanged hedging cost

---

## Documentation

| Document | Contents |
|----------|----------|
| [Math Derivations](docs/math_derivations.md) | Replication argument, gamma-theta identity, jump/stochastic-vol model risk, VaR/CVaR |
| [Parameter Guide](docs/parameter_guide.md) | Option parameters, cost rate, band width, simulation size, jump/Heston calibration |

---

## Extensions

- **Deep hedging**: replace the model-derived delta with a neural network policy trained by minimizing a convex risk measure (Buehler, Gonon, Teichmann and Wood, 2019)
- **No-transaction band networks**: build the band structure directly into a deep-hedging architecture (Imaki, Imajo, Ito, Minami and Nakagawa, 2021)
- **Reinforcement learning for hedging**: recover the cost-risk tradeoff without knowing the cost function's form (Kolm and Ritter, 2019)
- **Rough volatility**: swap the CIR variance process for a rough-kernel simulation and re-measure the error floor

---

## License

MIT License — see [LICENSE](LICENSE).
