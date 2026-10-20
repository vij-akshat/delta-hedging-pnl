"""
Discrete-Time Delta Hedging — Execution & Risk Terminal
=======================================================
Interactive companion to the research notebook. A trader sets the contract, the
hedge rule, and the world, and watches replication error, P&L attribution,
transaction-cost trade-offs, and model risk update live.

Sections
    PRICING & GREEKS   Black-Scholes engine, the one-day gamma-theta identity
    REPLICATION        the 1/sqrt(N) law and the frequency distribution
    ATTRIBUTION        exact convexity (gamma) vs decay (theta) streams
    COST-AWARE         the cost-risk frontier, optimal frequency, Leland vol
    MODEL RISK         jump floor, vega leak, drift invariance
    RISK REPORT        SPY calibration and the morning VaR/CVaR book
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm, skew
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================ EMBEDDED SPY DATA
SPY_EMBEDDED = [
    417.45, 420.2, 420.7, 425.92, 424.47, 422.26, 420.1, 421.62, 418.43, 416.72, 421.29, 421.5,
    423.16, 428.16, 428.65, 428.01, 424.66, 423.59, 424.66, 427.36, 430.8, 434.22, 433.95, 435.46,
    438.69, 439.67, 436.75, 436.75, 438.7, 439.9, 439.97, 437.05, 441.33, 442.17, 440.91, 434.77,
    433.53, 431.57, 435.33, 433.44, 430.54, 430.7, 430.44, 432.82, 427.78, 424.64, 421.4, 421.61,
    424.35, 423.2, 427.91, 421.98, 424.96, 427.65, 433.83, 435.62, 434.98, 435.8, 433.91, 431.0,
    429.67, 430.32, 433.15, 430.77, 431.28, 434.99, 429.75, 430.01, 429.11, 425.17, 418.14, 417.2,
    418.96, 412.8, 412.96, 415.36, 414.35, 414.19, 408.64, 411.62, 411.46, 416.35, 419.01, 421.19,
    422.92, 420.34, 418.25, 422.65, 422.63, 417.0, 413.33, 408.25, 407.55, 410.62, 404.73, 399.88,
    398.07, 402.83, 405.36, 409.68, 417.53, 421.34, 422.31, 423.51, 423.82, 420.52, 427.08, 426.67,
    434.95, 435.87, 436.4, 436.95, 440.31, 439.35, 441.05, 441.32, 440.52, 440.96, 440.65, 442.38,
    445.0, 442.66, 442.58, 440.79, 444.16, 446.07, 447.8, 449.85, 456.05, 457.51, 456.76, 459.33,
    462.12, 455.72, 460.04, 460.96, 462.91, 463.75, 463.92, 462.58, 459.99, 456.23, 454.76, 455.39,
    461.89, 461.19, 463.8, 463.59, 463.91, 462.21, 459.64, 463.73, 469.51, 470.5, 471.87, 472.39,
    474.96, 474.36, 478.11, 477.74, 469.95, 476.1, 481.11, 479.36, 480.75, 484.76, 484.97, 487.78,
    487.56, 480.85, 485.22, 488.56, 486.13, 483.46, 483.89, 493.91, 494.25, 492.44, 493.35, 492.7,
    494.47, 499.11, 498.58, 493.6, 496.1, 501.02, 498.01, 497.59, 502.94, 502.15, 501.16, 497.72,
    500.68, 503.46, 508.11, 509.79, 508.83, 507.42, 506.48, 510.74, 510.64, 509.75, 506.51, 507.07,
    500.88, 506.11, 506.4, 506.98, 501.91, 505.69, 498.71, 492.46, 491.57, 488.66, 487.65, 483.4,
    487.85, 493.64, 493.4, 491.53, 496.18, 497.94, 490.05, 488.46, 493.03, 499.14, 504.3, 504.85,
    504.9, 507.81, 508.47, 508.53, 510.87, 517.19, 516.13, 516.87, 517.47, 518.74, 517.24, 513.46,
    516.86, 517.22, 513.6, 510.19, 514.84, 515.26, 515.84, 521.97, 521.96, 521.32, 522.93, 524.19,
    528.5, 529.56, 529.88, 534.1, 535.46, 534.0, 533.29, 531.55, 533.6, 534.27, 535.11, 533.0,
    534.1, 537.7, 540.09, 543.21, 543.84, 544.36, 549.75, 545.01, 548.45, 549.96, 553.22, 545.46,
    541.27, 537.68, 543.22, 542.37, 530.08, 527.31, 533.22, 533.53, 530.83, 539.46, 531.82, 521.92,
    506.72, 511.39, 507.97, 519.71, 522.01, 522.28, 530.87, 532.54, 541.67, 542.89, 548.08, 547.19,
    549.07, 544.76, 550.54, 549.23, 549.99, 546.79, 546.84, 552.06, 540.7, 539.6, 538.28, 529.22,
    535.15, 537.48, 542.99, 547.57, 550.43, 551.24, 551.47, 549.83, 559.21, 558.25, 559.64, 561.24,
    560.0, 562.22, 561.41, 563.66, 558.61, 558.85, 557.82, 562.89, 557.8, 563.08, 566.98, 565.99,
    569.38, 574.03, 569.57, 572.05, 572.1, 574.3, 573.35, 573.05, 567.81, 569.04, 568.85, 570.6,
    571.53, 569.8, 558.63, 560.99, 559.78, 566.55, 580.63, 585.12, 587.66, 588.22, 586.39, 586.68,
    582.9, 575.44, 577.8, 579.91, 580.1, 583.22, 585.03, 587.01, 590.08, 588.29, 591.94, 593.0,
    593.28, 596.96, 595.98, 597.11, 594.03, 592.19, 596.77, 593.69, 593.57, 596.11, 593.65, 575.96,
    575.78, 582.7, 586.19, 592.7, 592.74, 586.5, 579.81, 577.7, 576.28, 583.49, 586.85, 580.21,
    581.06, 572.19, 573.08, 573.87, 584.3, 583.18, 589.04, 594.43, 597.77, 601.03, 599.28, 590.8,
    595.88, 593.2, 596.39, 593.21, 589.22, 593.18, 595.58, 597.65, 592.18, 596.2, 596.65, 594.73,
    601.01, 600.98, 602.75, 604.17, 601.65, 591.36, 588.67, 585.74, 586.04, 576.68, 585.68, 575.42,
    568.61, 574.72, 564.52, 567.68, 552.56, 547.97, 550.88, 543.54, 554.76, 559.04, 553.0, 559.02,
    557.4, 557.59, 567.57, 568.94, 562.15, 560.65, 549.36, 553.05, 554.61, 558.12, 530.62, 499.55,
    498.66, 490.85, 542.4, 518.63, 527.89, 533.01, 531.52, 519.7, 520.44, 508.06, 521.27, 529.35,
    540.49, 544.4, 544.61, 548.04, 548.26, 552.14, 560.34, 557.12, 552.47, 554.79, 558.66, 557.94,
    576.38, 580.19, 580.93, 583.77, 587.47, 588.11, 586.13, 576.25, 576.48, 572.55, 584.45, 581.07,
    583.36, 582.71, 585.99, 589.33, 589.18, 586.33, 592.35, 592.88, 596.24, 594.54, 596.91, 590.23,
    595.85, 590.76, 590.67, 589.28, 595.1, 601.68, 602.01, 606.72, 609.74, 612.65, 612.46, 615.23,
    620.08, 615.46, 615.12, 618.81, 620.56, 618.37, 619.55, 616.91, 618.97, 622.76, 622.3, 623.48,
    623.57, 628.88, 629.08, 631.74, 631.58, 629.92, 629.12, 626.76, 616.49, 625.86, 622.69, 627.46,
    626.93, 631.82, 630.57, 637.28, 639.47, 639.53, 638.03, 637.89, 634.43, 632.74, 630.2, 639.88,
    637.07, 639.73, 641.19, 643.46, 639.62, 634.88, 638.33, 643.66, 641.8, 643.37, 644.86, 646.72,
    652.1, 651.88, 655.35, 654.45, 653.64, 656.69, 659.94, 663.06, 659.46, 657.36, 654.32, 658.07,
    659.92, 662.41, 664.67, 665.43, 665.42, 667.81, 665.33, 669.3, 667.36, 649.32, 659.29, 658.48,
    661.4, 656.9, 660.63, 667.5, 667.49, 664.02, 667.96, 673.42, 681.36, 683.17, 683.5, 675.98,
    678.2, 679.47, 671.42, 673.74, 666.51, 667.17, 677.58, 679.13, 679.51, 668.24, 668.13, 661.9,
    656.34, 658.88, 648.84, 655.3, 664.94, 671.2, 675.83, 679.52, 676.42, 677.67, 680.02, 680.52,
    681.81, 679.76, 679.17, 683.68, 685.27, 677.9, 676.88, 675.03, 667.6, 672.64, 678.74, 682.96,
    686.09, 688.5, 688.43, 685.98, 685.14, 680.06, 681.31, 685.85, 689.93, 687.7, 687.63, 692.18,
    693.27, 691.88, 688.48, 690.35, 689.78, 675.73, 683.53, 687.1, 687.35, 690.84, 693.6, 693.53,
    692.15, 690.09, 693.52, 687.65, 684.32, 675.77, 688.74, 692.06, 690.23, 690.08, 679.41, 679.89,
    680.99, 684.42, 682.62, 687.55, 680.53, 685.48, 691.26, 687.42, 684.12, 684.51, 678.48, 683.26,
    679.45, 670.55, 676.42, 675.34, 674.49, 664.25, 660.49, 667.21, 668.96, 659.63, 658.0, 648.57,
    655.38, 653.18, 656.82, 645.09, 634.09, 631.97, 650.34, 655.24, 655.83, 658.93, 659.22, 676.01,
    679.91, 679.46, 686.1, 694.46, 699.94, 701.66, 710.14, 708.72, 704.08, 711.21, 708.45, 713.94,
    715.17, 711.69, 711.58, 718.66, 720.65, 718.01, 723.77, 733.83, 731.58, 737.62, 739.3, 738.18,
    742.31, 748.17, 739.17, 738.65, 733.73, 741.25, 742.72, 745.64, 750.59, 750.46, 754.6, 756.48,
    758.54, 759.57, 754.24, 757.09, 737.55, 739.22,
]


# ============================================================ PAGE / STYLE
st.set_page_config(
    page_title="Delta Hedging Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = dict(
    bg="#0a0e14",
    panel="#0d1117",
    panel2="#111722",
    border="#1c2433",
    border2="#2a3441",
    text="#c9d1d9",
    muted="#6e7681",
    blue="#58a6ff",
    red="#f85149",
    green="#3fb950",
    amber="#d29922",
    purple="#bc8cff",
    cyan="#39c5cf",
    orange="#db6d28",
    pink="#f778ba",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'IBM Plex Mono', monospace;
    }}
    .stApp {{ background: {C['bg']}; color: {C['text']}; }}

    /* Sidebar: widen it and scope the custom font to plain text only.
       Applying the font to every descendant (incl. BaseWeb slider/number-input
       internals) makes their fixed-width value bubbles/ticks overflow and
       overlap neighboring labels — so we deliberately do NOT use a `* {{ }}`
       selector here. */
    section[data-testid="stSidebar"] {{
        background: {C['panel']};
        border-right: 1px solid {C['border']};
        min-width: 340px !important;
        max-width: 400px !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        font-family: 'IBM Plex Mono', monospace;
    }}
    /* breathing room between stacked sidebar text blocks so consecutive
       section headers / captions never collide with the widget below them */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div {{
        line-height: 1.45;
        margin: 3px 0;
    }}
    section[data-testid="stSidebar"] label p {{
        line-height: 1.4;
        white-space: normal;
        overflow-wrap: break-word;
    }}
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
        margin-bottom: 2px;
    }}

    /* header bar */
    .term-head {{
        border: 1px solid {C['border2']};
        background: linear-gradient(90deg, {C['panel2']} 0%, {C['panel']} 100%);
        border-radius: 4px; padding: 14px 18px; margin-bottom: 4px;
        display: flex; justify-content: space-between; align-items: center;
    }}
    .term-title {{ font-size: 19px; font-weight: 700; letter-spacing: 1px; color: {C['text']}; }}
    .term-sub {{ font-size: 11px; color: {C['muted']}; letter-spacing: 1px; }}
    .term-tag {{
        font-size: 10px; letter-spacing: 1px; color: {C['green']};
        border: 1px solid {C['border2']}; border-radius: 3px;
        padding: 3px 9px; background: {C['panel']};
    }}

    /* status ribbon */
    .ribbon {{
        display: flex; gap: 0; border: 1px solid {C['border']};
        border-radius: 4px; overflow: hidden; margin: 8px 0 14px 0;
    }}
    .ribbon .cell {{
        flex: 1; padding: 9px 14px; border-right: 1px solid {C['border']};
        background: {C['panel']};
    }}
    .ribbon .cell:last-child {{ border-right: none; }}
    .ribbon .k {{ font-size: 10px; color: {C['muted']}; letter-spacing: 1px; }}
    .ribbon .v {{ font-size: 16px; font-weight: 600; color: {C['text']}; }}

    /* metric cards */
    [data-testid="stMetric"] {{
        background: {C['panel']}; border: 1px solid {C['border']};
        border-radius: 4px; padding: 12px 14px;
    }}
    [data-testid="stMetricLabel"] {{ color: {C['muted']}; }}
    [data-testid="stMetricValue"] {{ color: {C['text']}; font-size: 22px; }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px; background: {C['panel']};
        border: 1px solid {C['border']}; border-radius: 4px; padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; color: {C['muted']};
        font-size: 12px; letter-spacing: 1px; font-weight: 500;
        border-radius: 3px; padding: 8px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {C['panel2']}; color: {C['amber']};
        border: 1px solid {C['border2']};
    }}

    /* note box */
    .note {{
        border-left: 2px solid {C['border2']}; background: {C['panel']};
        padding: 10px 14px; margin: 6px 0 12px 0; font-size: 12px;
        color: {C['muted']}; line-height: 1.55; border-radius: 0 3px 3px 0;
    }}
    .note b {{ color: {C['text']}; }}

    h1, h2, h3 {{ color: {C['text']}; }}
    hr {{ border-color: {C['border']}; }}
    .stDataFrame {{ border: 1px solid {C['border']}; border-radius: 4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def dark(fig, title="", height=460, logx=False, logy=False):
    """Apply the Bloomberg-terminal theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=("\u25b8 " + title) if title else "",
                   font=dict(size=14, color=C["text"], family="IBM Plex Mono"), x=0.01),
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(family="IBM Plex Mono", size=12, color=C["text"]),
        height=height, margin=dict(l=58, r=24, t=44 if title else 18, b=46),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=C["muted"]),
                    orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["border2"],
                        font=dict(family="IBM Plex Mono", size=11, color=C["text"])),
    )
    axkw = dict(gridcolor=C["border"], zerolinecolor=C["border2"],
                linecolor=C["border2"], tickfont=dict(size=10, color=C["muted"]),
                title_font=dict(size=11, color=C["muted"]))
    fig.update_xaxes(**axkw)
    fig.update_yaxes(**axkw)
    if logx:
        fig.update_xaxes(type="log")
    if logy:
        fig.update_yaxes(type="log")
    return fig


def note(html):
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)


# ============================================================ NUMERIC CORE
def bs_price(S, K, T, r, sig):
    S = np.asarray(S, float)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sig):
    S = np.asarray(S, float)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return norm.cdf(d1)


def bs_gamma(S, K, T, r, sig):
    S = np.asarray(S, float)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return norm.pdf(d1) / (S * sig * np.sqrt(T))


def bs_theta(S, K, T, r, sig):
    S = np.asarray(S, float)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return -(S * norm.pdf(d1) * sig) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)


def bs_vega(S, K, T, r, sig):
    S = np.asarray(S, float)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def make_gbm_paths(N, n_paths, seed, S0, T, sigma, mu):
    rng = np.random.default_rng(seed)
    dt = T / N
    inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, N))
    return np.concatenate([np.full((n_paths, 1), S0),
                           np.exp(np.log(S0) + np.cumsum(inc, axis=1))], axis=1)


def make_merton_paths(N, n_paths, seed, S0, T, mu, sigma, lam, mu_j, sigma_j):
    rng = np.random.default_rng(seed)
    dt = T / N
    Z = rng.standard_normal((n_paths, N))
    counts = rng.poisson(lam * dt, (n_paths, N))
    ZJ = rng.standard_normal((n_paths, N))
    kappa_bar = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    inc = ((mu - lam * kappa_bar - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
           + counts * mu_j + np.sqrt(counts) * sigma_j * ZJ)
    return np.concatenate([np.full((n_paths, 1), S0),
                           np.exp(np.log(S0) + np.cumsum(inc, axis=1))], axis=1)


def make_heston_paths(N, n_paths, seed, S0, T, mu, v0, kappa, theta, xi, rho):
    rng = np.random.default_rng(seed)
    dt = T / N
    Z1 = rng.standard_normal((n_paths, N))
    Z2 = rng.standard_normal((n_paths, N))
    S = np.full(n_paths, S0)
    v = np.full(n_paths, v0)
    out = np.empty((n_paths, N + 1)); out[:, 0] = S0
    for k in range(N):
        vp = np.maximum(v, 0.0)
        Zs = rho * Z2[:, k] + np.sqrt(1.0 - rho**2) * Z1[:, k]
        S = S * np.exp((mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * Zs)
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z2[:, k]
        out[:, k + 1] = S
    return out


def hedge_pnl(S, S0, K, T, sigma_imp, r):
    n, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    held = np.full(n, float(bs_delta(S0, K, T, r, sigma_imp)))
    cash = -float(bs_price(S0, K, T, r, sigma_imp)) + held * S0
    for j in range(1, N):
        tgt = bs_delta(S[:, j], K, T - j * dt, r, sigma_imp)
        cash = cash * np.exp(r * dt) + (tgt - held) * S[:, j]
        held = tgt
    cash = cash * np.exp(r * dt)
    return cash + np.maximum(S[:, N] - K, 0.0) - held * S[:, N]


def hedge_with_costs(S, S0, K, T, sigma_imp, kc, r):
    n, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    held = np.full(n, float(bs_delta(S0, K, T, r, sigma_imp)))
    cash = -float(bs_price(S0, K, T, r, sigma_imp)) + held * S0
    cost = kc * held * S0
    for j in range(1, N):
        tgt = bs_delta(S[:, j], K, T - j * dt, r, sigma_imp)
        cash = cash * np.exp(r * dt) + (tgt - held) * S[:, j]
        cost = cost + kc * np.abs(tgt - held) * S[:, j]
        held = tgt
    cash = cash * np.exp(r * dt)
    cost = cost + kc * held * S[:, N]
    repl = cash + np.maximum(S[:, N] - K, 0.0) - held * S[:, N]
    return repl, cost


def band_hedge(S, S0, K, T, sigma_imp, kc, h, r):
    n, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    held = np.full(n, float(bs_delta(S0, K, T, r, sigma_imp)))
    cash = -float(bs_price(S0, K, T, r, sigma_imp)) + held * S0
    cost = kc * held * S0
    for j in range(1, N):
        cash = cash * np.exp(r * dt)
        tgt = bs_delta(S[:, j], K, T - j * dt, r, sigma_imp)
        m = np.abs(tgt - held) > h
        cash[m] += (tgt[m] - held[m]) * S[m, j]
        cost[m] += kc * np.abs(tgt[m] - held[m]) * S[m, j]
        held[m] = tgt[m]
    cash = cash * np.exp(r * dt)
    cost = cost + kc * held * S[:, N]
    repl = cash + np.maximum(S[:, N] - K, 0.0) - held * S[:, N]
    return repl, cost


def attribute_pnl(path, K, T, sigma):
    def V(S, tau):
        return float(bs_price(S, K, tau, 0.0, sigma)) if tau > 1e-12 else max(S - K, 0.0)
    path = np.asarray(path, float)
    N = len(path) - 1
    dt = T / N
    g = np.zeros(N); th = np.zeros(N)
    for k in range(1, N + 1):
        t_old, t_new = T - (k - 1) * dt, T - k * dt
        So, Sn = path[k - 1], path[k]
        d_prev = float(bs_delta(So, K, t_old, 0.0, sigma))
        g[k - 1] = (V(Sn, t_old) - V(So, t_old)) - d_prev * (Sn - So)
        th[k - 1] = V(Sn, t_new) - V(Sn, t_old)
    return np.cumsum(g), np.cumsum(th)


def realized_vol(path, T):
    lr = np.diff(np.log(np.asarray(path, float)))
    return np.sqrt(np.sum(lr**2) / T)


def var_cvar(pnl, level=0.95):
    q = np.quantile(pnl, 1.0 - level)
    return -q, -pnl[pnl <= q].mean()


# ============================================================ CACHED SCENARIOS
@st.cache_data(show_spinner=False)
def conv_sweep(Ns, n_paths, seed, S0, K, T, r, sigma):
    means, stds = [], []
    for N in Ns:
        pnl = hedge_pnl(make_gbm_paths(int(N), n_paths, seed, S0, T, sigma, r),
                        S0, K, T, sigma, r)
        means.append(float(pnl.mean())); stds.append(float(pnl.std()))
    return np.array(means), np.array(stds)


@st.cache_data(show_spinner=False)
def freq_distribution(n_paths, seed, S0, K, T, r, sigma):
    pm = hedge_pnl(make_gbm_paths(12, n_paths, seed, S0, T, sigma, r), S0, K, T, sigma, r)
    pd_ = hedge_pnl(make_gbm_paths(252, n_paths, seed, S0, T, sigma, r), S0, K, T, sigma, r)
    return pm, pd_


@st.cache_data(show_spinner=False)
def cost_frontier(Ns, n_paths, seed, S0, K, T, r, sigma, kc):
    risk, cost = [], []
    for N in Ns:
        repl, c = hedge_with_costs(make_gbm_paths(int(N), n_paths, seed, S0, T, sigma, r),
                                   S0, K, T, sigma, kc, r)
        risk.append(float(repl.std())); cost.append(float(c.mean()))
    return np.array(risk), np.array(cost)


@st.cache_data(show_spinner=False)
def band_frontier(bands, n_paths, seed, S0, K, T, r, sigma, kc):
    risk, cost = [], []
    S = make_gbm_paths(252, n_paths, seed, S0, T, sigma, r)
    for h in bands:
        repl, c = band_hedge(S, S0, K, T, sigma, kc, h, r)
        risk.append(float(repl.std())); cost.append(float(c.mean()))
    return np.array(risk), np.array(cost)


@st.cache_data(show_spinner=False)
def jump_compare(Ns, n_paths, seed, S0, K, T, r, sigma, lam, mu_j, sig_j):
    sig_diff = np.sqrt(max(sigma**2 - lam * (mu_j**2 + sig_j**2), 1e-8))
    g, m = [], []
    for N in Ns:
        g.append(float(hedge_pnl(make_gbm_paths(int(N), n_paths, seed, S0, T, sigma, r),
                                 S0, K, T, sigma, r).std()))
        m.append(float(hedge_pnl(make_merton_paths(int(N), n_paths, seed, S0, T, r, sig_diff, lam, mu_j, sig_j),
                                 S0, K, T, sigma, r).std()))
    pg = hedge_pnl(make_gbm_paths(252, n_paths, seed + 1, S0, T, sigma, r), S0, K, T, sigma, r)
    pm = hedge_pnl(make_merton_paths(252, n_paths, seed + 1, S0, T, r, sig_diff, lam, mu_j, sig_j),
                   S0, K, T, sigma, r)
    return np.array(g), np.array(m), pg, pm, sig_diff


@st.cache_data(show_spinner=False)
def heston_sweep(Ns, n_paths, seed, S0, K, T, r, sigma, v0, kappa, theta, xi, rho):
    hs = []
    for N in Ns:
        hs.append(float(hedge_pnl(make_heston_paths(int(N), n_paths, seed, S0, T, r, v0, kappa, theta, xi, rho),
                                  S0, K, T, sigma, r).std()))
    return np.array(hs)


@st.cache_data(show_spinner=False)
def drift_test(mus, n_paths, seed, S0, K, T, r, sigma):
    V0 = float(bs_price(S0, K, T, r, sigma))
    out = []
    for mu in mus:
        S = make_gbm_paths(126, n_paths, seed, S0, T, sigma, mu)
        hedged = hedge_pnl(S, S0, K, T, sigma, r)
        unhedged = np.maximum(S[:, -1] - K, 0.0) - V0 * np.exp(r * T)
        out.append((mu, hedged, float(unhedged.mean()), float(unhedged.std())))
    return out


@st.cache_data(show_spinner=True)
def load_spy(use_live, start_str, end_str):
    """start_str / end_str are ISO date strings ('YYYY-MM-DD'), or '' to fall
    back to the embedded snapshot. Cached per (use_live, start, end) so a new
    date range always triggers a fresh fetch rather than reusing stale data."""
    label = "SPY embedded snapshot, 2023-06-12 to 2026-06-09"
    closes = np.array(SPY_EMBEDDED, dtype=float)
    if use_live and start_str and end_str:
        try:
            import yfinance as yf
            last_err = None
            for _ in range(3):
                try:
                    df = yf.download("SPY", start=start_str, end=end_str, interval="1d",
                                    auto_adjust=True, progress=False)
                    if df is not None and len(df) > 100:
                        closes = df["Close"].dropna().to_numpy().ravel()
                        label = f"SPY via yfinance, {df.index[0].date()} to {df.index[-1].date()}"
                        break
                    else:
                        last_err = ValueError(
                            f"Only {0 if df is None else len(df)} rows returned for "
                            f"{start_str} to {end_str} — try a wider date range.")
                except Exception as e:
                    last_err = e
            if label.startswith("SPY embedded") and last_err is not None:
                label += f"  (live fetch failed: {last_err})"
        except Exception as e:
            label += f"  (yfinance unavailable: {e})"
    return closes, label


@st.cache_data(show_spinner=False)
def calibrate_jumps(closes):
    r_spy = np.diff(np.log(closes))
    n_years = len(r_spy) / 252
    mad = np.median(np.abs(r_spy - np.median(r_spy)))
    sig_daily = 1.4826 * mad
    thresh = 4.0 * sig_daily
    is_jump = np.abs(r_spy) > thresh
    lam = is_jump.sum() / n_years
    jr = r_spy[is_jump]
    mu_j = float(jr.mean()) if len(jr) else 0.0
    sig_j = float(jr.std(ddof=1)) if len(jr) > 1 else 0.0
    sig_diff = float(r_spy[~is_jump].std() * np.sqrt(252))
    sig_total = float(np.sqrt(sig_diff**2 + lam * (mu_j**2 + sig_j**2)))
    return dict(r=r_spy, sig_daily=float(sig_daily), thresh=float(thresh),
                is_jump=is_jump, lam=float(lam), mu_j=mu_j, sig_j=sig_j,
                sig_diff=sig_diff, sig_total=sig_total, n_flag=int(is_jump.sum()))


# ============================================================ SIDEBAR
st.sidebar.markdown(
    f"<div style='font-size:13px;font-weight:700;letter-spacing:1px;color:{C['amber']};"
    f"padding-bottom:6px'>HEDGING DESK &mdash; CONTROL</div>"
    f"<div style='font-size:10px;color:{C['muted']};letter-spacing:1px;padding-bottom:10px'>"
    f"long one European call, delta-hedged</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown(f"<div style='color:{C['muted']};font-size:10px;letter-spacing:1px'>CONTRACT</div>",
                    unsafe_allow_html=True)
S0 = st.sidebar.number_input("Spot  S0", 10.0, 1000.0, 100.0, 1.0)
K = st.sidebar.number_input("Strike  K", 10.0, 1000.0, 100.0, 1.0)
T = st.sidebar.slider("Maturity  T (years)", 0.1, 3.0, 1.0, 0.1)
r = st.sidebar.slider("Risk-free rate  r", 0.0, 0.10, 0.02, 0.005)
sigma = st.sidebar.slider("Implied vol  sigma", 0.05, 0.60, 0.20, 0.01)

st.sidebar.markdown(f"<div style='color:{C['muted']};font-size:10px;letter-spacing:1px;padding-top:8px'>HEDGE RULE</div>",
                    unsafe_allow_html=True)
N_single = st.sidebar.select_slider("Rebalances / yr  N",
                                    options=[12, 26, 52, 126, 252, 504], value=252)
band_h = st.sidebar.slider("Band width  h (delta)", 0.01, 0.20, 0.05, 0.01)
cost_bps = st.sidebar.slider("Transaction cost (bps, one-way)", 0.0, 50.0, 10.0, 1.0)
lam_risk = st.sidebar.slider("Risk charge  lambda", 0.5, 3.0, 1.65, 0.05)
kc = cost_bps / 1e4

st.sidebar.markdown(f"<div style='color:{C['muted']};font-size:10px;letter-spacing:1px;padding-top:8px'>MONTE CARLO</div>",
                    unsafe_allow_html=True)
n_paths = st.sidebar.select_slider("Paths", options=[1000, 2000, 4000, 8000], value=4000)
seed = st.sidebar.number_input("Seed", 0, 9999, 42, 1)

with st.sidebar.expander("MODEL-RISK WORLDS", expanded=False):
    st.caption("Merton jump-diffusion")
    lam_j = st.slider("Jump intensity  lambda_J / yr", 0.0, 3.0, 0.5, 0.1)
    mu_j = st.slider("Jump mean (log)  mu_J", -0.30, 0.10, -0.10, 0.01)
    sig_j = st.slider("Jump vol  sigma_J", 0.01, 0.40, 0.15, 0.01)
    st.caption("Heston stochastic vol")
    v0 = st.slider("v0", 0.01, 0.16, 0.04, 0.01)
    kappa = st.slider("kappa (mean reversion)", 0.5, 6.0, 3.0, 0.5)
    theta_h = st.slider("theta (long-run var)", 0.01, 0.16, 0.04, 0.01)
    xi = st.slider("xi (vol of vol)", 0.1, 0.8, 0.4, 0.05)
    rho = st.slider("rho (spot-vol corr)", -0.95, 0.0, -0.7, 0.05)

with st.sidebar.expander("MARKET DATA", expanded=False):
    use_live = st.checkbox("Pull live SPY (yfinance)", value=False)
    _today = datetime.now().date()
    if use_live:
        date_start = st.date_input("Start date", value=_today - timedelta(days=365 * 3),
                                   min_value=_today - timedelta(days=365 * 20), max_value=_today)
        date_end = st.date_input("End date", value=_today,
                                 min_value=date_start, max_value=_today)
    else:
        date_start, date_end = None, None
    st.caption("Off uses the embedded 3y snapshot; live needs network on Streamlit Cloud.")

# ============================================================ HEADER
price0 = float(bs_price(S0, K, T, r, sigma))
delta0 = float(bs_delta(S0, K, T, r, sigma))
gamma0 = float(bs_gamma(S0, K, T, r, sigma))
theta0 = float(bs_theta(S0, K, T, r, sigma))
vega0 = float(bs_vega(S0, K, T, r, sigma))
moneyness = "ATM" if abs(S0 / K - 1) < 0.02 else ("ITM" if S0 > K else "OTM")

st.markdown(
    f"""
    <div class="term-head">
      <div>
        <div class="term-title">DELTA HEDGING TERMINAL</div>
        <div class="term-sub">REPLICATION ERROR &middot; TRANSACTION COSTS &middot; MODEL RISK</div>
      </div>
      <div class="term-tag">LIVE BOOK &middot; LONG 1 CALL</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="ribbon">
      <div class="cell"><div class="k">CONTRACT</div><div class="v">{moneyness} &middot; K={K:.0f}</div></div>
      <div class="cell"><div class="k">MATURITY</div><div class="v">{T:.2g}y</div></div>
      <div class="cell"><div class="k">IMPLIED VOL</div><div class="v">{sigma:.0%}</div></div>
      <div class="cell"><div class="k">OPTION PRICE</div><div class="v">{price0:.3f}</div></div>
      <div class="cell"><div class="k">HEDGE SHARES</div><div class="v">{delta0:+.3f}</div></div>
      <div class="cell"><div class="k">REBAL / YR</div><div class="v">{N_single}</div></div>
      <div class="cell"><div class="k">COST</div><div class="v">{cost_bps:.0f}bp</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "PRICING & GREEKS", "REPLICATION", "ATTRIBUTION",
    "COST-AWARE", "MODEL RISK", "RISK REPORT",
])

# ------------------------------------------------------------ TAB 1: PRICING
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Price", f"{price0:.4f}")
    c[1].metric("Delta", f"{delta0:.4f}")
    c[2].metric("Gamma", f"{gamma0:.5f}")
    c[3].metric("Theta / yr", f"{theta0:.4f}")
    c[4].metric("Vega / vol pt", f"{vega0/100:.4f}")

    left, right = st.columns([1, 1])

    with left:
        Sg = np.linspace(0.5 * K, 1.5 * K, 200)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=Sg, y=bs_delta(Sg, K, T, r, sigma), name="delta",
                                 line=dict(color=C["blue"], width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=Sg, y=bs_gamma(Sg, K, T, r, sigma), name="gamma",
                                 line=dict(color=C["amber"], width=2)), secondary_y=True)
        fig.add_vline(x=S0, line=dict(color=C["muted"], width=1, dash="dot"))
        fig.add_vline(x=K, line=dict(color=C["red"], width=1, dash="dot"))
        fig = dark(fig, "Delta and gamma across spot", height=380)
        fig.update_yaxes(title_text="delta", secondary_y=False)
        fig.update_yaxes(title_text="gamma", secondary_y=True, showgrid=False)
        fig.update_xaxes(title_text="spot S")
        st.plotly_chart(fig, width="stretch", key="t1_greeks")

    with right:
        # Finite-difference validation
        hh = 1e-4
        fd_delta = (bs_price(S0 + hh, K, T, r, sigma) - bs_price(S0 - hh, K, T, r, sigma)) / (2 * hh)
        fd_gamma = (bs_price(S0 + hh, K, T, r, sigma) - 2 * bs_price(S0, K, T, r, sigma)
                    + bs_price(S0 - hh, K, T, r, sigma)) / hh**2
        fd_theta = -(bs_price(S0, K, T + hh, r, sigma) - bs_price(S0, K, T - hh, r, sigma)) / (2 * hh)
        fd_vega = (bs_price(S0, K, T, r, sigma + hh) - bs_price(S0, K, T, r, sigma - hh)) / (2 * hh)
        val = pd.DataFrame({
            "quantity": ["delta", "gamma", "theta", "vega"],
            "analytic": [delta0, gamma0, theta0, vega0],
            "finite diff": [float(fd_delta), float(fd_gamma), float(fd_theta), float(fd_vega)],
        })
        val["abs error"] = (val["analytic"] - val["finite diff"]).abs()
        st.markdown("**Greeks validated against central finite differences**")
        st.dataframe(val.style.format({"analytic": "{:.6f}", "finite diff": "{:.6f}",
                                       "abs error": "{:.2e}"}),
                     width="stretch", hide_index=True)
        put_parity = price0 - S0 + K * np.exp(-r * T)
        note(f"Put-call parity check: <b>C &minus; S + Ke<sup>&minus;rT</sup> = {put_parity:.4f}</b> "
             f"(the implied put value, positive as required).")

    st.markdown("---")
    st.markdown("**The one-day hedged P&L identity**")
    cc = st.columns([3, 1])
    with cc[1]:
        t_rem = st.slider("Time remaining (yr)", 0.05, T, min(0.5, T), 0.05, key="t1_trem")
        dt_day = 1 / 252
    V0_ = float(bs_price(S0, K, t_rem, r, sigma))
    d_ = float(bs_delta(S0, K, t_rem, r, sigma))
    g_ = float(bs_gamma(S0, K, t_rem, r, sigma))
    moves = np.linspace(-0.08 * S0, 0.08 * S0, 321)
    actual = np.array([
        (float(bs_price(S0 + dS, K, t_rem - dt_day, r, sigma)) - V0_)
        - d_ * dS + r * (d_ * S0 - V0_) * dt_day for dS in moves])
    identity = 0.5 * g_ * (moves**2 - sigma**2 * S0**2 * dt_day)
    be = sigma * S0 * np.sqrt(dt_day)
    with cc[0]:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=moves, y=actual, name="actual (full reprice)",
                                 line=dict(color=C["blue"], width=2.5)))
        fig.add_trace(go.Scatter(x=moves, y=identity, name="\u00bd\u0393(dS\u00b2 \u2212 \u03c3\u00b2S\u00b2dt)",
                                 line=dict(color=C["amber"], width=1.6, dash="dash")))
        fig.add_hline(y=0, line=dict(color=C["border2"], width=1))
        fig.add_vline(x=be, line=dict(color=C["muted"], width=1, dash="dot"))
        fig.add_vline(x=-be, line=dict(color=C["muted"], width=1, dash="dot"))
        fig = dark(fig, "One-day P&L of the delta-hedged call", height=380)
        fig.update_xaxes(title_text="one-day move  dS")
        fig.update_yaxes(title_text="hedged P&L")
        st.plotly_chart(fig, width="stretch", key="t1_identity")
    with cc[1]:
        st.metric("Breakeven move", f"±{be:.3f}", f"{be/S0:.2%} of spot")
        mask = np.abs(moves) <= 0.04 * S0
        st.metric("Max |actual − id| (3σ)", f"{np.abs(actual-identity)[mask].max():.4f}")
    note("Once delta-hedged, the day's P&L is pure convexity: the option earns "
         "<b>½&Gamma; dS²</b> from the realized move and pays <b>½&Gamma;&sigma;²S²dt</b> in theta rent. "
         "It breaks even exactly when the move equals one implied standard deviation.")

# ------------------------------------------------------------ TAB 2: REPLICATION
with tabs[1]:
    note("A frictionless discrete hedge leaves mean-zero noise whose standard deviation "
         "falls like <b>1/&radic;N</b> in the rebalancing frequency. It is the discretization "
         "error of quadratic variation, seen through gamma.")
    Ns = np.array([8, 16, 32, 64, 128, 256, 512])
    means, stds = conv_sweep(tuple(Ns.tolist()), n_paths, seed, S0, K, T, r, sigma)
    slope = float(np.polyfit(np.log(Ns), np.log(stds), 1)[0])
    b_const = float(np.mean(stds * np.sqrt(Ns)))

    cset = st.columns(3)
    cset[0].metric("Fitted log-log slope", f"{slope:.3f}", "theory −0.500")
    cset[1].metric("Error constant  b", f"{b_const:.2f}")
    cset[2].metric("Daily std (N=252)", f"{np.interp(252, Ns, stds):.3f}")

    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Ns, y=stds, name=f"measured (slope {slope:.3f})",
                                 mode="lines+markers", line=dict(color=C["blue"], width=2),
                                 marker=dict(size=7)))
        ref = stds[0] * np.sqrt(Ns[0] / Ns)
        fig.add_trace(go.Scatter(x=Ns, y=ref, name="reference ∝ 1/√N",
                                 line=dict(color=C["amber"], width=1.5, dash="dash")))
        fig = dark(fig, "The 1/√N law", height=400, logx=True, logy=True)
        fig.update_xaxes(title_text="rebalances per year  N")
        fig.update_yaxes(title_text="std of replication P&L")
        st.plotly_chart(fig, width="stretch", key="t2_law")
    with right:
        pm, pdl = freq_distribution(min(n_paths * 2, 20000), seed + 3, S0, K, T, r, sigma)
        fig = go.Figure()
        bins = dict(start=-7, end=7, size=0.25)
        fig.add_trace(go.Histogram(x=pm, name=f"monthly N=12  σ={pm.std():.2f}",
                                   marker_color=C["muted"], opacity=0.6, xbins=bins))
        fig.add_trace(go.Histogram(x=pdl, name=f"daily N=252  σ={pdl.std():.2f}",
                                   marker_color=C["blue"], opacity=0.8, xbins=bins))
        fig.add_vline(x=0, line=dict(color=C["text"], width=1))
        fig.update_layout(barmode="overlay")
        fig = dark(fig, "Same world, only frequency differs", height=400)
        fig.update_xaxes(title_text="replication P&L at expiry")
        fig.update_yaxes(title_text="paths")
        st.plotly_chart(fig, width="stretch", key="t2_dist")

    tbl = pd.DataFrame({"N": Ns, "mean": means, "std": stds, "std·√N": stds * np.sqrt(Ns)})
    st.dataframe(tbl.style.format({"mean": "{:+.4f}", "std": "{:.4f}", "std·√N": "{:.3f}"}),
                 width="stretch", hide_index=True)

# ------------------------------------------------------------ TAB 3: ATTRIBUTION
with tabs[2]:
    note("Repricing splits the hedged P&L exactly into a non-negative <b>convexity (gamma) "
         "stream</b> and a non-positive <b>decay (theta) stream</b>. Their sum telescopes to "
         "the realized hedge error to machine precision. Net sign follows realized vs implied vol.")
    cc = st.columns([1, 3])
    with cc[0]:
        path_seed = st.number_input("Path seed", 0, 9999, 11, 1, key="t3_seed")
        if st.button("Resample path", key="t3_resample"):
            st.session_state["t3_seed"] = int(np.random.default_rng().integers(0, 9999))
            st.rerun()
    path = make_gbm_paths(252, 1, int(path_seed), S0, T, sigma, 0.0)[0]
    cum_g, cum_t = attribute_pnl(path, K, T, sigma)
    exact = float(hedge_pnl(path[None, :], S0, K, T, sigma, 0.0)[0])
    tie = abs((cum_g[-1] + cum_t[-1]) - exact)
    rv = realized_vol(path, T)

    with cc[0]:
        st.metric("Net hedge P&L", f"{cum_g[-1]+cum_t[-1]:+.3f}")
        st.metric("Realized vol", f"{rv:.3f}", f"implied {sigma:.3f}")
        st.metric("Tie-out error", f"{tie:.1e}")

    with cc[1]:
        tgrid = np.linspace(0, T, 253)
        fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6],
                            subplot_titles=("the path", "exact attribution"))
        fig.add_trace(go.Scatter(x=tgrid, y=path, line=dict(color=C["blue"], width=1.4),
                                 showlegend=False), row=1, col=1)
        fig.add_hline(y=K, line=dict(color=C["muted"], width=1, dash="dot"), row=1, col=1)
        tg = np.linspace(T / 252, T, 252)
        fig.add_trace(go.Scatter(x=tg, y=cum_g, name=f"gamma {cum_g[-1]:+.2f}",
                                 line=dict(color=C["green"], width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=tg, y=cum_t, name=f"theta {cum_t[-1]:+.2f}",
                                 line=dict(color=C["red"], width=2)), row=1, col=2)
        fig.add_trace(go.Scatter(x=tg, y=cum_g + cum_t, name=f"net {cum_g[-1]+cum_t[-1]:+.2f}",
                                 line=dict(color=C["text"], width=2)), row=1, col=2)
        fig.add_hline(y=0, line=dict(color=C["border2"], width=1), row=1, col=2)
        fig = dark(fig, height=420)
        fig.update_xaxes(title_text="years", row=1, col=1)
        fig.update_xaxes(title_text="years", row=1, col=2)
        fig.update_yaxes(title_text="price", row=1, col=1)
        fig.update_yaxes(title_text="cumulative P&L", row=1, col=2)
        for ann in fig.layout.annotations:
            ann.font.size = 12; ann.font.color = C["muted"]
        st.plotly_chart(fig, width="stretch", key="t3_attr")

# ------------------------------------------------------------ TAB 4: COST-AWARE
with tabs[3]:
    note("Proportional costs grow like <b>&radic;N</b> while replication risk falls like "
         "<b>1/&radic;N</b>, so the all-in objective <b>J = E[cost] + &lambda;&middot;risk</b> has an "
         "interior optimum. A move-based <b>band</b> rule dominates any fixed clock at matched cost.")
    Ns_f = np.array([5, 10, 20, 40, 80, 160, 320, 640])
    bands = np.array([0.01, 0.02, 0.05, 0.10, 0.16])
    risk_f, cost_f = cost_frontier(tuple(Ns_f.tolist()), n_paths, seed, S0, K, T, r, sigma, kc)
    band_risk, band_cost = band_frontier(tuple(bands.tolist()), n_paths, seed, S0, K, T, r, sigma, kc)
    J = cost_f + lam_risk * risk_f
    N_star = int(Ns_f[int(np.argmin(J))])
    A = np.vstack([np.ones_like(Ns_f, dtype=float), np.sqrt(Ns_f)]).T
    c0_fit, a_fit = np.linalg.lstsq(A, cost_f, rcond=None)[0]
    b_for_star = float(np.mean(risk_f * np.sqrt(Ns_f)))
    N_star_theory = lam_risk * b_for_star / a_fit if a_fit > 0 else float("nan")

    cset = st.columns(3)
    cset[0].metric("Optimal frequency  N*", f"{N_star}", f"~every {252/N_star:.1f} trading days")
    cset[1].metric("Theory  N* = λb/a", f"{N_star_theory:.0f}")
    dt_d = 1 / 252
    sig_hat = np.sqrt(sigma**2 + 2 * kc * sigma * np.sqrt(2 / np.pi) / np.sqrt(dt_d))
    leland_prem = float(bs_price(S0, K, T, r, sig_hat) - bs_price(S0, K, T, r, sigma))
    cset[2].metric("Leland adj vol  σ̂", f"{sig_hat:.4f}", f"premium {leland_prem:+.3f}")

    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cost_f, y=risk_f, name="clock rule",
                                 mode="lines+markers", line=dict(color=C["muted"], width=2),
                                 marker=dict(size=7),
                                 text=[f"N={n}" for n in Ns_f], textposition="top right"))
        fig.add_trace(go.Scatter(x=band_cost, y=band_risk, name="band rule",
                                 mode="lines+markers", line=dict(color=C["blue"], width=2),
                                 marker=dict(size=8, symbol="square"),
                                 text=[f"h={h:.2f}" for h in bands], textposition="bottom right"))
        fig = dark(fig, "Cost-risk frontier", height=420)
        fig.update_xaxes(title_text="expected cost of running the hedge")
        fig.update_yaxes(title_text="replication risk (std, pre-cost)")
        st.plotly_chart(fig, width="stretch", key="t4_frontier")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Ns_f, y=J, name="J = cost + λ·risk",
                                 mode="lines+markers", line=dict(color=C["amber"], width=2),
                                 marker=dict(size=7)))
        fig.add_vline(x=N_star, line=dict(color=C["green"], width=1.5, dash="dash"))
        fig = dark(fig, "All-in objective vs frequency", height=420, logx=True)
        fig.update_xaxes(title_text="rebalances per year  N")
        fig.update_yaxes(title_text="J")
        st.plotly_chart(fig, width="stretch", key="t4_objective")

    order = np.argsort(cost_f)
    rows = []
    for h, rk, mc in zip(bands, band_risk, band_cost):
        rk_clock = float(np.interp(mc, cost_f[order], risk_f[order]))
        rows.append((h, mc, rk, rk_clock, (rk_clock - rk) / rk_clock if rk_clock else 0.0))
    bt = pd.DataFrame(rows, columns=["band h", "cost", "band risk", "clock risk @ cost", "improvement"])
    st.dataframe(bt.style.format({"band h": "{:.2f}", "cost": "{:.3f}", "band risk": "{:.4f}",
                                  "clock risk @ cost": "{:.4f}", "improvement": "{:.1%}"}),
                 width="stretch", hide_index=True)

# ------------------------------------------------------------ TAB 5: MODEL RISK
with tabs[4]:
    sub = st.radio("Risk factor", ["Jump-diffusion floor", "Stochastic-vol leak", "Drift invariance"],
                   horizontal=True, label_visibility="collapsed", key="t5_sub")
    Ns_m = np.array([8, 16, 32, 64, 128, 252])

    if sub == "Jump-diffusion floor":
        note("Variance-matched to the hedger's implied vol, a jump world delivers the same total "
             "variance but in <b>uncrossable gaps</b>. The error refuses to shrink with frequency, "
             "the long hedger's P&L turns violently right-skewed, and a systematic loss appears "
             "because gamma is local.")
        g_std, m_std, pg, pm, sig_diff = jump_compare(
            tuple(Ns_m.tolist()), n_paths, seed, S0, K, T, r, sigma, lam_j, mu_j, sig_j)
        cset = st.columns(3)
        cset[0].metric("Jump-world skew", f"{float(skew(pm)):+.2f}", f"GBM {float(skew(pg)):+.2f}")
        se = pm.std() / np.sqrt(len(pm))
        cset[1].metric("Mean P&L (jump)", f"{pm.mean():+.3f}", f"{abs(pm.mean())/se:.1f}σ from 0")
        cset[2].metric("Matched diffusion vol", f"{sig_diff:.3f}", f"prices at {sigma:.2f}")

        left, right = st.columns([1, 1])
        with left:
            fig = go.Figure()
            bins = dict(start=-6, end=10, size=0.3)
            fig.add_trace(go.Histogram(x=pg, name="Black-Scholes world",
                                       marker_color=C["muted"], opacity=0.6, xbins=bins))
            fig.add_trace(go.Histogram(x=pm, name="jump world (same variance)",
                                       marker_color=C["blue"], opacity=0.8, xbins=bins))
            fig.add_vline(x=0, line=dict(color=C["text"], width=1))
            fig.update_layout(barmode="overlay")
            fig = dark(fig, "Variance-matched, model-mismatched", height=400)
            fig.update_xaxes(title_text="daily-hedged P&L (long option)")
            fig.update_yaxes(title_text="paths")
            st.plotly_chart(fig, width="stretch", key="t5_jhist")
        with right:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=Ns_m, y=g_std, name="Black-Scholes",
                                     mode="lines+markers", line=dict(color=C["muted"], width=2)))
            fig.add_trace(go.Scatter(x=Ns_m, y=m_std, name="jump world",
                                     mode="lines+markers", line=dict(color=C["blue"], width=2),
                                     marker=dict(symbol="square")))
            fig.add_trace(go.Scatter(x=Ns_m, y=g_std[0] * np.sqrt(Ns_m[0] / Ns_m),
                                     name="∝ 1/√N", line=dict(color=C["amber"], width=1.4, dash="dash")))
            fig = dark(fig, "The 1/√N law dies under jumps", height=400, logx=True, logy=True)
            fig.update_xaxes(title_text="rebalances per year  N")
            fig.update_yaxes(title_text="std of replication P&L")
            st.plotly_chart(fig, width="stretch", key="t5_jconv")

    elif sub == "Stochastic-vol leak":
        note("Under Heston dynamics a <b>second risk factor</b> — the variance itself — leaks "
             "through as unhedged vega. Delta cannot reach it, so a floor appears. Unlike a jump it "
             "moves continuously, so part averages out and the floor sits below the jump floor.")
        feller = 2 * kappa * theta_h >= xi**2
        g_std, m_std, _, _, _ = jump_compare(
            tuple(Ns_m.tolist()), n_paths, seed, S0, K, T, r, sigma, lam_j, mu_j, sig_j)
        h_std = heston_sweep(tuple(Ns_m.tolist()), min(n_paths, 3000), seed,
                             S0, K, T, r, sigma, v0, kappa, theta_h, xi, rho)
        cset = st.columns(3)
        cset[0].metric("E[variance]", f"{theta_h:.3f}", f"implied {sigma**2:.3f}")
        cset[1].metric("Feller 2κθ ≥ ξ²", "holds" if feller else "violated",
                       f"{2*kappa*theta_h:.2f} vs {xi**2:.2f}")
        cset[2].metric("Heston floor (N=252)", f"{h_std[-1]:.3f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Ns_m, y=g_std, name="Black-Scholes",
                                 mode="lines+markers", line=dict(color=C["muted"], width=2)))
        fig.add_trace(go.Scatter(x=Ns_m, y=h_std, name="Heston (E[v] matched)",
                                 mode="lines+markers", line=dict(color=C["blue"], width=2),
                                 marker=dict(symbol="triangle-up", size=9)))
        fig.add_trace(go.Scatter(x=Ns_m, y=m_std, name="jump (for scale)",
                                 mode="lines+markers", line=dict(color=C["red"], width=1.6,
                                 dash="dot"), marker=dict(symbol="square")))
        fig = dark(fig, "The vega leak: a floor delta cannot reach", height=440, logx=True, logy=True)
        fig.update_xaxes(title_text="rebalances per year  N")
        fig.update_yaxes(title_text="std of replication P&L")
        st.plotly_chart(fig, width="stretch", key="t5_heston")

    else:
        note("The hedged P&L distribution is <b>invariant to the real-world drift &mu;</b>. A "
             "60-point swing in annual drift moves the unhedged book by tens of dollars and the "
             "hedged book by hundredths — the experimental content of risk-neutral pricing.")
        mus = (-0.20, 0.10, 0.40)
        res = drift_test(mus, n_paths, seed, S0, K, T, r, sigma)
        cols = [C["red"], C["muted"], C["blue"]]
        fig = go.Figure()
        rows = []
        for (mu, hedged, un_m, un_s), col in zip(res, cols):
            fig.add_trace(go.Histogram(x=hedged, name=f"drift {mu:+.0%}",
                                       marker_color=col, opacity=0.55,
                                       xbins=dict(start=-3, end=3, size=0.08)))
            rows.append((mu, float(hedged.mean()), float(hedged.std()), un_m, un_s))
        fig.add_vline(x=0, line=dict(color=C["text"], width=1))
        fig.update_layout(barmode="overlay")
        fig = dark(fig, "The hedged P&L does not know the drift", height=420)
        fig.update_xaxes(title_text="hedged P&L")
        fig.update_yaxes(title_text="paths")
        st.plotly_chart(fig, width="stretch", key="t5_drift")
        dt = pd.DataFrame(rows, columns=["drift μ", "hedged mean", "hedged std",
                                         "unhedged mean", "unhedged std"])
        st.dataframe(dt.style.format({"drift μ": "{:+.0%}", "hedged mean": "{:+.4f}",
                                      "hedged std": "{:.4f}", "unhedged mean": "{:+.2f}",
                                      "unhedged std": "{:.2f}"}),
                     width="stretch", hide_index=True)

# ------------------------------------------------------------ TAB 6: RISK REPORT
with tabs[5]:
    closes, label = load_spy(use_live,
                             str(date_start) if date_start else "",
                             str(date_end) if date_end else "")
    cal = calibrate_jumps(closes)
    st.markdown(f"<div style='font-size:11px;color:{C['muted']};letter-spacing:1px'>"
                f"DATA SOURCE &mdash; {label.upper()} &middot; {len(closes)} DAYS</div>",
                unsafe_allow_html=True)

    cset = st.columns(4)
    cset[0].metric("Jumps / year  λ̂", f"{cal['lam']:.2f}", f"{cal['n_flag']} flagged")
    cset[1].metric("Jump size (log)", f"{cal['mu_j']:+.3f}", f"±{cal['sig_j']:.3f}")
    cset[2].metric("Diffusion vol", f"{cal['sig_diff']:.3f}", "ex-jump")
    cset[3].metric("Total vol", f"{cal['sig_total']:.3f}", "diffusion + jump QV")

    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=closes, line=dict(color=C["blue"], width=1.1), showlegend=False))
        fig = dark(fig, "SPY, three years", height=360)
        fig.update_xaxes(title_text="trading day")
        fig.update_yaxes(title_text="adjusted close")
        st.plotly_chart(fig, width="stretch", key="t6_spy")
    with right:
        r_spy = cal["r"]
        jidx = np.where(cal["is_jump"])[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=r_spy, line=dict(color=C["muted"], width=0.7),
                                 name="daily log return"))
        fig.add_trace(go.Scatter(x=jidx, y=r_spy[jidx], mode="markers",
                                 marker=dict(color=C["red"], size=7),
                                 name=f"flagged jumps ({cal['n_flag']})"))
        fig.add_hline(y=cal["thresh"], line=dict(color=C["red"], width=1, dash="dot"))
        fig.add_hline(y=-cal["thresh"], line=dict(color=C["red"], width=1, dash="dot"))
        fig = dark(fig, "Threshold detection at 4 × 1.4826·MAD", height=360)
        fig.update_xaxes(title_text="trading day")
        fig.update_yaxes(title_text="daily log return")
        st.plotly_chart(fig, width="stretch", key="t6_jumps")

    st.markdown("---")
    st.markdown(f"**Morning risk report** — band hedge h={band_h:.2f}, cost {cost_bps:.0f}bp, "
                f"{n_paths} paths")

    sig_diff_bs = np.sqrt(max(sigma**2 - lam_j * (mu_j**2 + sig_j**2), 1e-8))
    worlds = [
        ("Black-Scholes", make_gbm_paths(252, n_paths, seed, S0, T, sigma, r), sigma),
        ("Merton jumps (stylized)",
         make_merton_paths(252, n_paths, seed, S0, T, r, sig_diff_bs, lam_j, mu_j, sig_j), sigma),
        ("Heston stochastic vol",
         make_heston_paths(252, min(n_paths, 4000), seed, S0, T, r, v0, kappa, theta_h, xi, rho), sigma),
        ("SPY-calibrated jumps",
         make_merton_paths(252, n_paths, seed, S0, T, r, cal["sig_diff"], cal["lam"], cal["mu_j"], cal["sig_j"]),
         cal["sig_total"]),
    ]
    results, rows = [], []
    for name, Sw, sig_use in worlds:
        repl, c = band_hedge(Sw, S0, K, T, sig_use, kc, band_h, r)
        net = repl - c
        va, cv = var_cvar(net)
        results.append((name, net, cv))
        rows.append((name, float(net.mean()), float(net.std()), va, cv, float(c.mean())))
    rep = pd.DataFrame(rows, columns=["world", "mean", "std", "VaR95", "CVaR95", "E[cost]"])
    st.dataframe(rep.style.format({"mean": "{:+.3f}", "std": "{:.3f}", "VaR95": "{:.3f}",
                                   "CVaR95": "{:.3f}", "E[cost]": "{:.3f}"}),
                 width="stretch", hide_index=True)

    cv_bs = results[0][2]
    tail = ",  ".join(f"{nm.split(' ')[0]} {cv/cv_bs:.1f}×" for nm, _, cv in results[1:])
    note(f"Tail multiple vs the priced (Black-Scholes) world: <b>{tail}</b>. A hedge does not "
         f"remove risk; it chooses which risks remain, and CVaR prices the remainder.")

    fig = make_subplots(rows=1, cols=4, shared_yaxes=True,
                        subplot_titles=[f"{nm}<br>CVaR95={cv:.2f}" for nm, _, cv in results])
    for i, (nm, net, cv) in enumerate(results, start=1):
        fig.add_trace(go.Histogram(x=net, marker_color=C["blue"], opacity=0.8,
                                   xbins=dict(start=-8, end=10, size=0.3), showlegend=False),
                      row=1, col=i)
        fig.add_vline(x=-cv, line=dict(color=C["red"], width=1.4, dash="dash"), row=1, col=i)
    fig = dark(fig, height=320)
    for ann in fig.layout.annotations:
        ann.font.size = 10; ann.font.color = C["muted"]
    st.plotly_chart(fig, width="stretch", key="t6_report")

st.markdown(
    f"<div style='text-align:center;color:{C['muted']};font-size:10px;letter-spacing:1px;"
    f"padding-top:18px'>DELTA HEDGING TERMINAL &middot; MONTE CARLO RESEARCH TOOL &middot; "
    f"NOT INVESTMENT ADVICE</div>",
    unsafe_allow_html=True,
)
