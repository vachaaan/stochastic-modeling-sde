#!/usr/bin/env python3
"""
CL 677 Project: Smooth Random Noise and the Stratonovich Limit
Author: Vachan & Avdhoot

This script reproduces the key figures and statistical validations
for smooth random functions and their application to the geometric
random walk problem, comparing the Itô and Stratonovich limits.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm, lognorm

# Set consistent plotting parameters
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "lines.linewidth": 1.0,
    "figure.dpi": 130,
})


# --- Core Noise Generator ---

def make_srfun(x, lam, L, seed=0, big=False):
    """
    Generates smooth random noise based on the Filip et al. paper.
    Uses an extended domain L' = 1.2L to avoid periodic artifacts.
    """
    x = np.asarray(x, dtype=float)
    Lp = 1.2 * L
    m = int(np.floor(Lp / lam))

    # Toggle variance based on normalization needed
    var = (2.0 / ((2 * m + 1) * lam)) if big else (1.0 / (2 * m + 1))
    std = np.sqrt(var)

    rng = np.random.default_rng(seed)
    coeffs = rng.normal(0.0, std, 2 * m + 1)

    a0 = coeffs[0]
    a = coeffs[1::2]
    b = coeffs[2::2]

    j = np.arange(1, m + 1, dtype=float)
    theta = (2 * np.pi / Lp) * np.outer(x, j)

    return a0 + np.sqrt(2.0) * (np.cos(theta) @ a + np.sin(theta) @ b)


def _integral_weights(T, Lp, m):
    """Pre-calculates weights for exact analytical integration to speed up the MC simulation."""
    j = np.arange(1, m + 1, dtype=float)
    phase = 2 * np.pi * j * T / Lp
    return (Lp / (2 * np.pi * j)) * np.sin(phase), (Lp / (2 * np.pi * j)) * (1 - np.cos(phase))


def mc_ensemble(lam, N_mc, mu, sigma, T, u0, seed, t_eval=None):
    """
    Vectorized Monte Carlo solver for the geometric random walk.
    If t_eval is given, it computes the running integral for the full path.
    Otherwise, it just calculates the final value at T.
    """
    Lp = 1.2 * T
    m = int(np.floor(Lp / lam))
    std = np.sqrt(2.0 / ((2 * m + 1) * lam))

    rng = np.random.default_rng(seed)
    C = rng.normal(0.0, std, (N_mc, 2 * m + 1))

    a0 = C[:, 0]
    a = C[:, 1::2]
    b = C[:, 2::2]

    if t_eval is None:
        wc, ws = _integral_weights(T, Lp, m)
        integral = a0 * T + np.sqrt(2.0) * (a @ wc + b @ ws)
        return u0 * np.exp(mu * T + sigma * integral)

    t_eval = np.asarray(t_eval, dtype=float)
    j = np.arange(1, m + 1, dtype=float)
    phase = np.outer(t_eval, 2 * np.pi * j / Lp)
    Lp_2pij = Lp / (2 * np.pi * j)

    wc = Lp_2pij * np.sin(phase)
    ws = Lp_2pij * (1 - np.cos(phase))

    integral = (a0[:, None] * t_eval[None, :] + np.sqrt(2.0) * (a @ wc.T + b @ ws.T))
    return u0 * np.exp(mu * t_eval[None, :] + sigma * integral)


def mc_stats(lam, N_mc, mu, sigma, T, u0, seed=0, t_eval=None):
    """Helper function to extract mean and std metrics for the ensemble."""
    U = mc_ensemble(lam, N_mc, mu, sigma, T, u0, seed, t_eval)
    return U.mean(axis=0), U.std(axis=0, ddof=1), U.std(axis=0, ddof=1) / np.sqrt(N_mc)


# --- Plotting Functions ---

def fig1_normalisations(seed=5, out="fig1_normalisations.png"):
    x = np.linspace(-1.0, 1.0, 5000)
    L = 2.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5))
    fig.suptitle(
        "Fig 1 – Smooth random functions on [−1,1]\n"
        "Left: standard norm  $a_j,b_j\\sim\\mathcal{N}(0,1/(2m{+}1))$   |   "
        "Right: big norm  $a_j,b_j\\sim\\mathcal{N}(0,\\,2/((2m{+}1)\\lambda))$",
        fontsize=10, y=1.02,
    )

    titles = ["standard  (amplitude $O(1)$)", "big  (amplitude $O(\\lambda^{-1/2})$)"]
    for c, tt in enumerate(titles):
        axes[0, c].set_title(tt, fontsize=10)

    for r, lam in enumerate([0.1, 0.025]):
        m = int(np.floor(1.2 * L / lam))
        for c, big in enumerate([False, True]):
            f = make_srfun(x, lam, L, seed=seed, big=big)
            ax = axes[r, c]
            ax.plot(x, f, color="#2171b5", lw=0.7)
            ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.4)
            ax.set_xlim(-1, 1)

            amp = f"$O(1)$" if not big else f"$O(\\lambda^{{-1/2}})={1 / lam ** 0.5:.1f}$"
            ax.text(0.97, 0.95, f"$\\lambda={lam}$,  $m={m}$\namp {amp}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))
            ax.set_xlabel("$x$")
        axes[r, 0].set_ylabel("$f(x)$")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Fig 1] saved → {out}")


def fig2_statistical_validation(out="fig2_statistics.png"):
    np.random.seed(0)
    N_rep = 20_000
    L = 2.0
    x0 = np.linspace(-1.0, 1.0, 600)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle("Fig 2 – Statistical Properties of Smooth Random Functions", fontsize=12)

    # (a) Pointwise distribution
    ax = fig.add_subplot(gs[0, 0])
    lam = 0.1
    vals = [make_srfun(np.array([0.0]), lam, L, seed=k, big=False)[0] for k in range(N_rep)]
    vals = np.array(vals)
    ax.hist(vals, bins=60, density=True, color="#4292c6", alpha=0.75, label=f"Empirical  (n={N_rep})")
    xx = np.linspace(-4, 4, 300)
    ax.plot(xx, norm.pdf(xx, 0, 1), "r-", lw=2.0, label=r"$\mathcal{N}(0,1)$ theory")
    ax.set_xlabel("$f(0)$")
    ax.set_ylabel("density")
    ax.set_title(f"(a) Pointwise distribution  $\\lambda={lam}$ (standard)")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.97, f"mean={vals.mean():.3f}\nstd={vals.std():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.7))

    # (b) Covariance Kernel
    ax = fig.add_subplot(gs[0, 1])
    lam = 0.2
    Lp = 1.2 * L
    m = int(np.floor(Lp / lam))

    rng = np.random.default_rng(7)
    N_c = 5_000
    std = np.sqrt(1.0 / (2 * m + 1))

    C_mat = rng.normal(0, std, (N_c, 2 * m + 1))
    a0_c = C_mat[:, 0]
    a_c = C_mat[:, 1::2]
    b_c = C_mat[:, 2::2]
    j = np.arange(1, m + 1, dtype=float)

    theta = (2 * np.pi / Lp) * np.outer(x0, j)
    F_x = a0_c[:, None] + np.sqrt(2.0) * (a_c @ np.cos(theta).T + b_c @ np.sin(theta).T)
    F_0 = a0_c + np.sqrt(2.0) * a_c.sum(axis=1)

    cov_emp = (F_x * F_0[:, None]).mean(axis=0)

    x_safe = x0.copy()
    x_safe[np.abs(x_safe) < 1e-10] = 1e-10
    D_theory = np.sin((2 * m + 1) * np.pi * x_safe / Lp) / ((2 * m + 1) * np.sin(np.pi * x_safe / Lp))

    ax.plot(x0, cov_emp, color="#4292c6", lw=1.2, label="Empirical $C(x,0)$")
    ax.plot(x0, D_theory, "r--", lw=1.8, label="Dirichlet kernel $D(x)$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$C(x,0)$")
    ax.set_title(f"(b) Covariance kernel  $\\lambda={lam}$,  $m={m}$")
    ax.legend(fontsize=9)

    # (c) Power spectrum
    ax = fig.add_subplot(gs[1, 0])
    x_long = np.linspace(-1.0, 1.0, 2048)
    h = x_long[1] - x_long[0]
    colors = ["#08519c", "#2171b5", "#4292c6", "#6baed6"]

    for col, lam_ps in zip(colors, [0.4, 0.2, 0.1, 0.05]):
        f_ps = make_srfun(x_long, lam_ps, L, seed=3, big=False)
        freqs = np.fft.rfftfreq(len(x_long), d=h)
        power = np.abs(np.fft.rfft(f_ps)) ** 2 / len(x_long)
        k_plot = 2 * np.pi * freqs
        ax.semilogy(k_plot, power + 1e-12, lw=0.85, alpha=0.85,
                    label=f"$\\lambda={lam_ps}$  cutoff $k^*={2 * np.pi / lam_ps:.0f}$")
        ax.axvline(2 * np.pi / lam_ps, color=col, ls=":", lw=1.0, alpha=0.6)

    ax.set_xlim(0, 400)
    ax.set_ylim(1e-7, 1e1)
    ax.set_xlabel("Angular wavenumber $k$")
    ax.set_ylabel("Power")
    ax.set_title("(c) Power spectrum  (flat below $k^*=2\\pi/\\lambda$, zero above)")
    ax.legend(fontsize=8)

    # (d) Big-norm pointwise variance vs lambda
    ax = fig.add_subplot(gs[1, 1])
    lams = np.array([0.5, 0.2, 0.1, 0.05, 0.025, 0.01])
    vars_emp, vars_theory = [], []

    for la in lams:
        v = [make_srfun(np.array([0.5]), la, L, seed=k, big=True)[0] ** 2 for k in range(3000)]
        vars_emp.append(np.mean(v))
        vars_theory.append(2.0 / la)

    ax.loglog(lams, vars_emp, "o-", color="#2171b5", ms=7, lw=1.5, label="Empirical Var$[f(x)]$")
    ax.loglog(lams, vars_theory, "r--", lw=2.0, label=r"Theory $2/\lambda$")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Pointwise variance")
    ax.set_title("(d) Big-norm pointwise variance $\\to 2/\\lambda$")
    ax.legend(fontsize=9)
    ax.invert_xaxis()

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Fig 2] saved → {out}")


def fig3_random_walks(seed=42, out="fig3_random_walks.png"):
    t = np.linspace(0.0, 1.0, 10_000)
    L = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    fig.suptitle(
        r"Fig 3 – Smooth random walks  $u(t)=\int_0^t f_{\rm big}(s)\,ds$  "
        r"$\longrightarrow$  Brownian path as $\lambda\to 0$   (same seed)",
        fontsize=11,
    )
    for ax, lam in zip(axes, [0.2, 0.04, 0.008]):
        f = make_srfun(t, lam, L, seed=seed, big=True)
        u = cumulative_trapezoid(f, t, initial=0.0)
        m = int(np.floor(1.2 * L / lam))

        ax.plot(t, u, color="#2171b5", lw=0.75)
        ax.fill_between(t, u, alpha=0.12, color="#2171b5")
        ax.axhline(0, color="k", lw=0.4, ls="--", alpha=0.5)
        ax.set_title(f"$\\lambda={lam}$,  $m={m}$ modes")
        ax.set_xlabel("$t$")
        ax.set_xlim(0, 1)
        ax.text(0.03, 0.03, f"$u(1)={u[-1]:.3f}$", transform=ax.transAxes, fontsize=9, bbox=dict(fc="white", alpha=0.7))

    axes[0].set_ylabel("$u(t)$")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Fig 3] saved → {out}")


def fig4_brownian_convergence(out="fig4_brownian_convergence.png"):
    from scipy.stats import kstest

    T = 1.0
    L = 1.0
    N_mc = 5_000
    lams = np.array([0.5, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.002])

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle("Fig 4 – Convergence of ∫₀ᵗ f_big to Brownian Motion", fontsize=12)

    # (a) Sample path refinement
    ax = fig.add_subplot(gs[0, 0])
    t_plt = np.linspace(0, T, 3000)
    pal = {"0.20": "#08519c", "0.04": "#2171b5", "0.008": "#6baed6"}

    for lam_s, col in zip([0.2, 0.04, 0.008], pal.values()):
        for k in range(10):
            f = make_srfun(t_plt, lam_s, L, seed=k, big=True)
            u = cumulative_trapezoid(f, t_plt, initial=0.0)
            kw = dict(lw=0.6, alpha=0.6, color=col)
            ax.plot(t_plt, u, **kw, label=f"$\\lambda={lam_s}$" if k == 0 else "")

    ax.set_xlabel("$t$")
    ax.set_ylabel("$W(t)=\\int_0^t f_{\\rm big}$")
    ax.set_title("(a) 10 paths per λ  (same seeds)")
    ax.legend(fontsize=8, loc="upper left")

    # (b) Var[W(T)] vs lambda
    ax = fig.add_subplot(gs[0, 1])
    vars_W, sems_W = [], []
    print("\n  Brownian convergence: Var[W(1)] → T = 1.0")
    print(f"  {'λ':>8}   {'Var[W(T)]':>11}   {'±SEM':>8}")
    print("  " + "-" * 35)

    for la in lams:
        Lp = 1.2 * T
        m = int(np.floor(Lp / la))
        std = np.sqrt(2.0 / ((2 * m + 1) * la))
        rng = np.random.default_rng(0)

        C = rng.normal(0, std, (N_mc, 2 * m + 1))
        a0 = C[:, 0]
        a = C[:, 1::2]
        b = C[:, 2::2]

        wc, ws = _integral_weights(T, Lp, m)
        W = a0 * T + np.sqrt(2.0) * (a @ wc + b @ ws)
        v = W.var(ddof=1)
        se = v * np.sqrt(2 / (N_mc - 1))

        vars_W.append(v)
        sems_W.append(se)
        print(f"  {la:8.3f}   {v:11.5f}   {se:8.5f}")

    ax.errorbar(lams, vars_W, yerr=2 * np.array(sems_W), fmt="o-", color="#2171b5", ms=7, capsize=3, lw=1.8,
                label="Empirical Var$[W(T)]$  (±2 SEM)")
    ax.axhline(T, color="r", ls="--", lw=2, label=f"Theory  $T={T}$")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Var$[W(T)]$")
    ax.set_title(f"(b) Variance of $W(T)=\\int_0^T f_{{\\rm big}}\\,dt$  →  $T={T}$")
    ax.legend(fontsize=9)

    # (c) KS test
    ax = fig.add_subplot(gs[1, 0])
    pvals = []

    for la in lams:
        Lp = 1.2 * T
        m = int(np.floor(Lp / la))
        std = np.sqrt(2.0 / ((2 * m + 1) * la))
        rng = np.random.default_rng(1)

        C = rng.normal(0, std, (N_mc, 2 * m + 1))
        a0 = C[:, 0]
        a = C[:, 1::2]
        b = C[:, 2::2]

        wc, ws = _integral_weights(T, Lp, m)
        W = a0 * T + np.sqrt(2.0) * (a @ wc + b @ ws)
        stat, p = kstest(W / np.sqrt(T), 'norm')
        pvals.append(p)

    ax.semilogx(lams, pvals, "s-", color="#2171b5", ms=8, lw=1.5)
    ax.axhline(0.05, color="r", ls="--", lw=1.5, label="p = 0.05 threshold")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("KS test p-value")
    ax.set_title("(c) KS test:  $W(T)/\\sqrt{T} \\sim \\mathcal{N}(0,1)$?")
    ax.invert_xaxis()
    ax.legend(fontsize=9)
    ax.text(0.05, 0.05, "p ≫ 0.05 → fail to reject normality", transform=ax.transAxes, fontsize=8, color="#2171b5")

    # (d) Var[W(t)] vs t
    ax = fig.add_subplot(gs[1, 1])
    t_eval = np.linspace(0.02, T, 40)
    la = 0.01
    Lp = 1.2 * T
    m = int(np.floor(Lp / la))
    std = np.sqrt(2.0 / ((2 * m + 1) * la))
    rng = np.random.default_rng(2)
    N2 = 8_000

    C = rng.normal(0, std, (N2, 2 * m + 1))
    a0c = C[:, 0]
    ac = C[:, 1::2]
    bc = C[:, 2::2]

    var_t = []
    for ti in t_eval:
        wc_i, ws_i = _integral_weights(ti, Lp, m)
        W_i = a0c * ti + np.sqrt(2.0) * (ac @ wc_i + bc @ ws_i)
        var_t.append(W_i.var(ddof=1))

    ax.plot(t_eval, var_t, "o-", color="#2171b5", ms=4, lw=1.2, label=f"Empirical Var$[W(t)]$  $\\lambda={la}$")
    ax.plot(t_eval, t_eval, "r--", lw=2.0, label="Theory  $t$  (Brownian)")
    ax.set_xlabel("$t$")
    ax.set_ylabel("Var$[W(t)]$")
    ax.set_title("(d) Var$[W(t)] \\to t$  (linear = Brownian signature)")
    ax.legend(fontsize=9)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Fig 4] saved → {out}")


def fig5_stratonovich_limit(
        mu=0.0, sigma=1.0, T=1.0, u0=1.0,
        N_mc=20_000, out="fig5_stratonovich_limit.png"
):
    E_ito = u0 * np.exp(mu * T)
    E_strat = u0 * np.exp((mu + 0.5 * sigma ** 2) * T)
    correction = 0.5 * sigma ** 2 * T
    lams = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002])

    print(f"\n  Geometric random walk  μ={mu}, σ={sigma}, T={T}, u₀={u0}")
    print(f"  Itô prediction        E[u(T)] = u₀·exp(μT)         = {E_ito:.4f}")
    print(f"  Stratonovich target   E[u(T)] = u₀·exp((μ+σ²/2)T) = {E_strat:.4f}")
    print(f"  Itô–Strat correction  σ²/2·T = {correction:.4f}")
    print(f"\n  {'λ':>7}  {'m':>5}  {'E[u(T)]':>10}  {'±2SEM':>9}  "
          f"{'→ Strat ratio':>14}  {'Bias from Strat':>16}")
    print("  " + "-" * 66)

    means, errs = [], []
    for la in lams:
        mu_e, std_e, sem_e = mc_stats(la, N_mc, mu, sigma, T, u0)
        means.append(float(mu_e))
        errs.append(float(sem_e))
        m = int(np.floor(1.2 * T / la))
        print(f"  {la:7.3f}  {m:5d}  {float(mu_e):10.4f}  {2 * float(sem_e):9.4f}  "
              f"{float(mu_e) / E_strat:14.4f}  {float(mu_e) - E_strat:16.4f}")

    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(lams, means, yerr=2 * np.array(errs),
                 fmt="o-", color="#2171b5", ms=8, lw=2.0, capsize=4,
                 label=f"MC  $E[u(T)]$  ±2 SEM  (N={N_mc:,})")
    ax1.axhline(E_strat, color="#d62728", ls="--", lw=2.2,
                label=rf"Stratonovich:  $u_0 e^{{(\mu+\sigma^2/2)T}} = {E_strat:.4f}$")
    ax1.axhline(E_ito, color="#2ca02c", ls="--", lw=2.2,
                label=rf"Itô:           $u_0 e^{{\mu T}} = {E_ito:.4f}$")

    ax1.fill_between([lams[-1] * 0.7, lams[0] * 1.3], E_ito, E_strat, alpha=0.07, color="gray",
                     label=f"Itô–Strat gap  = {E_strat - E_ito:.4f}")
    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.set_xlabel("$\\lambda$  (decreasing $\\rightarrow$ white-noise limit)", labelpad=6)
    ax1.set_ylabel("$E[u(T)]$")
    ax1.set_title(
        f"Fig 5 – Stratonovich limit  ($\\mu={mu},\\;\\sigma={sigma},\\;T={T}$)\n"
        r"$du/dt = (\mu + \sigma f)u$   [Smooth ODE $\to$ Stratonovich SDE as $\lambda\to 0$]"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(gs[1])
    bias = np.abs(np.array(means) - E_strat)
    ax2.loglog(lams, bias, "s-", color="#6a51a3", ms=8, lw=1.8, label="|MC − Stratonovich|")
    ax2.loglog(lams, lams * (bias[0] / lams[0]), "k--", lw=1.2, alpha=0.6, label="$\\propto \\lambda$ reference")
    ax2.invert_xaxis()
    ax2.set_xlabel("$\\lambda$")
    ax2.set_ylabel(r"$|E_{\rm MC}[u(T)] - E_{\rm Strat}[u(T)]|$")
    ax2.set_title("Bias toward Stratonovich\n(decreases as $\\lambda \\to 0$)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Fig 5] saved → {out}")


def fig6_time_evolution(
        mu=0.0, sigma=1.0, T=1.0, u0=1.0,
        N_mc=8_000, out="fig6_time_evolution.png"
):
    t_eval = np.linspace(0.0, T, 60)
    E_ito = u0 * np.exp(mu * t_eval)
    E_strat = u0 * np.exp((mu + 0.5 * sigma ** 2) * t_eval)

    lams = [0.5, 0.1, 0.02, 0.005]
    colors = ["#c7e9c0", "#74c476", "#238b45", "#00441b"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        r"Fig 6 – Time evolution of $E[u(t)]$  ($\mu=0,\;\sigma=1$)"
        "\nSmooth ODE tracks Stratonovich mean → not Itô mean",
        fontsize=11,
    )

    for ax in axes:
        ax.plot(t_eval, E_strat, "r--", lw=2.5, label=r"Stratonovich  $e^{(\mu+\sigma^2/2)t}$", zorder=10)
        ax.plot(t_eval, E_ito, "g--", lw=2.5, label=r"Itô  $e^{\mu t} = 1$", zorder=10)

    for ax_idx, ax in enumerate(axes):
        for lam, col in zip(lams, colors):
            U = mc_ensemble(lam, N_mc, mu, sigma, T, u0, seed=0, t_eval=t_eval)
            E = U.mean(axis=0)
            se = U.std(axis=0, ddof=1) / np.sqrt(N_mc)
            ax.plot(t_eval, E, "-", color=col, lw=1.8, label=f"$\\lambda={lam}$")

            if ax_idx == 1:
                ax.fill_between(t_eval, E - 2 * se, E + 2 * se, color=col, alpha=0.20)

        ax.set_xlabel("$t$")
        ax.set_xlim(0, T)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("$E[u(t)]$")
    axes[0].set_title("Mean trajectories")
    axes[1].set_title("Mean ± 2 SEM  shading")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Fig 6] saved → {out}")


def fig7_lognormal(
        mu=0.0, sigma=1.0, T=1.0, u0=1.0,
        N_mc=30_000, out="fig7_lognormal.png"
):
    lams_cdf = [0.2, 0.01]
    lams_scan = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])

    ln_mu = mu * T
    ln_sig = sigma * np.sqrt(T)

    E_strat = u0 * np.exp(ln_mu + 0.5 * ln_sig ** 2)
    E2_strat = u0 ** 2 * np.exp(2 * ln_mu + 2 * ln_sig ** 2)
    V_strat = E2_strat - E_strat ** 2
    skew_strat = (np.exp(ln_sig ** 2) + 2) * np.sqrt(np.exp(ln_sig ** 2) - 1)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle(
        "Fig 7 – Log-normal Distribution of Geometric Random Walk\n"
        r"$u(T) = u_0 e^{\mu T + \sigma W(T)}$  (Stratonovich)",
        fontsize=12,
    )

    # (a) PDF of u(T)
    ax = fig.add_subplot(gs[0, 0])
    cols = ["#2171b5", "#d62728"]

    for la, col in zip(lams_cdf, cols):
        U = mc_ensemble(la, N_mc, mu, sigma, T, u0, seed=0)
        ax.hist(U, bins=80, density=True, alpha=0.45, color=col, label=f"$\\lambda={la}$  (MC)")

    u_grid = np.linspace(0.01, np.percentile(mc_ensemble(0.01, 5000, mu, sigma, T, u0, seed=0), 99.5), 400)
    pdf_ln = lognorm.pdf(u_grid, s=ln_sig, scale=u0 * np.exp(ln_mu))

    ax.plot(u_grid, pdf_ln, "k-", lw=2.5,
            label=f"Log-Normal  $\\ln(u/u_0)\\sim\\mathcal{{N}}({ln_mu:.1f},{ln_sig ** 2:.2f})$")
    ax.axvline(E_strat, color="r", ls=":", lw=1.5, label=f"$E_{{\\rm Strat}}={E_strat:.3f}$")
    ax.set_xlabel("$u(T)$")
    ax.set_ylabel("density")
    ax.set_title("(a)  PDF of $u(T)$")
    ax.legend(fontsize=8)

    # (b) PDF of log u(T)
    ax = fig.add_subplot(gs[0, 1])
    for la, col in zip(lams_cdf, cols):
        U = mc_ensemble(la, N_mc, mu, sigma, T, u0, seed=0)
        logU = np.log(U / u0)
        ax.hist(logU, bins=80, density=True, alpha=0.45, color=col, label=f"$\\lambda={la}$")

    zz = np.linspace(ln_mu - 4 * ln_sig, ln_mu + 4 * ln_sig, 300)
    ax.plot(zz, norm.pdf(zz, ln_mu, ln_sig), "k-", lw=2.5, label=f"$\\mathcal{{N}}({ln_mu:.1f},{ln_sig ** 2:.2f})$")
    ax.set_xlabel("$\\ln(u(T)/u_0)$")
    ax.set_ylabel("density")
    ax.set_title("(b)  PDF of $\\ln(u(T)/u_0)$  →  normal")
    ax.legend(fontsize=8)

    # (c) Var[u(T)] vs lambda
    ax = fig.add_subplot(gs[1, 0])
    vars_u, sem_v = [], []
    print(f"\n  Var[u(T)] convergence  (theory = {V_strat:.4f})")
    print(f"  {'λ':>7}  {'Var[u(T)]':>11}  {'→ Strat':>9}")
    print("  " + "-" * 32)

    for la in lams_scan:
        U = mc_ensemble(la, N_mc, mu, sigma, T, u0, seed=1)
        v = float(U.var(ddof=1))
        vars_u.append(v)
        sem_v.append(v * np.sqrt(2 / (N_mc - 1)))
        print(f"  {la:7.3f}  {v:11.4f}  {v / V_strat:9.4f}")

    ax.errorbar(lams_scan, vars_u, yerr=2 * np.array(sem_v), fmt="o-", color="#2171b5", ms=7, capsize=3, lw=1.8,
                label="Empirical Var$[u(T)]$  ±2 SEM")
    ax.axhline(V_strat, color="r", ls="--", lw=2, label=f"Stratonovich  Var = {V_strat:.3f}")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Var$[u(T)]$")
    ax.set_title("(c) Variance of $u(T)$  →  log-normal prediction")
    ax.legend(fontsize=9)

    # (d) Skewness of u(T) vs lambda
    ax = fig.add_subplot(gs[1, 1])
    from scipy.stats import skew as sp_skew
    skews, sem_sk = [], []

    for la in lams_scan:
        U = mc_ensemble(la, N_mc, mu, sigma, T, u0, seed=2)
        sk = float(sp_skew(U))
        skews.append(sk)
        sem_sk.append(np.sqrt(6.0 / N_mc))

    ax.errorbar(lams_scan, skews, yerr=2 * np.array(sem_sk), fmt="o-", color="#6a51a3", ms=7, capsize=3, lw=1.8,
                label="Empirical skewness  ±2 SEM")
    ax.axhline(skew_strat, color="r", ls="--", lw=2, label=f"Log-normal skewness = {skew_strat:.3f}")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Skewness of $u(T)$")
    ax.set_title("(d) Skewness  →  log-normal value")
    ax.legend(fontsize=9)

    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Fig 7] saved → {out}")


# --- Execution ---

if __name__ == "__main__":
    t_start = time.time()

    SECTIONS = [
        ("Fig 1  Standard vs Big normalisation", fig1_normalisations),
        ("Fig 2  Statistical validation", fig2_statistical_validation),
        ("Fig 3  Smooth random walks", fig3_random_walks),
        ("Fig 4  Brownian convergence", fig4_brownian_convergence),
        ("Fig 5  Stratonovich limit  E[u(T)] vs λ", lambda: fig5_stratonovich_limit(N_mc=20_000)),
        ("Fig 6  Time evolution  E[u(t)]", lambda: fig6_time_evolution(N_mc=8_000)),
        ("Fig 7  Log-normal PDF + variance", lambda: fig7_lognormal(N_mc=30_000)),
    ]

    for title, fn in SECTIONS:
        print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")
        fn()

    print(f"\n{'=' * 62}")
    print(f"  Total runtime:  {time.time() - t_start:.1f} s")
    print(f"{'=' * 62}")