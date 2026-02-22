"""
experiment_2b_moment_convergence.py
====================================
Produces: TABLE 2b — Convergence of Simulated Moments to Analytical Moments
          FIGURE   — Log-log convergence plot (results/figure_convergence.pdf)

PURPOSE
-------
Fix parameters at their true values. Increase (N, T) and show that
the maximum absolute discrepancy between Path A (occupancy measure)
and Path B (simulated panel) shrinks toward zero. The absence of a
persistent floor rules out systematic bias and confirms that both
paths target the same population moment.

WHAT THIS FILE PRODUCES
-----------------------
1. results/table2b_convergence.csv — one row per (N, T) configuration
2. results/figure_convergence.pdf  — log-log plot of max|Δ| vs N×T
"""

import numpy as np
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.economic_model import RustBusEngine


def run():

    # ─────────────────────────────────────────────────────
    # Fixed parameters (true DGP)
    # ─────────────────────────────────────────────────────
    beta = 0.95
    trans_probs = (0.4, 0.5, 0.1)
    n_s = 30
    theta1 = 2.0
    RC = 15.0
    seed = 42

    model = RustBusEngine(n_s=n_s, beta=beta, trans_probs=trans_probs)

    # Path A: exact analytical moments
    V, policy = model.solve_bellman(theta1, RC)
    lam = model.compute_occupancy_measure(policy)
    m_exact = model.compute_moments(lam)

    # ─────────────────────────────────────────────────────
    # Sample size configurations
    # ─────────────────────────────────────────────────────
    configs = [
        (100,       50),
        (500,      100),
        (1_000,    200),
        (5_000,    500),
        (10_000,  1_000),
        (20_000,  1_000),
        (50_000,  2_000),
    ]

    # ─────────────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/table2b_convergence.csv"
    fieldnames = [
        "N", "T", "NxT", "max_diff", "rmse", "shrinkage",
        "diff_m1", "diff_m2", "diff_m3", "diff_m4", "diff_m5",
    ]

    sep = "=" * 80
    print(f"\n{sep}")
    print("TABLE 2b: MOMENT CONVERGENCE IN SAMPLE SIZE".center(80))
    print(f"{sep}")
    print(f"\n  Parameters: theta1 = {theta1}, RC = {RC}")
    print(f"  Model: n_s = {n_s}, beta = {beta}")

    print(f"\n  Exact population moments:")
    m_names = ['m1(repl)', 'm2(mile)', 'm3(m×r)', 'm4(m²)', 'm5(m²×r)']
    for i, name in enumerate(m_names):
        print(f"    {name} = {m_exact[i]:.10f}")

    print(f"\n  {'N':>8} {'T':>6} {'N×T':>12}"
          f"  {'max|Δ|':>10} {'RMSE':>10} {'Shrink':>8}")
    print(f"  {'─'*8} {'─'*6} {'─'*12}"
          f"  {'─'*10} {'─'*10} {'─'*8}")

    rows = []
    prev_max = None
    # For the figure
    NxT_list = []
    maxdiff_list = []

    for N, T in configs:

        states, actions = model.simulate_panel(theta1, RC, N=N, T=T, seed=seed)
        m_sim = model.compute_data_moments(states, actions)

        diffs = np.abs(m_exact - m_sim)
        max_diff = float(np.max(diffs))
        rmse = float(np.sqrt(np.mean(diffs**2)))

        if prev_max is not None and max_diff > 1e-15:
            shrinkage = prev_max / max_diff
            shrink_str = f"{shrinkage:.1f}x"
        else:
            shrinkage = None
            shrink_str = "---"

        print(f"  {N:8,} {T:6,} {N*T:12,}"
              f"  {max_diff:10.6f} {rmse:10.6f} {shrink_str:>8}")

        NxT_list.append(N * T)
        maxdiff_list.append(max_diff)

        rows.append({
            "N": N, "T": T, "NxT": N * T,
            "max_diff": f"{max_diff:.8f}",
            "rmse": f"{rmse:.8f}",
            "shrinkage": f"{shrinkage:.2f}" if shrinkage else "",
            "diff_m1": f"{diffs[0]:.8f}", "diff_m2": f"{diffs[1]:.8f}",
            "diff_m3": f"{diffs[2]:.8f}", "diff_m4": f"{diffs[3]:.8f}",
            "diff_m5": f"{diffs[4]:.8f}",
        })

        prev_max = max_diff

    # ── Write CSV ──
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV saved to: {csv_path}")

    # ─────────────────────────────────────────────────────
    # FIGURE: Log-log convergence plot
    # ─────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        NxT_arr = np.array(NxT_list, dtype=float)
        maxdiff_arr = np.array(maxdiff_list, dtype=float)

        # Data points
        ax.loglog(NxT_arr, maxdiff_arr, 'o-', color='#2c3e50',
                  markersize=7, linewidth=1.8, label=r'$\max_\ell |\Delta^\ell|$')

        # Reference line: 1/sqrt(N*T) rate
        ref_x = np.logspace(np.log10(NxT_arr[0]), np.log10(NxT_arr[-1]), 50)
        # Anchor reference line at the first data point
        ref_y = maxdiff_arr[0] * np.sqrt(NxT_arr[0] / ref_x)
        ax.loglog(ref_x, ref_y, '--', color='#95a5a6', linewidth=1.2,
                  label=r'$\mathcal{O}(1/\sqrt{N \times T})$ reference')

        ax.set_xlabel(r'Effective sample size $N \times T$', fontsize=11)
        ax.set_ylabel(r'$\max_\ell \, |\Delta^\ell|$', fontsize=11)
        ax.set_title('Convergence of Simulated Moments\nto Occupancy-Measure Moments',
                      fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, which='both', alpha=0.3)

        fig.tight_layout()
        fig_path = "results/figure_convergence.pdf"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Figure saved to: {fig_path}\n")

    except ImportError:
        print("  (matplotlib not available — figure skipped)\n")


if __name__ == "__main__":
    run()