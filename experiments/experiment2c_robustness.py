"""
experiment_2c_moment_statespace.py
===================================
Produces: TABLE 2c — Moment Consistency across State Space Dimensions

PURPOSE
-------
Verify that the agreement between analytical occupancy-measure moments
(Path A) and simulated discounted moments (Path B) holds across
different state space resolutions. This rules out the possibility
that consistency is an artifact of a particular discretization.

WHAT THIS FILE PRODUCES
-----------------------
A CSV file (results/table2c_statespace.csv) with one row per n_s.
"""

import numpy as np
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.economic_model import RustBusEngine


def run():

    # ─────────────────────────────────────────────────────
    # Fixed parameters
    # ─────────────────────────────────────────────────────
    beta = 0.95
    trans_probs = (0.4, 0.5, 0.1)
    theta1 = 2.0
    RC = 15.0

    # Large panel
    N = 10_000
    T = 1_000
    seed = 42

    # ─────────────────────────────────────────────────────
    # State space sizes to test
    # ─────────────────────────────────────────────────────
    state_sizes = [10, 20, 30, 50, 75, 100, 150, 200, 300]

    # ─────────────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/table2c_statespace.csv"
    fieldnames = [
        "n_s", "threshold", "max_diff",
        "diff_m1", "diff_m2", "diff_m3", "diff_m4", "diff_m5",
        "exact_m1", "exact_m2", "exact_m3", "exact_m4", "exact_m5",
    ]

    sep = "=" * 90
    print(f"\n{sep}")
    print("TABLE 2c: MOMENT CONSISTENCY ACROSS STATE SPACE DIMENSIONS".center(90))
    print(f"{sep}")
    print(f"\n  Parameters: theta1 = {theta1}, RC = {RC}")
    print(f"  Simulation: N = {N:,}, T = {T:,}, seed = {seed}")

    rows = []

    for ns in state_sizes:

        model = RustBusEngine(n_s=ns, beta=beta, trans_probs=trans_probs)

        # Path A: analytical
        V, policy = model.solve_bellman(theta1, RC)
        lam = model.compute_occupancy_measure(policy)
        m_exact = model.compute_moments(lam)

        replace_states = np.where(policy == 1)[0]
        threshold = int(replace_states[0]) if len(replace_states) > 0 else -1

        # Path B: simulation
        states, actions = model.simulate_panel(theta1, RC, N=N, T=T, seed=seed)
        m_sim = model.compute_data_moments(states, actions)

        diffs = np.abs(m_exact - m_sim)
        max_diff = float(np.max(diffs))

        # ── Print block for this n_s ──
        print(f"\n  ┌─ n_s = {ns}, threshold s* = {threshold}")
        print(f"  │")
        print(f"  │  {'Moment':>10} {'Exact (Path A)':>16} {'Simulated (Path B)':>18} {'|Δ|':>12}")
        print(f"  │  {'─'*10} {'─'*16} {'─'*18} {'─'*12}")
        m_labels = ['m¹(repl)', 'm²(mile)', 'm³(m×r)', 'm⁴(m²)', 'm⁵(m²×r)']
        for i, label in enumerate(m_labels):
            print(f"  │  {label:>10} {m_exact[i]:16.10f} {m_sim[i]:18.10f} {diffs[i]:12.8f}")
        print(f"  │")
        print(f"  └─ max|Δ| = {max_diff:.8f}")

        rows.append({
            "n_s": ns, "threshold": threshold,
            "max_diff": f"{max_diff:.8f}",
            "diff_m1": f"{diffs[0]:.8f}", "diff_m2": f"{diffs[1]:.8f}",
            "diff_m3": f"{diffs[2]:.8f}", "diff_m4": f"{diffs[3]:.8f}",
            "diff_m5": f"{diffs[4]:.8f}",
            "exact_m1": f"{m_exact[0]:.10f}", "exact_m2": f"{m_exact[1]:.10f}",
            "exact_m3": f"{m_exact[2]:.10f}", "exact_m4": f"{m_exact[3]:.10f}",
            "exact_m5": f"{m_exact[4]:.10f}",
        })

    # ── Write CSV ──
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    run()