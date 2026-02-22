"""
experiment_2a_moment_params.py
===============================
Produces: TABLE 2a — Moment Consistency across Parameter Values

PURPOSE
-------
Verify that the two moment computation paths agree across a range of
economically distinct parameterizations:

    Path A (analytical):  parameters → Bellman → policy → occupancy measure → moments
    Path B (simulation):  parameters → Bellman → policy → simulate_panel → data moments

Both paths should target the same population quantity. The discrepancy
should be small (sampling noise only) with no systematic bias.

WHAT THIS FILE PRODUCES
-----------------------
A CSV file (results/table2a_moment_params.csv) with one row per (theta1, RC):
    theta1, RC, threshold, max_diff, diff_m1..m5, exact_m1..m5
"""

import numpy as np
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.economic_model import RustBusEngine


def run():

    # ─────────────────────────────────────────────────────
    # Fixed model specification
    # ─────────────────────────────────────────────────────
    beta = 0.95
    trans_probs = (0.4, 0.5, 0.1)
    n_s = 30

    # Large panel for simulation path
    N = 10_000
    T = 1_000
    seed = 42

    model = RustBusEngine(n_s=n_s, beta=beta, trans_probs=trans_probs)

    # ─────────────────────────────────────────────────────
    # Parameter configurations spanning diverse regimes
    # ─────────────────────────────────────────────────────
    param_pairs = [
        (0.10,  3.0),   # very low cost → never or rarely replace
        (0.50,  5.0),   # low cost, low RC → frequent replacement
        (1.00, 10.0),   # moderate
        (2.00, 15.0),   # baseline (true parameters)
        (3.00, 20.0),   # high cost, high RC
        (5.00, 30.0),   # very high cost → replace early
    ]

    # ─────────────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/table2a_moment_params.csv"
    fieldnames = [
        "theta1", "RC", "threshold", "max_diff",
        "diff_m1", "diff_m2", "diff_m3", "diff_m4", "diff_m5",
        "exact_m1", "exact_m2", "exact_m3", "exact_m4", "exact_m5",
    ]

    sep = "=" * 90
    print(f"\n{sep}")
    print("TABLE 2a: MOMENT CONSISTENCY ACROSS PARAMETER VALUES".center(90))
    print(f"{sep}")
    print(f"\n  Model: n_s = {n_s}, beta = {beta}, trans_probs = {trans_probs}")
    print(f"  Simulation: N = {N:,}, T = {T:,}, seed = {seed}")

    rows = []

    for theta1, RC in param_pairs:

        # Path A: analytical via occupancy measure
        V, policy = model.solve_bellman(theta1, RC)
        lam = model.compute_occupancy_measure(policy)
        m_exact = model.compute_moments(lam)

        # Replacement threshold
        replace_states = np.where(policy == 1)[0]
        threshold = int(replace_states[0]) if len(replace_states) > 0 else -1

        # Path B: simulation with discounted weights
        states, actions = model.simulate_panel(theta1, RC, N=N, T=T, seed=seed)
        m_sim = model.compute_data_moments(states, actions)

        diffs = np.abs(m_exact - m_sim)
        max_diff = float(np.max(diffs))

        # ── Print block for this (theta1, RC) ──
        print(f"\n  ┌─ theta1 = {theta1}, RC = {RC}, threshold s* = {threshold}")
        print(f"  │")
        print(f"  │  {'Moment':>10} {'Exact (Path A)':>16} {'Simulated (Path B)':>18} {'|Δ|':>12}")
        print(f"  │  {'─'*10} {'─'*16} {'─'*18} {'─'*12}")
        m_labels = ['m¹(repl)', 'm²(mile)', 'm³(m×r)', 'm⁴(m²)', 'm⁵(m²×r)']
        for i, label in enumerate(m_labels):
            print(f"  │  {label:>10} {m_exact[i]:16.10f} {m_sim[i]:18.10f} {diffs[i]:12.8f}")
        print(f"  │")
        print(f"  └─ max|Δ| = {max_diff:.8f}")

        rows.append({
            "theta1": theta1, "RC": RC, "threshold": threshold,
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