"""
experiment_1_validation.py
===========================
Produces: TABLE 1 — Exact Validation: MIQP versus Exhaustive Enumeration

PURPOSE
-------
Verify that the MIQP formulation finds the SAME global optimum as
exhaustive enumeration across a range of state-space sizes and grid
resolutions. Both methods receive the SAME data moment vector, computed
from a simulated panel at the true parameters.

WHAT THIS FILE PRODUCES
-----------------------
A CSV file (results/table1_validation.csv) with one row per configuration:
    n_s, J, K, grid_size, Q_enum, Q_miqp, match, t_enum, t_miqp, mip_gap

The match column is 1 if |Q_enum - Q_miqp| < 1e-4, else 0.
"""

import numpy as np
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.economic_model import RustBusEngine
from src.exhaustive_search import ExhaustiveEnumerator

try:
    from src.miqp_solver import MIQPSolver
    HAS_MIQP = True
except ImportError:
    HAS_MIQP = False


def run():

    if not HAS_MIQP:
        print("ERROR: Gurobi is required. Install with: pip install gurobipy")
        return

    # ─────────────────────────────────────────────────────
    # Model calibration (same across all experiments)
    # ─────────────────────────────────────────────────────
    beta = 0.95
    trans_probs = (0.4, 0.5, 0.1)
    theta1_true = 2.0
    RC_true = 15.0

    # Data moments: simulated panel
    N_buses = 5_000
    T_periods = 500
    sim_seed = 42

    # ─────────────────────────────────────────────────────
    # Configurations: (n_s, K1, K2)
    # ─────────────────────────────────────────────────────
    configs = [
        (10,    8,   8),
        (15,   10,  10),
        (20,   12,  12),
        (25,   12,  12),
        (30,   15,  15),
        (30,   19,  21),
        (50,   15,  15),
        (50,   25,  25),
        (75,   20,  20),
        (100,  30,  30),
        (100,  50,  50),
        (150,  40,  40),
        (150,  75,  75),
        (200, 100, 100),
        (300, 150, 150),
    ]

    W = np.eye(5)
    MATCH_TOL = 1e-4

    # ─────────────────────────────────────────────────────
    # Output setup
    # ─────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv_path = "results/table1_validation.csv"
    fieldnames = [
        "n_s", "J", "K", "grid_size",
        "Q_enum", "Q_miqp", "match",
        "t_enum", "t_miqp", "mip_gap",
        "theta1_enum", "RC_enum", "theta1_miqp", "RC_miqp",
    ]

    sep = "=" * 90
    print(f"\n{sep}")
    print("TABLE 1: MIQP vs EXHAUSTIVE ENUMERATION".center(90))
    print(f"{sep}")
    print(f"\n  True parameters: theta1 = {theta1_true}, RC = {RC_true}")
    print(f"  Beta = {beta}, trans_probs = {trans_probs}")
    print(f"  Data moments: simulated panel (N={N_buses:,}, T={T_periods:,}, seed={sim_seed})")

    print(f"\n  {'n_s':>4} {'JxK':>10} {'Grid':>7}"
          f"  {'Q_enum':>14} {'Q_miqp':>14} {'Match':>5}"
          f"  {'t_enum':>8} {'t_miqp':>8} {'MIP Gap':>10}")
    print(f"  {'─'*4} {'─'*10} {'─'*7}"
          f"  {'─'*14} {'─'*14} {'─'*5}"
          f"  {'─'*8} {'─'*8} {'─'*10}")

    rows = []

    for n_s, K1, K2 in configs:

        total_grid = K1 * K2

        # Build model and simulate data moments
        model = RustBusEngine(n_s=n_s, beta=beta, trans_probs=trans_probs)
        states, actions = model.simulate_panel(
            theta1_true, RC_true, N=N_buses, T=T_periods, seed=sim_seed
        )
        data_moments = model.compute_data_moments(states, actions)

        theta1_grid = np.linspace(0.5, 5.0, K1)
        RC_grid = np.linspace(5.0, 30.0, K2)

        # ── Exhaustive enumeration ──
        enum = ExhaustiveEnumerator(model, theta1_grid, RC_grid, data_moments, W)
        res_enum = enum.solve()

        # ── MIQP ──
        try:
            miqp = MIQPSolver(model, theta1_grid, RC_grid, data_moments, W)
            res_miqp = miqp.solve(time_limit=600, mip_gap=0.0, verbose=False)
            if 'error' in res_miqp:
                raise RuntimeError(res_miqp['error'])
        except Exception as e:
            print(f"  {n_s:4d} {K1:>4}x{K2:<4} {total_grid:7,}"
                  f"  {res_enum['Q_min']:14.8e}"
                  f"  {'SKIPPED':>14} {'--':>5}"
                  f"  {res_enum['solve_time']:7.2f}s {'--':>8} {'--':>10}")
            continue

        # ── Compare ──
        diff = abs(res_miqp['Q_min'] - res_enum['Q_min'])
        match = 1 if diff < MATCH_TOL else 0

        print(f"  {n_s:4d} {K1:>4}x{K2:<4} {total_grid:7,}"
              f"  {res_enum['Q_min']:14.8e} {res_miqp['Q_min']:14.8e}"
              f" {'  YES' if match else '** NO'}"
              f"  {res_enum['solve_time']:7.2f}s {res_miqp['solve_time']:7.2f}s"
              f" {res_miqp['optimality_gap']:10.2e}")

        rows.append({
            "n_s": n_s, "J": K1, "K": K2, "grid_size": total_grid,
            "Q_enum": f"{res_enum['Q_min']:.10e}",
            "Q_miqp": f"{res_miqp['Q_min']:.10e}",
            "match": match,
            "t_enum": f"{res_enum['solve_time']:.4f}",
            "t_miqp": f"{res_miqp['solve_time']:.4f}",
            "mip_gap": f"{res_miqp['optimality_gap']:.6e}",
            "theta1_enum": f"{res_enum['theta1_best']:.6f}",
            "RC_enum": f"{res_enum['RC_best']:.6f}",
            "theta1_miqp": f"{res_miqp['theta1_best']:.6f}",
            "RC_miqp": f"{res_miqp['RC_best']:.6f}",
        })

    # ── Write CSV ──
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_match = sum(r["match"] for r in rows)
    print(f"\n  Results: {n_match}/{len(rows)} configurations match.")
    print(f"  CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    run()