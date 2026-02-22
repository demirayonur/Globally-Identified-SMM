"""
experiment_3_sa_vs_miqp.py
===========================
Produces: TABLE 3 — Large-Scale Comparison: SA versus MIQP
          TABLE 4 — MIQP Problem Dimensions and Solver Performance

PURPOSE
-------
Head-to-head comparison of simulated annealing and MIQP on large
instances where exhaustive enumeration is infeasible. Both methods
receive the SAME data moment vector (simulated panel). SA uses a
single random restart — representative of standard applied practice.

The MIQP solution, when MIP gap = 0, is certified ground truth.
Any SA solution with a strictly higher criterion value is a confirmed
failure to locate the global optimum.

WHAT THIS FILE PRODUCES
-----------------------
1. results/table3_sa_vs_miqp.csv   — one row per configuration
2. results/table4_miqp_dimensions.csv — MIQP problem sizes for selected configs
"""

import numpy as np
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.economic_model import RustBusEngine
from src.simulated_annealing import SMMSimulatedAnnealing

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
    # Model calibration
    # ─────────────────────────────────────────────────────
    beta = 0.95
    trans_probs = (0.4, 0.5, 0.1)
    theta1_true = 2.0
    RC_true = 15.0

    # Data moments: large simulated panel
    N_buses = 50_000
    T_periods = 5_000
    sim_seed = 42

    # SA settings: single restart, generous iterations
    SA_RESTARTS = 1
    SA_T_INIT = 1.0
    SA_COOLING = 0.9995
    SA_SEED = 42

    # ─────────────────────────────────────────────────────
    # Configurations: (n_s, K1, K2, sa_max_iter)
    # ─────────────────────────────────────────────────────
    configs = [
        (100, 50,   50,   50_000),
        (200, 100, 100,   50_000),
        (300, 150, 150,   50_000),
        (400, 200, 200,   50_000),
        (500, 250, 250,   50_000),
        (600, 300, 300,   50_000),
        (700, 350, 350,   50_000),
        (800, 400, 400,   50_000),
        (900, 450, 450,   50_000),
        (1000, 500, 500,  50_000),
    ]

    W = np.eye(5)
    MATCH_TOL = 1e-5

    # ─────────────────────────────────────────────────────
    # Output
    # ─────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    csv3 = "results/table3_sa_vs_miqp.csv"
    csv4 = "results/table4_miqp_dimensions.csv"

    fields3 = [
        "n_s", "J", "K", "grid_size",
        "Q_sa", "t_sa", "n_eval",
        "Q_miqp", "t_miqp", "mip_gap",
        "match", "speedup",
        "theta1_sa", "RC_sa", "theta1_miqp", "RC_miqp",
    ]
    fields4 = [
        "n_s", "J", "K", "grid_size",
        "n_binary", "n_continuous", "n_constraints",
        "t_miqp", "mip_gap",
    ]

    sep = "=" * 100
    print(f"\n{sep}")
    print("TABLE 3: SIMULATED ANNEALING vs MIQP".center(100))
    print(f"{sep}")
    print(f"\n  True parameters: theta1 = {theta1_true}, RC = {RC_true}")
    print(f"  Beta = {beta}, trans_probs = {trans_probs}")
    print(f"  Data moments: simulated panel (N={N_buses:,}, T={T_periods:,}, seed={sim_seed})")
    print(f"  SA: {SA_RESTARTS} restart, cooling = {SA_COOLING}, T0 = {SA_T_INIT}")

    print(f"\n  {'n_s':>4} {'JxK':>10} {'Grid':>7}"
          f"  │ {'Q_SA':>14} {'t_SA':>7} {'Evals':>6}"
          f"  │ {'Q_MIQP':>14} {'t_MIQP':>7} {'Gap':>10}"
          f"  │ {'Match':>5} {'Speed':>7}")
    print(f"  {'─'*4} {'─'*10} {'─'*7}"
          f"  │ {'─'*14} {'─'*7} {'─'*6}"
          f"  │ {'─'*14} {'─'*7} {'─'*10}"
          f"  │ {'─'*5} {'─'*7}")

    rows3 = []
    rows4 = []

    for n_s, K1, K2, sa_iter in configs:

        total_grid = K1 * K2

        # Build model and simulate data moments
        model = RustBusEngine(n_s=n_s, beta=beta, trans_probs=trans_probs)
        states, actions = model.simulate_panel(
            theta1_true, RC_true, N=N_buses, T=T_periods, seed=sim_seed
        )
        data_moments = model.compute_data_moments(states, actions)

        theta1_grid = np.linspace(0.5, 5.0, K1)
        RC_grid = np.linspace(5.0, 30.0, K2)

        # ── Simulated Annealing ──
        sa = SMMSimulatedAnnealing(model, theta1_grid, RC_grid, data_moments, W)
        res_sa = sa.solve(
            n_restarts=SA_RESTARTS,
            max_iter=sa_iter,
            T_init=SA_T_INIT,
            cooling_rate=SA_COOLING,
            seed=SA_SEED,
        )

        # ── MIQP ──
        try:
            miqp = MIQPSolver(model, theta1_grid, RC_grid, data_moments, W)
            res_miqp = miqp.solve(time_limit=600, mip_gap=0.0, verbose=False)
            if 'error' in res_miqp:
                raise RuntimeError(res_miqp['error'])
        except Exception:
            print(f"  {n_s:4d} {K1:>4}x{K2:<4} {total_grid:7,}"
                  f"  │ {res_sa['Q_min']:14.8e} {res_sa['solve_time']:6.2f}s"
                  f" {res_sa['n_evaluations']:6d}"
                  f"  │ {'SKIPPED':>14} {'--':>7} {'--':>10}"
                  f"  │ {'--':>5} {'--':>7}")
            continue

        # ── Compare ──
        diff = abs(res_sa['Q_min'] - res_miqp['Q_min'])
        match = 1 if diff < MATCH_TOL else 0
        t_sa = res_sa['solve_time']
        t_miqp = res_miqp['solve_time']
        speedup = t_sa / t_miqp if t_miqp > 0.001 else float('inf')

        print(f"  {n_s:4d} {K1:>4}x{K2:<4} {total_grid:7,}"
              f"  │ {res_sa['Q_min']:14.8e} {t_sa:6.2f}s"
              f" {res_sa['n_evaluations']:6d}"
              f"  │ {res_miqp['Q_min']:14.8e} {t_miqp:6.2f}s"
              f" {res_miqp['optimality_gap']:10.2e}"
              f"  │ {'  YES' if match else '** NO':>5}"
              f" {speedup:6.1f}x")

        rows3.append({
            "n_s": n_s, "J": K1, "K": K2, "grid_size": total_grid,
            "Q_sa": f"{res_sa['Q_min']:.10e}",
            "t_sa": f"{t_sa:.4f}",
            "n_eval": res_sa['n_evaluations'],
            "Q_miqp": f"{res_miqp['Q_min']:.10e}",
            "t_miqp": f"{t_miqp:.4f}",
            "mip_gap": f"{res_miqp['optimality_gap']:.6e}",
            "match": match,
            "speedup": f"{speedup:.2f}",
            "theta1_sa": f"{res_sa['theta1_best']:.6f}",
            "RC_sa": f"{res_sa['RC_best']:.6f}",
            "theta1_miqp": f"{res_miqp['theta1_best']:.6f}",
            "RC_miqp": f"{res_miqp['RC_best']:.6f}",
        })

        # Table 4 data: MIQP dimensions
        rows4.append({
            "n_s": n_s, "J": K1, "K": K2, "grid_size": total_grid,
            "n_binary": res_miqp['n_binary'],
            "n_continuous": res_miqp['n_continuous'],
            "n_constraints": res_miqp['n_constraints'],
            "t_miqp": f"{t_miqp:.4f}",
            "mip_gap": f"{res_miqp['optimality_gap']:.6e}",
        })

    # ── Write CSVs ──
    with open(csv3, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields3)
        writer.writeheader()
        writer.writerows(rows3)

    with open(csv4, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields4)
        writer.writeheader()
        writer.writerows(rows4)

    # ── Summary ──
    n_compared = len(rows3)
    n_match = sum(int(r["match"]) for r in rows3)
    n_fail = n_compared - n_match
    fail_rate = 100.0 * n_fail / n_compared if n_compared > 0 else 0

    print(f"\n  {'─'*80}")
    print(f"  SUMMARY")
    print(f"  {'─'*80}")
    print(f"  Configurations compared:    {n_compared}")
    print(f"  SA found global optimum:    {n_match}/{n_compared}")
    print(f"  SA failure rate:            {fail_rate:.0f}%")

    if rows3:
        speedups = [float(r["speedup"]) for r in rows3 if float(r["speedup"]) < 1e6]
        if speedups:
            print(f"  Avg MIQP speedup:           {np.mean(speedups):.1f}x")
            print(f"  Max MIQP speedup:           {np.max(speedups):.1f}x")

    print(f"\n  Table 3 saved to: {csv3}")
    print(f"  Table 4 saved to: {csv4}\n")


if __name__ == "__main__":
    run()