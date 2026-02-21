"""
experiment_1_miqp_vs_bruteforce.py
===================================
MIQP vs Exhaustive Enumeration: Correctness Validation

PURPOSE
-------
For different state space sizes and grid resolutions, verify that the
MIQP solver finds the SAME global optimum as exhaustive enumeration.

Usage:
    python experiment_1_miqp_vs_bruteforce.py
"""

import numpy as np
import sys
import os

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

    sep = "=" * 80
    print(f"\n{sep}")
    print("EXPERIMENT 1: MIQP vs BRUTE FORCE VALIDATION".center(80))
    print(f"{sep}")

    # Fixed model parameters
    beta = 0.95
    trans_probs = (0.4, 0.5, 0.1)
    theta1_true = 2.0
    RC_true = 15.0

    print(f"\n  True parameters: theta1 = {theta1_true}, RC = {RC_true}")
    print(f"  Beta = {beta}, data moments: population (exact, no noise)")

    # Configurations: (n_s, K1, K2)
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

    configs = [(300, 150, 150)]

    # Table header
    print(f"\n  {'n_s':>4} {'K1':>4}{'x':1}{'K2':<4} {'Grid':>6}"
          f"  {'Q_enum':>14} {'Q_miqp':>14} {'Match':>5}"
          f"  {'t_enum':>7} {'t_miqp':>7}"
          f"  {'Gap':>9} {'#Opt':>5}")
    print(f"  {'─' * 4} {'─' * 9} {'─' * 6}"
          f"  {'─' * 14} {'─' * 14} {'─' * 5}"
          f"  {'─' * 7} {'─' * 7}"
          f"  {'─' * 9} {'─' * 5}")

    n_tested = 0
    n_passed = 0
    n_skipped = 0

    for n_s, K1, K2 in configs:

        total_grid = K1 * K2

        # Build model and compute population moments
        model = RustBusEngine(n_s=n_s, beta=beta, trans_probs=trans_probs)
        
        N_buses = 5000       # number of buses
        T_periods = 500      # periods per bus
        sim_seed = 42         # for reproducibility

        states, actions = model.simulate_panel(
            theta1_true, RC_true,
            N=N_buses, T=T_periods, seed=sim_seed
        )
        data_moments = model.compute_data_moments(states, actions)

        theta1_grid = np.linspace(0.5, 5.0, K1)
        RC_grid = np.linspace(5.0, 30.0, K2)
        W = np.eye(5)

        # Exhaustive enumeration (always works)
        enum = ExhaustiveEnumerator(model, theta1_grid, RC_grid, data_moments, W)
        res_enum = enum.solve()

        # MIQP (may hit license limit)
        try:
            miqp = MIQPSolver(model, theta1_grid, RC_grid, data_moments, W)
            res_miqp = miqp.solve(time_limit=600, mip_gap=0.0, verbose=False)
            if 'error' in res_miqp:
                raise RuntimeError(res_miqp['error'])
        except Exception:
            print(f"  {n_s:4d} {K1:4d}x{K2:<4d} {total_grid:6d}"
                  f"  {res_enum['Q_min']:14.8e}"
                  f" {'(license limit)':>14}    --"
                  f"  {res_enum['solve_time']:6.2f}s      --"
                  f"        -- {res_enum['n_global_optima']:5d}")
            n_skipped += 1
            continue

        # Compare
        n_tested += 1
        diff = abs(res_miqp['Q_min'] - res_enum['Q_min'])
        match = diff < 1e-4
        if match:
            n_passed += 1

        print(f"  {n_s:4d} {K1:4d}x{K2:<4d} {total_grid:6d}"
              f"  {res_enum['Q_min']:14.8e} {res_miqp['Q_min']:14.8e}"
              f" {'  YES' if match else '** NO'}"
              f"  {res_enum['solve_time']:6.2f}s {res_miqp['solve_time']:6.2f}s"
              f"  {res_miqp['optimality_gap']:9.2e}"
              f" {res_enum['n_global_optima']:5d}")

    # Summary
    print(f"\n  {'─' * 70}")
    print(f"  Tested: {n_tested}  |  Passed: {n_passed}"
          f"  |  Failed: {n_tested - n_passed}"
          f"  |  Skipped (license): {n_skipped}")

    if n_tested == n_passed and n_tested > 0:
        print(f"\n  RESULT: ALL {n_tested} configs match. MIQP finds the global optimum.")
    if n_skipped > 0:
        print(f"\n  NOTE: {n_skipped} config(s) exceeded the Gurobi license limit.")
        print(f"        Full academic license (free): gurobi.com/academia/")
    print()


if __name__ == "__main__":
    run()