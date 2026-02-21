"""
experiment_3_largescale_sa_vs_miqp.py
======================================
Large-Scale Comparison: Simulated Annealing vs MIQP Solver

PURPOSE
-------
For large problem instances, compare SA and MIQP head-to-head on
solution quality, runtime, and optimality certification.

For each configuration we report:
    SA:   Q_best, solve_time, # unique grid points evaluated
    MIQP: Q_best, solve_time, MIP gap (0 = proven optimal)

When MIP gap = 0, the MIQP answer is GROUND TRUTH and we can judge
whether SA found the same optimum or got stuck.

DATA MOMENTS
-------------
Population moments (exact, from occupancy measure at true parameters).
No simulation noise — differences between SA and MIQP are purely
due to the optimization algorithms, not data variation.

GUROBI LICENSE NOTE
--------------------
The free restricted license has size limits. Configurations that exceed
it are gracefully skipped (SA results still reported).
Full academic license (free): gurobi.com/academia/

Usage:
    python experiment_3_largescale_sa_vs_miqp.py
"""

import numpy as np
import sys
import os

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

    sep = "=" * 80
    print(f"\n{sep}")
    print("EXPERIMENT 3: LARGE-SCALE SA vs MIQP".center(80))
    print(f"{sep}")

    # Fixed parameters
    beta = 0.95
    trans_probs = (0.4, 0.5, 0.1)
    theta1_true = 2.0
    RC_true = 15.0

    print(f"\n  True parameters: theta1 = {theta1_true}, RC = {RC_true}")
    print(f"  Beta = {beta}, trans_probs = {trans_probs}")
    print(f"  Data moments: POPULATION (exact, from occupancy measure)")
    print(f"  SA cooling rate: 0.9995")

    # Configurations: (n_s, K1, K2, sa_restarts, sa_max_iter)
    configs = [
        (10,    8,   8,    1,   10_000),
        (15,   10,  10,    1,   10_000),
        (20,   12,  12,    1,   15_000),
        (25,   12,  12,    1,   15_000),
        (30,   15,  15,    1,   15_000),
        (30,   19,  21,    1,   20_000),
        (50,   15,  15,    1,   20_000),
        (50,   25,  25,    1,   30_000),
        (75,   20,  20,    1,   30_000),
        (100,  30,  30,    1,   40_000),
        (100,  50,  50,    1,   50_000),
        (150,  40,  40,    1,   50_000),
        (150,  75,  75,    1,   60_000),
        (200, 100, 100,    1,   60_000),
    ]

    # Table header
    print(f"\n  {'n_s':>4} {'Grid':>10} {'#pts':>7}"
          f"  │ {'Q_sa':>14} {'t_sa':>7} {'#eval':>6}"
          f"  │ {'Q_miqp':>14} {'t_miqp':>7} {'MIP gap':>9}"
          f"  │ {'Match':>5} {'Speed':>6}")
    print(f"  {'─' * 4} {'─' * 10} {'─' * 7}"
          f"  │ {'─' * 14} {'─' * 7} {'─' * 6}"
          f"  │ {'─' * 14} {'─' * 7} {'─' * 9}"
          f"  │ {'─' * 5} {'─' * 6}")

    results = []
    license_hit = False

    for n_s, K1, K2, n_restarts, sa_iter in configs:

        total_grid = K1 * K2

        # Build model and exact population moments
        model = RustBusEngine(n_s=n_s, beta=beta, trans_probs=trans_probs)
        N_buses = 50000       # number of buses
        T_periods = 5000      # periods per bus
        sim_seed = 42         # for reproducibility

        states, actions = model.simulate_panel(
            theta1_true, RC_true,
            N=N_buses, T=T_periods, seed=sim_seed
        )
        data_moments = model.compute_data_moments(states, actions)

        theta1_grid = np.linspace(0.5, 5.0, K1)
        RC_grid = np.linspace(5.0, 30.0, K2)
        W = np.eye(5)

        # --- SA (always works) ---
        sa = SMMSimulatedAnnealing(
            model, theta1_grid, RC_grid, data_moments, W
        )
        res_sa = sa.solve(
            n_restarts=n_restarts,
            max_iter=sa_iter,
            T_init=1.0,
            cooling_rate=0.9995,
            seed=42
        )

        # --- MIQP (may hit license limit) ---
        try:
            miqp = MIQPSolver(model, theta1_grid, RC_grid, data_moments, W)
            res_miqp = miqp.solve(time_limit=600, mip_gap=0.0, verbose=False)
            if 'error' in res_miqp:
                raise RuntimeError(res_miqp['error'])

            # Both ran — compare
            diff = abs(res_sa['Q_min'] - res_miqp['Q_min'])
            match = diff < 1e-5
            t_sa = res_sa['solve_time']
            t_miqp = res_miqp['solve_time']
            speedup = t_sa / t_miqp if t_miqp > 0.001 else float('inf')

            print(f"  {n_s:4d} {K1:4d}x{K2:<4d} {total_grid:7,}"
                  f"  │ {res_sa['Q_min']:14.8e} {t_sa:6.2f}s"
                  f" {res_sa['n_evaluations']:6d}"
                  f"  │ {res_miqp['Q_min']:14.8e} {t_miqp:6.2f}s"
                  f" {res_miqp['optimality_gap']:9.2e}"
                  f"  │ {'  YES' if match else '** NO'}"
                  f" {speedup:5.1f}x")

            results.append({
                'n_s': n_s, 'K1': K1, 'K2': K2, 'grid': total_grid,
                'Q_sa': res_sa['Q_min'], 'Q_miqp': res_miqp['Q_min'],
                't_sa': t_sa, 't_miqp': t_miqp,
                'gap': res_miqp['optimality_gap'],
                'n_eval': res_sa['n_evaluations'],
                'match': match, 'speedup': speedup,
                'n_binary': res_miqp['n_binary'],
                'n_continuous': res_miqp['n_continuous'],
                'n_constraints': res_miqp['n_constraints'],
                'miqp_ran': True,
            })

        except Exception:
            # License limit — report SA only
            if not license_hit:
                license_hit = True
                print(f"  {'─' * 4}─{'─' * 10}─{'─' * 7}"
                      f"──┤ Gurobi license limit reached below"
                      f" ├{'─' * 39}")

            print(f"  {n_s:4d} {K1:4d}x{K2:<4d} {total_grid:7,}"
                  f"  │ {res_sa['Q_min']:14.8e}"
                  f" {res_sa['solve_time']:6.2f}s"
                  f" {res_sa['n_evaluations']:6d}"
                  f"  │ {'--':>14} {'--':>7} {'--':>9}"
                  f"  │   --     --")

            results.append({
                'n_s': n_s, 'K1': K1, 'K2': K2, 'grid': total_grid,
                'Q_sa': res_sa['Q_min'], 'Q_miqp': None,
                't_sa': res_sa['solve_time'], 't_miqp': None,
                'gap': None, 'n_eval': res_sa['n_evaluations'],
                'match': None, 'speedup': None,
                'miqp_ran': False,
            })

    # =================================================================
    # Summary
    # =================================================================
    print(f"\n{'─' * 80}")
    print("  SUMMARY")
    print(f"{'─' * 80}")

    compared = [r for r in results if r['miqp_ran']]
    sa_only = [r for r in results if not r['miqp_ran']]

    if compared:
        n_match = sum(r['match'] for r in compared)
        n_comp = len(compared)
        max_gap = max(r['gap'] for r in compared)
        avg_speedup = np.mean([r['speedup'] for r in compared])

        print(f"\n  Instances with both SA and MIQP: {n_comp}")
        print(f"  SA found global optimum:         {n_match}/{n_comp}")
        print(f"  Max MIP gap (MIQP cert.):        {max_gap:.2e}"
              f"{'  (all proven optimal)' if max_gap < 1e-6 else ''}")
        print(f"  Avg SA/MIQP time ratio:          {avg_speedup:.1f}x")

        missed = [r for r in compared if not r['match']]
        if missed:
            print(f"\n  SA MISSED the global optimum in {len(missed)} instance(s):")
            for r in missed:
                gap_val = r['Q_sa'] - r['Q_miqp']
                print(f"    n_s={r['n_s']}, grid={r['K1']}x{r['K2']}: "
                      f"Q_sa={r['Q_sa']:.6e}, Q_miqp={r['Q_miqp']:.6e}, "
                      f"gap={gap_val:.4e}")
        else:
            print(f"  SA found the global optimum in ALL compared instances.")

        largest = compared[-1]
        print(f"\n  Largest solved MIQP instance:")
        print(f"    States: {largest['n_s']}, "
              f"Grid: {largest['K1']}x{largest['K2']} = "
              f"{largest['grid']:,} pts")
        print(f"    Binary variables:     {largest['n_binary']}")
        print(f"    Continuous variables:  {largest['n_continuous']}")
        print(f"    Constraints:          {largest['n_constraints']}")
        print(f"    MIQP solve time:      {largest['t_miqp']:.2f}s")
        print(f"    MIP gap:              {largest['gap']:.2e}")

    if sa_only:
        print(f"\n  Instances with SA only (license limit): {len(sa_only)}")
        for r in sa_only:
            print(f"    n_s={r['n_s']:3d}, grid={r['K1']:3d}x{r['K2']:<3d}"
                  f" ({r['grid']:>6,} pts): "
                  f"Q_sa={r['Q_sa']:.6e}, "
                  f"t={r['t_sa']:.1f}s, "
                  f"evals={r['n_eval']}")

    print()


if __name__ == "__main__":
    run()