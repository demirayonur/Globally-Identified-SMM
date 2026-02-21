"""
main.py — Demo: Compare All Three Estimators
=============================================

This script demonstrates the complete workflow:

    1. Set up the economic model
    2. Choose true parameters and simulate panel data
    3. Define the parameter grid 
    4. Run exhaustive enumeration (gold standard)
    5. Run simulated annealing (traditional approach)
    6. Run the MIQP solver (proposed method)
    7. Compare results

Run with:
    python main.py
"""

from src.economic_model import RustBusEngine
from src.exhaustive_search import ExhaustiveEnumerator
from src.simulated_annealing import SMMSimulatedAnnealing

try:
    from src.miqp_solver import MIQPSolver
    HAS_MIQP = True
except ImportError:
    HAS_MIQP = False
    print("NOTE: Gurobi not found. MIQP solver will be skipped.")
    print("      Install with: pip install gurobipy\n")

import numpy as np

"""Run the full comparison."""

sep = "=" * 70
dash = "-" * 70

print("\n")
print(sep)
print("Rust (1987) Bus Engine Replacement -- Estimation Comparison")
print(sep)

# =================================================================
# STEP 1: Create the economic model
# =================================================================
# This sets up the state space, action space, transitions, and
# discount factor. These are FIXED — not estimated.

model = RustBusEngine(
    n_s=200,                         # 100 mileage bins
    beta=0.75,                       # discount factor
    trans_probs=(0.4, 0.5, 0.1)      # mileage transition probs
)

print(f"\nModel setup:")
print(f"  State space:      {model.n_s} mileage bins")
print(f"  Action space:     {{0=keep, 1=replace}}")
print(f"  Discount factor:  beta = {model.beta}")
print(f"  Transition probs: {model.trans_probs}")
print(f"  nu_bar = 1/(1-beta) = {model.nu_bar:.2f}")

# =================================================================
# STEP 2: Choose true parameters and examine the model
# =================================================================
# In a Monte Carlo experiment, we know the "truth" and check
# whether our estimator can recover it from simulated data.

theta1_true = 2.0    # maintenance cost grows at rate 2 per bin
RC_true = 15.0       # replacement costs 15

print(f"\nTrue parameters:")
print(f"  theta1 = {theta1_true}   (maintenance cost rate)")
print(f"  RC     = {RC_true}  (replacement cost)")

# Solve the model at true parameters to see what the optimal
# policy looks like and what moments the model predicts.
V_true, policy_true = model.solve_bellman(theta1_true, RC_true)
lam_true = model.compute_occupancy_measure(policy_true)
pop_moments = model.compute_moments(lam_true)

# Find the replacement threshold (first state where policy = replace)
replace_states = np.where(policy_true == 1)[0]
threshold = replace_states[0] if len(replace_states) > 0 else None

print(f"\nAt true parameters:")
print(f"  Optimal policy: replace at mileage >= {threshold}")
print(f"  Population moments (exact, no simulation noise):")
print(f"    m1 (replace freq)    = {pop_moments[0]:.6f}")
print(f"    m2 (avg mileage)     = {pop_moments[1]:.6f}")
print(f"    m3 (mileage*replace) = {pop_moments[2]:.6f}")
print(f"    m4 (mileage^2)       = {pop_moments[3]:.6f}")
print(f"    m5 (mileage^2*repl)  = {pop_moments[4]:.6f}")

# =================================================================
# STEP 3: Simulate panel data
# =================================================================
# Generate fake data by simulating N buses for T periods.
# The data moments will differ from population moments due to
# sampling noise (finite N and T).

N_buses = 5000       # number of buses
T_periods = 500      # periods per bus
sim_seed = 42         # for reproducibility

print(f"\nSimulating panel data:")
print(f"  N = {N_buses} buses, T = {T_periods} periods, "
        f"seed = {sim_seed}")

states, actions = model.simulate_panel(
    theta1_true, RC_true,
    N=N_buses, T=T_periods, seed=sim_seed
)
data_moments = model.compute_data_moments(states, actions)

print(f"  Data moments (with sampling noise):")
print(f"    m1 = {data_moments[0]:.6f}  "
        f"(pop: {pop_moments[0]:.6f}, "
        f"diff: {data_moments[0]-pop_moments[0]:+.6f})")
print(f"    m2 = {data_moments[1]:.6f}  "
        f"(pop: {pop_moments[1]:.6f}, "
        f"diff: {data_moments[1]-pop_moments[1]:+.6f})")
print(f"    m3 = {data_moments[2]:.6f}  "
        f"(pop: {pop_moments[2]:.6f}, "
        f"diff: {data_moments[2]-pop_moments[2]:+.6f})")
print(f"    m4 = {data_moments[3]:.6f}  "
        f"(pop: {pop_moments[3]:.6f}, "
        f"diff: {data_moments[3]-pop_moments[3]:+.6f})")
print(f"    m5 = {data_moments[4]:.6f}  "
        f"(pop: {pop_moments[4]:.6f}, "
        f"diff: {data_moments[4]-pop_moments[4]:+.6f})")

# =================================================================
# STEP 4: Define the parameter grid
# =================================================================
# THIS IS THE SAME GRID FOR ALL THREE ESTIMATORS.
# Fair comparison requires searching the same set of candidates.

theta1_grid = np.linspace(0.5, 5.0, 100)   # 19 values from 0.5 to 5.0
RC_grid = np.linspace(5.0, 30.0, 100)      # 21 values from 5.0 to 30.0
total_grid = len(theta1_grid) * len(RC_grid)

print(f"\nParameter grid (SAME for all estimators):")
print(f"  theta1: {len(theta1_grid)} values in "
        f"[{theta1_grid[0]:.2f}, {theta1_grid[-1]:.2f}]")
print(f"  RC:     {len(RC_grid)} values in "
        f"[{RC_grid[0]:.2f}, {RC_grid[-1]:.2f}]")
print(f"  Total:  {total_grid} grid points")

# Check if true values are on the grid
on_grid_t1 = np.any(np.isclose(theta1_grid, theta1_true))
on_grid_rc = np.any(np.isclose(RC_grid, RC_true))
print(f"  theta1_true on grid: {'yes' if on_grid_t1 else 'NO'}")
print(f"  RC_true on grid:     {'yes' if on_grid_rc else 'NO'}")

# Weighting matrix: identity (all moments weighted equally)
W = np.eye(5)

# =================================================================
# ESTIMATOR 1: Exhaustive Enumeration (gold standard)
# =================================================================
print(f"\n{dash}")
print("ESTIMATOR 1: EXHAUSTIVE ENUMERATION (gold standard)")
print(dash)
print("  Evaluating SMM at all", total_grid, "grid points...")

enum = ExhaustiveEnumerator(
    model, theta1_grid, RC_grid, data_moments, W
)
res_enum = enum.solve()

print(f"  Time:            {res_enum['solve_time']:.2f} s")
print(f"  Q_min:           {res_enum['Q_min']:.10e}")
print(f"  theta1_best:     {res_enum['theta1_best']:.4f}")
print(f"  RC_best:         {res_enum['RC_best']:.4f}")
print(f"  # global optima: {res_enum['n_global_optima']}")

if res_enum['n_global_optima'] > 1:
    if res_enum['n_global_optima'] <= 10:
        print("  All global optima:")
        for opt in res_enum['global_optima']:
            print(f"    theta1={opt['theta1']:.4f}, "
                    f"RC={opt['RC']:.4f}, Q={opt['Q']:.10e}")
    else:
        for opt in res_enum['global_optima'][:5]:
            print(f"    theta1={opt['theta1']:.4f}, "
                    f"RC={opt['RC']:.4f}, Q={opt['Q']:.10e}")


# =================================================================
# ESTIMATOR 2: Simulated Annealing (traditional approach)
# =================================================================
print(f"\n{dash}")
print("ESTIMATOR 2: SIMULATED ANNEALING (5 restarts)")
print(dash)

sa = SMMSimulatedAnnealing(
    model, theta1_grid, RC_grid, data_moments, W
)
res_sa = sa.solve(
    n_restarts=5,           # 5 independent runs
    max_iter=10_000,        # 10,000 steps per run
    T_init=1.0,             # initial temperature
    cooling_rate=0.995,     # cool by 0.5% each step
    seed=123                # for reproducibility
)

print(f"  Time:            {res_sa['solve_time']:.2f} s")
print(f"  Q_min:           {res_sa['Q_min']:.10e}")
print(f"  theta1_best:     {res_sa['theta1_best']:.4f}")
print(f"  RC_best:         {res_sa['RC_best']:.4f}")
print(f"  # grid pts evaluated: {res_sa['n_evaluations']} "
        f"out of {total_grid}")

# Did SA find the global optimum?
sa_is_global = abs(res_sa['Q_min'] - res_enum['Q_min']) < 1e-8
print(f"  Found global optimum: {'YES' if sa_is_global else 'NO'}")
if not sa_is_global:
    gap = res_sa['Q_min'] - res_enum['Q_min']
    print(f"  Gap to global:   {gap:.6e}")

# Show results from each restart
print("  Per-restart results:")
for run in res_sa['all_runs']:
    is_best = " <-- best" if abs(run['Q'] - res_sa['Q_min']) < 1e-12 else ""
    print(f"    Restart {run['restart']}: "
            f"theta1={run['theta1']:.4f}, RC={run['RC']:.4f}, "
            f"Q={run['Q']:.6e}{is_best}")

# =================================================================
# ESTIMATOR 3: MIQP Solver (proposed method)
# =================================================================
if HAS_MIQP:
    print(f"\n{dash}")
    print("ESTIMATOR 3: MIQP SOLVER (proposed method)")
    print(dash)

    miqp = MIQPSolver(
        model, theta1_grid, RC_grid, data_moments, W
    )
    res_miqp = miqp.solve(
        time_limit=300,     # max 5 minutes
        mip_gap=0.0,        # require exact global optimum
        verbose=False        # suppress solver output
    )

    if 'error' not in res_miqp:
        print(f"  Time:            {res_miqp['solve_time']:.2f} s")
        print(f"  Q_min:           {res_miqp['Q_min']:.10e}")
        print(f"  theta1_best:     {res_miqp['theta1_best']:.4f}")
        print(f"  RC_best:         {res_miqp['RC_best']:.4f}")
        print(f"  MIP gap:         {res_miqp['optimality_gap']:.2e}")
        print(f"  # binary vars:   {res_miqp['n_binary']}")
        print(f"  # continuous:    {res_miqp['n_continuous']}")
        print(f"  # constraints:   {res_miqp['n_constraints']}")

        # Verify against exhaustive enumeration
        miqp_matches = abs(res_miqp['Q_min'] - res_enum['Q_min']) < 1e-6
        print(f"\n  Matches exhaustive enumeration: "
                f"{'YES' if miqp_matches else 'NO'}")
        if not miqp_matches:
            diff = abs(res_miqp['Q_min'] - res_enum['Q_min'])
            print(f"  MIQP Q:  {res_miqp['Q_min']:.10e}")
            print(f"  Enum Q:  {res_enum['Q_min']:.10e}")
            print(f"  Diff:    {diff:.6e}")

        # Check if MIQP solution is among the global optima
        miqp_in_optima = any(
            abs(opt['theta1'] - res_miqp['theta1_best']) < 1e-8 and
            abs(opt['RC'] - res_miqp['RC_best']) < 1e-8
            for opt in res_enum['global_optima']
        )
        print(f"  MIQP solution in enumerated optima set: "
                f"{'YES' if miqp_in_optima else 'NO'}")
    else:
        print(f"  Error: {res_miqp['error']}")
