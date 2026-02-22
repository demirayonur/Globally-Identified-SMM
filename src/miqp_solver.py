"""
miqp_solver.py — Globally Optimal SMM via Mixed-Integer Quadratic Programming
==============================================================================

This is the PROPOSED METHOD in the paper. Instead of searching the
parameter grid with a heuristic (like Simulated Annealing), it encodes
the ENTIRE estimation problem — including the Bellman equation — as a
single optimization problem that a solver (Gurobi) can solve to
CERTIFIED global optimality.

HOW IT WORKS (the big picture)
------------------------------
Traditional SMM is a BILEVEL problem:
    Outer level: choose theta to minimize moment distance
    Inner level: solve the Bellman equation at that theta

This is hard because the inner level is a "black box" — the optimizer
can't see inside it. The MIQP reformulation OPENS the black box by
replacing the Bellman equation with its optimality conditions (primal
feasibility + dual feasibility + complementary slackness) expressed
as linear constraints with binary variables.

The result is a SINGLE-LEVEL optimization problem:
    - Quadratic objective (the SMM criterion, quadratic in lambda)
    - Linear constraints (flow balance, Bellman, Big-M, etc.)
    - Binary variables (which action is optimal at each state,
                        which parameters are selected)

This is a Mixed-Integer Quadratic Program (MIQP), which Gurobi's
Branch-and-Bound algorithm can solve to proven global optimality.

THE DECISION VARIABLES
----------------------
    V[s]       : value function at each state (continuous, free)
    lam[s, a]  : occupancy measure for each (state, action) pair
                 (continuous, non-negative)
    x[s, a]    : is action a optimal at state s? (binary: 0 or 1)
    y1[j]      : is theta1 grid point j selected? (binary: 0 or 1)
    y2[k]      : is RC grid point k selected? (binary: 0 or 1)
    e[l]       : moment gap = data moment - model moment (continuous, free)

THE CONSTRAINTS (mapping to the paper)
--------------------------------------
    (i)   Parameter selection: sum y1 = 1, sum y2 = 1
    (ii)  Flow balance: occupancy measure conservation law
    (iii) Bellman feasibility + Big-M for keep action
    (iv)  Bellman feasibility + Big-M for replace action
    (v)   Occupancy-policy linking: lam(s,a) <= nu_bar * x(s,a)
    (vi)  Moment gap definitions: e[l] = m_hat[l] - m_tilde[l]

THE OBJECTIVE
-------------
    min e' W e    (quadratic in the continuous variables e)

REQUIREMENTS
------------
    gurobipy (pip install gurobipy)
    Free academic license at: https://www.gurobi.com/academia/
"""

import numpy as np
import time
import warnings

# Try to import Gurobi. If not installed, the class will raise
# a helpful error when you try to use it.
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    warnings.warn(
        "gurobipy not found. MIQPSolver will not be available. "
        "Install with: pip install gurobipy"
    )


class MIQPSolver:
    """
    Globally optimal SMM estimation via Mixed-Integer Quadratic Programming.

    Parameters
    ----------
    model : RustBusEngine
        The economic model (from economic_model.py).

    theta1_grid : array-like
        Candidate values for theta1.

    RC_grid : array-like
        Candidate values for RC. 

    data_moments : ndarray, shape (L,)
        Empirical moments from data.

    W : ndarray, shape (L, l) or None
        Weighting matrix. None defaults to identity.
    """

    def __init__(self, model, theta1_grid, RC_grid, data_moments, W=None):

        # Check that Gurobi is available
        if not HAS_GUROBI:
            raise ImportError(
                "Gurobi is required for MIQPSolver. "
                "Install with: pip install gurobipy. "
                "Free academic licenses at gurobi.com."
            )

        self.model = model
        self.theta1_grid = np.asarray(theta1_grid, dtype=float)
        self.RC_grid = np.asarray(RC_grid, dtype=float)
        self.data_moments = np.asarray(data_moments, dtype=float)
        self.W = W if W is not None else np.eye(len(data_moments))

        # Store frequently used quantities
        self.n_s = model.n_s            # number of states
        self.beta = model.beta          # discount factor
        self.nu_bar = model.nu_bar      # = 1/(1-beta)
        self.K1 = len(self.theta1_grid) # number of theta1 candidates
        self.K2 = len(self.RC_grid)     # number of RC candidates

        # ---------------------------------------------------------------
        # Big-M constant
        # ---------------------------------------------------------------
        # M_bar is an upper bound on the Bellman residual.
        # It must be large enough that the Big-M constraints are
        # non-binding when they should be, but as small as possible
        # (tighter Big-M = faster Branch-and-Bound).
        #
        # From the paper: M_bar = 2 * u_bar / (1 - beta)
        # where u_bar = max possible |u(s, a)| over all parameters.
        #
        # For keep: |u(s,0)| = theta1 * s <= max_theta1 * (n_s - 1)
        # For replace: |u(s,1)| = RC <= max_RC
        u_bar = max(
            float(np.max(self.theta1_grid)) * (self.n_s - 1),
            float(np.max(self.RC_grid))
        )
        self.M_bar = 2.0 * u_bar / (1.0 - self.beta)

    def solve(self, time_limit=300, mip_gap=0.0, verbose=False):
        """
        Build and solve the MIQP.

        This is the main method. It:
            1. Creates all decision variables
            2. Adds all constraints
            3. Sets the quadratic objective
            4. Calls Gurobi's Branch-and-Bound solver
            5. Extracts and returns the solution

        Parameters
        ----------
        time_limit : float
            Maximum solver time in seconds. If Gurobi hasn't finished
            by then, it returns the best solution found so far
            (which may not be globally optimal — check the MIP gap).

        mip_gap : float
            Target optimality gap. 0.0 means "prove global optimality."
            A positive value (e.g., 0.01) means "stop when you've proven
            the solution is within 1% of optimal."

        verbose : bool
            If True, Gurobi prints its solver log (useful for debugging).
            If False, solver runs silently.

        Returns
        -------
        results : dict with keys:
            'theta1_best'     : estimated theta1
            'RC_best'         : estimated RC
            'Q_min'           : optimal SMM criterion value
            'optimality_gap'  : MIP gap (0.0 = CERTIFIED global optimum)
            'V'               : value function at the solution
            'lambda'          : occupancy measure at the solution
            'policy'          : optimal policy at the solution
            'model_moments'   : model-implied moments at the solution
            'solve_time'      : Gurobi solve time in seconds
            'n_binary'        : number of binary variables
            'n_continuous'    : number of continuous variables
            'n_constraints'   : number of constraints
        """
        # Shortcuts for frequently used values
        n_s = self.n_s
        beta = self.beta
        nu_bar = self.nu_bar
        M_bar = self.M_bar
        P_keep = self.model.P_keep
        mu = self.model.mu

        # =============================================================
        # CREATE THE GUROBI MODEL
        # =============================================================
        grb = gp.Model("Rust_MIQP")

        # Suppress solver output unless verbose mode is on
        if not verbose:
            grb.Params.OutputFlag = 0

        # Set solver parameters
        grb.Params.TimeLimit = time_limit
        grb.Params.MIPGap = mip_gap

        # =============================================================
        # DECISION VARIABLES
        # =============================================================

        # ----- V[s]: Value function -----
        # One continuous variable per state. Can be any real number
        # (lb=-infinity, ub=+infinity).
        # V[s] represents the lifetime value of being at mileage s.
        V = {}
        for s in range(n_s):
            V[s] = grb.addVar(
                lb=-GRB.INFINITY,    # no lower bound
                ub=GRB.INFINITY,     # no upper bound
                vtype=GRB.CONTINUOUS, # continuous variable
                name=f"V_{s}"        # name for debugging
            )

        # ----- lam[s, a]: Occupancy measure -----
        # One continuous, non-negative variable per (state, action) pair.
        # lam[s, a] = discounted frequency of visiting (s, a).
        lam = {}
        for s in range(n_s):
            for a in range(2):
                lam[s, a] = grb.addVar(
                    lb=0.0,               # non-negative
                    ub=GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"lam_{s}_{a}"
                )

        # ----- x[s, a]: Policy indicator -----
        # One BINARY variable per (state, action) pair.
        # x[s, a] = 1 means "action a is optimal at state s."
        # x[s, a] = 0 means "action a is NOT optimal at state s."
        #
        # These encode the optimal policy and are the key to
        # linearizing complementary slackness.
        x = {}
        for s in range(n_s):
            for a in range(2):
                x[s, a] = grb.addVar(
                    vtype=GRB.BINARY,     # can only be 0 or 1
                    name=f"x_{s}_{a}"
                )

        # ----- y1[j]: Theta1 selection -----
        # One binary variable per theta1 grid point.
        # y1[j] = 1 means "we select theta1 = theta1_grid[j]."
        # Exactly one y1[j] will be 1; the rest will be 0.
        y1 = {}
        for j in range(self.K1):
            y1[j] = grb.addVar(
                vtype=GRB.BINARY,
                name=f"y1_{j}"
            )

        # ----- y2[k]: RC selection -----
        # Same idea as y1, but for the RC parameter.
        y2 = {}
        for k in range(self.K2):
            y2[k] = grb.addVar(
                vtype=GRB.BINARY,
                name=f"y2_{k}"
            )

        # ----- e[l]: Moment gap -----
        # L continuous variables, one per moment.
        # e[l] = data_moment[l] - model_moment[l]
        # The objective minimizes e'We, so these capture the
        # "distance" between data and model.
        e = {}
        for l in range(5):
            e[l] = grb.addVar(
                lb=-GRB.INFINITY,
                ub=GRB.INFINITY,
                vtype=GRB.CONTINUOUS,
                name=f"e_{l}"
            )

        # Tell Gurobi to process the variables we just added
        grb.update()

        # =============================================================
        # CONSTRAINT (i): PARAMETER SELECTION
        # =============================================================
        # Exactly one theta1 value must be selected:
        #     y1[0] + y1[1] + ... + y1[K1-1] = 1
        #
        # Exactly one RC value must be selected:
        #     y2[0] + y2[1] + ... + y2[K2-1] = 1
        #
        # This is what makes the parameters "discrete" — we're
        # choosing from a finite grid, not optimizing continuously.
        grb.addConstr(
            gp.quicksum(y1[j] for j in range(self.K1)) == 1,
            name="select_theta1"
        )
        grb.addConstr(
            gp.quicksum(y2[k] for k in range(self.K2)) == 1,
            name="select_RC"
        )

        # =============================================================
        # CONSTRAINT (ii): FLOW BALANCE
        # =============================================================
        # This is the conservation law for occupancy measures:
        #
        #   lam(s,0) + lam(s,1)
        #     - beta * [inflow from keep] - beta * [inflow from replace]
        #     = mu(s)
        #
        # for every state s.
        #
        # Inflow from keep actions:
        #   sum_{s'} P(s | s', keep) * lam(s', 0)
        #   = sum_{s'} P_keep[s', s] * lam(s', 0)
        #   (Note: P_keep[s', s] = prob of going FROM s' TO s)
        #
        # Inflow from replace actions:
        #   sum_{s'} P(s | s', replace) * lam(s', 1)
        #   Since P(s | s', replace) = P_keep[0, s] for ALL s',
        #   this simplifies to: P_keep[0, s] * sum_{s'} lam(s', 1)
        for s in range(n_s):
            # Inflow from keep: sum over predecessor states s'
            inflow_keep = gp.quicksum(
                P_keep[sp, s] * lam[sp, 0]
                for sp in range(n_s)
            )

            # Inflow from replace: P_keep[0,s] * total replace occupancy
            # This uses the special structure of replacement transitions:
            # all replacements reset to state 0 then evolve the same way.
            inflow_replace = P_keep[0, s] * gp.quicksum(
                lam[sp, 1]
                for sp in range(n_s)
            )

            grb.addConstr(
                lam[s, 0] + lam[s, 1]
                - beta * (inflow_keep + inflow_replace)
                == mu[s],
                name=f"flow_{s}"
            )

        # =============================================================
        # SHARED EXPRESSIONS (used in multiple constraints)
        # =============================================================
        # These are linear expressions in the decision variables.
        # Building them once avoids duplication.

        # The "selected" theta1 value = sum_j theta1_grid[j] * y1[j]
        # Since exactly one y1[j] = 1, this equals the selected theta1.
        theta1_expr = gp.quicksum(
            self.theta1_grid[j] * y1[j]
            for j in range(self.K1)
        )

        # The "selected" RC value = sum_k RC_grid[k] * y2[k]
        RC_expr = gp.quicksum(
            self.RC_grid[k] * y2[k]
            for k in range(self.K2)
        )

        # Continuation value after replacement.
        # This is: beta * sum_{s'} P(s'|0, keep) * V(s')
        # It's the SAME for all states (because replacement always
        # resets to mileage 0, regardless of current mileage).
        # Computing it once saves n_s redundant computations.
        cont_replace = gp.quicksum(
            P_keep[0, sp] * V[sp]
            for sp in range(n_s)
        )

        # =============================================================
        # CONSTRAINT (iii): BELLMAN + BIG-M FOR KEEP (a=0)
        # =============================================================
        # The Bellman residual for keeping at state s is:
        #
        #   delta(s, 0) = V(s) + theta1*s - beta * sum P(s'|s,0) V(s')
        #
        # We need two things:
        #   1. delta(s, 0) >= 0           (primal feasibility)
        #   2. delta(s, 0) <= M_bar * (1 - x(s, 0))   (Big-M)
        #
        # Together these enforce complementary slackness:
        #   If x(s,0) = 1 (keep is optimal): delta(s,0) = 0
        #   If x(s,0) = 0 (keep is not optimal): delta(s,0) free, lam(s,0) = 0
        for s in range(n_s):
            # Continuation value if we keep at state s:
            # sum_{s'} P_keep[s, s'] * V(s')
            cont_keep_s = gp.quicksum(
                P_keep[s, sp] * V[sp]
                for sp in range(n_s)
            )

            # Bellman residual for keep:
            # V(s) - [-theta1*s + beta * cont] = V(s) + theta1*s - beta*cont
            #
            # Note: s * theta1_expr multiplies the integer s by a
            # linear expression in y1 variables. Gurobi handles this.
            residual_keep = V[s] + s * theta1_expr - beta * cont_keep_s

            # Lower bound: residual >= 0
            grb.addConstr(
                residual_keep >= 0,
                name=f"bellman_lb_keep_{s}"
            )

            # Upper bound (Big-M): residual <= M_bar * (1 - x(s, 0))
            grb.addConstr(
                residual_keep <= M_bar * (1 - x[s, 0]),
                name=f"bellman_ub_keep_{s}"
            )

        # =============================================================
        # CONSTRAINT (iv): BELLMAN + BIG-M FOR REPLACE (a=1)
        # =============================================================
        # Same logic as keep, but the residual is:
        #
        #   delta(s, 1) = V(s) + RC - beta * sum P(s'|0,0) V(s')
        #
        # Key observation: the continuation value (cont_replace) is
        # the SAME for all states s. Only V(s) changes across states.
        for s in range(n_s):
            # Bellman residual for replace:
            residual_replace = V[s] + RC_expr - beta * cont_replace

            # Lower bound: residual >= 0
            grb.addConstr(
                residual_replace >= 0,
                name=f"bellman_lb_replace_{s}"
            )

            # Upper bound (Big-M): residual <= M_bar * (1 - x(s, 1))
            grb.addConstr(
                residual_replace <= M_bar * (1 - x[s, 1]),
                name=f"bellman_ub_replace_{s}"
            )

        # =============================================================
        # CONSTRAINT (v): OCCUPANCY-POLICY LINKING
        # =============================================================
        # 0 <= lam(s, a) <= nu_bar * x(s, a)
        #
        # This links the occupancy measure to the policy:
        #   If x(s,a) = 0 (action a not optimal at s):
        #       lam(s,a) <= 0, so lam(s,a) = 0.
        #       Zurcher never takes this action here.
        #   If x(s,a) = 1 (action a is optimal at s):
        #       lam(s,a) <= nu_bar, which is non-binding.
        #       Zurcher may visit this (s,a) pair.
        #
        # Note: the lower bound lam >= 0 was already set when we
        # created the variable (lb=0.0). We only need the upper bound.
        for s in range(n_s):
            for a in range(2):
                grb.addConstr(
                    lam[s, a] <= nu_bar * x[s, a],
                    name=f"occ_policy_{s}_{a}"
                )

        # =============================================================
        # CONSTRAINT (vi): MOMENT GAP DEFINITIONS
        # =============================================================
        # e[l] = m_hat[l] - m_tilde[l]
        #
        # where the model-implied moments are LINEAR in lambda:
        #
        #   m_tilde_1 = nu_inv * sum_s lam(s, 1)
        #   m_tilde_2 = nu_inv * sum_s s * [lam(s,0) + lam(s,1)]
        #   m_tilde_3 = nu_inv * sum_s s * lam(s, 1)
        #
        # CRITICAL: h_l(s, a) does NOT depend on theta.
        # So m_tilde is purely linear in lambda.
        # No McCormick auxiliary variables z needed.
        nu_inv = 1.0 / nu_bar

        # Moment 1: replacement frequency
        m_tilde_1 = nu_inv * gp.quicksum(
            lam[s, 1] for s in range(n_s)
        )

        # Moment 2: average mileage
        m_tilde_2 = nu_inv * gp.quicksum(
            s * (lam[s, 0] + lam[s, 1]) for s in range(n_s)
        )

        # Moment 3: mileage-replacement cross-moment
        m_tilde_3 = nu_inv * gp.quicksum(
            s * lam[s, 1] for s in range(n_s)
        )

        # Moment 4: second moment of mileage (s^2)
        m_tilde_4 = nu_inv * gp.quicksum(
            s * s * (lam[s, 0] + lam[s, 1]) for s in range(n_s)
        )

        # Moment 5: second moment of mileage at replacement (s^2 * 1{a=1})
        m_tilde_5 = nu_inv * gp.quicksum(
            s * s * lam[s, 1] for s in range(n_s)
        )

        # Link the e variables to the moments
        grb.addConstr(
            e[0] == self.data_moments[0] - m_tilde_1,
            name="gap_m1"
        )
        grb.addConstr(
            e[1] == self.data_moments[1] - m_tilde_2,
            name="gap_m2"
        )
        grb.addConstr(
            e[2] == self.data_moments[2] - m_tilde_3,
            name="gap_m3"
        )
        grb.addConstr(
            e[3] == self.data_moments[3] - m_tilde_4,
            name="gap_m4"
        )
        grb.addConstr(
            e[4] == self.data_moments[4] - m_tilde_5,
            name="gap_m5"
        )

        # =============================================================
        # OBJECTIVE: min Q = e' W e
        # =============================================================
        # This is a QUADRATIC function of the continuous variables e.
        # Since e is affine in lambda (linear plus constants from m_hat),
        # Q is quadratic in lambda.
        #
        # Together with binary variables for policy (x) and parameters
        # (y1, y2), this gives us a Mixed-Integer Quadratic Program.
        #
        # Gurobi's Branch-and-Bound solves this to certified global
        # optimality: when MIP gap = 0, we KNOW no better solution exists.
        obj = gp.QuadExpr()
        for l in range(5):
            for lp in range(5):
                # W[l, lp] * e[l] * e[lp]
                obj += self.W[l, lp] * e[l] * e[lp]

        grb.setObjective(obj, GRB.MINIMIZE)

        # =============================================================
        # SOLVE
        # =============================================================
        grb.optimize()

        # Count variables for reporting
        n_binary = sum(
            1 for v in grb.getVars() if v.vType == GRB.BINARY
        )
        n_continuous = sum(
            1 for v in grb.getVars() if v.vType == GRB.CONTINUOUS
        )
        n_constraints = grb.NumConstrs

        # =============================================================
        # EXTRACT SOLUTION
        # =============================================================
        if grb.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):

            # Which theta1 was selected?
            # Find the j where y1[j] = 1 (the rest are 0).
            theta1_idx = None
            for j in range(self.K1):
                # .X gives the solution value. Binary vars should be
                # 0 or 1, but due to numerical precision they might be
                # 0.999... or 0.001..., so we threshold at 0.5.
                if y1[j].X > 0.5:
                    theta1_idx = j
                    break

            # Which RC was selected?
            RC_idx = None
            for k in range(self.K2):
                if y2[k].X > 0.5:
                    RC_idx = k
                    break

            theta1_est = self.theta1_grid[theta1_idx]
            RC_est = self.RC_grid[RC_idx]

            # Extract the full occupancy measure
            lam_sol = np.zeros((n_s, 2))
            for s in range(n_s):
                for a in range(2):
                    lam_sol[s, a] = lam[s, a].X

            # Extract the value function
            V_sol = np.array([V[s].X for s in range(n_s)])

            # Extract the policy from the x variables
            policy_sol = np.zeros(n_s, dtype=int)
            for s in range(n_s):
                if x[s, 1].X > 0.5:
                    policy_sol[s] = 1

            # Compute model-implied moments at the solution
            # (using the model class for consistency)
            model_moments = self.model.compute_moments(lam_sol)

            return {
                'theta1_best': theta1_est,
                'RC_best': RC_est,
                'theta1_idx': theta1_idx,
                'RC_idx': RC_idx,
                'Q_min': grb.ObjVal,
                'optimality_gap': grb.MIPGap,
                'V': V_sol,
                'lambda': lam_sol,
                'policy': policy_sol,
                'model_moments': model_moments,
                'solve_time': grb.Runtime,
                'status': grb.status,
                'n_binary': n_binary,
                'n_continuous': n_continuous,
                'n_constraints': n_constraints,
            }
        else:
            # Solver failed (infeasible, time limit, etc.)
            return {
                'status': grb.status,
                'solve_time': grb.Runtime,
                'error': f"Gurobi terminated with status {grb.status}",
                'n_binary': n_binary,
                'n_continuous': n_continuous,
                'n_constraints': n_constraints,
            }
    