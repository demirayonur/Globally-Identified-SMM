"""
simulated_annealing.py — SMM Estimation with Simulated Annealing
================================================================

This is the TRADITIONAL estimation approach used in practice.
It combines two components:

    INNER LOOP (at each candidate parameter):
        Solve the Bellman equation -> occupancy measure -> moments -> Q(theta)

    OUTER LOOP (searching over parameters):
        Simulated Annealing explores the parameter grid trying to find
        the theta that minimizes Q.

WHAT IS SIMULATED ANNEALING?
----------------------------
Imagine you're blindfolded on a hilly landscape and want to find the
lowest valley. You can:
    1. Take a step in a random direction
    2. If you went downhill, great — keep going
    3. If you went uphill, MAYBE keep going anyway (with some probability)

The "maybe keep going uphill" part is what makes SA different from
simple hill-climbing. It lets the algorithm escape local minima by
occasionally accepting worse solutions. The probability of accepting
an uphill move is controlled by a "temperature" parameter:

    P(accept uphill move) = exp(-delta / temperature)

where delta = how much worse the new point is.

At high temperature: almost any move is accepted (exploration).
At low temperature: only downhill moves are accepted (exploitation).

The temperature starts high and decreases over iterations (cooling).
This is why it's called "annealing" — like cooling metal slowly.

FAIR COMPARISON WITH MIQP
--------------------------
Both SA and MIQP search the SAME finite parameter grid.
SA navigates it by random local moves on the grid indices.
MIQP encodes the entire grid in binary variables and solves exactly.
The comparison is apples-to-apples.

THE PROBLEM WITH SA
-------------------
SA can get STUCK in local minima. Even with random restarts, there's
no guarantee it finds the global minimum. And crucially, there's no
way to KNOW whether you found it or not. The MIQP solver fixes this
with a certificate of optimality.
"""

import numpy as np
import time


class SMMSimulatedAnnealing:
    """
    Traditional SMM estimation using Simulated Annealing on a discrete grid.

    Parameters
    ----------
    model : RustBusEngine
        The economic model (from economic_model.py).

    theta1_grid : array-like
        Candidate values for theta1. This is the SAME grid the MIQP uses.

    RC_grid : array-like
        Candidate values for RC. This is the SAME grid the MIQP uses.

    data_moments : ndarray, shape (5,)
        Empirical moments from data.

    W : ndarray, shape (5, 5) or None
        Weighting matrix. None defaults to identity.
    """

    def __init__(self, model, theta1_grid, RC_grid, data_moments, W=None):
        self.model = model
        self.theta1_grid = np.asarray(theta1_grid)
        self.RC_grid = np.asarray(RC_grid)
        self.data_moments = np.asarray(data_moments)
        self.W = W if W is not None else np.eye(len(data_moments))

        # Grid dimensions
        self.K1 = len(self.theta1_grid)
        self.K2 = len(self.RC_grid)

        # Cache: stores previously computed Q values to avoid
        # solving the Bellman equation at the same point twice.
        # Key = (j, k) grid indices, Value = Q(theta1_j, RC_k).
        self._cache = {}

    def _evaluate(self, j, k):
        """
        Evaluate the SMM objective at grid indices (j, k).

        Uses a cache to avoid redundant computation. If we've already
        evaluated Q at this grid point, just return the stored value.
        Otherwise, run the full inner loop and store the result.

        Parameters
        ----------
        j : int
            Index into theta1_grid (0 <= j < K1).
        k : int
            Index into RC_grid (0 <= k < K2).

        Returns
        -------
        Q : float
            SMM criterion value at (theta1_grid[j], RC_grid[k]).
        """
        if (j, k) not in self._cache:
            # First time seeing this grid point: compute and store
            self._cache[(j, k)] = self.model.smm_objective(
                self.theta1_grid[j], self.RC_grid[k],
                self.data_moments, self.W
            )
        return self._cache[(j, k)]

    def _propose_neighbor(self, j, k, rng):
        """
        Propose a random neighbor on the DISCRETE parameter grid.

        This is how SA explores the grid. From the current position
        (j, k), we randomly pick one of two moves:
            - Move theta1 up or down by one grid step (j +/- 1)
            - Move RC up or down by one grid step (k +/- 1)

        If the move would go off the edge of the grid, we "reflect"
        back (stay at the boundary). This ensures we always stay
        within the valid grid.

        IMPORTANT: We move by ONE grid step at a time. This is a
        local search — SA explores the grid one step at a time,
        relying on the temperature schedule to eventually explore
        the whole space.

        Parameters
        ----------
        j, k : int
            Current grid indices.
        rng : numpy Generator
            Random number generator.

        Returns
        -------
        j_new, k_new : int
            Proposed new grid indices.
        """
        if rng.random() < 0.5:
            # Perturb theta1 index by +1 or -1
            j_new = j + rng.choice([-1, 1])
            # Clamp to valid range [0, K1-1]
            j_new = max(0, min(self.K1 - 1, j_new))
            return j_new, k
        else:
            # Perturb RC index by +1 or -1
            k_new = k + rng.choice([-1, 1])
            # Clamp to valid range [0, K2-1]
            k_new = max(0, min(self.K2 - 1, k_new))
            return j, k_new

    def _single_run(self, max_iter, T_init, cooling_rate, seed=None):
        """
        Execute a single run of Simulated Annealing.

        Starting from a random grid point, the algorithm takes max_iter
        steps, gradually cooling the temperature.

        Parameters
        ----------
        max_iter : int
            Number of SA iterations (steps to take).
        T_init : float
            Initial temperature. Higher = more exploration initially.
        cooling_rate : float
            Temperature multiplier each iteration: T_{i+1} = cooling_rate * T_i.
            Must be in (0, 1). Closer to 1 = slower cooling.
            Example: 0.995 means temperature drops by 0.5% each step.
        seed : int or None
            Random seed for this run.

        Returns
        -------
        j_best, k_best : int
            Grid indices of the best solution found during this run.
        Q_best : float
            Best objective value found.
        """
        rng = np.random.default_rng(seed)

        # Start at a RANDOM grid point
        j = rng.integers(0, self.K1)
        k = rng.integers(0, self.K2)
        Q_current = self._evaluate(j, k)

        # Track the best solution seen so far in this run
        j_best, k_best, Q_best = j, k, Q_current
        temperature = T_init

        for iteration in range(max_iter):
            # --- Step 1: Propose a neighbor ---
            j_prop, k_prop = self._propose_neighbor(j, k, rng)
            Q_prop = self._evaluate(j_prop, k_prop)

            # --- Step 2: Decide whether to accept the move ---
            delta = Q_prop - Q_current  # positive = worse, negative = better

            if delta <= 0:
                # The proposed point is BETTER (lower Q). Always accept.
                j, k, Q_current = j_prop, k_prop, Q_prop
            elif temperature > 1e-15:
                # The proposed point is WORSE. Accept with probability
                # exp(-delta / temperature).
                #
                # High temperature -> exp is close to 1 -> usually accept
                # Low temperature  -> exp is close to 0 -> rarely accept
                #
                # This is the Metropolis criterion.
                acceptance_prob = np.exp(-delta / temperature)
                if rng.random() < acceptance_prob:
                    j, k, Q_current = j_prop, k_prop, Q_prop

            # --- Step 3: Update the best solution if current is better ---
            if Q_current < Q_best:
                j_best, k_best, Q_best = j, k, Q_current
            
            # --- Step 4: Cool the temperature ---
            temperature *= cooling_rate

        return j_best, k_best, Q_best

    def solve(self, n_restarts=5, max_iter=10_000, T_init=1.0,
              cooling_rate=0.995, seed=None):
        """
        Run Simulated Annealing with multiple random restarts.

        WHY MULTIPLE RESTARTS?
        A single SA run might get stuck in a local minimum. Running
        multiple independent runs from different starting points
        increases the chance of finding the global minimum.
        We keep the best solution across all runs.

        BUT: even with many restarts, there's NO GUARANTEE of finding
        the global minimum. This is the fundamental limitation that
        the MIQP solver addresses.

        Parameters
        ----------
        n_restarts : int
            Number of independent SA runs. Each starts from a random point.
            More restarts = higher chance of finding global optimum,
            but proportionally more computation time.

        max_iter : int
            Number of iterations per run.

        T_init : float
            Initial temperature for each run.

        cooling_rate : float
            Geometric cooling factor.

        seed : int or None
            Master random seed. Each restart gets a different sub-seed
            derived from this master seed, ensuring reproducibility.

        Returns
        -------
        results : dict with keys:
            'theta1_best'   : best theta1 found across all restarts
            'RC_best'       : best RC found
            'Q_min'         : best (lowest) objective value
            'all_runs'      : list of per-restart results
            'n_evaluations' : total unique grid points evaluated
            'solve_time'    : wall-clock seconds
        """
        t0 = time.time()
        rng = np.random.default_rng(seed)

        # Clear the cache so we can count fresh evaluations
        self._cache = {}

        best_j, best_k, best_Q = None, None, np.inf
        all_runs = []

        for r in range(n_restarts):
            # Generate a unique seed for this restart
            run_seed = int(rng.integers(0, 2**31))

            # Run SA from a random starting point
            j, k, Q = self._single_run(
                max_iter, T_init, cooling_rate, seed=run_seed
            )

            # Record this run's result
            all_runs.append({
                'restart': r,
                'theta1': self.theta1_grid[j],
                'RC': self.RC_grid[k],
                'theta1_idx': j,
                'RC_idx': k,
                'Q': Q,
            })

            # Update the overall best if this run found something better
            if Q < best_Q:
                best_j, best_k, best_Q = j, k, Q

        elapsed = time.time() - t0

        return {
            'theta1_best': self.theta1_grid[best_j],
            'RC_best': self.RC_grid[best_k],
            'theta1_idx': best_j,
            'RC_idx': best_k,
            'Q_min': best_Q,
            'all_runs': all_runs,
            'n_evaluations': len(self._cache),
            'solve_time': elapsed,
        }