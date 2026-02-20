"""
exhaustive.py — Exhaustive Enumeration (Gold Standard)
======================================================

This is the simplest possible estimation method: try EVERY parameter
combination on the grid, evaluate the SMM objective at each one,
and pick the one with the lowest value.

WHY IS THIS THE "GOLD STANDARD"?
Because it is guaranteed to find the global optimum by construction.
If you check every point, you literally cannot miss the best one.
The MIQP solver claims to find the global optimum via a clever
reformulation. Exhaustive enumeration is how we VERIFY that claim.

WHY NOT ALWAYS USE THIS?
It works fine when the grid is small (say 15 x 15 = 225 points).
But if you have 5 parameters each with 30 grid points, you'd need
30^5 = 24,300,000 evaluations. Each one requires solving a Bellman
equation. That would take days or weeks.

The MIQP solver achieves the same answer (certified global optimum)
without checking every point, using Branch-and-Bound to prune.

IMPORTANT ON MULTIPLE GLOBAL OPTIMA
------------------------------------
It is possible that several parameter vectors achieve the SAME minimum
objective value. This happens when different parameters produce the
same optimal policy (same replacement threshold), hence the same
occupancy measure, hence the same moments. The enumerator collects
ALL such ties, not just the first one found.
"""

import numpy as np
import time


class ExhaustiveEnumerator:
    """
    Evaluate the SMM objective at every point on the parameter grid.

    Parameters
    ----------
    model : RustBusEngine
        The economic model (from economic_model.py).

    theta1_grid : array-like
        List of candidate values for theta1.
        Example: np.linspace(0.5, 5.0, 19) gives 19 evenly spaced values.

    RC_grid : array-like
        List of candidate values for RC.
        Example: np.linspace(5.0, 30.0, 21) gives 21 evenly spaced values.

    data_moments : ndarray, shape (n_moments,)
        Empirical moments computed from data.

    W : ndarray, shape (n_moments, n_moments) or None
        Weighting matrix. None defaults to identity.
    """

    def __init__(self, model, theta1_grid, RC_grid, data_moments, W=None):
        self.model = model
        self.theta1_grid = np.asarray(theta1_grid)
        self.RC_grid = np.asarray(RC_grid)
        self.data_moments = np.asarray(data_moments)

        # Default weighting matrix: identity (all moments weighted equally)
        self.W = W if W is not None else np.eye(len(data_moments))

        # Grid dimensions
        self.K1 = len(self.theta1_grid)  # number of theta1 candidates
        self.K2 = len(self.RC_grid)       # number of RC candidates

    def solve(self, tol=1e-10):
        """
        Evaluate SMM at every grid point and find all global minima.

        The algorithm:
            for each theta1 on the grid:
                for each RC on the grid:
                    1. Solve Bellman equation at (theta1, RC)
                    2. Compute occupancy measure
                    3. Compute model-implied moments
                    4. Evaluate Q = [m_hat - m_tilde]' W [m_hat - m_tilde]
                    5. Store Q in a table

            Find the minimum Q in the table.
            Collect all (theta1, RC) pairs that achieve this minimum.

        Parameters
        ----------
        tol : float
            Tolerance for identifying ties. Two Q-values are considered
            equal if they differ by less than tol.

        Returns
        -------
        results : dict with keys:
            'Q_surface'       : (K1, K2) array of all objective values
            'Q_min'           : the minimum objective value found
            'global_optima'   : list of dicts, one per global optimum
            'n_global_optima' : how many global optima exist
            'theta1_best'     : theta1 of the first global optimum
            'RC_best'         : RC of the first global optimum
            'solve_time'      : wall-clock seconds
        """
        t0 = time.time()

        # Create a 2D table to store Q(theta1, RC) for every combination
        Q_surface = np.zeros((self.K1, self.K2))

        # Loop over every grid point
        for j, theta1 in enumerate(self.theta1_grid):
            for k, RC in enumerate(self.RC_grid):
                # Evaluate the SMM objective at this parameter pair.
                # This calls: Bellman -> policy -> occupancy -> moments -> Q
                Q_surface[j, k] = self.model.smm_objective(
                    theta1, RC, self.data_moments, self.W
                )

        # Find the global minimum
        Q_min = np.min(Q_surface)

        # Collect ALL parameter pairs that achieve this minimum
        # (within numerical tolerance). There may be ties.
        global_optima = []
        for j in range(self.K1):
            for k in range(self.K2):
                if Q_surface[j, k] <= Q_min + tol:
                    global_optima.append({
                        'theta1': self.theta1_grid[j],
                        'RC': self.RC_grid[k],
                        'theta1_idx': j,
                        'RC_idx': k,
                        'Q': Q_surface[j, k],
                    })

        elapsed = time.time() - t0

        return {
            'Q_surface': Q_surface,
            'Q_min': Q_min,
            'global_optima': global_optima,
            'n_global_optima': len(global_optima),
            'theta1_best': global_optima[0]['theta1'],
            'RC_best': global_optima[0]['RC'],
            'solve_time': elapsed,
        }
