"""
economic_model.py — The Rust (1987) Bus Engine Replacement Model
================================================================

This file contains the economics of the problem. It knows nothing about
estimation methods (no simulated annealing, no MIQP). It only knows:

    1. What the model looks like (states, actions, payoffs, transitions)
    2. How to solve the Bellman equation (find the optimal policy)
    3. How to compute occupancy measures (long-run behavior)
    4. How to compute moments (summary statistics the model predicts)
    5. How to simulate fake data (for Monte Carlo experiments)

THE STORY
---------
Onur Demiray manages a fleet of buses. Each month he looks at each
engine, sees its accumulated mileage, and decides:

    - KEEP the engine (a=0): pay maintenance cost that grows with mileage
    - REPLACE it (a=1): pay a fixed cost for a brand new engine

The maintenance cost is:   u(s, 0) = -theta1 * s
The replacement cost is:   u(s, 1) = -RC

where s is the mileage bin (0 = new, 29 = very old for a 30-bin grid).

The key question: at what mileage should Onur replace the engine?
The answer depends on the parameters (theta1, RC), which we want to
estimate from data.
"""

import numpy as np
from scipy.linalg import solve


class RustBusEngine:
    """
    The economic model. Everything about the bus engine problem.

    Parameters
    ----------
    n_s : int
        Number of mileage bins. The state space is {0, 1, 2, ..., n_s - 1}.
        State 0 = brand new engine. State n_s-1 = highest mileage bin.
        Each bin represents roughly 10,000 miles.
        Example: n_s=30 covers 0 to 290,000 miles.

    beta : float
        Discount factor, must be between 0 and 1 (exclusive).
        Controls how much the agent cares about the future.
        beta=0.95 means $1 next period is worth $0.95 today.
        Higher beta = more patient agent = replaces later.

    trans_probs : tuple of 3 floats
        (p0, p1, p2) = probabilities that mileage increases by 0, 1, or 2
        bins in one period.
        Must sum to 1.
        Example: (0.4, 0.5, 0.1) means 40% chance mileage stays same,
        50% chance it goes up one bin, 10% chance it goes up two bins.
    """

    def __init__(self, n_s=30, beta=0.95, trans_probs=(0.4, 0.5, 0.1)):

        # ---------------------------------------------------------------
        # Input validation
        # ---------------------------------------------------------------
        # These checks catch common mistakes before they cause
        # mysterious errors later.

        assert 0 < beta < 1, f"Discount factor must be in (0,1), got {beta}"
        assert len(trans_probs) == 3, f"Need exactly 3 transition probabilities, got {len(trans_probs)}"
        assert abs(sum(trans_probs) - 1.0) < 1e-12, f"Transition probs must sum to 1, got {sum(trans_probs)}"
        assert all(p >= 0 for p in trans_probs), "Transition probabilities must be non-negative"

        # ---------------------------------------------------------------
        # Store model primitives
        # ---------------------------------------------------------------
        self.n_s = n_s                 # number of states
        self.beta = beta               # discount factor
        self.trans_probs = trans_probs # mileage increment probabilities

        # ---------------------------------------------------------------
        # Derived constants 
        # ---------------------------------------------------------------
        # nu_bar = 1/(1-beta): the total discounted mass.
        self.nu_bar = 1.0 / (1.0 - beta)

        # ---------------------------------------------------------------
        # Build transition matrices 
        # ---------------------------------------------------------------
        # P_keep[s, s']    = probability of going from state s to s' if KEEP
        # P_replace[s, s'] = probability of going from state s to s' if REPLACE
        #
        # These are n_s x n_s matrices. Each row sums to 1.
        self.P_keep, self.P_replace = self._build_transition_matrices()

        # A convenient dictionary so we can write self.P[action]
        # instead of if/else blocks everywhere.
        self.P = {0: self.P_keep, 1: self.P_replace}

        # ---------------------------------------------------------------
        # Initial distribution over states (Assumption A3 in the paper)
        # ---------------------------------------------------------------
        # mu[s] = probability of starting in state s.
        # We use uniform: every mileage bin is equally likely initially.
        # This satisfies Assumption A3 (strictly positive for all states).
        self.mu = np.ones(n_s) / n_s

    def _build_transition_matrices(self):
        """
        Build the keep and replace transition matrices.

        KEEP (a=0): mileage drifts UP from wherever it currently is.
            If currently at mileage bin s, next period you're at:
                s     with probability p0  (no increase)
                s+1   with probability p1  (increase by 1 bin)
                s+2   with probability p2  (increase by 2 bins)
            BOUNDARY RULE: if s+j would exceed n_s-1, you stay at n_s-1.
            This is called "absorption" — the highest bin is a ceiling.

        REPLACE (a=1): mileage RESETS to 0, then drifts.
            Regardless of current mileage, after replacement you're at:
                0     with probability p0
                1     with probability p1
                2     with probability p2
            This is the same as starting from state 0 and keeping.

        Returns
        -------
        P_keep : ndarray, shape (n_s, n_s)
            P_keep[s, s'] = P(next state = s' | current state = s, action = keep)

        P_replace : ndarray, shape (n_s, n_s)
            P_replace[s, s'] = P(next state = s' | current state = s, action = replace)
        """
        n_s = self.n_s
        p = self.trans_probs  # (p0, p1, p2)

        # --- Keep transition matrix ---
        P_keep = np.zeros((n_s, n_s))
        for s in range(n_s):
            for j in range(3):  # j = 0, 1, 2 (mileage increment)
                # Next state: s+j, but capped at the highest bin
                s_next = min(s + j, n_s - 1)
                # We use += because at the boundary, multiple increments
                # can lead to the same state (e.g., s=29, j=1 and j=2
                # both go to 29, so P[29,29] = p0 + p1 + p2 = 1).
                P_keep[s, s_next] += p[j]

        # --- Replace transition matrix ---
        # After replacement, mileage resets to 0 and then evolves.
        # So the transition from ANY state s is the same as from state 0:
        #   P_replace[s, :] = P_keep[0, :]   for all s
        P_replace = np.zeros((n_s, n_s))
        for s in range(n_s):
            P_replace[s, :] = P_keep[0, :]

        return P_keep, P_replace

    # ===================================================================
    # STEP 1: Flow payoffs
    # ===================================================================

    def flow_payoff(self, theta1, RC):
        """
        Compute the flow payoff (immediate reward) for every (state, action).

        This is the "one-period profit" before considering the future.

        KEEP (a=0): you pay maintenance cost proportional to mileage.
            u(s, 0) = -theta1 * s
            At mileage 0: cost is 0 (brand new engine, no maintenance).
            At mileage 15: cost is -15*theta1.
            At mileage 29: cost is -29*theta1 (very expensive!).

        REPLACE (a=1): you pay a flat replacement cost regardless of mileage.
            u(s, 1) = -RC
            Same cost whether mileage is 5 or 25.

        SEPARABILITY: theta1 ONLY appears in the keep payoff.
                      RC ONLY appears in the replace payoff.
        This is a key structural feature that simplifies the MIQP
        (we can select theta1 and RC independently).

        Parameters
        ----------
        theta1 : float
            How fast maintenance costs grow with mileage. 
        RC : float
            Cost of replacing the engine. 

        Returns
        -------
        u : ndarray, shape (n_s, 2)
            u[s, a] = flow payoff at state s when taking action a.
        """
        u = np.zeros((self.n_s, 2))
        u[:, 0] = -theta1 * np.arange(self.n_s)  # keep payoffs
        u[:, 1] = -RC                              # replace payoffs
        return u

    # ===================================================================
    # STEP 2: Solve the Bellman equation
    # ===================================================================

    def solve_bellman(self, theta1, RC, tol=1e-12, max_iter=100_000, verbose=False):
        """
        Find the optimal value function and policy by value function iteration.

        The Bellman equation says: the value of being at mileage s is
        the BEST of keeping or replacing:

            V(s) = max{ -theta1*s + beta * E[V(s') | s, keep],
                         -RC      + beta * E[V(s') | s, replace] }

        We solve this by iterating: start with V=0 everywhere, plug into
        the right side, get a new V, plug that in, repeat until V stops
        changing. This is guaranteed to converge (contraction mapping).

        Parameters
        ----------
        theta1, RC : float
            The structural parameters.
        tol : float
            Stop when the biggest change in V is less than tol.
            Smaller = more accurate but slower. 1e-12 is very precise.
        max_iter : int
            Safety limit on iterations (should never be reached).

        Returns
        -------
        V : ndarray, shape (n_s,)
            V[s] = the discounted lifetime value of being at mileage s.
        policy : ndarray of int, shape (n_s,)
            policy[s] = 0 (keep) or 1 (replace) = the optimal action at s.
        """
        u = self.flow_payoff(theta1, RC)
        V = np.zeros(self.n_s)  # initial guess: all zeros

        for iter in range(max_iter):
            # Compute the "Q-value" for each action at each state.
            # Q(s, a) = immediate payoff + discounted expected future value
            #
            # self.P_keep @ V  is a vector where element s =
            #   sum_{s'} P_keep[s, s'] * V[s'] = E[V(s') | state=s, keep]
            Q_keep = u[:, 0] + self.beta * (self.P_keep @ V)
            Q_replace = u[:, 1] + self.beta * (self.P_replace @ V)

            # New value function: take the better action at each state
            V_new = np.maximum(Q_keep, Q_replace)

            # Compute error
            err = np.max(np.abs(V_new - V))

            # Verbose,
            if verbose:
                print(f"Iteration {iter} with error {round(err, 4)}")

            # Check convergence: has V stopped changing?
            if err < tol:
                V = V_new
                break

            V = V_new

        # Extract the optimal policy from the converged value function.
        # At each state, which action gives a higher Q-value?
        # We recompute Q-values to be precise with the final V.
        Q_keep = u[:, 0] + self.beta * (self.P_keep @ V)
        Q_replace = u[:, 1] + self.beta * (self.P_replace @ V)

        # policy[s] = 1 if replacing is STRICTLY better than keeping.
        # Ties go to keep (a=0). This is a convention; it doesn't matter
        # for the results since ties happen at measure zero.
        policy = (Q_replace > Q_keep).astype(int)

        return V, policy

    # ===================================================================
    # STEP 3: Compute the occupancy measure
    # ===================================================================

    def compute_occupancy_measure(self, policy):
        """
        Compute the occupancy measure lambda(s, a) for a given policy.

        WHAT IS AN OCCUPANCY MEASURE?
        Imagine running the model forever. At each period, Zurcher is at
        some mileage s and takes an action a. The occupancy measure
        lambda(s, a) is the DISCOUNTED frequency of visiting (s, a):

            lambda(s, a) = sum_{t=0}^{infinity} beta^t * Prob(s_t = s, a_t = a)

        Think of it as a "heat map" over state-action pairs, where earlier
        visits count more (because of discounting).

        WHY DO WE NEED IT?
        The occupancy measure lets us compute model-implied moments
        WITHOUT simulation. Instead of simulating millions of periods
        and computing sample averages, we can compute expected values
        analytically using lambda as weights.

        HOW TO COMPUTE IT?
        For a deterministic policy (like the optimal one), only
        lambda(s, policy[s]) can be positive. Define d(s) = lambda(s, policy[s]).
        Then d satisfies the flow balance equation:

            d(s) - beta * sum_{s'} P(s | s', policy[s']) * d(s') = mu(s)

        In matrix form: (I - beta * P_pi^T) * d = mu

        This is a system of n_s linear equations in n_s unknowns.

        Parameters
        ----------
        policy : ndarray of int, shape (n_s,)
            The deterministic policy. policy[s] in {0, 1}.

        Returns
        -------
        lam : ndarray, shape (n_s, 2)
            The occupancy measure.
            lam[s, 0] = discounted frequency of (state=s, action=keep).
            lam[s, 1] = discounted frequency of (state=s, action=replace).
            lam[s, a] = 0 whenever a != policy[s].
        """
        # Build the transition matrix UNDER THE POLICY.
        # P_pi[s, s'] = probability of going from s to s' when following policy.
        # If policy[s]=0 (keep): P_pi[s,:] = P_keep[s,:]
        # If policy[s]=1 (replace): P_pi[s,:] = P_replace[s,:]
        P_pi = np.zeros((self.n_s, self.n_s))
        for s in range(self.n_s):
            P_pi[s, :] = self.P[policy[s]][s, :]

        # Solve the flow balance equation: (I - beta * P_pi^T) * d = mu
        #
        # Why P_pi^T (transpose)?
        #   P_pi[s', s] = prob of going FROM s' TO s under the policy
        #   P_pi^T[s, s'] = P_pi[s', s] = prob of arriving AT s FROM s'
        #   The transpose switches "where I came from" to "where I go to".
        #
        # The flow balance says: the occupancy at state s (left side)
        # minus the discounted inflow (middle term) equals the initial
        # mass at s (right side). This is a conservation law.
        A = np.eye(self.n_s) - self.beta * P_pi.T
        d = solve(A, self.mu)

        # Build the full (n_s x 2) occupancy measure.
        # Under a deterministic policy, only one action is ever taken
        # at each state, so one column is zero.
        lam = np.zeros((self.n_s, 2))
        for s in range(self.n_s):
            lam[s, policy[s]] = d[s]

        return lam

    # ===================================================================
    # STEP 4: Compute model-implied moments
    # ===================================================================

    def compute_moments(self, lam):
        """
        Compute model-implied moments analytically from an occupancy measure.

        A "moment" is a summary statistic that we can compute both from
        data and from the model. The SMM estimator finds parameters that
        make the model moments match the data moments.

        CRITICAL PROPERTY: all five moments are LINEAR in lambda.
        This means the MIQP objective (which is quadratic in the moments)
        is quadratic in lambda — exactly the structure Branch-and-Bound
        can handle.

        Why "no McCormick variables needed":
        The moment functions h(s,a) below depend only on (s,a), NOT on
        theta. So the model-implied moment is just a weighted sum of
        lambda with known coefficients. If h depended on theta, we'd
        have bilinear terms (theta * lambda) that need extra tricks.

        Parameters
        ----------
        lam : ndarray, shape (n_s, 2)
            An occupancy measure (from compute_occupancy_measure).

        Returns
        -------
        moments : ndarray, shape (5,)
            moments[0] = replacement frequency
            moments[1] = average mileage
            moments[2] = mileage-replacement cross-moment
            moments[3] = second moment of mileage
            moments[4] = second moment of mileage at replacement
        """
        # nu_inv = 1/nu_bar normalizes the occupancy measure into
        # a proper probability distribution (sums to 1).
        nu_inv = 1.0 / self.nu_bar

        # s_vals[s] = s (just the numbers 0, 1, 2, ..., n_s-1)
        s_vals = np.arange(self.n_s)

        # s_sq[s] = s^2 (squared mileage: 0, 1, 4, 9, 16, ...)
        s_sq = s_vals ** 2

        # ----- Moment 1: Replacement frequency -----
        # h_1(s, a) = 1{a = 1}
        # "How often does Onur replace the engine?"
        # Data analog: fraction of bus-month observations with a replacement.
        # Identifies RC: higher RC -> fewer replacements -> lower m1.
        m1 = nu_inv * np.sum(lam[:, 1])

        # ----- Moment 2: Average mileage -----
        # h_2(s, a) = s
        # "What is the average mileage across all observations?"
        # Data analog: mean mileage across all bus-months.
        # Identifies theta1: higher theta1 -> replace sooner -> lower mileage.
        m2 = nu_inv * np.sum(s_vals * (lam[:, 0] + lam[:, 1]))

        # ----- Moment 3: Mileage-replacement cross-moment -----
        # h_3(s, a) = s * 1{a = 1}
        # "Mileage times the replacement indicator."
        # NOT the same as "average mileage at replacement" (that would be
        # a ratio m3/m1, which is nonlinear). This unconditional version
        # is LINEAR in lambda, which is essential for the MIQP.
        # Identifies both parameters jointly.
        m3 = nu_inv * np.sum(s_vals * lam[:, 1])

        # ----- Moment 4: Second moment of mileage -----
        # h_4(s, a) = s^2
        # "What is the average SQUARED mileage?"
        # Combined with m2, this pins down the VARIANCE of mileage:
        #   Var(s) = E[s^2] - (E[s])^2 = m4 - m2^2
        # Policies with the same mean mileage but different thresholds
        # produce different variances. A threshold at s*=5 concentrates
        # mass on bins 0-5 (low variance), while s*=20 spreads mass
        # over 0-20 (high variance). This helps separate parameter
        # vectors that moments 1-3 cannot distinguish.
        m4 = nu_inv * np.sum(s_sq * (lam[:, 0] + lam[:, 1]))

        # ----- Moment 5: Second moment of mileage at replacement -----
        # h_5(s, a) = s^2 * 1{a = 1}
        # "Squared mileage times the replacement indicator."
        # Captures whether replacements happen at a narrow range of
        # mileages (low m5) or spread across many mileage bins (high m5).
        # Under a sharp threshold policy, replacements cluster at the
        # threshold and the few states just above it.
        m5 = nu_inv * np.sum(s_sq * lam[:, 1])

        return np.array([m1, m2, m3, m4, m5])

    # ===================================================================
    # STEP 5: Evaluate the SMM objective at a single parameter point
    # ===================================================================

    def smm_objective(self, theta1, RC, data_moments, W):
        """
        Evaluate the SMM criterion Q(theta) at a single parameter vector.

        Q(theta) = [m_hat - m_tilde(theta)]' W [m_hat - m_tilde(theta)]

        This is the "distance" between data moments and model moments.
        The estimator finds theta that MINIMIZES this distance.

        This function performs the FULL INNER LOOP:
            1. Solve Bellman equation at (theta1, RC)  -> V, policy
            2. Compute occupancy measure from policy   -> lambda
            3. Compute model-implied moments           -> m_tilde
            4. Evaluate quadratic distance             -> Q

        In traditional SMM, this function is called THOUSANDS of times
        (once for each candidate parameter the optimizer tries).

        Parameters
        ----------
        theta1, RC : float
            The candidate parameter values to evaluate.
        data_moments : ndarray, shape (5,)
            The empirical moments computed from observed data.
        W : ndarray, shape (5, 5)
            Weighting matrix. Positive semi-definite.
            Identity matrix (W = I) is a common simple choice.
            Optimal W is the inverse of the moment covariance matrix.

        Returns
        -------
        Q : float
            The SMM criterion value. Lower is better.
        """
        # Inner loop: solve the model at this parameter guess
        V, policy = self.solve_bellman(theta1, RC)
        lam = self.compute_occupancy_measure(policy)
        model_moments = self.compute_moments(lam)

        # Compute the moment gap
        e = data_moments - model_moments  # shape (5,)

        # Quadratic form: e' W e = sum_{l,l'} W[l,l'] * e[l] * e[l']
        # This is a scalar (a single number).
        return float(e @ W @ e)

    # ===================================================================
    # STEP 6: Simulate panel data
    # ===================================================================

    def simulate_panel(self, theta1, RC, N=1000, T=500, seed=None):
        """
        Simulate a panel dataset of N buses observed for T periods.

        This is how we generate "fake data" for Monte Carlo experiments.
        In a real application, you'd have actual bus maintenance records.

        For each bus n = 1, ..., N:
            1. Draw initial mileage from mu (uniform over all bins)
            2. For t = 1, ..., T:
                a. Observe current mileage s_nt
                b. Apply optimal policy: a_nt = policy[s_nt]
                c. Draw next mileage from P(.|s_nt, a_nt)

        Parameters
        ----------
        theta1, RC : float
            The TRUE parameters that generate the data.
        N : int
            Number of buses in the panel.
        T : int
            Number of time periods each bus is observed.
        seed : int or None
            Random seed for reproducibility. Same seed = same data.

        Returns
        -------
        states : ndarray of int, shape (N, T)
            states[n, t] = mileage bin of bus n at time t.
        actions : ndarray of int, shape (N, T)
            actions[n, t] = action taken (0=keep, 1=replace) for bus n at time t.
        """
        # Create a random number generator with the given seed.
        rng = np.random.default_rng(seed)

        # Solve the model to get the optimal policy
        V, policy = self.solve_bellman(theta1, RC)

        # Allocate output arrays
        states = np.zeros((N, T), dtype=int)
        actions = np.zeros((N, T), dtype=int)

        # Precompute cumulative distribution functions (CDFs) for
        # fast random sampling. cdf[s, a, :] is the CDF of the
        # transition distribution from (s, a).
        #
        # Example: if P(s'|s,a) = [0.4, 0.5, 0.1, 0, ...]
        # then CDF = [0.4, 0.9, 1.0, 1.0, ...]
        # To sample: draw u ~ Uniform(0,1), find smallest s' with CDF[s'] >= u.
        cdf = np.zeros((self.n_s, 2, self.n_s))
        for s in range(self.n_s):
            for a in range(2):
                cdf[s, a, :] = np.cumsum(self.P[a][s, :])

        # Draw initial states for all buses at once
        states[:, 0] = rng.choice(self.n_s, size=N, p=self.mu)

        # Simulate forward
        for t in range(T):
            # Assign actions according to the optimal policy.
            # This is vectorized: policy[states[:, t]] looks up the
            # optimal action for each bus simultaneously.
            actions[:, t] = policy[states[:, t]]

            # Transition to next period (skip at the last period)
            if t < T - 1:
                s_arr = states[:, t]    # current states, shape (N,)
                a_arr = actions[:, t]   # current actions, shape (N,)

                # Vectorized CDF inversion sampling:
                # 1. Draw uniform random numbers for all buses
                # 2. Look up each bus's CDF
                # 3. Find where the uniform falls in the CDF
                u = rng.random(N).reshape(-1, 1)     # shape (N, 1)
                bus_cdfs = cdf[s_arr, a_arr, :]       # shape (N, n_s)

                # Count how many CDF values are below u.
                # This gives the next state index.
                states[:, t + 1] = (bus_cdfs < u).sum(axis=1)

        return states, actions

    def compute_data_moments(self, states, actions):
        """
        Compute empirical moments from panel data using DISCOUNTED weights.

        These are sample analogs of the population moments computed
        by compute_moments(). The SMM estimator finds parameters
        that make the model moments close to these data moments.

        IMPORTANT: We use discounted weights (1-beta)*beta^t to match
        the discounted occupancy measure used in compute_moments().
        Simple (1/NT) averages would converge to ERGODIC moments,
        which differ from the discounted population moments whenever
        the initial distribution mu is not the stationary distribution.

        m_hat_l = (1/N) * sum_n sum_t (1-beta)*beta^t * h_l(s_nt, a_nt)

        For large T, the truncation error is negligible since beta^T ≈ 0.

        Parameters
        ----------
        states : ndarray of int, shape (N, T)
            From simulate_panel().
        actions : ndarray of int, shape (N, T)
            From simulate_panel().

        Returns
        -------
        moments : ndarray, shape (5,)
            The empirical moment vector.
        """
        N, T = states.shape

        # Build the discounted weight for each period:
        #   w[t] = (1 - beta) * beta^t
        #
        # These sum to approximately 1 for large T:
        #   sum_{t=0}^{T-1} (1-beta)*beta^t = 1 - beta^T ≈ 1
        #
        # The (1-beta) factor normalizes so that weights sum to 1,
        # making the discounted moment a proper weighted average.
        # This matches nu_bar^{-1} * lambda in compute_moments().
        w = (1.0 - self.beta) * (self.beta ** np.arange(T))  # shape (T,)

        # Broadcast weights across all buses: shape (1, T) * (N, T) = (N, T)
        # Each bus gets the same time weights.
        weighted_replace = (actions == 1).astype(float) * w[np.newaxis, :]
        weighted_states = states.astype(float) * w[np.newaxis, :]
        weighted_all = w[np.newaxis, :] * np.ones((N, T))
        states_sq = states.astype(float) ** 2
        weighted_states_sq = states_sq * w[np.newaxis, :]

        # Average across buses (each bus is an independent draw from mu)
        # m_hat = (1/N) * sum_n [sum_t w[t] * h(s_nt, a_nt)]
        m1 = np.sum(weighted_replace) / N                          # replace freq
        m2 = np.sum(weighted_states) / N                            # avg mileage
        m3 = np.sum(states.astype(float) * weighted_replace) / N   # mileage x replace
        m4 = np.sum(weighted_states_sq) / N                         # mileage^2
        m5 = np.sum(states_sq * weighted_replace) / N               # mileage^2 x replace

        return np.array([m1, m2, m3, m4, m5])