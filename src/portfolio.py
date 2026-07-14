import numpy as np
import pyomo.environ as pyo
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt

from .validator import PortfolioValidator
from .plot_utils import plot_heatmap


class Portfolio:
    def __init__(self, covariance_matrix, return_array=None, asset_list=None, check_psd=True):
        PortfolioValidator.validate_covariance_matrix(covariance_matrix, check_psd=check_psd)
        self.covariance_matrix = covariance_matrix
        self.num_assets = covariance_matrix.shape[0]

        self.returns = PortfolioValidator.validate_return_array(return_array, self.num_assets)
        self.assets = PortfolioValidator.validate_asset_list(asset_list, self.num_assets)

        self.positions = None
        self.variance = None
        self.portfolio_return = None

        self.position_horizon = None
        self.actual_horizon = []
        self.estimate_horizon = []
        self.return_horizon = []

        self.sigma, self.omega = None, None
        self.calculate_sigma_and_omega()

    def calculate_portfolio_positions(self):
        raise NotImplementedError("Implement in subclass")

    def calculate_portfolio_variance(self):
        self._check_positions()
        var_calc = self.positions.T @ self.covariance_matrix @ self.positions
        self.variance = var_calc[0][0]
        return self.variance

    def calculate_portfolio_return(self):
        self._check_positions()
        exp_calc = self.returns.T @ self.positions
        self.portfolio_return = exp_calc[0][0]
        return self.portfolio_return

    def calculate_sigma_and_omega(self):
        # sigma = per-asset volatility, omega = correlation matrix. Used
        # throughout this module (e.g. theta = 1/sigma in the closed-form
        # solvers below).
        self.sigma = np.sqrt(np.diag(self.covariance_matrix))
        self.omega = self.covariance_matrix / np.outer(self.sigma, self.sigma)
        return self.sigma, self.omega

    def reset_horizons(self):
        self.position_horizon = np.ones((self.num_assets, 1))
        self.actual_horizon = []
        self.estimate_horizon = []
        self.return_horizon = []

    def _check_positions(self):
        assert self.positions is not None, 'Positions have not been calculated'
        assert isinstance(self.positions, np.ndarray), 'Positions array should be a numpy array'
        self.positions = self.positions.reshape(-1, 1)


class EqualWeightPortfolio(Portfolio):
    def calculate_portfolio_positions(self):
        self.positions = np.ones(self.num_assets) / self.num_assets
        return self.positions


class LongOnlyMarkowitz(Portfolio):
    """Long-only global minimum-variance portfolio.

        minimize    x' V x
        subject to  sum(x) = 1, x >= 0

    Default solver is `'pounce'`: an in-process QP solve via
    `pounce.solve_qp` (https://github.com/jkitchin/pounce), with no external
    solver executable or license required. Any other value is treated as a
    Pyomo solver name (e.g. `'ipopt'`, `'gurobi_direct'`) and solved via a
    Pyomo NLP/QP model instead.
    """

    def __init__(self, covariance_matrix, return_array=None, asset_list=None, solver='pounce', tee=False):
        super().__init__(covariance_matrix, return_array, asset_list)
        self.solver = solver
        self.tee = tee
        self.m = None

    def _solve_pounce(self):
        # Imported lazily so importing this module doesn't require pounce to
        # be installed unless solver='pounce' is actually used.
        import pounce

        n = self.num_assets
        # pounce's convention is minimize 0.5*x'Px + c'x, so P = 2V matches
        # our objective x'Vx exactly (c=0).
        result = pounce.solve_qp(
            P=2.0 * self.covariance_matrix,
            c=np.zeros(n),
            A=np.ones((1, n)),
            b=[1.0],
            lb=np.zeros(n),
        )
        assert result.status == 'optimal', f"pounce did not converge: {result.status}"
        self.positions = np.asarray(result.x).reshape(-1, 1)
        return self.positions

    def _create_optimization_model(self):
        self.m = pyo.ConcreteModel()
        self.m.I = pyo.Set(initialize=list(range(self.num_assets)))
        self.m.x = pyo.Var(self.m.I, within=pyo.NonNegativeReals, initialize=1 / self.num_assets)

        V = {index: value for index, value in np.ndenumerate(self.covariance_matrix)}
        self.m.V = pyo.Param(self.m.I, self.m.I, initialize=V, mutable=True)

        @self.m.Objective(sense=pyo.minimize)
        def obj(m):
            return sum(m.x[i] * m.x[j] * m.V[i, j] for i in m.I for j in m.I)

        @self.m.Constraint()
        def sum_to_one(m):
            return sum(m.x[i] for i in m.I) == 1

    def calculate_portfolio_positions(self):
        if self.solver == 'pounce':
            return self._solve_pounce()

        if self.m is None:
            self._create_optimization_model()
        else:
            for (i, j), value in np.ndenumerate(self.covariance_matrix):
                self.m.V[i, j]._value = value

        solver_object = pyo.SolverFactory(self.solver)
        if self.solver == 'gurobi_direct':
            solver_object.options['Threads'] = 8
        results = solver_object.solve(self.m, tee=self.tee)
        assert results.solver.termination_condition in ['optimal', 'locallyOptimal']

        x_sol = self.m.x.extract_values()
        self.positions = np.array([x_sol[i] for i in sorted(x_sol.keys())]).reshape(-1, 1)
        return self.positions


class IgnoreAllCorrelation(Portfolio):
    def calculate_portfolio_positions(self):
        ones = np.ones((self.num_assets, 1))
        diag_V_inv = np.diag(1 / np.diag(self.covariance_matrix))
        self.positions = diag_V_inv @ ones / (ones.T @ diag_V_inv @ ones)
        return self.positions


class ConstantCorrelationBlock(Portfolio):
    """Single-block constant-correlation min-variance portfolio (Proposition 1).

    Estimates rho by averaging the off-diagonal entries of the correlation
    matrix, then computes the long-only minimum-variance portfolio via the
    sorting-based closed-form solution.
    """

    def __init__(self, covariance_matrix, return_array=None, asset_list=None):
        super().__init__(covariance_matrix, return_array, asset_list, check_psd=False)
        if self.num_assets > 1:
            non_diag = ~np.eye(self.num_assets, dtype=bool)
            self.mean_rho = float(self.omega[non_diag].mean())
        else:
            self.mean_rho = 1.0

    def _compute_threshold_via_sorting(self):
        """Return bar_theta per Proposition 1 (eq. fixed.point.sol).

        Sort theta = 1/sigma in decreasing order and find the largest i such
        that bar_theta_i = rho/(1+(i-1)*rho) * sum_{j<=i} theta_j < theta_i.
        The condition is monotone so we stop at the first failure.
        """
        rho = self.mean_rho
        theta = 1.0 / self.sigma
        theta_sorted = theta[np.argsort(theta)[::-1]]

        best_i, theta_sum = 1, 0.0
        for i in range(1, self.num_assets + 1):
            theta_sum += theta_sorted[i - 1]
            bar_theta_i = rho * theta_sum / (1.0 + (i - 1) * rho)
            if bar_theta_i <= theta_sorted[i - 1]:
                best_i = i
            else:
                break

        return rho * np.sum(theta_sorted[:best_i]) / (1.0 + (best_i - 1) * rho)

    def calculate_portfolio_positions(self):
        if self.num_assets == 1:
            self.positions = np.ones((1, 1))
            return self.positions

        bar_theta = self._compute_threshold_via_sorting()
        theta = 1.0 / self.sigma

        # y = (1/(1-rho)) * (theta - bar_theta)^+  (eq. sol.y); 1/(1-rho) cancels on normalisation
        y = np.maximum(theta - bar_theta, 0.0)
        # x_i = y_i / sigma_i  (back-transform from correlation-normalised space)
        x = y / self.sigma
        x /= x.sum()

        self.positions = x.reshape(-1, 1)
        return self.positions


class CorrelationAwareBlock(ConstantCorrelationBlock):
    """Correlation-aware robustification of ConstantCorrelationBlock.

    After computing the fixed-point threshold bar_theta via sorting, projects
    it onto the robust-feasible set F (Algorithm 1) so that it stays at least
    epsilon away from every breakpoint theta_i = 1/sigma_i.

    Parameters
    ----------
    epsilon : float or None
        Robustness margin. None uses the heuristic epsilon = max(theta) / n.
    """

    def __init__(self, covariance_matrix, epsilon=None, return_array=None, asset_list=None,
                 plot_threshold=False, plot_title=None):
        super().__init__(covariance_matrix, return_array, asset_list)
        self.epsilon = epsilon
        self.plot_threshold = plot_threshold
        self._plot_title = plot_title

    def _robustify_threshold(self, bar_theta):
        """Project bar_theta onto the robust-feasible set (Algorithm 1)."""
        theta = 1.0 / self.sigma
        eps = self.epsilon if self.epsilon is not None else np.max(theta) / self.num_assets

        bp = np.sort(np.unique(theta))  # breakpoints theta_1 <= ... <= theta_n

        # Feasible regions:
        #   (-inf, bp[0] - eps]                          always present (theta_0 = -inf)
        #   [bp[j] + eps, bp[j+1] - eps]  for each interior gap >= 2*eps
        feasible = [(-np.inf, bp[0] - eps)]
        for j in range(len(bp) - 1):
            if bp[j + 1] - bp[j] >= 2.0 * eps:
                feasible.append((bp[j] + eps, bp[j + 1] - eps))

        # Already feasible?
        for lo, hi in feasible:
            if (lo == -np.inf or bar_theta >= lo) and bar_theta <= hi:
                return bar_theta

        # Project onto closest feasible point
        best, best_dist = bar_theta, np.inf
        for lo, hi in feasible:
            candidate = min(bar_theta, hi) if lo == -np.inf else np.clip(bar_theta, lo, hi)
            dist = abs(candidate - bar_theta)
            if dist < best_dist:
                best_dist, best = dist, candidate

        return best

    def _plot_threshold(self, bar_theta, theta_eps, title=None):
        theta = 1.0 / self.sigma
        plt.figure()
        if title:
            plt.title(title)
        plt.scatter(self.assets, theta)
        plt.axhline(y=bar_theta, color='red', label=r'Original Threshold $\bar{\theta}$')
        plt.axhline(y=theta_eps, color='green', linestyle='--',
                    label=r'Robust Threshold $\theta_{\epsilon}$')
        plt.xlabel('Asset')
        plt.ylabel(r'$\theta = 1/\sigma$')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    def calculate_portfolio_positions(self):
        if self.num_assets == 1:
            self.positions = np.ones((1, 1))
            return self.positions

        bar_theta = self._compute_threshold_via_sorting()
        theta_eps = self._robustify_threshold(bar_theta)
        theta = 1.0 / self.sigma

        if self.plot_threshold:
            self._plot_threshold(bar_theta, theta_eps, title=self._plot_title)

        y = np.maximum(theta - theta_eps, 0.0)
        x = y / self.sigma
        x /= x.sum()

        self.positions = x.reshape(-1, 1)
        return self.positions


class BlockDiagonalPortfolio(Portfolio):
    """Min-variance portfolio via block-diagonal approximation.

    Clusters assets with hierarchical clustering (default: single-linkage), approximates
    the correlation matrix as block-diagonal (Section sec:block.approx), then
    solves the min-variance problem using one of three methods:

    'direct'   — Proposition K.block: solves the coupled fixed-point system
                 (eq.fixed.point.block) exactly via bisection on the global
                 threshold theta_hat.
    'aware'    — Two-tier scheme: CorrelationAwareBlock within each cluster
                 (Tier 1) and across clusters (Tier 2).
    'oblivious'— Two-tier scheme: IgnoreAllCorrelation at both tiers.

    Parameters
    ----------
    method : {'direct', 'oblivious', 'aware'}
        Sets both tiers for the two-tier scheme.  Ignored when tier1_method /
        tier2_method are provided explicitly.  Defaults to 'aware'.
    tier1_method : {'oblivious', 'aware'} or None
        Within-cluster method for the two-tier scheme.  Must not be 'direct'.
    tier2_method : {'oblivious', 'aware'} or None
        Cross-cluster method for the two-tier scheme.  Must not be 'direct'.
    clustering_parameter : float or None
        Distance threshold for cutting the dendrogram.  None triggers the
        automatic largest-gap rule.
    epsilon : float or None
        Robustness margin for 'aware' tiers (ignored for 'direct').
        None uses the per-block heuristic max(theta)/n.
    use_approx_cov : bool
        Two-tier only.  True — block variances use the approximated
        constant-rho_i covariance; False — use the original submatrix.
    plot_clustering : bool
        Display a dendrogram and clustered covariance heatmap after clustering.
    """

    _VALID      = ('oblivious', 'aware', 'direct')
    _VALID_TIER = ('oblivious', 'aware')

    def __init__(self, covariance_matrix, method='aware', tier1_method=None,
                 tier2_method=None, clustering_parameter=None, epsilon=None,
                 use_approx_cov=True, return_array=None, asset_list=None,
                 clustering_method='single', plot_clustering=False):
        super().__init__(covariance_matrix, return_array, asset_list)
        assert method in self._VALID, f"method must be one of {self._VALID}"
        self.method = method
        if method == 'direct':
            self.tier1_method = None
            self.tier2_method = None
        else:
            self.tier1_method = tier1_method if tier1_method is not None else method
            self.tier2_method = tier2_method if tier2_method is not None else method
            assert self.tier1_method in self._VALID_TIER, \
                f"tier1_method must be one of {self._VALID_TIER}"
            assert self.tier2_method in self._VALID_TIER, \
                f"tier2_method must be one of {self._VALID_TIER}"
        self.clustering_parameter = clustering_parameter
        self.epsilon = epsilon
        self.use_approx_cov = use_approx_cov
        assert clustering_method in ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'], 'Select a valid clustering_method'
        self.clustering_method = clustering_method
        self.plot_clustering = plot_clustering
        self.clusters = None
        self._linkage_matrix = None
        self._threshold = None
        # Set by _calculate_direct; available for inspection / testing
        self._theta_hat = None
        self._bar_thetas = None
        # Set by calculate_portfolio_positions; used by _plot_clusters
        self._global_rho = None
        self._block_info = None

    # ------------------------------------------------------------------ #
    #  Clustering                                                          #
    # ------------------------------------------------------------------ #

    def _cluster(self):
        d = np.sqrt(2.0 * np.clip(1.0 - self.omega, 0.0, None))
        np.fill_diagonal(d, 0.0)
        Z = sch.linkage(ssd.squareform(d, checks=False), method=self.clustering_method)
        self._linkage_matrix = Z

        if self.clustering_parameter is None:
            link_dists = Z[:, 2]
            k = int(np.argmax(np.diff(link_dists)))
            threshold = 0.5 * (link_dists[k] + link_dists[k + 1]) + 1e-10
        else:
            threshold = float(self.clustering_parameter)

        self._threshold = threshold
        self.clusters = sch.fcluster(Z, threshold, criterion='distance')

    # ------------------------------------------------------------------ #
    #  Block-diagonal approximation (Section sec:block.approx)            #
    # ------------------------------------------------------------------ #

    def _block_approximation(self):
        """Return global_rho and per-cluster info dict.

        global_rho : mean of all off-block correlation entries.
        block_info : {cluster_id: {'indices': array, 'rho_i': float}}
            rho_i = (tilde_rho_i - global_rho) / (1 - global_rho)
        """
        n = self.num_assets
        unique_clusters = sorted(np.unique(self.clusters))

        in_block = np.zeros((n, n), dtype=bool)
        for c in unique_clusters:
            idx = np.where(self.clusters == c)[0]
            in_block[np.ix_(idx, idx)] = True
        off_block = ~in_block
        np.fill_diagonal(off_block, False)

        global_rho = float(self.omega[off_block].mean()) if off_block.any() else 0.0

        block_info = {}
        for c in unique_clusters:
            idx = np.where(self.clusters == c)[0]
            n_i = len(idx)
            if n_i > 1:
                omega_b = self.omega[np.ix_(idx, idx)]
                off_diag = ~np.eye(n_i, dtype=bool)
                tilde_rho_i = float(omega_b[off_diag].mean())
                denom = 1.0 - global_rho
                rho_i = (tilde_rho_i - global_rho) / denom if denom > 1e-12 else 0.0
            else:
                rho_i = 0.0
            block_info[c] = {'indices': idx, 'rho_i': rho_i}

        return global_rho, block_info

    # ------------------------------------------------------------------ #
    #  Direct solver (Proposition K.block)                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _single_block_bar_theta(phi, rho):
        """Fixed-point threshold for one block on shifted thetas phi = theta - theta_hat.

        Solves: bar = rho/(1-rho) * sum_j (phi_j - bar)^+
        using the sorting algorithm.  phi may contain negative entries
        (assets with phi_j <= 0 are always inactive).
        """
        if rho == 0.0:
            return 0.0
        phi_sorted = np.sort(phi)[::-1]
        if phi_sorted[0] <= 0:
            return 0.0
        best_i, phi_sum = 0, 0.0
        for i in range(1, len(phi) + 1):
            v = phi_sorted[i - 1]
            if v <= 0:
                break
            phi_sum += v
            bar = rho * phi_sum / (1.0 + (i - 1) * rho)
            if bar <= v:
                best_i = i
            else:
                break
        if best_i == 0:
            return 0.0
        return rho * np.sum(phi_sorted[:best_i]) / (1.0 + (best_i - 1) * rho)

    def _direct_f(self, theta_hat, global_rho, block_info):
        """Residual of the cross-block fixed-point (eq.fixed.point.block):

          hat_theta = rho/(1-rho) * sum_i 1/(1-rho_i) * 1_i'*(phi_i - bar_i*1_i)^+

        Algebraically equals rho/(1-rho)*sum_i bar_i/rho_i, but written in the
        same form as the single-block equation.  F is strictly increasing,
        F(0) <= 0, F(max theta) > 0, so there is always a unique root.
        """
        total = 0.0
        for _, info in sorted(block_info.items()):
            idx   = info['indices']
            rho_i = info['rho_i']
            phi_i = 1.0 / self.sigma[idx] - theta_hat
            bar_i = self._single_block_bar_theta(phi_i, rho_i)
            excess_i = float(np.sum(np.maximum(phi_i - bar_i, 0.0)))
            total += excess_i / (1.0 - rho_i)
        return theta_hat - global_rho / (1.0 - global_rho) * total

    def _calculate_direct(self, global_rho, block_info):
        """Solve the block fixed-point system (eq.fixed.point.block) via bisection.

        Finds theta_hat such that F(theta_hat) = 0, then computes per-block
        bar_theta_i and final portfolio weights.

        Weights: x_j proportional to (theta_j - theta_hat - bar_i)^+ / ((1-rho_i)*sigma_j)
        for asset j in block i.  The (1-rho_i) factor is required for cross-block
        allocation; it can be shown that sigma_i^2 = (1-rho_i) / C_i where
        C_i = sum_j (theta_j - combined_i)^+ * theta_j, so this formula is
        algebraically identical to the two-tier inverse-variance weighting on
        true block-diagonal matrices.
        """
        if global_rho < 1e-15:
            theta_hat = 0.0
        else:
            lo, hi = 0.0, float(np.max(1.0 / self.sigma))
            tol = 1e-12
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                if self._direct_f(mid, global_rho, block_info) < 0:
                    lo = mid
                else:
                    hi = mid
            theta_hat = 0.5 * (lo + hi)

        self._theta_hat = theta_hat
        self._bar_thetas = {}
        x = np.zeros(self.num_assets)

        for c, info in sorted(block_info.items()):
            idx   = info['indices']
            rho_i = info['rho_i']
            theta_i = 1.0 / self.sigma[idx]
            phi_i   = theta_i - theta_hat
            bar_i   = self._single_block_bar_theta(phi_i, rho_i)

            self._bar_thetas[c] = bar_i
            excess = np.maximum(theta_i - theta_hat - bar_i, 0.0)
            x[idx] = excess / ((1.0 - rho_i) * self.sigma[idx])

        total = x.sum()
        if total < 1e-15:
            x = np.ones(self.num_assets) / self.num_assets
        else:
            x /= total

        self.positions = x.reshape(-1, 1)

        if self.plot_clustering:
            self._plot_direct_thresholds()

        return self.positions

    # ------------------------------------------------------------------ #
    #  Direct threshold visualisation                                     #
    # ------------------------------------------------------------------ #

    def _plot_direct_thresholds(self, title=None):
        """Plot per-asset thetas grouped by block with theta_hat and theta_hat + bar_i."""
        _, ax = plt.subplots(figsize=(max(10, self.num_assets * 0.6), 5))

        x_cursor = 0
        tick_positions, tick_labels = [], []
        first = True

        for c, info in sorted(self._block_info.items()):
            idx    = info['indices']
            n_i    = len(idx)
            thetas = 1.0 / self.sigma[idx]
            x0, x1 = x_cursor - 0.4, x_cursor + n_i - 0.6

            ax.scatter(np.arange(x_cursor, x_cursor + n_i), thetas,
                       color='steelblue', zorder=3, s=50)

            combined = self._theta_hat + self._bar_thetas[c]
            ax.hlines(combined, x0, x1, colors='darkorange', linewidths=2,
                      label=r'$\hat\theta + \bar\theta_i$' if first else None)

            if x_cursor > 0:
                ax.axvline(x=x_cursor - 0.5, color='gray', linewidth=0.8,
                           linestyle=':', alpha=0.6)

            tick_positions.append(x_cursor + (n_i - 1) / 2.0)
            tick_labels.append(f'C{c}')
            x_cursor += n_i
            first = False

        ax.axhline(y=self._theta_hat, color='red', linewidth=1.5,
                   label=r'$\hat\theta$')

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel(r'$\theta = 1/\sigma$')
        ax.set_title(title or 'Direct Solver Thresholds')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Single-block solver dispatch (two-tier methods)                    #
    # ------------------------------------------------------------------ #

    def _single_block_weights(self, V_approx, method, asset_list=None, plot_threshold=False, plot_title=None):
        n = V_approx.shape[0]
        if n == 1:
            return np.ones((1, 1))
        if method == 'oblivious':
            port = IgnoreAllCorrelation(V_approx, check_psd=False)
        else:
            port = CorrelationAwareBlock(V_approx, epsilon=self.epsilon,
                                         asset_list=asset_list,
                                         plot_threshold=plot_threshold,
                                         plot_title=plot_title)
        port.calculate_portfolio_positions()
        return port.positions

    # ------------------------------------------------------------------ #
    #  Main routine                                                        #
    # ------------------------------------------------------------------ #

    def _build_approx_matrix(self, global_rho, block_info):
        """Reconstruct the full n×n approximated covariance from the block decomposition."""
        Omega = np.full((self.num_assets, self.num_assets), global_rho)
        np.fill_diagonal(Omega, 1.0)
        for info in block_info.values():
            idx = info['indices']
            tilde_rho = info['rho_i'] * (1.0 - global_rho) + global_rho
            n_i = len(idx)
            block = (1.0 - tilde_rho) * np.eye(n_i) + tilde_rho * np.ones((n_i, n_i))
            Omega[np.ix_(idx, idx)] = block
        D = np.diag(self.sigma)
        return D @ Omega @ D

    def calculate_portfolio_positions(self):
        self._cluster()
        self._global_rho, self._block_info = self._block_approximation()
        global_rho, block_info = self._global_rho, self._block_info

        if self.plot_clustering:
            self._plot_clusters()

        if self.method == 'direct':
            return self._calculate_direct(global_rho, block_info)

        xs, agg_sigmas, index_map = [], [], []

        # Tier 1: within-block portfolios
        for block_id, info in sorted(block_info.items()):
            idx = info['indices']
            rho_i = info['rho_i']
            n_i = len(idx)
            sigma_i = self.sigma[idx]
            block_assets = np.array(self.assets)[idx].tolist()

            Omega_i = (1.0 - rho_i) * np.eye(n_i) + rho_i * np.ones((n_i, n_i))
            V_approx_i = np.diag(sigma_i) @ Omega_i @ np.diag(sigma_i)

            x_i = self._single_block_weights(V_approx_i, self.tier1_method,
                                             asset_list=block_assets,
                                             plot_threshold=self.plot_clustering,
                                             plot_title=f'Tier 1 — Block {block_id}')

            V_src = V_approx_i if self.use_approx_cov else self.covariance_matrix[np.ix_(idx, idx)]
            var_i = (x_i.T @ V_src @ x_i).item()

            xs.append(x_i)
            agg_sigmas.append(np.sqrt(max(var_i, 0.0)))
            index_map.extend(idx.tolist())

        # Tier 2: aggregate cross-block portfolio
        K = len(xs)
        sig_agg = np.array(agg_sigmas)
        Omega_agg = (1.0 - global_rho) * np.eye(K) + global_rho * np.ones((K, K))
        V_agg = np.diag(sig_agg) @ Omega_agg @ np.diag(sig_agg)
        x_agg = self._single_block_weights(V_agg, self.tier2_method,
                                            plot_threshold=self.plot_clustering,
                                            plot_title='Tier 2 — Aggregate')

        # Reconstruct: scale each block by its aggregate weight, then reorder
        final = np.concatenate([xs[i] * x_agg[i].item() for i in range(K)])
        reordered = np.zeros(self.num_assets)
        reordered[np.array(index_map)] = final.ravel()
        self.positions = reordered.reshape(-1, 1)
        return self.positions

    # ------------------------------------------------------------------ #
    #  Plotting                                                            #
    # ------------------------------------------------------------------ #

    def _plot_clusters(self):
        plt.figure(figsize=(10, 5))
        sch.dendrogram(self._linkage_matrix, labels=np.arange(self.num_assets))
        plt.axhline(y=self._threshold, color='red', linestyle='--',
                    label=f'Threshold = {self._threshold:.4f}')
        plt.title(f'BlockDiagonalPortfolio clustering  (tier1={self.tier1_method}, tier2={self.tier2_method})')
        plt.xlabel('Asset')
        plt.ylabel('Distance')
        plt.legend()
        plt.tight_layout()
        plt.show()

        sorted_idx = np.argsort(self.clusters)
        labels = np.array(self.assets)[sorted_idx]
        V_approx = self._build_approx_matrix(self._global_rho, self._block_info)

        cov_actual = self.covariance_matrix[np.ix_(sorted_idx, sorted_idx)]
        cov_approx = V_approx[np.ix_(sorted_idx, sorted_idx)]
        cov_vmin = min(cov_actual.min(), cov_approx.min())
        cov_vmax = max(cov_actual.max(), cov_approx.max())

        _, axes = plt.subplots(1, 2, figsize=(18, 8))
        plot_heatmap(cov_actual, 'Clustered Covariance', labels, ax=axes[0],
                     vmin=cov_vmin, vmax=cov_vmax)
        plot_heatmap(cov_approx, 'Approximated Block-Diagonal Covariance', labels, ax=axes[1],
                     vmin=cov_vmin, vmax=cov_vmax)
        plt.tight_layout()
        plt.show()

        sigma_approx = np.sqrt(np.diag(V_approx))
        Omega_approx = V_approx / np.outer(sigma_approx, sigma_approx)
        corr_actual = self.omega[np.ix_(sorted_idx, sorted_idx)]
        corr_approx = Omega_approx[np.ix_(sorted_idx, sorted_idx)]
        corr_vmin = min(corr_actual.min(), corr_approx.min())

        _, axes = plt.subplots(1, 2, figsize=(18, 8))
        plot_heatmap(corr_actual, 'Clustered Correlation', labels, ax=axes[0],
                     vmin=corr_vmin, vmax=1.0)
        plot_heatmap(corr_approx, 'Approximated Block-Diagonal Correlation', labels, ax=axes[1],
                     vmin=corr_vmin, vmax=1.0)
        plt.tight_layout()
        plt.show()
