import numpy as np


class PortfolioEvaluator:
    def __init__(
        self,
        covariance_matrix: np.ndarray,
        correlation_matrix: np.ndarray,
        periods_per_year: int = 252,
    ):
        self.covariance_matrix = covariance_matrix
        self.correlation_matrix = correlation_matrix
        self.periods_per_year = periods_per_year

    def evaluate(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        eps: float = 1e-6,
    ) -> dict:
        """
        returns : (T,) or (T, 1)  — per-period portfolio returns
        weights : (n, T)           — portfolio weights at each period
        """
        returns = np.asarray(returns).flatten()
        T = len(returns)
        _, T_w = weights.shape
        assert T == T_w

        cov = self.covariance_matrix
        corr = self.correlation_matrix

        def annualized_volatility(ret):
            return np.sqrt(self.periods_per_year) * np.std(ret, ddof=1)

        def support_size(w):
            return np.sum(np.abs(w) > eps)

        def gini(x):
            x = np.abs(x) + 1e-12
            diff = np.abs(x[:, None] - x[None, :])
            return diff.sum() / (2 * len(x) * x.sum())

        def risk_contributions(w):
            m = cov @ w
            var = w @ m
            return w * m / (var + 1e-12)

        ann_vol = annualized_volatility(returns)

        supports = np.array([support_size(weights[:, t]) for t in range(T)])
        avg_support = supports.mean()

        hhi = np.sum(weights ** 2, axis=0)
        enb = 1 / (hhi + 1e-12)
        gini_weights = np.array([gini(weights[:, t]) for t in range(T)])

        turnover = np.zeros(T - 1)
        for t in range(1, T):
            turnover[t - 1] = np.abs(weights[:, t] - weights[:, t - 1]).sum()
        cumulative_turnover = turnover.sum()

        rc_hhi = np.zeros(T)
        rc_enb = np.zeros(T)
        for t in range(T):
            rc = risk_contributions(weights[:, t])
            rc_hhi[t] = np.sum(rc ** 2)
            rc_enb[t] = 1 / (rc_hhi[t] + 1e-12)

        avg_corr = np.zeros(T)
        for t in range(T):
            w = weights[:, t]
            W = np.abs(w[:, None] * w[None, :])
            avg_corr[t] = (W * corr).sum() / (W.sum() + 1e-12)

        # Diversification ratio: weighted-average asset vol / portfolio vol.
        # DR = 1 means zero diversification benefit (all assets move together);
        # higher values mean the portfolio is exploiting low correlations.
        sigma = np.sqrt(np.diag(cov))
        dr = np.zeros(T)
        for t in range(T):
            w = weights[:, t]
            port_vol = np.sqrt(max(w @ cov @ w, 0.0))
            dr[t] = (np.abs(w) @ sigma) / (port_vol + 1e-12)

        # Average squared weight: mean(w_i^2) = HHI / n.
        # Complements HHI with an absolute (not relative) concentration measure.
        avg_sq_weight = np.mean(weights ** 2, axis=0)

        # Maximum weight: largest absolute position at each step.
        max_weight = np.max(np.abs(weights), axis=0)

        return {
            "annualized_volatility": ann_vol,
            "avg_support_size": avg_support,
            "hhi": hhi.mean(),
            "effective_num_assets": enb.mean(),
            "gini_weights": gini_weights.mean(),
            "turnover": turnover.mean(),
            "cumulative_turnover": cumulative_turnover,
            "risk_contribution_hhi": rc_hhi.mean(),
            "risk_contribution_enb": rc_enb.mean(),
            "weighted_avg_correlation": avg_corr.mean(),
            "diversification_ratio": dr.mean(),
            "avg_squared_weight": avg_sq_weight.mean(),
            "max_weight": max_weight.mean(),
        }
