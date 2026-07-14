from dataclasses import dataclass, field
import copy

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from tqdm import tqdm

from .portfolio import Portfolio, LongOnlyMarkowitz
from .estimator import CovarianceEstimator, SampleCovarianceEstimator
from .evaluator import PortfolioEvaluator
from .validator import PortfolioValidator


@dataclass
class BacktestResult:
    name: str
    actual_horizon: list = field(default_factory=list)
    estimate_horizon: list = field(default_factory=list)
    return_horizon: list = field(default_factory=list)
    position_horizon: np.ndarray = None
    metrics: dict = field(default_factory=dict)


class BacktestEngine:
    """Rolling-window backtest against a known ("true") covariance matrix.

    At each step, the covariance is re-estimated from a fresh simulated
    return window and every portfolio is re-optimized; realized variance is
    tracked against the true covariance and against the LongOnly benchmark.
    """

    def __init__(
        self,
        portfolios: dict,
        covariance_matrix: np.ndarray,
        n_samples: int = 200,
        window_size: int = None,
        window_shift: int = 1,
        return_array: np.ndarray = None,
        asset_list: list = None,
        estimator: CovarianceEstimator = None,
        solver: str = 'pounce',
    ):
        assert isinstance(portfolios, dict), 'Portfolios must be a dict with name as key and Portfolio as value'
        self.portfolios = portfolios
        self.covariance_matrix = covariance_matrix
        self.num_assets = covariance_matrix.shape[0]
        self.n_samples = n_samples
        self.window_size = self.num_assets * 5 if window_size is None else window_size
        assert self.window_size >= self.num_assets, 'Window size must be >= number of assets'
        self.window_shift = window_shift
        self.solver = solver
        self.estimator = estimator or SampleCovarianceEstimator()

        num_assets = covariance_matrix.shape[0]
        self.returns = PortfolioValidator.validate_return_array(return_array, num_assets)
        self.assets = PortfolioValidator.validate_asset_list(asset_list, num_assets)

        self.sigma = np.sqrt(np.diag(covariance_matrix))
        self.omega = covariance_matrix / np.outer(self.sigma, self.sigma)

        self.benchmark = None
        self.deltas = None

        self._sync_portfolio_data()

    def _sync_portfolio_data(self):
        for portfolio in self.portfolios.values():
            portfolio.num_assets = self.num_assets
            portfolio.covariance_matrix = self.covariance_matrix
            portfolio.returns = self.returns
            portfolio.assets = self.assets

    def get_benchmark(self) -> Portfolio:
        for portfolio in self.portfolios.values():
            if isinstance(portfolio, LongOnlyMarkowitz):
                self.benchmark = copy.deepcopy(portfolio)
                return self.benchmark
        self.benchmark = LongOnlyMarkowitz(
            self.covariance_matrix, return_array=self.returns, asset_list=self.assets, solver=self.solver
        )
        return self.benchmark

    def run(self, sampled_returns: np.ndarray = None) -> dict[str, BacktestResult]:
        self.get_benchmark()
        self.benchmark.calculate_portfolio_positions()
        self.benchmark.calculate_portfolio_variance()

        if sampled_returns is None:
            sampled_returns = np.random.multivariate_normal(
                np.zeros(self.num_assets), self.covariance_matrix, self.n_samples + self.window_shift
            )

        self.deltas = np.arange(0, self.n_samples - self.window_size + self.window_shift, self.window_shift)
        benchmark_var_vector = np.ones(len(self.deltas)) * self.benchmark.variance
        evaluator = PortfolioEvaluator(self.covariance_matrix, self.omega)

        results = {}

        for name, portfolio in self.portfolios.items():
            portfolio.reset_horizons()
            result = BacktestResult(name=name)

            if hasattr(portfolio, 'plot_clustering'):
                portfolio.plot_clustering = False

            for delta in tqdm(self.deltas, desc=name):
                window = sampled_returns[delta: self.window_size + delta]
                estimated_cov = self.estimator.fit(window)
                out_of_sample_ret = sampled_returns[self.window_size + delta].reshape(-1, 1)

                portfolio.returns = out_of_sample_ret
                portfolio.covariance_matrix = estimated_cov
                portfolio.calculate_sigma_and_omega()
                portfolio.calculate_portfolio_positions()
                portfolio.calculate_portfolio_variance()
                portfolio.calculate_portfolio_return()

                portfolio.position_horizon = np.hstack([portfolio.position_horizon, portfolio.positions])
                result.estimate_horizon.append(portfolio.variance)

                portfolio.covariance_matrix = self.covariance_matrix
                portfolio.calculate_portfolio_variance()
                result.actual_horizon.append(portfolio.variance)
                result.return_horizon.append(portfolio.portfolio_return)

            result.position_horizon = portfolio.position_horizon[:, 1:]

            area = np.abs(simpson(np.array(result.actual_horizon) - benchmark_var_vector))
            returns_arr = np.array(result.return_horizon).reshape(-1, 1)
            metrics = evaluator.evaluate(returns_arr, result.position_horizon)
            metrics['area'] = area
            result.metrics = metrics

            results[name] = result

        return results

    def results_to_dataframe(self, results: dict[str, BacktestResult]) -> pd.DataFrame:
        return pd.DataFrame({name: r.metrics for name, r in results.items()})
