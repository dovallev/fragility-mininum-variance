"""
figures.py

Standalone, reproducible small illustrative instances from "Fragility of
Minimum-Variance Portfolios" (Ovalle, Laird, Grossmann & Peña, 2026): each
is a hand-built low-dimensional covariance chosen to make the threshold
effects behind minimum-variance fragility visible. Each example is a
function; main() runs all of them. Figures are saved as PDFs under figures/.

Run from the repo root:
    python figures.py
"""

from pathlib import Path

import numpy as np

from src.generator import ConstantCorrelationGenerator
from src.portfolio import LongOnlyMarkowitz, IgnoreAllCorrelation, CorrelationAwareBlock
from src.backtest import BacktestEngine
from src.visualizer import PortfolioVisualizer
from src.plot_utils import plot_heatmap

FIGURES_DIR = Path(__file__).parent / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

SOLVER = 'pounce'


class _DynamicCAB(CorrelationAwareBlock):
    """CorrelationAwareBlock that refreshes mean_rho from the estimated cov each step."""
    def calculate_portfolio_positions(self):
        if self.num_assets > 1:
            non_diag = ~np.eye(self.num_assets, dtype=bool)
            self.mean_rho = float(self.omega[non_diag].mean())
        return super().calculate_portfolio_positions()


# ── 2x2 unstable ──────────────────────────────────────────────────────────────

def fig_2x2_unstable():
    """Variance-area and position plots for the near-singular 2x2 instance.

    sigma_1=1, sigma_2=0.99, rho=0.99
    n_samples=313, window_size=63, window_shift=1
    Outputs: var_2x2_unstable.pdf, por_2x2_unstable.pdf
    """
    rho = 0.99
    sigma_1 = 1
    sigma_2 = 0.99
    Omega = np.array([[1, rho], [rho, 1]])
    Sigma = np.diag([sigma_1, sigma_2])
    cov = Sigma @ Omega @ Sigma
    ret = None
    assets = ['Asset 2', 'Asset 1']

    portfolios = {
        'Long-Only Markowitz': LongOnlyMarkowitz(cov, return_array=ret, asset_list=assets, solver=SOLVER),
    }

    engine = BacktestEngine(
        portfolios=portfolios,
        covariance_matrix=cov,
        return_array=ret,
        asset_list=assets,
        n_samples=313,
        window_size=63,
        window_shift=1,
    )
    np.random.seed(2)
    results = engine.run()
    print(engine.results_to_dataframe(results))

    viz = PortfolioVisualizer(assets=assets)
    figsize = (5, 4)

    viz.plot_variance_performance(
        results, engine.deltas,
        benchmark=engine.benchmark,
        plot_estimated=False,
        save=True, plot_name='var_2x2_unstable',
        figsize=figsize,
    )
    viz.plot_position_performance(
        results, engine.deltas,
        assets=assets,
        benchmark=engine.benchmark,
        plot_benchmark=False,
        save=True, plot_name='por_2x2_unstable',
        figsize=figsize,
    )


# ── 3x3 unstable ──────────────────────────────────────────────────────────────

def fig_3x3_unstable():
    """Variance-area and position plots for the 3x3 constant-correlation instance.

    sigma_1=1, sigma_2=1.8, sigma_3=2, rho=0.48
    n_samples=313, window_size=63, window_shift=1
    Outputs: var_3x3_unstable.pdf, por_3x3_unstable.pdf
    """
    rho = 0.48
    sigma_1 = 1.0
    sigma_2 = 1.8
    sigma_3 = 2.0
    Omega = np.array([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])
    Sigma = np.diag([sigma_1, sigma_2, sigma_3])
    cov = Sigma @ Omega @ Sigma
    ret = None
    assets = ['Asset 1', 'Asset 2', 'Asset 3']

    portfolios = {
        'Long-Only Markowitz': LongOnlyMarkowitz(cov, return_array=ret, asset_list=assets, solver=SOLVER),
    }

    engine = BacktestEngine(
        portfolios=portfolios,
        covariance_matrix=cov,
        return_array=ret,
        asset_list=assets,
        n_samples=313,
        window_size=63,
        window_shift=1,
    )
    np.random.seed(2)
    results = engine.run()
    print(engine.results_to_dataframe(results))

    viz = PortfolioVisualizer(assets=assets)
    figsize = (5, 4)

    viz.plot_variance_performance(
        results, engine.deltas,
        benchmark=engine.benchmark,
        plot_estimated=False,
        save=True, plot_name='var_3x3_unstable',
        figsize=figsize,
    )
    viz.plot_position_performance(
        results, engine.deltas,
        assets=assets,
        benchmark=engine.benchmark,
        plot_benchmark=False,
        save=True, plot_name='por_3x3_unstable',
        figsize=figsize,
    )


# ── cov_ex3: 7x7 block-diagonal example ──────────────────────────────────────

def fig_cov_ex3():
    """Heatmap + variance/position plots for the 7x7 block-diagonal instance.

    3 blocks (sizes 2, 2, 3), within-block rho = [0.96, 0.92, 0.96],
    cross-block rho = 0.6, sigma = I_7
    n_samples=313, window_size=63, window_shift=1
    Outputs: cov_ex3.pdf, var_block_ex3.pdf, por_block_ex3.pdf
    """
    omega = np.array([
        [1.  , 0.96, 0.6 , 0.6 , 0.6 , 0.6 , 0.6 ],
        [0.96, 1.  , 0.6 , 0.6 , 0.6 , 0.6 , 0.6 ],
        [0.6 , 0.6 , 1.  , 0.92, 0.6 , 0.6 , 0.6 ],
        [0.6 , 0.6 , 0.92, 1.  , 0.6 , 0.6 , 0.6 ],
        [0.6 , 0.6 , 0.6 , 0.6 , 1.  , 0.96, 0.96],
        [0.6 , 0.6 , 0.6 , 0.6 , 0.96, 1.  , 0.96],
        [0.6 , 0.6 , 0.6 , 0.6 , 0.96, 0.96, 1.  ],
    ])
    sigma = np.diag(np.ones(7))
    cov = sigma @ omega @ sigma
    ret = None
    assets = [f'Ast. {i+1}' for i in range(7)]

    plot_heatmap(cov, '', assets, save=True, plot_name='cov_ex3',
                 vmin=0, vmax=1, font_size=16)

    portfolios = {
        'Long-Only Markowitz': LongOnlyMarkowitz(cov, return_array=ret, asset_list=assets, solver=SOLVER),
    }

    engine = BacktestEngine(
        portfolios=portfolios,
        covariance_matrix=cov,
        return_array=ret,
        asset_list=assets,
        n_samples=313,
        window_size=63,
        window_shift=1,
    )
    np.random.seed(2)
    results = engine.run()
    print(engine.results_to_dataframe(results))

    viz = PortfolioVisualizer(assets=assets)
    figsize = (5, 4)

    viz.plot_variance_performance(
        results, engine.deltas,
        benchmark=engine.benchmark,
        plot_estimated=False,
        save=True, plot_name='var_block_ex3',
        figsize=figsize,
    )
    viz.plot_position_performance(
        results, engine.deltas,
        assets=assets,
        benchmark=engine.benchmark,
        plot_benchmark=False,
        save=True, plot_name='por_block_ex3',
        figsize=figsize,
    )


# ── 8x8 constant-rho examples ────────────────────────────────────────────────

def _8x8_portfolios(cov, ret, assets):
    """Shared portfolio dict for the 8x8 constant-rho examples."""
    return {
        'Long-Only Markowitz': LongOnlyMarkowitz(cov, return_array=ret, asset_list=assets, solver=SOLVER),
        'Corr-Oblivious':      IgnoreAllCorrelation(cov, return_array=ret, asset_list=assets),
        'Corr-Aware':          _DynamicCAB(cov, epsilon=1/8, return_array=ret, asset_list=assets),
    }


def _run_8x8(cov, ret, assets, var_name, por_name):
    portfolios = _8x8_portfolios(cov, ret, assets)
    engine = BacktestEngine(
        portfolios=portfolios,
        covariance_matrix=cov,
        return_array=ret,
        asset_list=assets,
        n_samples=313,
        window_size=63,
        window_shift=1,
    )
    np.random.seed(2)
    results = engine.run()
    print(engine.results_to_dataframe(results))

    viz = PortfolioVisualizer(assets=assets)
    figsize = (15, 4)
    viz.plot_variance_performance(
        results, engine.deltas,
        benchmark=engine.benchmark,
        plot_estimated=False,
        save=True, plot_name=var_name,
        figsize=figsize,
    )
    viz.plot_position_performance(
        results, engine.deltas,
        assets=assets,
        benchmark=engine.benchmark,
        plot_benchmark=False,
        save=True, plot_name=por_name,
        figsize=figsize,
    )


def fig_8x8_one():
    """8x8 constant-rho=0.90 instance with uniform sigmas (all 1).

    Portfolios: LongOnly, Corr-Oblivious, Corr-Aware (epsilon=1/8).
    Outputs: var_8x8_one.pdf, por_8x8_one.pdf
    """
    assets = [f'Ast. {i+1}' for i in range(8)]
    gen = ConstantCorrelationGenerator(sigmas=[1] * 8, rho=0.90, asset_list=assets)
    cov, ret, assets = gen.generate()
    _run_8x8(cov, ret, assets, 'var_8x8_one', 'por_8x8_one')


def fig_8x8_onetwo():
    """8x8 constant-rho=0.90 instance with sigmas [1,1,1,1,2,2,2,2].

    Portfolios: LongOnly, Corr-Oblivious, Corr-Aware (epsilon=1/8).
    Outputs: var_8x8_onetwo.pdf, por_8x8_onetwo.pdf
    """
    assets = [f'Ast. {i+1}' for i in range(8)]
    gen = ConstantCorrelationGenerator(sigmas=[1, 1, 1, 1, 2, 2, 2, 2], rho=0.90, asset_list=assets)
    cov, ret, assets = gen.generate()
    _run_8x8(cov, ret, assets, 'var_8x8_onetwo', 'por_8x8_onetwo')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    fig_2x2_unstable()
    fig_3x3_unstable()
    fig_cov_ex3()
    fig_8x8_one()
    fig_8x8_onetwo()


if __name__ == '__main__':
    main()
