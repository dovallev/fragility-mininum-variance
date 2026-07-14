"""
main.py

Backtests LongOnly, EqualWeight, and InvVol against the block-diagonal
shrinkage methods (BDS-Obliv, BDS-Aware, BDS-Direct) from "Fragility of
Minimum-Variance Portfolios" (Ovalle, Laird, Grossmann & Peña, 2026) on a
synthetic block-diagonal covariance, sample covariance only.
"""

import numpy as np

from src.generator import BlockCovarianceGenerator
from src.portfolio import (
    EqualWeightPortfolio,
    LongOnlyMarkowitz,
    IgnoreAllCorrelation,
    BlockDiagonalPortfolio,
)
from src.backtest import BacktestEngine
from src.visualizer import PortfolioVisualizer
from src.estimator import SampleCovarianceEstimator
from src.plot_utils import plot_heatmap


# 12-block synthetic covariance structure: block sizes, within-block correlation
# ranges, and (optional) per-asset volatilities used to build heterogeneous-vol
# blocks below.
BLOCK_SIZES  = [1, 2, 3, 2, 1, 2, 3, 2, 5, 3, 1, 3]
BLOCK_RANGES = [
    (0.4, 0.8), (0.5, 0.8), (0.6, 0.8), (0.6, 0.9),
    (0.4, 0.8), (0.5, 0.8), (0.6, 0.8), (0.6, 0.9),
    (0.4, 0.8), (0.5, 0.8), (0.6, 0.8), (0.6, 0.9),
]
# Low-vol blocks (1,3,5,9): sigmas ~0.10-0.15; high-vol blocks (2,4,6,8,10,12): ~0.30-0.50;
# mixed within blocks where size allows to stress within-cluster vol dispersion.
BLOCK_SIGMAS = [
    [0.10],                          # block 1  - low
    [0.10, 0.40],                    # block 2  - mixed
    [0.10, 0.12, 0.15],              # block 3  - low
    [0.30, 0.50],                    # block 4  - high
    [0.35],                          # block 5  - high
    [0.30, 0.45],                    # block 6  - high
    [0.10, 0.12, 0.14],              # block 7  - low
    [0.35, 0.50],                    # block 8  - high
    [0.10, 0.11, 0.40, 0.42, 0.45], # block 9  - mixed (large block)
    [0.30, 0.35, 0.40],              # block 10 - high
    [0.12],                          # block 11 - low
    [0.10, 0.35, 0.50],              # block 12 - mixed
]

N_SAMPLES    = 400
WINDOW_SIZE  = 100
WINDOW_SHIFT = 1


def main(heterogeneous_vol: bool = False):
    generator = BlockCovarianceGenerator(
        block_sizes=BLOCK_SIZES,
        block_ranges=BLOCK_RANGES,
        off_block_range=(0.01, 0.2),
        shuffle=False,
        seed=2,
        block_sigmas=BLOCK_SIGMAS if heterogeneous_vol else None,
    )
    covariance_matrix, return_array, asset_list = generator.generate()

    # Shuffle asset order so the block structure carries no positional information,
    # then plot it before it reaches the portfolios: clustering below has to recover
    # the blocks from the covariance values alone, with nothing to read off row/column order.
    n = covariance_matrix.shape[0]
    shuffle_idx = np.random.permutation(n)
    covariance_matrix = covariance_matrix[np.ix_(shuffle_idx, shuffle_idx)]
    return_array = return_array[shuffle_idx]
    if asset_list is not None:
        asset_list = [asset_list[i] for i in shuffle_idx]

    plot_heatmap(covariance_matrix, 'Shuffled covariance (before clustering)', asset_list)

    kw = dict(return_array=return_array, asset_list=asset_list)

    # One-shot visualisation on the full covariance matrix (clustering + per-block thresholds)
    viz_port = BlockDiagonalPortfolio(covariance_matrix, method='direct', plot_clustering=True, **kw)
    viz_port.calculate_portfolio_positions()

    portfolios = {
        'LongOnly':    LongOnlyMarkowitz(covariance_matrix, solver='pounce', **kw),
        'EqualWeight': EqualWeightPortfolio(covariance_matrix, **kw),
        'InvVol':      IgnoreAllCorrelation(covariance_matrix, **kw),
        'BDS-Obliv':    BlockDiagonalPortfolio(covariance_matrix, method='oblivious', **kw),
        'BDS-Aware':    BlockDiagonalPortfolio(covariance_matrix, method='aware', **kw),
        'BDS-Direct':   BlockDiagonalPortfolio(covariance_matrix, method='direct', **kw),
    }

    engine = BacktestEngine(
        portfolios=portfolios,
        covariance_matrix=covariance_matrix,
        return_array=return_array,
        asset_list=asset_list,
        estimator=SampleCovarianceEstimator(),
        n_samples=N_SAMPLES,
        window_size=WINDOW_SIZE,
        window_shift=WINDOW_SHIFT,
    )
    results = engine.run()
    print(engine.results_to_dataframe(results))

    viz = PortfolioVisualizer(assets=asset_list)
    viz.plot_variance_performance(results, engine.deltas, benchmark=engine.benchmark)
    viz.plot_return_performance(results, engine.deltas)
    viz.plot_position_performance(results, engine.deltas, assets=engine.assets,
                                  benchmark=engine.benchmark, plot_benchmark=True)


if __name__ == '__main__':
    main(heterogeneous_vol=False)


# PLACEHOLDER — update once the paper has a venue/DOI/arXiv id (see CITATION.cff).
#
# @article{ovalle2026fragility,
#   title   = {Fragility of Minimum-Variance Portfolios},
#   author  = {Ovalle, Daniel and Laird, Carl D. and Grossmann, Ignacio E. and Pe{\~n}a, Javier},
#   year    = {2026},
#   note    = {Carnegie Mellon University},
# }
