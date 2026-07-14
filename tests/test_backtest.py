"""Backtest integration tests.

Runs BacktestEngine on a known block-diagonal covariance matrix with the six
methods used in main.py (LongOnly, EqualWeight, InvVol, BD-Obliv, BD-Aware,
BD-Direct) on sample covariance, and checks basic structural and economic
sanity of the results.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestEngine
from src.portfolio import (
    EqualWeightPortfolio,
    LongOnlyMarkowitz,
    IgnoreAllCorrelation,
    BlockDiagonalPortfolio,
)
from src.estimator import SampleCovarianceEstimator
from src.generator import BlockCovarianceGenerator


SEED         = 42
N_ASSETS     = 8          # 3 blocks: sizes [2, 3, 3]
N_SAMPLES    = 120
WINDOW_SIZE  = 40
WINDOW_SHIFT = 10          # (120 - 40) / 10 = 8 rebalance steps

METHOD_NAMES = {'LongOnly', 'EqualWeight', 'InvVol', 'BD-Obliv', 'BD-Aware', 'BD-Direct'}


@pytest.fixture(scope='module')
def covariance():
    gen = BlockCovarianceGenerator(
        block_sizes=[2, 3, 3],
        block_ranges=[(0.5, 0.7), (0.4, 0.6), (0.6, 0.8)],
        off_block_range=(0.05, 0.15),
        seed=SEED,
    )
    cov, return_array, asset_list = gen.generate()
    return cov, return_array, asset_list


def _make_portfolios(cov, return_array, asset_list):
    kw = dict(return_array=return_array, asset_list=asset_list)
    return {
        'LongOnly':    LongOnlyMarkowitz(cov, **kw),
        'EqualWeight': EqualWeightPortfolio(cov, **kw),
        'InvVol':      IgnoreAllCorrelation(cov, **kw),
        'BD-Obliv':    BlockDiagonalPortfolio(cov, method='oblivious', **kw),
        'BD-Aware':    BlockDiagonalPortfolio(cov, method='aware', **kw),
        'BD-Direct':   BlockDiagonalPortfolio(cov, method='direct', **kw),
    }


def test_backtest_sanity(covariance):
    cov, return_array, asset_list = covariance
    n = cov.shape[0]

    engine = BacktestEngine(
        portfolios=_make_portfolios(cov, return_array, asset_list),
        covariance_matrix=cov,
        return_array=return_array,
        asset_list=asset_list,
        estimator=SampleCovarianceEstimator(),
        n_samples=N_SAMPLES,
        window_size=WINDOW_SIZE,
        window_shift=WINDOW_SHIFT,
    )
    results = engine.run()

    n_steps      = len(engine.deltas)
    n_metrics    = len(next(iter(results.values())).metrics)
    n_portfolios = len(results)

    # --- structural assertions -------------------------------------------

    assert set(results.keys()) == METHOD_NAMES

    for name, result in results.items():
        ctx = f"[{name}]"
        assert len(result.return_horizon)   == n_steps,       f"{ctx} return_horizon length"
        assert len(result.estimate_horizon) == n_steps,       f"{ctx} estimate_horizon length"
        assert len(result.actual_horizon)   == n_steps,       f"{ctx} actual_horizon length"
        assert result.position_horizon.shape == (n, n_steps), f"{ctx} position_horizon shape"

    # --- economic sanity checks: all six methods here are long-only -------

    for name, result in results.items():
        ctx = f"[{name}]"

        assert all(v > 0 for v in result.estimate_horizon), \
            f"{ctx} non-positive estimated variance"
        assert all(v > 0 for v in result.actual_horizon), \
            f"{ctx} non-positive realized variance"

        np.testing.assert_allclose(
            result.position_horizon.sum(axis=0), 1.0, atol=1e-4,
            err_msg=f"{ctx} weights do not sum to 1",
        )
        assert (result.position_horizon >= -1e-6).all(), \
            f"{ctx} negative weight in a long-only portfolio"

    # --- metric finiteness, including 'area' (only present here, not in
    #     the empirical engine) ---------------------------------------------

    for name, result in results.items():
        assert 'area' in result.metrics, f"[{name}] missing 'area' metric"
        for metric, value in result.metrics.items():
            assert np.isfinite(value), f"[{name}] metric '{metric}' = {value} is not finite"

    # --- results_to_dataframe shape --------------------------------------

    df = engine.results_to_dataframe(results)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (n_metrics, n_portfolios)
