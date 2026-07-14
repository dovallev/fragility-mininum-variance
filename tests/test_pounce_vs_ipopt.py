"""Cross-check LongOnlyMarkowitz's default solver ('pounce') against IPOPT.

`pounce` (https://github.com/jkitchin/pounce) is used here in-process via
`LongOnlyMarkowitz(solver='pounce')` -> `pounce.solve_qp`.

Skipped entirely if either `pounce-solver` or a working `ipopt` is not
installed.
"""

import numpy as np
import pytest
import pyomo.environ as pyo

from src.portfolio import LongOnlyMarkowitz
from src.generator import BlockCovarianceGenerator, ConstantCorrelationGenerator

pytest.importorskip("pounce", reason="pounce-solver not installed")

IPOPT_AVAILABLE = pyo.SolverFactory('ipopt').available(exception_flag=False)

pytestmark = pytest.mark.skipif(
    not IPOPT_AVAILABLE, reason="ipopt is not available on this system"
)


# A handful of instances spanning different sizes and correlation structures.
INSTANCES = [
    ConstantCorrelationGenerator([0.10, 0.20, 0.15, 0.30, 0.25], rho=0.40),
    ConstantCorrelationGenerator([0.10, 0.11, 0.12, 0.13], rho=0.70),
    ConstantCorrelationGenerator([0.05, 0.20, 0.50], rho=0.20),
    BlockCovarianceGenerator(
        block_sizes=[2, 3, 3], block_ranges=[(0.5, 0.7), (0.4, 0.6), (0.6, 0.8)],
        off_block_range=(0.05, 0.15), seed=1,
    ),
    BlockCovarianceGenerator(
        block_sizes=[1, 2, 3, 2], block_ranges=[(0.4, 0.8), (0.5, 0.8), (0.6, 0.8), (0.6, 0.9)],
        off_block_range=(0.01, 0.2), seed=7,
    ),
]


def _instance_id(gen):
    if isinstance(gen, ConstantCorrelationGenerator):
        return f"ccg_n={len(gen.sigmas)},rho={gen.rho}"
    return f"block_n={sum(gen.block_sizes)}"


@pytest.mark.parametrize("gen", INSTANCES, ids=_instance_id)
def test_ipopt_matches_pounce(gen):
    V, _, _ = gen.generate()

    lom_ipopt = LongOnlyMarkowitz(V, solver='ipopt')
    lom_ipopt.calculate_portfolio_positions()

    lom_pounce = LongOnlyMarkowitz(V, solver='pounce')
    lom_pounce.calculate_portfolio_positions()

    np.testing.assert_allclose(
        lom_ipopt.positions.ravel(),
        lom_pounce.positions.ravel(),
        atol=1e-4,
        err_msg="LongOnlyMarkowitz(ipopt) differs from LongOnlyMarkowitz(pounce)",
    )


def test_pounce_is_default_solver():
    V, _, _ = ConstantCorrelationGenerator([0.10, 0.20, 0.15], rho=0.5).generate()
    assert LongOnlyMarkowitz(V).solver == 'pounce'
