import numpy as np
import pytest

from src.portfolio import ConstantCorrelationBlock, LongOnlyMarkowitz
from src.generator import ConstantCorrelationGenerator


CASES = [
    ConstantCorrelationGenerator([0.10, 0.20, 0.15, 0.30, 0.25], rho=0.40),
    ConstantCorrelationGenerator([0.10, 0.11, 0.12, 0.13],        rho=0.70),
    ConstantCorrelationGenerator([0.05, 0.20, 0.50],               rho=0.20),
    ConstantCorrelationGenerator([0.10, 0.20, 0.15, 0.30],         rho=0.10),
    ConstantCorrelationGenerator([0.10, 0.10, 0.10, 0.10],         rho=0.60),
]


def _case_id(gen):
    return f"n={len(gen.sigmas)},rho={gen.rho}"


# --------------------------------------------------------------------------- #
#  Test 1 – sorting threshold satisfies the fixed-point equation (eq.fixed.point)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", CASES, ids=_case_id)
def test_sorting_threshold_satisfies_fixed_point(gen):
    V, _, _ = gen.generate()
    p = ConstantCorrelationBlock(V)

    rho = p.mean_rho
    theta = 1.0 / p.sigma
    n = p.num_assets

    bar_theta = p._compute_threshold_via_sorting()

    # Fixed-point: bar_theta = rho/(1+(n-1)*rho) * e' * max{theta, bar_theta*e}
    rhs = rho / (1.0 + (n - 1) * rho) * np.sum(np.maximum(theta, bar_theta))
    np.testing.assert_allclose(bar_theta, rhs, atol=1e-12,
                               err_msg="bar_theta does not satisfy the fixed-point equation")


# --------------------------------------------------------------------------- #
#  Test 2 – sorting weights match LongOnlyMarkowitz on the true constant-rho matrix
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", CASES, ids=_case_id)
def test_sorting_matches_longonly_markowitz(gen):
    V, _, _ = gen.generate()

    sp = ConstantCorrelationBlock(V)
    sp.calculate_portfolio_positions()

    lo = LongOnlyMarkowitz(V)
    lo.calculate_portfolio_positions()

    # atol=1e-5 accommodates typical residual on inactive-asset bounds
    np.testing.assert_allclose(
        sp.positions.ravel(),
        lo.positions.ravel(),
        atol=1e-5,
        err_msg="SubPortfolio sorting weights differ from LongOnlyMarkowitz",
    )
