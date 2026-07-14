import numpy as np
import pytest

from src.portfolio import (
    ConstantCorrelationBlock,
    CorrelationAwareBlock,
    BlockDiagonalPortfolio,
    IgnoreAllCorrelation,
    LongOnlyMarkowitz,
)
from src.generator import ConstantCorrelationGenerator, BlockDiagonalStructureGenerator


# --------------------------------------------------------------------------- #
#  Helpers / fixtures                                                           #
# --------------------------------------------------------------------------- #

def _make_block_diagonal_cov(block_specs):
    """True block-diagonal covariance with zero off-block correlations.

    block_specs : list of (sigmas, rho) tuples, one per block.
    Returns (V, idx_list) where idx_list[k] = np.arange of block k asset indices.
    """
    n = sum(len(sigmas) for sigmas, _ in block_specs)
    V = np.zeros((n, n))
    idx_list = []
    start = 0
    for sigmas, rho in block_specs:
        k = len(sigmas)
        end = start + k
        V_block, _, _ = ConstantCorrelationGenerator(list(sigmas), rho).generate()
        V[start:end, start:end] = V_block
        idx_list.append(np.arange(start, end))
        start = end
    return V, idx_list


# True block-diagonal cases: list of (sigmas, rho) per block
BLOCK_SPECS = [
    [([0.10, 0.20, 0.15], 0.60), ([0.25, 0.30], 0.50), ([0.12, 0.18, 0.22], 0.70)],
    [([0.10, 0.12], 0.80), ([0.20, 0.25, 0.30, 0.15], 0.65)],
    # ([0.15, 0.25], 0.6) avoided: 1/0.25 = bar_theta exactly → ipopt degenerate
    [([0.10, 0.20, 0.30], 0.70), ([0.15, 0.22], 0.60), ([0.05, 0.08, 0.12, 0.10], 0.75)],
]

# Constant-correlation generators (single-block matrices)
CCB_GENS = [
    ConstantCorrelationGenerator([0.10, 0.20, 0.15, 0.30, 0.25], rho=0.40),
    ConstantCorrelationGenerator([0.05, 0.20, 0.50], rho=0.20),
    ConstantCorrelationGenerator([0.10, 0.11, 0.12, 0.13], rho=0.70),
]

# Known block-structure generators: cover approximation recovery and direct-method cases
BLOCK_STRUCT_GENS = [
    BlockDiagonalStructureGenerator(
        [[0.10, 0.20, 0.15], [0.25, 0.30], [0.12, 0.18, 0.22]],
        [0.60, 0.50, 0.70], global_rho=0.10,
    ),
    BlockDiagonalStructureGenerator(
        [[0.10, 0.12], [0.20, 0.25, 0.30, 0.15]],
        [0.80, 0.65], global_rho=0.05,
    ),
    # Singleton first block: rho_i is always 0.0 for n_i=1 regardless of input value
    BlockDiagonalStructureGenerator(
        [[0.15], [0.10, 0.20, 0.30], [0.05, 0.08]],
        [0.0, 0.70, 0.45], global_rho=0.15,
    ),
    # Higher global_rho cases
    BlockDiagonalStructureGenerator(
        [[0.10, 0.20, 0.15], [0.25, 0.30]],
        [0.50, 0.40], global_rho=0.30,
    ),
    BlockDiagonalStructureGenerator(
        [[0.10, 0.12, 0.14], [0.20, 0.25], [0.08, 0.10, 0.12]],
        [0.60, 0.55, 0.70], global_rho=0.25,
    ),
]


def _spec_id(spec):
    sizes = [len(s) for s, _ in spec]
    return "blocks=" + "+".join(map(str, sizes))


def _gen_id(gen):
    return f"n={len(gen.sigmas)},rho={gen.rho}"


def _bd_gen_id(gen):
    return "blocks=" + "+".join(str(len(s)) for s in gen.block_sigmas) + f",rho={gen.global_rho}"


# --------------------------------------------------------------------------- #
#  Test 1 – BD-aware (epsilon=0) on true block-diagonal matches LongOnlyMarkowitz
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec", BLOCK_SPECS, ids=_spec_id)
def test_bd_aware_true_block_diagonal_matches_longonly(spec):
    """BD-aware with epsilon=0 is exact on a true block-diagonal covariance.

    When the covariance is truly block-diagonal, all off-block entries are zero,
    so global_rho = 0 and rho_i = tilde_rho_i.  The block-diagonal approximation
    is then exact, and CorrelationAwareBlock with epsilon=0 recovers the same
    solution as the full long-only Markowitz QP.

    A failure here means either the clustering is not finding the true blocks,
    or the within-block solver is not matching the unconstrained optimum.
    """
    V, _ = _make_block_diagonal_cov(spec)

    bd = BlockDiagonalPortfolio(V, method='aware', epsilon=0)
    bd.calculate_portfolio_positions()

    lo = LongOnlyMarkowitz(V)
    lo.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd.positions.ravel(),
        lo.positions.ravel(),
        atol=1e-4,
        err_msg="BD-aware (epsilon=0) on true block-diagonal differs from LongOnlyMarkowitz",
    )


# --------------------------------------------------------------------------- #
#  Test 2 – BD-oblivious with singleton clusters matches IgnoreAllCorrelation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", CCB_GENS, ids=_gen_id)
def test_bd_oblivious_singleton_clusters_matches_ignore_correlation(gen):
    """BD-oblivious collapses to inverse-variance weighting when every asset is its own cluster.

    With clustering_parameter=1e-10, every pair of assets is too far apart to
    merge, so each asset forms a singleton cluster.  Tier-1 trivially returns
    weight 1 for a single asset.  The Tier-2 aggregate matrix is diagonal
    (no off-block entries → global_rho = 0), and 'oblivious' then assigns
    inverse-variance weights equal to IgnoreAllCorrelation.

    A failure here means Tier-1 singleton handling or the Tier-2 aggregate
    diagonal structure is broken.
    """
    V, _, _ = gen.generate()

    # clustering_parameter=1e-10 forces every asset into its own cluster
    bd = BlockDiagonalPortfolio(V, method='oblivious', clustering_parameter=1e-10)
    bd.calculate_portfolio_positions()

    iac = IgnoreAllCorrelation(V)
    iac.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd.positions.ravel(),
        iac.positions.ravel(),
        atol=1e-12,
        err_msg="BD-oblivious (singleton clusters) differs from IgnoreAllCorrelation",
    )


# --------------------------------------------------------------------------- #
#  Test 3 – BD-aware with single cluster matches CorrelationAwareBlock
# --------------------------------------------------------------------------- #

_EPSILONS = [None, 0.01, 0.05]


@pytest.mark.parametrize("gen", CCB_GENS, ids=_gen_id)
@pytest.mark.parametrize("eps", _EPSILONS, ids=lambda e: f"eps={e}")
def test_bd_aware_single_cluster_matches_correlation_aware_block(gen, eps):
    """BD-aware reduces to CorrelationAwareBlock when all assets form one cluster.

    With clustering_parameter=inf, all assets merge into K=1.  Tier-1 runs
    CorrelationAwareBlock on the full matrix.  Tier-2 has a single block so
    x_agg = [1] regardless of the method, contributing no additional weighting.
    The final positions therefore equal those of a standalone CorrelationAwareBlock.

    A failure here means the single-cluster code path deviates from the
    direct solver, or global_rho is incorrectly computed when there are no
    off-block entries.
    """
    V, _, _ = gen.generate()

    # clustering_parameter=np.inf merges all assets into a single cluster
    bd = BlockDiagonalPortfolio(V, method='aware', epsilon=eps, clustering_parameter=np.inf)
    bd.calculate_portfolio_positions()

    cab = CorrelationAwareBlock(V, epsilon=eps)
    cab.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd.positions.ravel(),
        cab.positions.ravel(),
        atol=1e-12,
        err_msg=f"BD-aware (single cluster, eps={eps}) differs from CorrelationAwareBlock",
    )


# --------------------------------------------------------------------------- #
#  Test 4 – CorrelationAwareBlock(epsilon=0) matches ConstantCorrelationBlock
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", CCB_GENS, ids=_gen_id)
def test_aware_epsilon_zero_matches_ccb(gen):
    """With epsilon=0 the robustification step is a no-op and both solvers agree.

    CorrelationAwareBlock projects bar_theta onto the epsilon-feasible set.
    When epsilon=0 every real number is feasible (the only excluded points are
    the breakpoints themselves, which have measure zero), so bar_theta is never
    moved.  The result must equal ConstantCorrelationBlock which computes bar_theta
    without any projection.

    A failure here means the projection logic incorrectly perturbs the threshold
    even when epsilon=0.
    """
    V, _, _ = gen.generate()

    cab = CorrelationAwareBlock(V, epsilon=0)
    cab.calculate_portfolio_positions()

    ccb = ConstantCorrelationBlock(V)
    ccb.calculate_portfolio_positions()

    np.testing.assert_allclose(
        cab.positions.ravel(),
        ccb.positions.ravel(),
        atol=1e-12,
        err_msg="CorrelationAwareBlock(epsilon=0) differs from ConstantCorrelationBlock",
    )


# --------------------------------------------------------------------------- #
#  Test 5 – Robustified threshold is at least epsilon away from every breakpoint
# --------------------------------------------------------------------------- #

_EPSILONS_MARGIN = [0.01, 0.05, 0.10]


@pytest.mark.parametrize("gen", CCB_GENS, ids=_gen_id)
@pytest.mark.parametrize("eps", _EPSILONS_MARGIN, ids=lambda e: f"eps={e}")
def test_robustified_threshold_respects_epsilon_margin(gen, eps):
    """The robustified threshold theta_eps satisfies |theta_eps - theta_i| >= eps for all i.

    This is the core guarantee of Algorithm 1 (Section sec.constant.correl):
    the output must lie in one of the feasible regions, each of which excludes
    an epsilon-ball around every breakpoint theta_i = 1/sigma_i.

    A failure here means the projection in _robustify_threshold does not
    correctly respect the margin, making the portfolio susceptible to active-set
    instability under small perturbations of sigma.
    """
    V, _, _ = gen.generate()
    cab = CorrelationAwareBlock(V, epsilon=eps)

    bar_theta = cab._compute_threshold_via_sorting()
    theta_eps = cab._robustify_threshold(bar_theta)

    breakpoints = 1.0 / cab.sigma  # theta_i = 1/sigma_i
    for bp in breakpoints:
        assert abs(theta_eps - bp) >= eps - 1e-14, (
            f"Robustified threshold {theta_eps:.6f} is within epsilon={eps} "
            f"of breakpoint {bp:.6f} (gap = {abs(theta_eps - bp):.2e})"
        )


# --------------------------------------------------------------------------- #
#  Test 6 – Equal volatilities give equal weights (1/n)
# --------------------------------------------------------------------------- #

_EQUAL_SIG_CASES = [
    (4, 0.30, 0.05),
    (5, 0.60, 0.10),
    (3, 0.80, 0.20),
]


@pytest.mark.parametrize("n,rho,sigma", _EQUAL_SIG_CASES, ids=lambda x: str(x))
def test_equal_sigmas_give_equal_weights_ccb(n, rho, sigma):
    """With identical volatilities the minimum-variance portfolio is equally weighted.

    When all sigma_i are equal the objective is symmetric in all assets, so
    the optimal long-only portfolio assigns weight 1/n to every asset.
    ConstantCorrelationBlock should recover this by setting bar_theta low enough
    that all assets clear the threshold.

    A failure here means the sorting-based threshold computation breaks symmetry.
    """
    V, _, _ = ConstantCorrelationGenerator([sigma] * n, rho=rho).generate()

    ccb = ConstantCorrelationBlock(V)
    ccb.calculate_portfolio_positions()

    np.testing.assert_allclose(
        ccb.positions.ravel(),
        np.full(n, 1.0 / n),
        atol=1e-12,
        err_msg="ConstantCorrelationBlock does not give equal weights for equal sigmas",
    )


@pytest.mark.parametrize("n,rho,sigma", _EQUAL_SIG_CASES, ids=lambda x: str(x))
@pytest.mark.parametrize("eps", [None, 0.01, 0.05], ids=lambda e: f"eps={e}")
def test_equal_sigmas_give_equal_weights_aware(n, rho, sigma, eps):
    """CorrelationAwareBlock also gives 1/n weights under equal volatilities.

    The robustification step should not break symmetry: when all breakpoints
    are identical there is only one distinct theta_i, and the projection must
    push bar_theta to the same side of it for every asset, preserving equal
    weighting.

    A failure here means the robustification procedure treats equal-sigma assets
    asymmetrically.
    """
    V, _, _ = ConstantCorrelationGenerator([sigma] * n, rho=rho).generate()

    cab = CorrelationAwareBlock(V, epsilon=eps)
    cab.calculate_portfolio_positions()

    np.testing.assert_allclose(
        cab.positions.ravel(),
        np.full(n, 1.0 / n),
        atol=1e-12,
        err_msg=f"CorrelationAwareBlock(eps={eps}) does not give equal weights for equal sigmas",
    )


# --------------------------------------------------------------------------- #
#  Test 7 – tier1_method / tier2_method override method (backward compat)     #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", CCB_GENS, ids=_gen_id)
@pytest.mark.parametrize("m", ['oblivious', 'aware'], ids=lambda m: m)
def test_explicit_tier_methods_match_method_shorthand(gen, m):
    """tier1_method=m, tier2_method=m should produce identical results to method=m."""
    V, _, _ = gen.generate()

    bd_shorthand = BlockDiagonalPortfolio(V, method=m)
    bd_shorthand.calculate_portfolio_positions()

    bd_explicit = BlockDiagonalPortfolio(V, tier1_method=m, tier2_method=m)
    bd_explicit.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd_shorthand.positions.ravel(),
        bd_explicit.positions.ravel(),
        atol=1e-12,
        err_msg=f"method='{m}' shorthand differs from explicit tier1_method=tier2_method='{m}'",
    )


# --------------------------------------------------------------------------- #
#  Test 8 – Single cluster: tier1 drives output, tier2 is irrelevant (K=1)   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", CCB_GENS, ids=_gen_id)
@pytest.mark.parametrize("eps", _EPSILONS, ids=lambda e: f"eps={e}")
def test_mixed_obliv_aware_single_cluster_matches_aware_block(gen, eps):
    """tier1='aware', tier2='oblivious', single cluster → matches CorrelationAwareBlock.

    With K=1 the Tier-2 problem has a single block so x_agg=[1] regardless of
    tier2_method, and the result is entirely determined by Tier 1.
    """
    V, _, _ = gen.generate()

    bd = BlockDiagonalPortfolio(V, tier1_method='aware', tier2_method='oblivious',
                                epsilon=eps, clustering_parameter=np.inf)
    bd.calculate_portfolio_positions()

    cab = CorrelationAwareBlock(V, epsilon=eps)
    cab.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd.positions.ravel(),
        cab.positions.ravel(),
        atol=1e-12,
        err_msg=f"BD(tier1=aware,tier2=oblivious,single cluster,eps={eps}) differs from CorrelationAwareBlock",
    )


@pytest.mark.parametrize("gen", CCB_GENS, ids=_gen_id)
def test_mixed_aware_obliv_single_cluster_matches_ignore_correlation(gen):
    """tier1='oblivious', tier2='aware', single cluster → matches IgnoreAllCorrelation.

    With K=1 the aggregate weight is always 1, so the output equals the Tier-1
    result which is inverse-variance weighting.
    """
    V, _, _ = gen.generate()

    bd = BlockDiagonalPortfolio(V, tier1_method='oblivious', tier2_method='aware',
                                clustering_parameter=np.inf)
    bd.calculate_portfolio_positions()

    iac = IgnoreAllCorrelation(V)
    iac.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd.positions.ravel(),
        iac.positions.ravel(),
        atol=1e-12,
        err_msg="BD(tier1=oblivious,tier2=aware,single cluster) differs from IgnoreAllCorrelation",
    )


# --------------------------------------------------------------------------- #
#  Test 9 – Singleton clusters: tier2 drives output, tier1 is trivial (n_i=1) #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", CCB_GENS, ids=_gen_id)
def test_mixed_aware_obliv_singleton_matches_ignore_correlation(gen):
    """tier1='aware', tier2='oblivious', singleton clusters → matches IgnoreAllCorrelation.

    Each cluster has exactly one asset so Tier-1 always returns weight 1.
    The aggregate Tier-2 with 'oblivious' then assigns inverse-variance weights
    over aggregate sigmas which equal the individual sigmas → same as IgnoreAllCorrelation.
    """
    V, _, _ = gen.generate()

    bd = BlockDiagonalPortfolio(V, tier1_method='aware', tier2_method='oblivious',
                                clustering_parameter=1e-10)
    bd.calculate_portfolio_positions()

    iac = IgnoreAllCorrelation(V)
    iac.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd.positions.ravel(),
        iac.positions.ravel(),
        atol=1e-12,
        err_msg="BD(tier1=aware,tier2=oblivious,singleton) differs from IgnoreAllCorrelation",
    )


# --------------------------------------------------------------------------- #
#  Test 10 – All four combinations are valid portfolios (sum=1, non-negative)  #
# --------------------------------------------------------------------------- #

_ALL_COMBOS = [
    ('oblivious', 'oblivious'),
    ('oblivious', 'aware'),
    ('aware',     'oblivious'),
    ('aware',     'aware'),
]


@pytest.mark.parametrize("spec", BLOCK_SPECS, ids=_spec_id)
@pytest.mark.parametrize("t1,t2", _ALL_COMBOS, ids=lambda x: f"{x[0]}/{x[1]}" if isinstance(x, tuple) else x)
def test_all_tier_combos_valid_portfolio(spec, t1, t2):
    """All tier1 x tier2 combinations sum to 1 and are non-negative."""
    V, _ = _make_block_diagonal_cov(spec)

    bd = BlockDiagonalPortfolio(V, tier1_method=t1, tier2_method=t2)
    bd.calculate_portfolio_positions()

    w = bd.positions.ravel()
    assert np.all(w >= -1e-10), f"Negative weights for tier1={t1}, tier2={t2}: {w.min():.4e}"
    np.testing.assert_allclose(
        w.sum(), 1.0, atol=1e-10,
        err_msg=f"Weights do not sum to 1 for tier1={t1}, tier2={t2}",
    )


# --------------------------------------------------------------------------- #
#  Test 11 – Aware/Oblivious vs Oblivious/Aware produce different allocations  #
# --------------------------------------------------------------------------- #

def test_mixed_combos_differ_from_each_other():
    """tier1=aware/tier2=obliv and tier1=obliv/tier2=aware give different weights
    for a multi-block matrix with non-trivial within- and cross-block correlations.
    """
    spec = BLOCK_SPECS[0]  # 3 blocks: sizes 3+2+3
    V, _ = _make_block_diagonal_cov(spec)

    # Add a small but non-zero off-block correlation so global_rho != 0
    rng = np.random.default_rng(42)
    n = V.shape[0]
    noise = rng.uniform(0.02, 0.06, (n, n))
    noise = (noise + noise.T) / 2
    np.fill_diagonal(noise, 0.0)
    V_noisy = V + noise
    # Restore PSD-ness via nearest-PSD projection
    eigvals, eigvecs = np.linalg.eigh(V_noisy)
    eigvals = np.maximum(eigvals, 1e-8)
    V_noisy = eigvecs @ np.diag(eigvals) @ eigvecs.T

    bd_ao = BlockDiagonalPortfolio(V_noisy, tier1_method='aware',    tier2_method='oblivious')
    bd_oa = BlockDiagonalPortfolio(V_noisy, tier1_method='oblivious', tier2_method='aware')
    bd_ao.calculate_portfolio_positions()
    bd_oa.calculate_portfolio_positions()

    assert not np.allclose(bd_ao.positions.ravel(), bd_oa.positions.ravel(), atol=1e-6), (
        "aware/oblivious and oblivious/aware gave identical weights — "
        "the tier2_method is not being applied."
    )


# --------------------------------------------------------------------------- #
#  Test 12 – _block_approximation recovers global_rho and rho_i from a known  #
#            block-structured matrix (paper eq. block.structure)               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", BLOCK_STRUCT_GENS, ids=_bd_gen_id)
def test_block_approximation_recovers_known_parameters(gen):
    """_block_approximation exactly recovers the parameters used to build the matrix.

    The paper (Section sec:block.approx) defines the block-diagonal approximation via:
        rho       = mean of all off-block correlation entries
        tilde_rho_i = mean of within-block off-diagonal entries for block i
        rho_i     = (tilde_rho_i - rho) / (1 - rho)

    This test constructs a covariance matrix whose Omega is exactly of the form
        Omega = rho * 1*1^T + (1-rho) * block_diag(Omega_1, ..., Omega_K)
    with each Omega_i = (1-rho_i)*I + rho_i*1*1^T, using known values of
    global_rho and rho_i.  It then injects the true cluster labels directly
    (bypassing the clustering step) and checks that _block_approximation
    recovers the exact input parameters up to floating-point precision.

    A failure here means the global_rho estimate, the tilde_rho_i computation,
    or the rho_i normalisation formula do not match the paper.
    """
    V, _, _ = gen.generate()
    global_rho = gen.global_rho
    true_clusters = gen.true_clusters

    bd = BlockDiagonalPortfolio(V)
    bd.clusters = true_clusters  # inject known clusters, bypassing _cluster()

    recovered_global_rho, block_info = bd._block_approximation()

    assert abs(recovered_global_rho - global_rho) < 1e-12, (
        f"global_rho: expected {global_rho}, got {recovered_global_rho:.15f}"
    )

    for block_id, (sigmas, rho_i) in enumerate(
            zip(gen.block_sigmas, gen.block_rhos), start=1):
        expected_rho_i = rho_i if len(sigmas) > 1 else 0.0
        got = block_info[block_id]['rho_i']
        assert abs(got - expected_rho_i) < 1e-12, (
            f"Block {block_id}: rho_i expected {expected_rho_i}, got {got:.15f}"
        )


# --------------------------------------------------------------------------- #
#  Test 13 – BD-direct on a true block-diagonal matrix matches LongOnlyMarkowitz
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec", BLOCK_SPECS, ids=_spec_id)
def test_direct_true_block_diagonal_matches_longonly(spec):
    """BD-direct (Proposition K.block) is exact on a true block-diagonal covariance.

    When the covariance is truly block-diagonal, global_rho = 0 and theta_hat = 0.
    The fixed-point system reduces to independent single-block problems, and the
    positions (theta_j - bar_i)^+ / ((1-rho_i)*sigma_j) are the exact KKT solution
    of the global long-only min-variance problem.

    It can be shown algebraically that sigma_i^2 = (1-rho_i)/C_i where
    C_i = sum_j (theta_j - bar_i)^+ * theta_j, which means the (1-rho_i) scaling
    is equivalent to inverse-variance cross-block weighting — the same result as
    BD-aware on the same matrix (Test 1).

    A failure here means either theta_hat is not converging to 0, the per-block
    bar_theta computation is wrong, or the (1-rho_i) factor is missing.
    """
    V, _ = _make_block_diagonal_cov(spec)

    bd = BlockDiagonalPortfolio(V, method='direct')
    bd.calculate_portfolio_positions()

    lo = LongOnlyMarkowitz(V)
    lo.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd.positions.ravel(),
        lo.positions.ravel(),
        atol=1e-4,
        err_msg="BD-direct on true block-diagonal differs from LongOnlyMarkowitz",
    )


# --------------------------------------------------------------------------- #
#  Test 14 – The two expressions for theta_hat agree at the fixed point        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", BLOCK_STRUCT_GENS, ids=_bd_gen_id)
def test_direct_theta_hat_expressions_agree(gen):
    """At the fixed point, the simple and long forms of theta_hat are equal.

    The paper gives two equivalent expressions (eq.fixed.point.block):

      Simple:  theta_hat = rho/(1-rho) * sum_i bar_i/rho_i
                         = rho/(1-rho) * sum_i S_i/(1+(n_i-1)*rho_i)

      Long:    theta_hat = rho / D * sum_i [sum_j max{theta_j, theta_hat+bar_i}]
                                          / (1+(n_i-1)*rho_i)
               where D = 1-rho + rho * sum_i n_i/(1+(n_i-1)*rho_i)

    The equivalence is an algebraic identity that holds at every fixed point, not
    just at the solution.  This test builds a matrix with the paper's exact block
    structure (bypassing the clustering step), runs the direct solver, and verifies
    that both expressions evaluate to the same theta_hat.

    A failure here means the fixed-point iteration did not converge, or there is
    a bug in computing S_i (the sum-of-maxima term).
    """
    V, _, _ = gen.generate()
    global_rho = gen.global_rho
    true_clusters = gen.true_clusters

    bd = BlockDiagonalPortfolio(V, method='direct')
    bd.clusters = true_clusters
    _, block_info = bd._block_approximation()
    bd._calculate_direct(global_rho, block_info)

    theta_hat = bd._theta_hat
    bar_thetas = bd._bar_thetas

    # ---- Simple form: rho/(1-rho) * sum_i S_i / (1+(n_i-1)*rho_i)
    simple_total = 0.0
    for c, info in sorted(block_info.items()):
        idx   = info['indices']
        rho_i = info['rho_i']
        n_i   = len(idx)
        phi_i = 1.0 / bd.sigma[idx] - theta_hat
        bar_i = bar_thetas[c]
        S_i   = float(np.sum(np.maximum(phi_i, bar_i)))
        simple_total += S_i / (1.0 + (n_i - 1) * rho_i)
    theta_hat_simple = global_rho / (1.0 - global_rho) * simple_total if global_rho > 1e-15 else 0.0

    # ---- Long form: rho/D * sum_i [sum_j max{theta_j, theta_hat+bar_i}] / (1+(n_i-1)*rho_i)
    D = 1.0 - global_rho
    long_num = 0.0
    for c, info in sorted(block_info.items()):
        idx   = info['indices']
        rho_i = info['rho_i']
        n_i   = len(idx)
        theta_i = 1.0 / bd.sigma[idx]
        bar_i   = bar_thetas[c]
        inner   = float(np.sum(np.maximum(theta_i, theta_hat + bar_i)))
        denom_i = 1.0 + (n_i - 1) * rho_i
        long_num += inner / denom_i
        D        += global_rho * n_i / denom_i
    theta_hat_long = global_rho / D * long_num if global_rho > 1e-15 else 0.0

    np.testing.assert_allclose(
        theta_hat_simple, theta_hat,
        atol=1e-10,
        err_msg=f"Simple form ({theta_hat_simple:.10f}) differs from solved theta_hat ({theta_hat:.10f})",
    )
    np.testing.assert_allclose(
        theta_hat_long, theta_hat,
        atol=1e-10,
        err_msg=f"Long form ({theta_hat_long:.10f}) differs from solved theta_hat ({theta_hat:.10f})",
    )


# --------------------------------------------------------------------------- #
#  Test 15 – BD-direct matches LongOnlyMarkowitz for any exact block structure #
#            (global_rho > 0, not just the truly block-diagonal case)          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("gen", BLOCK_STRUCT_GENS, ids=_bd_gen_id)
def test_direct_exact_block_structure_matches_longonly(gen):
    """BD-direct (Proposition K.block) is exact for any matrix of the paper's block form.

    Proposition K.block covers all matrices of the form
        Omega = (1-rho)*blockdiag(Omega_1,...,Omega_K) + rho*ee^T
    with Omega_i = (1-rho_i)*I + rho_i*ee^T — not just the rho=0 case of Test 13.

    The test constructs such a matrix with known rho > 0, injects the true cluster
    labels to bypass the heuristic clustering step (which is a separate concern),
    and verifies the direct solver matches long-only min-variance solution.

    A failure here means theta_hat is not correctly capturing the coupling between
    blocks introduced by the non-zero global correlation rho.
    """
    V, _, _ = gen.generate()
    global_rho = gen.global_rho
    true_clusters = gen.true_clusters

    # Bypass clustering: inject exact partition and call the solve directly
    bd = BlockDiagonalPortfolio(V, method='direct')
    bd.clusters = true_clusters
    recovered_rho, block_info = bd._block_approximation()
    bd._calculate_direct(recovered_rho, block_info)

    lo = LongOnlyMarkowitz(V)
    lo.calculate_portfolio_positions()

    np.testing.assert_allclose(
        bd.positions.ravel(),
        lo.positions.ravel(),
        atol=1e-4,
        err_msg=f"BD-direct (global_rho={global_rho}) differs from LongOnlyMarkowitz",
    )


# --------------------------------------------------------------------------- #
#  Test 16 – Corollary 1, Case 1: rho=0 → theta_hat=0, blocks decouple       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec", BLOCK_SPECS, ids=_spec_id)
def test_corollary1_case1_rho_zero_decouples_blocks(spec):
    """Corollary 1 (Case 1): global_rho=0 implies theta_hat=0 and blocks are independent.

    On a truly block-diagonal covariance the direct solver must satisfy:
      (a) theta_hat = 0
      (b) within each block, relative weights match CorrelationAwareBlock(epsilon=0)
          applied to the block sub-covariance (Proposition 1 per block)
      (c) full portfolio matches LongOnlyMarkowitz
    """
    V, idx_list = _make_block_diagonal_cov(spec)

    n = V.shape[0]
    clusters = np.zeros(n, dtype=int)
    for block_id, idx in enumerate(idx_list, start=1):
        clusters[idx] = block_id

    bd = BlockDiagonalPortfolio(V, method='direct')
    bd.clusters = clusters
    _, block_info = bd._block_approximation()
    bd._calculate_direct(0.0, block_info)

    # (a) theta_hat must be 0
    assert abs(bd._theta_hat) < 1e-12, (
        f"theta_hat should be 0 when global_rho=0, got {bd._theta_hat:.2e}"
    )

    # (b) within-block relative weights match CorrelationAwareBlock(epsilon=0)
    bd_w = bd.positions.ravel()
    for idx in idx_list:
        block_w = bd_w[idx]
        if block_w.sum() < 1e-15:
            continue
        block_w_norm = block_w / block_w.sum()

        cab = CorrelationAwareBlock(V[np.ix_(idx, idx)], epsilon=0)
        cab.calculate_portfolio_positions()

        np.testing.assert_allclose(
            block_w_norm,
            cab.positions.ravel(),
            atol=1e-10,
            err_msg="Within-block relative weights differ from CorrelationAwareBlock(epsilon=0)",
        )

    # (c) full positions match LongOnlyMarkowitz
    lo = LongOnlyMarkowitz(V)
    lo.calculate_portfolio_positions()
    np.testing.assert_allclose(
        bd_w,
        lo.positions.ravel(),
        atol=1e-4,
        err_msg="BD-direct (global_rho=0) full positions differ from LongOnlyMarkowitz",
    )


# --------------------------------------------------------------------------- #
#  Test 17 – Corollary 1, Case 2: all rho_i=0 → reduces to Proposition 1    #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("sigmas,rho,block_sizes", [
    ([0.10, 0.20, 0.15, 0.25, 0.30], 0.40, [2, 3]),
    ([0.05, 0.20, 0.50, 0.10, 0.30], 0.30, [2, 2, 1]),
    ([0.10, 0.12, 0.14, 0.20, 0.25, 0.08], 0.50, [3, 3]),
], ids=["n5_rho0.4_2+3", "n5_rho0.3_2+2+1", "n6_rho0.5_3+3"])
def test_corollary1_case2_all_rho_i_zero_reduces_to_prop1(sigmas, rho, block_sizes):
    """Corollary 1 (Case 2): all rho_i=0 → BD-direct reduces to Proposition 1.

    When every within-block correlation is zero, the block structure collapses to
        Omega = (1-rho)*I + rho*11^T
    which is a single constant-correlation matrix with parameter rho.  The
    fixed-point system then equals Proposition 1 applied to all assets, so:
      (a) all bar_theta_i = 0
      (b) BD-direct matches CorrelationAwareBlock(epsilon=0) on the full matrix
      (c) BD-direct matches LongOnlyMarkowitz on the full matrix
    """
    sigmas = np.array(sigmas)
    n = len(sigmas)
    Omega = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    V = np.diag(sigmas) @ Omega @ np.diag(sigmas)

    clusters = np.zeros(n, dtype=int)
    start = 0
    for block_id, size in enumerate(block_sizes, start=1):
        clusters[start:start + size] = block_id
        start += size

    bd = BlockDiagonalPortfolio(V, method='direct')
    bd.clusters = clusters
    recovered_rho, block_info = bd._block_approximation()
    bd._calculate_direct(recovered_rho, block_info)

    # (a) all bar_theta_i must be 0 (rho_i=0 for every block)
    for c, bar_i in bd._bar_thetas.items():
        assert abs(bar_i) < 1e-12, (
            f"Block {c}: bar_theta should be 0 when rho_i=0, got {bar_i:.2e}"
        )

    # (b) matches CorrelationAwareBlock(epsilon=0) on full matrix (Proposition 1)
    cab = CorrelationAwareBlock(V, epsilon=0)
    cab.calculate_portfolio_positions()
    np.testing.assert_allclose(
        bd.positions.ravel(),
        cab.positions.ravel(),
        atol=1e-8,
        err_msg="BD-direct (all rho_i=0) differs from CorrelationAwareBlock(epsilon=0)",
    )

    # (c) matches LongOnlyMarkowitz on full matrix
    lo = LongOnlyMarkowitz(V)
    lo.calculate_portfolio_positions()
    np.testing.assert_allclose(
        bd.positions.ravel(),
        lo.positions.ravel(),
        atol=1e-4,
        err_msg="BD-direct (all rho_i=0) differs from LongOnlyMarkowitz",
    )
