# fragility-minimum-variance

A runnable companion to:

> Ovalle, D., Laird, C.D., Grossmann, I.E., & Peña, J. (2026). *Fragility of
> Minimum-Variance Portfolios.* Carnegie Mellon University.

Minimum-variance portfolios are known to be highly sensitive to covariance
estimation error. The paper shows this fragility arises because the optimal
solution is **piecewise-defined**, with fixed-point thresholds in the
closed-form long-only minimum-variance solution determining the active set:
when a threshold approaches one of the breakpoints defined by the ordered
inverse volatilities, small perturbations in volatilities or correlations
flip assets in/out of the active set (via the positive-part operator),
causing discontinuous changes in the support of the optimal portfolio and,
consequently, discontinuous reallocations of weight. Instability is
therefore not solely a consequence of covariance estimation error itself,
but of how that error interacts with the threshold structure of the
optimization problem.

This motivates **structured shrinkage** methods that modify the correlation
structure so that the induced thresholds remain uniformly separated from
active-set breakpoints, rather than modifying the optimization problem
itself:

- **Correlation-oblivious shrinkage** sets the threshold to zero, which is
  equivalent to shrinking the correlation to zero (inverse-variance
  weights). This removes the truncation mechanism entirely: the active set
  becomes the full asset set, invariant to perturbations, at the cost of a
  fully dense portfolio.
- **Correlation-aware shrinkage** instead computes the smallest correlation
  shrinkage for which the induced threshold remains at least `epsilon` away
  from every breakpoint (equivalently, shrinks the correlation to an implied
  `rho_epsilon`). This preserves the positive-part truncation, and
  therefore sparsity, while still stabilizing the active set.

The framework extends from a single constant-correlation block to
block-diagonal structures via a two-tier scheme (within-block, then
cross-block shrinkage), and from there to *any* covariance matrix via
clustering-based block-diagonal approximation, which is what this repo
backtests, following the paper's **Block-Diagonal Shrinkage (BDS)** family.
The block-diagonal approximation itself is already a form of shrinkage: it
replaces the original covariance matrix with a structured approximation
parameterized by one within-block correlation and one cross-block
correlation per cluster, regardless of how the portfolio weights are then
computed on top of it.

**Methods compared**

| Method | Description |
| --- | --- |
| `LongOnly`   | Long-only global minimum-variance (Markowitz), no structure assumed |
| `EqualWeight`| 1/n portfolio |
| `InvVol`     | Inverse-variance portfolio (ignores all correlation) |
| `BDS-Obliv`  | correlation-oblivious block shrinkage: adds an allocation-level shrinkage step on top of the block-diagonal approximation, ignoring cross-block correlation in the allocation itself |
| `BDS-Aware`  | correlation-aware block shrinkage: same allocation-level shrinkage step, but incorporates within-block correlation, allowing sparser portfolios |
| `BDS-Direct` | direct block-diagonal shrinkage: solves the long-only minimum-variance problem (using the fixed-point system, Proposition 2) directly on the block-diagonal covariance approximation , with no further allocation-level shrinkage on top |

All three BDS variants are shrinkage methods, they differ only in *where*
the regularization is introduced. BDS-Obliv and BDS-Aware combine the
structural shrinkage of the block-diagonal approximation with an additional,
allocation-level shrinkage step. BDS-Direct places all of the regularization
at the covariance-approximation stage and then solves the resulting problem
exactly, with no further adjustment, the structural shrinkage alone is
enough to stabilize it.

`heterogeneous_vol=False` (default) vs. `heterogeneous_vol=True` in `main()`
switches between a homogeneous-volatility and a heterogeneous-volatility
synthetic universe, the two regimes the paper studies these methods under.

## Install

Requires Python >= 3.9.

```bash
pip install -r requirements.txt
```

`LongOnly` solves a small QP; the default solver is
[`pounce`](https://github.com/jkitchin/pounce) (`pounce-solver` on PyPI, in
`requirements.txt`), an in-process interior-point QP solver that needs no
external solver executable or license.

Pass `solver='ipopt'`, `solver='gurobi_direct'`, or any other
[Pyomo](https://www.pyomo.org/)-supported NLP/QP solver name to
`LongOnlyMarkowitz` instead if you'd rather use one you already have
installed (e.g. `conda install -c conda-forge ipopt`).

## Run

```bash
python main.py
```

This generates a synthetic 12-block covariance matrix, runs a rolling-window
backtest (sample covariance re-estimated at each step), and produces:

- a bar chart of the `BDS-Direct` portfolio's positions on the true covariance,
  plus its clustering dendrogram and per-block threshold diagnostics,
- a printed table of backtest metrics (realized volatility, turnover,
  concentration, diversification ratio, area between realized and true
  variance, ...) for all six methods,
- variance-tracking, cumulative-return, and position-stacking plots per
  method.

Set `heterogeneous_vol=True` in the `main()` call to use blocks with
dispersed within-block volatilities instead of the default homogeneous-vol
setup.

## Figures

```bash
python figures.py
```

Reproduces five small, illustrative instances from the paper as standalone
figures (saved as PDFs under `figures/`):

| Function | Instance | Outputs |
| --- | --- | --- |
| `fig_2x2_unstable`  | near-singular 2x2 (`rho=0.99`)                              | `var_2x2_unstable.pdf`, `por_2x2_unstable.pdf` |
| `fig_3x3_unstable`  | 3x3 constant-correlation (`rho=0.48`, heterogeneous sigmas) | `var_3x3_unstable.pdf`, `por_3x3_unstable.pdf` |
| `fig_cov_ex3`       | 7x7 block-diagonal (3 blocks, cross-block `rho=0.6`)        | `cov_ex3.pdf`, `var_block_ex3.pdf`, `por_block_ex3.pdf` |
| `fig_8x8_one`       | 8x8 constant-correlation (`rho=0.90`), uniform sigmas       | `var_8x8_one.pdf`, `por_8x8_one.pdf` |
| `fig_8x8_onetwo`    | 8x8 constant-correlation (`rho=0.90`), two-group sigmas     | `var_8x8_onetwo.pdf`, `por_8x8_onetwo.pdf` |

Each is a standalone, callable function (`python -c "from figures import fig_cov_ex3; fig_cov_ex3()"`).
Running the module reproduces all five.

## Tests

```bash
pytest
```

Covers the closed-form solvers behind `BDS-Aware`/`BDS-Direct` against known
cases (true block-diagonal matrices, single/singleton clusters, equal
volatilities, ...) with checks against `LongOnlyMarkowitz` and
`IgnoreAllCorrelation` as ground truth, plus an end-to-end sanity check of
`BacktestEngine` (weights sum to 1, long-only, finite metrics) for all six
methods used in `main.py`.

An additional cross-check, `tests/test_pounce_vs_ipopt.py`, verifies that
`LongOnlyMarkowitz`'s default solver
([`pounce`](https://github.com/jkitchin/pounce)) agrees with the `ipopt`
solver path. It's skipped automatically if IPOPT isn't available on your
system. IPOPT is an optional alternative, not a dependency of this repo.

## Layout

```
main.py            - entry point: builds the DGP, runs the backtest, plots results
figures.py         - standalone reproductions of 5 small illustrative paper instances
src/
  generator.py     - synthetic covariance generators (block-diagonal DGP + test fixtures)
  portfolio.py     - portfolio classes (LongOnly, EqualWeight, InvVol, BDS variants)
  estimator.py     - covariance estimator (sample covariance only)
  backtest.py       - rolling-window backtest engine
  evaluator.py     - backtest metrics
  visualizer.py    - plotting
  validator.py     - input validation helpers
tests/             - pytest suite (BDS solver correctness + backtest sanity checks)
```

## Citation

See [CITATION.cff](CITATION.cff) (placeholder, to be updated once the paper
has a venue/DOI). A BibTeX entry to fill in is also at the bottom of `main.py`.

## License

MIT. See [LICENSE](LICENSE).
