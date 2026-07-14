import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import simpson

from .backtest import BacktestResult
from .portfolio import Portfolio


class PortfolioVisualizer:
    def __init__(self, assets: list = None):
        self.assets = assets
        self.pal = sns.color_palette("Set2")

    def plot_single_portfolio(self, portfolio: Portfolio, ax=None):
        if portfolio.positions is None:
            portfolio.calculate_portfolio_positions()

        portfolio.calculate_portfolio_return()
        portfolio.calculate_portfolio_variance()

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
            show = True
        else:
            show = False

        assets = portfolio.assets
        positions = portfolio.positions.flatten()
        colors = ["green" if w > 0 else "red" for w in positions]

        ax.barh(assets, positions, color=colors)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title(
            f"{portfolio.__class__.__name__} | "
            f"Exp Ret: {round(100 * portfolio.portfolio_return, 4)}% | "
            f"Vol: {round(np.sqrt(252*portfolio.variance), 4)}"
        )
        ax.set_xlabel("Portfolio Weight")

        num_assets = len(assets)
        max_display = 50
        if num_assets > max_display:
            ax.set_yticklabels([])
        else:
            font_size = max(5, 12 * (max_display / max(num_assets, max_display)))
            ax.tick_params(axis='y', labelsize=font_size)

        if num_assets <= 40:
            for j, v in enumerate(positions):
                ax.text(v, j, f"{v:.1%}", va='center', ha="left" if v >= 0 else "right", color="black")

        min_x, max_x = positions.min(), positions.max()
        if min_x == max_x:
            ax.set_xlim(min_x - 0.1, max_x + 0.1)
        else:
            padding = (max_x - min_x) * 0.2
            ax.set_xlim(min_x - padding, max_x + padding)

        if show:
            plt.show()

    def plot_variance_performance(
        self,
        results: dict[str, BacktestResult],
        deltas,
        benchmark=None,
        plot_estimated: bool = True,
        save: bool = False,
        plot_name: str = '',
        figsize: tuple = None,
    ):
        cols = min(3, len(results))
        rows = (len(results) + cols - 1) // cols
        if figsize is None:
            figsize = (cols * 5, rows * 4)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True)
        axes = np.array(axes).flatten()

        benchmark_var_vector = (
            np.ones(len(deltas)) * benchmark.variance if benchmark is not None else None
        )

        area = np.nan
        for i, (name, result) in enumerate(results.items()):
            if benchmark_var_vector is not None:
                axes[i].plot(deltas, benchmark_var_vector, color='red', label='True')
                area = simpson(np.array(result.actual_horizon) - benchmark_var_vector)
                axes[i].fill_between(
                    deltas, result.actual_horizon, benchmark_var_vector, color='blue', alpha=0.2
                )
                tail_str = f' | Area: {round(np.abs(area), 3)}'
            else:
                avg = np.sqrt(np.mean(result.estimate_horizon)) * np.sqrt(252)
                tail_str = f' | Inst. Vol. Est: {round(avg, 5)}'

            if getattr(result, 'actual_horizon', []):
                axes[i].plot(deltas, result.actual_horizon, color='blue', label='Realized')
            if plot_estimated and result.estimate_horizon:
                axes[i].plot(deltas, result.estimate_horizon, color='gray', label='Estimated', alpha=0.7)

            axes[i].set_xlabel("Day")
            axes[i].set_ylabel("Variance")
            axes[i].legend(loc='upper left')
            axes[i].set_title(name + tail_str)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            plt.savefig(f"figures/{plot_name}.pdf", format="pdf", bbox_inches="tight", dpi=1200)
        plt.show()
        return np.abs(area)

    def plot_return_performance(
        self,
        results: dict[str, BacktestResult],
        deltas,
        save: bool = False,
        plot_name: str = '',
    ):
        cols = min(3, len(results))
        rows = (len(results) + cols - 1) // cols + 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), sharey=True)
        axes = np.array(axes).flatten()

        for i, (name, result) in enumerate(results.items()):
            ret_arr = np.array(result.return_horizon)
            axes[i].plot(deltas, np.cumsum(ret_arr), color='blue', label='Return')
            axes[i].plot(deltas, np.zeros(len(deltas)), color='red')
            vol = np.std(ret_arr, ddof=1) * np.sqrt(252)
            axes[i].set_xlabel("Day")
            axes[i].set_ylabel("Expected Return")
            axes[i].legend(loc='upper right')
            axes[i].set_title(name + f' | Ann. Vol.: {round(vol, 5)}')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            plt.savefig(f"figures/{plot_name}.pdf", format="pdf", bbox_inches="tight", dpi=1200)
        plt.show()

    def plot_position_performance(
        self,
        results: dict[str, BacktestResult],
        deltas,
        assets: list,
        benchmark=None,
        plot_benchmark: bool = True,
        save: bool = False,
        plot_name: str = '',
        figsize: tuple = None,
    ):
        adder = 1 if (benchmark is not None and plot_benchmark) else 0
        cols = min(3, len(results))
        rows = (len(results) + cols - 1) // cols + adder
        if figsize is None:
            figsize = (cols * 5, (rows + 1) * 4)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=True)
        axes = np.array(axes).flatten()

        for i, (name, result) in enumerate(results.items()):
            pos = result.position_horizon
            x_pos = pos * (pos > 0)
            x_neg = pos * (pos < 0)

            axes[i].stackplot(deltas, *x_pos, colors=self.pal, labels=assets, alpha=1)
            axes[i].stackplot(deltas, *x_neg, colors=self.pal, alpha=1)

            avg_turnover = np.abs(np.diff(pos, axis=1)).sum(axis=0).mean()
            avg_nz = (np.abs(pos) > 1e-6).sum(axis=0).mean()

            axes[i].set_xlabel("Day")
            axes[i].set_ylabel("Weights")
            axes[i].set_title(name + f' | TO: {round(avg_turnover * 100, 2)}% | Avg. NZ: {round(avg_nz, 2)}')
            if len(assets) <= 8:
                axes[i].legend(loc='lower left', fontsize=8 if len(assets) >= 4 else None)

        if benchmark is not None and plot_benchmark:
            bench_axis = axes[i + 1]
            bp = benchmark.positions
            bench_axis.stackplot(deltas, *(bp * (bp > 0)), colors=self.pal, labels=assets, alpha=1)
            bench_axis.stackplot(deltas, *(bp * (bp < 0)), colors=self.pal, alpha=1)
            avg_nz_bench = (np.abs(bp) > 1e-6).sum().item()
            bench_axis.set_xlabel("Day")
            bench_axis.set_ylabel("Weights")
            bench_axis.set_title(f'"True" Portfolio | TO: 0.0% | Avg. NZ: {round(avg_nz_bench, 2)}')
            if len(assets) <= 8:
                bench_axis.legend(loc='lower left')
            cleanup_start = i + 2
        else:
            cleanup_start = i + 1

        for j in range(cleanup_start, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            plt.savefig(f"figures/{plot_name}.pdf", format="pdf", bbox_inches="tight", dpi=1200)
        plt.show()
