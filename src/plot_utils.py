import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def truncate_colormap(cmap_name, min_val=0.0, max_val=0.5):
    cmap = plt.get_cmap(cmap_name)
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc_{cmap_name}", cmap(np.linspace(min_val, max_val, 256))
    )


def plot_heatmap(matrix, title, tickers=None, ax=None, save=False, plot_name='heatmap', vmin=None, vmax=None, font_size=None):
    tickers = list(range(matrix.shape[0])) if tickers is None else tickers
    num_assets = len(tickers)

    if font_size is None:
        font_size = max(5, 12 * (50 / max(num_assets, 50)))

    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    cmap = truncate_colormap("inferno_r", 0., 0.75)
    sns.heatmap(
        matrix, annot=False, cmap=cmap, vmin=vmin, vmax=vmax,
        linewidths=0.5, square=True,
        xticklabels=tickers, yticklabels=tickers, ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.collections[0].colorbar.ax.tick_params(labelsize=font_size)

    plt.tight_layout()
    plt.draw()

    if save:
        plt.savefig(f"figures/{plot_name}.pdf", format="pdf", bbox_inches="tight", dpi=1200)
    if show:
        plt.show()
