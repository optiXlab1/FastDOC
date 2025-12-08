import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial",
    "mathtext.bf": "Arial",
})

# ============================================================
# CSV loader
# ============================================================
def load_csv(basename: str) -> pd.DataFrame:
    """Load a CSV file, supporting both the base name and the '.csv' extension."""
    directory = 'numerical_result'

    for candidate in [basename, f"{basename}.csv"]:
        candidate_path = os.path.join(directory, candidate)
        if os.path.exists(candidate_path):
            return pd.read_csv(candidate_path)

    raise FileNotFoundError(
        f"Could not find {basename} or {basename}.csv in {directory}"
    )


# ============================================================
# Plotting function
# ============================================================
def plot_bench_logy(df, xcol, title, xlabel, ax, index, savepath=None):
    """Plot computation times (log scale) with uniform markers and thick lines."""
    x = np.array(df[xcol])

    idoc_mu = np.array(df['IDOC_mean'])
    FastDOC_mu = np.array(df['FastDOC_mean'])
    SafePDP_mu = np.array(df['SafePDP_mean'])

    idoc_std = df['IDOC_std'] if 'IDOC_std' in df.columns else np.zeros_like(idoc_mu)
    FastDOC_std = df['FastDOC_std'] if 'FastDOC_std' in df.columns else np.zeros_like(FastDOC_mu)
    SafePDP_std = df['SafePDP_std'] if 'SafePDP_std' in df.columns else np.zeros_like(SafePDP_mu)

    COLOR_FastDOC = '#CA0E12'
    COLOR_IDOC = '#25377F'
    COLOR_SafePDP = '#F6BD12'

    start = 1

    # Plot first two points without error bar
    ax.plot(x[:start+1], FastDOC_mu[:start+1], '.-', color=COLOR_FastDOC, linewidth=1.8, markersize=4)
    ax.plot(x[:start+1], idoc_mu[:start+1], '.-', color=COLOR_IDOC, linewidth=1.8, markersize=4)
    ax.plot(x[:start+1], SafePDP_mu[:start+1], '.-', color=COLOR_SafePDP, linewidth=1.8, markersize=4)

    # Errorbars for remaining points
    ax.errorbar(x[start:], FastDOC_mu[start:], yerr=FastDOC_std[start:], fmt='.-',
                label='FastDOC', color=COLOR_FastDOC, linewidth=2.2, markersize=4, capsize=4)
    ax.errorbar(x[start:], idoc_mu[start:], yerr=idoc_std[start:], fmt='.-',
                label='IDOC', color=COLOR_IDOC, linewidth=2.2, markersize=4, capsize=4)
    ax.errorbar(x[start:], SafePDP_mu[start:], yerr=SafePDP_std[start:], fmt='.-',
                label='SafePDP', color=COLOR_SafePDP, linewidth=2.2, markersize=4, capsize=4)

    ax.set_yscale('log')
    ax.set_xlabel(xlabel)

    if index == 0:
        ax.set_ylabel('Computation Time (s)')

    # Only for the 3rd subplot: show only 1e-1 and 5e-2
    if index == 2:
        ticks = [5e-2, 1e-1, 2e-1]
        ax.set_yticks(ticks)
        ax.set_yticklabels([r"$5\times 10^{-2}$", r"$10^{-1}$", r"$2\times 10^{-1}$"])

    ax.xaxis.grid(False)
    ax.yaxis.grid(True, which='major', linestyle=(0, (6, 4)), linewidth=0.8, color='#666666', alpha=0.75)
    ax.yaxis.grid(True, which='minor', linestyle=(0, (4, 4)), linewidth=0.5, color='#999999', alpha=0.5)

    # ax.yaxis.set_major_locator(LogLocator(base=10.0))
    # ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))

    ax.legend()


# ============================================================
# Dataset processing
# ============================================================
def process_dataset(name, xcol, title, xlabel, ax, index):
    """Load dataset, plot, compute speedups + relative errors."""
    df = load_csv(name)
    df = df.sort_values(by=xcol).reset_index(drop=True)

    # Plot to SVG
    svg_path = f"{name}.svg"
    plot_bench_logy(df, xcol, title, xlabel, ax, index, svg_path)

    # Compute speedups
    eps = 1e-12
    FastDOC_vs_IDOC = (df["IDOC_mean"] / (df["FastDOC_mean"] + eps)).mean()
    FastDOC_vs_SafePDP = (df["SafePDP_mean"] / (df["FastDOC_mean"] + eps)).mean()

    # Relative errors (optional)
    if "IDOC_relerr_mean" in df.columns and "SafePDP_relerr_mean" in df.columns:
        IDOC_relerr_mean = df["IDOC_relerr_mean"].mean()
        SafePDP_relerr_mean = df["SafePDP_relerr_mean"].mean()
    else:
        IDOC_relerr_mean = SafePDP_relerr_mean = np.nan

    return {
        "dataset": name,
        "rows": len(df),
        "svg_path": svg_path,
        "FastDOC speedup vs IDOC (avg)": float(FastDOC_vs_IDOC),
        "FastDOC speedup vs SafePDP (avg)": float(FastDOC_vs_SafePDP),
        "IDOC_relerr_mean": float(IDOC_relerr_mean),
        "SafePDP_relerr_mean": float(SafePDP_relerr_mean),
        "df": df
    }


# ============================================================
# Main
# ============================================================
def main():
    datasets = {
        "bench_time_vs_N": {
            "xcol": "N",
            "title": "Compute Time vs Horizon",
            "xlabel": "Horizon N"
        },
        "bench_time_vs_ns": {
            "xcol": "ns",
            "title": "Compute Time vs State Dimension",
            "xlabel": "State Dimension n"
        },
        "bench_time_vs_ntheta": {
            "xcol": "ntheta",
            "title": "Compute Time vs Parameter Number",
            "xlabel": "Number of parameters d"
        },
    }

    summaries = []
    all_rows = []

    fig, axs = plt.subplots(1, 3, figsize=(14.4, 3.6), dpi=150)

    titles = [
        "(a) Horizon length scalability.",
        "(b) Model size scalability.",
        "(c) Parameter size scalability.",
    ]

    for index, (name, meta) in enumerate(datasets.items()):
        try:
            result = process_dataset(
                name,
                meta["xcol"],
                meta["title"],
                meta["xlabel"],
                axs[index],
                index
            )

            # 设置子图标题并放在下方
            axs[index].set_title(titles[index], fontsize=18, pad=-20, y=-0.3)

            # 设置边框宽度为1.5
            for spine in axs[index].spines.values():
                spine.set_linewidth(1.5)

            summaries.append(result)
            all_rows.append(result["df"][["IDOC_mean", "FastDOC_mean", "SafePDP_mean"]])

        except Exception as e:
            print(f"[WARN] {name}: {e}")

    plt.tight_layout()
    plt.savefig("numerical_result/bench_results.svg", format="svg")
    plt.show()

    # Global averages
    if all_rows:
        cat = pd.concat(all_rows, ignore_index=True)
        eps = 1e-12
        overall_FastDOC_vs_IDOC = float((cat["IDOC_mean"] / (cat["FastDOC_mean"] + eps)).mean())
        overall_FastDOC_vs_SafePDP = float((cat["SafePDP_mean"] / (cat["FastDOC_mean"] + eps)).mean())


if __name__ == "__main__":
    main()
