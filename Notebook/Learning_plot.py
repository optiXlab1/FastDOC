# analyze_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

from LearningExperiment.Learning_exp import Bicycle_Model, build_coc_from_env, mpc_rollout
from LearningExperiment.util import RoutePredictor


RESULT_DIR = "learning_result"
os.makedirs(RESULT_DIR, exist_ok=True)


# ================================
# 1. Training Loss
# ================================
def plot_losses(npz_path: str, out_svg: str = None, ax=None):
    """Plot loss curves (FastDOC/FastDOC, IDOC, PDP) from saved NPZ, save as SVG (2.0x2.0)."""
    z = np.load(npz_path, allow_pickle=True)
    loss_FastDOC = z["loss_FastDOC"]
    loss_IDOC = z["loss_IDOC"]
    loss_SafePDP = z["loss_SafePDP"]

    COLOR_OUR = "#CA0E12"
    COLOR_IDOC = "#25377F"
    COLOR_PDP = "#F6BD12"

    LS_SOLID = "-"
    LS_DASH = "--"

    iters = np.arange(len(loss_FastDOC))
    internal_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=120)
        internal_fig = True

    ax.plot(iters, loss_FastDOC, label="FastDOC",
            color=COLOR_OUR, linestyle=LS_SOLID, linewidth=1.8)
    ax.plot(iters, loss_IDOC, label="IDOC",
            color=COLOR_IDOC, linestyle=LS_SOLID, linewidth=1.8)
    ax.plot(iters, loss_SafePDP, label="SafePDP",
            color=COLOR_PDP, linestyle=LS_DASH, linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    title_tag = os.path.basename(npz_path).replace("training_results_all_", "").replace(".npz", "")
    ax.set_title(f"Training Loss - {title_tag}")
    ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.6)
    ax.legend(loc="upper right")
    if internal_fig:
        plt.tight_layout()
        if out_svg is None:
            out_svg = f"loss_from_npz_{title_tag}.svg"
        plt.savefig(out_svg, format="svg", bbox_inches="tight")
        print(f"[plot_losses] saved: {out_svg}")
        plt.close()


# ================================
# 2. Build/Solve Time Bar
# ================================
def plot_time_bars(npz_path: str, out_svg: str = None, ax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    z = np.load(npz_path, allow_pickle=True)
    methods = ["FastDOC", "IDOC", "SafePDP"]
    COLORS = {"FastDOC": "#CA0E12", "IDOC": "#25377F", "SafePDP": "#F6BD12"}

    build_avg = np.array([float(z[f"build_time_avg_{m}"]) * 1e3 for m in methods])
    solve_avg = np.array([float(z[f"solve_time_avg_{m}"]) * 1e3 for m in methods])

    n_methods = len(methods)
    width = 0.95
    group_gap = 1
    x_build = np.arange(n_methods)
    x_solve = x_build + n_methods + group_gap

    internal_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=150)
        internal_fig = True

    ax.set_facecolor("#ffffff")

    for i, m in enumerate(methods):
        ax.bar(x_build[i], build_avg[i], width, color=COLORS[m], edgecolor="none")
        ax.bar(x_solve[i], solve_avg[i], width, color=COLORS[m], edgecolor="none")

        ax.text(x_build[i], build_avg[i] + 0.01 * max(build_avg),
                f"{build_avg[i]:.2f}", ha="center", va="bottom", fontsize=12, color=COLORS[m])
        ax.text(x_solve[i], solve_avg[i] + 0.01 * max(solve_avg),
                f"{solve_avg[i]:.2f}", ha="center", va="bottom", fontsize=12, color=COLORS[m])

    mid_build = np.mean(x_build)
    mid_solve = np.mean(x_solve)
    ax.set_xticks([mid_build, mid_solve])
    ax.set_xticklabels(["Build", "Solve"], fontsize=16, fontweight="bold")

    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, max(np.concatenate([build_avg, solve_avg])) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in methods]
    leg = ax.legend(handles, methods, loc="upper right")

    ax.set_title("Backward Time (ms)")

    if internal_fig:
        plt.tight_layout()
        title_tag = os.path.basename(npz_path).replace(".npz", "")
        if out_svg is None:
            out_svg = os.path.join(RESULT_DIR, f"time_bars_grouped_{title_tag}.svg")
        plt.savefig(out_svg, format="svg", bbox_inches="tight")
        print(f"[plot_time_bars] saved: {out_svg}")
        plt.close()

# ================================
# 3. MPC Rollout Comparison
# ================================
def rollout_and_compare(npz_path, vehicle_csv, route_csv, x_variable="t", y_variable="v",
                        save_prefix="mpc_eval", ax=None):
    import matplotlib as mpl
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Polygon
    import matplotlib.lines as mlines

    z = np.load(npz_path, allow_pickle=True)
    dt = float(z["dt"])
    start_index, end_index = [int(v) for v in z["road_range"]]
    title_tag = os.path.basename(npz_path).replace(".npz", "")

    rollout_root = os.path.join("learning_result", f"rollout_{title_tag}")
    csv_dir = os.path.join(rollout_root, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    vehicle_data = pd.read_csv(vehicle_csv)
    rp = RoutePredictor(route_csv)
    env = Bicycle_Model()
    env.initDyn()
    env.initCost()
    env.initConstraints()
    coc = build_coc_from_env(env, dt=dt, gamma=1e-2)

    df_real = vehicle_data.iloc[start_index:end_index].copy().reset_index(drop=True)
    if x_variable == "t":
        x_real = np.arange(len(df_real)) * dt
    else:
        x_real = df_real[x_variable].to_numpy()
    y_real = df_real[y_variable].to_numpy()

    key = "params_FastDOC"
    params_traj = np.asarray(z[key], dtype=float)
    n_total = len(params_traj)
    idxs = [1, 2, 5, 10] + [i for i in range(20, 101, 20)] + [i for i in range(200, n_total, 200)]
    idxs = [i for i in idxs if i < n_total]
    if idxs[-1] != n_total - 1:
        idxs.append(n_total - 1)
    selected_params = [params_traj[i] for i in idxs]

    c_init = np.array(to_rgb("#8BCFED"))
    c_final = np.array(to_rgb("#25377F"))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("grad_FastDOC_blue", [c_init, c_final])
    colors_traj = [cmap(i / (len(selected_params) - 1)) for i in range(len(selected_params))]
    alphas = np.linspace(0.2, 0.8, len(selected_params))
    line_widths = np.linspace(0.5, 1.5, len(selected_params))

    rollout_dfs = []
    for i, params in enumerate(selected_params):
        cache_file = os.path.join(csv_dir, f"step_{idxs[i]}.csv")
        if os.path.exists(cache_file):
            df_tmp = pd.read_csv(cache_file)
        else:
            df_tmp = mpc_rollout("FastDOC", params, start_index, end_index, 5, dt,
                                 vehicle_data, rp, coc)
            df_tmp.to_csv(cache_file, index=False)
        rollout_dfs.append(df_tmp)

    internal_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=240)
        internal_fig = True

    df_init, df_final = rollout_dfs[0], rollout_dfs[-1]
    if x_variable == "t":
        x_init = (df_init["index"].to_numpy() - start_index) * dt
        x_final = (df_final["index"].to_numpy() - start_index) * dt
    else:
        x_init = df_init[x_variable].to_numpy()
        x_final = df_final[x_variable].to_numpy()
    y_init, y_final = df_init[y_variable].to_numpy(), df_final[y_variable].to_numpy()

    min_len = min(len(x_init), len(x_final))
    xi, yi = x_init[:min_len], y_init[:min_len]
    xf, yf = x_final[:min_len], y_final[:min_len]

    poly_x = np.concatenate([xi, xf[::-1]])
    poly_y = np.concatenate([yi, yf[::-1]])
    verts = np.column_stack([poly_x, poly_y])
    poly = Polygon(verts, closed=True, facecolor="#A1A1A1", edgecolor="none", alpha=0.25)
    ax.add_patch(poly)

    for df_tmp, c, alpha, lw in zip(rollout_dfs[:-1], colors_traj[:-1], alphas[:-1], line_widths[:-1]):
        if x_variable == "t":
            x = (df_tmp["index"].to_numpy() - start_index) * dt
        else:
            x = df_tmp[x_variable].to_numpy()
        y = df_tmp[y_variable].to_numpy()
        ax.plot(x, y, color=c, alpha=alpha, linewidth=lw, zorder=1)

    ax.plot(xf, yf, color="#25377F", linewidth=1.5, zorder=2, label="Final")
    ax.plot(x_real, y_real, color="#CA0E12", linestyle="--", linewidth=1.2, zorder=3, label="Human")

    import matplotlib.lines as mlines
    line_init = mlines.Line2D([], [], color="#8BCFED", lw=1.6, label="Init")
    line_final = mlines.Line2D([], [], color="#25377F", lw=1.6, label="Final")
    line_human = mlines.Line2D([], [], color="#CA0E12", lw=1.6, linestyle="--", label="Human")

    ax.legend(handles=[line_init, line_final, line_human])

    unit_map = {"v": "Velocity (m/s)", "a": "Acceleration (m/s²)", "x": "Position X (m)", "y": "Position Y (m)"}
    xlabel = "Time (s)" if x_variable == "t" else x_variable
    xlabel = unit_map.get(xlabel, xlabel)
    ylabel = unit_map.get(y_variable, y_variable)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{x_variable}-{y_variable} Gradient Rollout ({title_tag})")
    ax.grid(True, linestyle="--", alpha=0.6)

    if internal_fig:
        plt.tight_layout()
        out_svg = os.path.join(rollout_root, f"{save_prefix}_{x_variable}_{y_variable}_grad.svg")
        plt.savefig(out_svg, format="svg", bbox_inches="tight")
        print(f"[rollout_and_compare] saved gradient rollout: {out_svg}")
        plt.close()


# ================================
# 4. main
# ================================
def main():
    npz_straight = "learning_result/training_results_all_straight.npz"
    vehicle_straight = "../LearningExperiment/vehicle_log_straight.csv"
    route_csv = "../LearningExperiment/route_points.csv"

    fig1, axs1 = plt.subplots(2, 2, figsize=(8.4, 6.4), height_ratios=[1,1], dpi=150)

    plot_losses(npz_straight, ax=axs1[0, 0])
    plot_time_bars(npz_straight, ax=axs1[0, 1])
    rollout_and_compare(npz_straight, vehicle_straight, route_csv, x_variable="t", y_variable="v", ax=axs1[1, 0])
    rollout_and_compare(npz_straight, vehicle_straight, route_csv, x_variable="t", y_variable="a", ax=axs1[1, 1])

    axs1[0, 0].set_title("(a) Imitation Loss", fontsize=18, pad=16)
    axs1[0, 1].set_title("(b) Backward Time", fontsize=18, pad=16)
    axs1[1, 0].set_title("(c) Velocity Comparison", fontsize=18, pad=16)
    axs1[1, 1].set_title("(d) Acceleration Comparison", fontsize=18, pad=16)

    axs1[0, 0].set_ylim(180, 320)  # Loss
    axs1[0, 1].set_ylim(0, 3)  # Backward Time
    axs1[1, 0].set_ylim(3, 8)  # Velocity
    axs1[1, 1].set_ylim(-0.5, 2)  # Acceleration

    for ax in axs1.flat:
        ax.title.set_position([0.5, 1.05])

    for ax in axs1.flat:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = os.path.join(RESULT_DIR, "summary_straight_2x2.svg")
    plt.savefig(out1, format="svg", bbox_inches="tight")
    print(f"[main] saved: {out1}")
    plt.show()
    plt.close(fig1)

    npz_curve = "learning_result/training_results_all_curve.npz"
    vehicle_curve = "../LearningExperiment/vehicle_log_curve.csv"

    fig2 = plt.figure(figsize=(8.4, 7.2), dpi=150)
    gs = fig2.add_gridspec(2, 2 , height_ratios=[0.8,1])

    ax_loss = fig2.add_subplot(gs[0, 0])
    ax_time = fig2.add_subplot(gs[0, 1])
    ax_traj = fig2.add_subplot(gs[1, :])

    plot_losses(npz_curve, ax=ax_loss)
    plot_time_bars(npz_curve, ax=ax_time)
    rollout_and_compare(npz_curve, vehicle_curve, route_csv,
                        x_variable="x", y_variable="y", ax=ax_traj)
    ax_traj.set_xlim(40, 80)

    ax_loss.set_title("(a) Imitation Loss", fontsize=18, pad=16)
    ax_time.set_title("(b) Backward Time", fontsize=18, pad=16)
    ax_traj.set_title("(c) Trajectory Comparison", fontsize=18, pad=16)

    for ax in [ax_loss, ax_time, ax_traj]:
        ax.title.set_position([0.5, 1.05])

    ax_loss.set_ylim(35, 200)
    ax_time.set_ylim(0, 3)
    ax_traj.set_ylim(60, 105)

    for ax in [ax_loss, ax_time, ax_traj]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = os.path.join(RESULT_DIR, "summary_curve_2x2.svg")
    plt.savefig(out2, format="svg", bbox_inches="tight")
    print(f"[main] saved: {out2}")
    plt.show()
    plt.close(fig2)

if __name__ == "__main__":
    main()
