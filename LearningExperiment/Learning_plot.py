# analyze_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Global font configuration =====
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

from LearningExperiment.Learning_exp import Bicycle_Model, build_coc_from_env, mpc_rollout
from LearningExperiment.util import RoutePredictor


# Create output directory
RESULT_DIR = "learning_result"
os.makedirs(RESULT_DIR, exist_ok=True)


# ================================
# 1. Training Loss
# ================================
def plot_losses(npz_path: str, out_svg: str = None):
    """Plot loss curves (FastDOC/FastDOC, IDOC, PDP) from saved NPZ, save as SVG (2.0x2.0)."""
    z = np.load(npz_path, allow_pickle=True)
    loss_FastDOC = z["loss_FastDOC"]
    loss_IDOC    = z["loss_IDOC"]
    loss_SafePDP     = z["loss_SafePDP"]

    COLOR_OUR  = "#d62728"
    COLOR_IDOC = "#1f77b4"
    COLOR_PDP  = "#f1c40f"
    LS_SOLID   = "-"
    LS_DASH    = "--"

    iters = np.arange(len(loss_FastDOC))
    plt.figure(figsize=(2.4, 2.4), dpi=120)
    plt.plot(iters, loss_FastDOC, label="FastDOC",
             color=COLOR_OUR, linestyle=LS_SOLID, linewidth=1.2)
    plt.plot(iters, loss_IDOC,    label="IDOC",
             color=COLOR_IDOC, linestyle=LS_SOLID, linewidth=1.4)
    plt.plot(iters, loss_SafePDP,     label="SafePDP",
             color=COLOR_PDP, linestyle=LS_DASH, linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    title_tag = os.path.basename(npz_path).replace("training_results_all_", "").replace(".npz", "")
    plt.title(f"Training Loss - {title_tag}")
    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.6)
    plt.legend()
    plt.tight_layout()

    if out_svg is None:
        out_svg = f"loss_from_npz_{title_tag}.svg"
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    print(f"[plot_losses] saved: {out_svg}")
    plt.close()


# ================================
# 2. Build/Solve Time Bar
# ================================
def plot_time_bars(npz_path: str, out_svg: str = None):
    """
    Plot grouped bar charts:
      - Left group: Build time
      - Right group: Solve time
      - Three bars per group: FastDOC / IDOC / SafePDP
      - X-axis labels: 'Build' and 'Solve'
    """
    z = np.load(npz_path, allow_pickle=True)
    methods = ["FastDOC", "IDOC", "SafePDP"]
    COLORS = {"FastDOC": "#CA0E12", "IDOC": "#25377F", "SafePDP": "#F6BD12"}

    build_avg = np.array([float(z[f"build_time_avg_{m}"]) * 1e3 for m in methods])
    solve_avg = np.array([float(z[f"solve_time_avg_{m}"]) * 1e3 for m in methods])
    
    n_methods = len(methods)
    width = 0.95  # more wide
    group_gap = 1
    x_build = np.arange(n_methods)
    x_solve = x_build + n_methods + group_gap

    fig, ax = plt.subplots(figsize=(3.6, 2.4), dpi=150)
    ax.set_facecolor("#ffffff")

    # Build bars
    for i, m in enumerate(methods):
        ax.bar(x_build[i], build_avg[i], width, color=COLORS[m], edgecolor="none")

    # Solve bars
    for i, m in enumerate(methods):
        ax.bar(x_solve[i], solve_avg[i], width, color=COLORS[m], edgecolor="none")

    # top text
    for i, m in enumerate(methods):
        ax.text(x_build[i], build_avg[i] + 0.01 * max(build_avg),
                f"{build_avg[i]:.2f}", ha="center", va="bottom",
                fontsize=8, color=COLORS[m])
        ax.text(x_solve[i], solve_avg[i] + 0.01 * max(solve_avg),
                f"{solve_avg[i]:.2f}", ha="center", va="bottom",
                fontsize=8, color=COLORS[m])

    mid_build = np.mean(x_build)
    mid_solve = np.mean(x_solve)
    ax.set_xticks([mid_build, mid_solve])
    ax.set_xticklabels(["Build", "Solve"], fontsize=9, fontweight="bold")

    ax.set_ylabel("Time (ms)")
    ax.set_ylim(0, max(np.concatenate([build_avg, solve_avg])) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in methods]
    ax.legend(handles, methods, frameon=False, loc="upper right", fontsize=8.5)

    plt.tight_layout()

    title_tag = os.path.basename(npz_path).replace(".npz", "")
    if out_svg is None:
        out_svg = os.path.join(RESULT_DIR, f"time_bars_grouped_{title_tag}.svg")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    print(f"[plot_time_bars] saved: {out_svg}")
    plt.show()
    plt.close(fig)

# ================================
# 3. MPC Rollout Comparison
# ================================
def rollout_and_compare(
    npz_path: str,
    vehicle_csv: str,
    route_csv: str,
    x_variable: str = "t",
    y_variable: str = "v",
    save_prefix: str = "mpc_eval",
):
    """
    Visualize FastDOC rollouts by comparing the NMPC-generated trajectories
    before and after training with the human driving demonstration.
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
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
    if key not in z:
        raise KeyError(f"{key} not found in npz")
    params_traj = np.asarray(z[key], dtype=float)
    n_total = len(params_traj)

    idxs = [1, 2, 5, 10] + [i for i in range(20, 101, 20)] + [i for i in range(200, n_total, 200)]
    idxs = [i for i in idxs if i < n_total]
    if idxs[-1] != n_total - 1:
        idxs.append(n_total - 1)
    idxs = sorted(set(idxs))
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
            print(f"[cache hit] {cache_file}")
        else:
            print(f"[recompute] step {idxs[i]}")
            df_tmp = mpc_rollout(
                method_name=f"FastDOC_step_{idxs[i]}",
                params=params,
                start_index=start_index,
                end_index=end_index,
                horizon=5,
                dt=dt,
                vehicle_data=vehicle_data,
                rp=rp,
                coc=coc
            )
            df_tmp.to_csv(cache_file, index=False)
        rollout_dfs.append(df_tmp)

    fig, ax = plt.subplots(figsize=(3.6, 2.4), dpi=240)

    df_init, df_final = rollout_dfs[0], rollout_dfs[-1]
    if x_variable == "t":
        x_init = (df_init["index"].to_numpy() - start_index) * dt
        x_final = (df_final["index"].to_numpy() - start_index) * dt
    else:
        x_init = df_init[x_variable].to_numpy()
        x_final = df_final[x_variable].to_numpy()
    y_init = df_init[y_variable].to_numpy()
    y_final = df_final[y_variable].to_numpy()

    min_len = min(len(x_init), len(x_final))
    xi, yi = x_init[:min_len], y_init[:min_len]
    xf, yf = x_final[:min_len], y_final[:min_len]

    poly_x = np.concatenate([xi, xf[::-1]])
    poly_y = np.concatenate([yi, yf[::-1]])
    verts = np.column_stack([poly_x, poly_y])
    fill_color = np.array(to_rgb("#A1A1A1"))  # 浅灰色填充
    poly = Polygon(verts, closed=True, facecolor=fill_color, edgecolor="none", alpha=0.25, zorder=0)
    ax.add_patch(poly)

    for i, (df_tmp, c, alpha,line_width) in enumerate(zip(rollout_dfs[:-1], colors_traj[:-1], alphas[:-1], line_widths[:-1])):
        if x_variable == "t":
            x = (df_tmp["index"].to_numpy() - start_index) * dt
        else:
            x = df_tmp[x_variable].to_numpy()
        y = df_tmp[y_variable].to_numpy()
        ax.plot(x, y, color=c, alpha=alpha, linewidth=line_width, zorder=1)

    ax.plot(xf, yf, color="#25377F", linewidth=1.5, zorder=2, label="Final")

    ax.plot(x_real, y_real, color="#CA0E12", linestyle="--",linewidth=1.2, zorder=3, label="Human")

    line_init = mlines.Line2D([], [], color="#9ecae1", lw=1.6, label="Init")
    line_final = mlines.Line2D([], [], color="#08519c", lw=1.6, label="Final")
    line_human = mlines.Line2D([], [], color="#4d4d4d", lw=1.6, label="Human")
    ax.legend(handles=[line_init, line_final, line_human], frameon=False)

    ax.set_xlabel("Time (s)" if x_variable == "t" else x_variable)
    ax.set_ylabel(y_variable)
    ax.set_title(f"{x_variable}-{y_variable} Gradient Rollout ({title_tag})")
    ax.grid(True, linestyle="--", alpha=0.6)
    xlabel = "Time (s)" if x_variable == "t" else x_variable

    unit_map = {
        "v": "Velocity (m/s)",
        "a": "Acceleration (m/s²)",
        "x": "Position X (m)",
        "y": "Position Y (m)"
    }
    ylabel = unit_map.get(y_variable, y_variable)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{x_variable}-{y_variable} Gradient Rollout ({title_tag})")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_svg = os.path.join(rollout_root, f"{save_prefix}_{x_variable}_{y_variable}_grad.svg")
    plt.savefig(out_svg, format="svg", bbox_inches="tight")
    print(f"[rollout_and_compare] saved gradient rollout: {out_svg}")
    plt.show()
    plt.close(fig)


# ================================
# 4. main
# ================================
def main():
    npz_path = "Saved/training_results_all_straight.npz"
    vehicle_csv = "vehicle_log_straight.csv"
    route_csv = "route_points.csv"

    # plot_losses(npz_path)
    plot_time_bars(npz_path)

    # rollout_and_compare(npz_path, vehicle_csv, route_csv, x_variable="t", y_variable="v")
    # rollout_and_compare(npz_path, vehicle_csv, route_csv, x_variable="t", y_variable="a")

    npz_path = "Saved/training_results_all_curve.npz"
    vehicle_csv = "vehicle_log_curve.csv"
    route_csv = "route_points.csv"

    # rollout_and_compare(npz_path, vehicle_csv, route_csv, x_variable="x", y_variable="y")

    # plot_losses(npz_path)
    plot_time_bars(npz_path)


if __name__ == "__main__":
    main()
