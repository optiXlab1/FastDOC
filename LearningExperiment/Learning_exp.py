import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from casadi import *
from FastDOC import FastDOC
from FastDOC import SafePDP
from LearningExperiment.util import RoutePredictor, align_yaw_to_init


# ----------------------------- Bicycle model -----------------------------
class Bicycle_Model:
    def __init__(self, project_name='bicycle'):
        self.project_name = project_name

    def initDyn(self, L=2.5):
        """Define states X=[X,Y,phi,v,a,delta], controls U=[j,ddelta], and continuous-time dynamics f."""
        self.dyn_auxvar = vcat([])
        X, Y, phi, v, a, delta = SX.sym('X'), SX.sym('Y'), SX.sym('phi'), SX.sym('v'), SX.sym('a'), SX.sym('delta')
        self.X = vertcat(X, Y, phi, v, a, delta)

        j, ddelta = SX.sym('j'), SX.sym('ddelta')
        self.U = vertcat(j, ddelta)

        f = vertcat(
            v * cos(phi),
            v * sin(phi),
            v / L * tan(delta),
            a,
            j,
            ddelta
        )
        self.f = f

    def initCost(self):
        """Define quadratic tracking cost with look-ahead and control effort regularization."""
        parameter = []
        wX = SX.sym('wX'); wY = SX.sym('wY'); wphi = SX.sym('wphi'); wv = SX.sym('wv')
        wa = SX.sym('wa'); wdelta = SX.sym('wdelta'); wj = SX.sym('wj'); wddelta = SX.sym('wddelta')
        lookahead_distance = SX.sym('lookahead_distance')
        alpha = SX.sym('alpha')
        parameter += [wX, wY, wphi, wv, wa, wdelta, wj, wddelta, lookahead_distance, alpha]
        self.cost_auxvar = vcat(parameter)

        ref_state = [SX.sym('ref_x'), SX.sym('ref_y'), SX.sym('ref_phi'),
                     SX.sym('ref_v'), SX.sym('ref_a'), SX.sym('ref_delta'), SX.sym('cur_delta')]
        self.cost_other_value = vcat(ref_state)

        # stage cost
        self.path_cost = (
            wX * (self.X[0] + lookahead_distance * cos(self.X[2] + alpha * self.cost_other_value[6]) - self.cost_other_value[0]) ** 2 +
            wY * (self.X[1] + lookahead_distance * sin(self.X[2] + alpha * self.cost_other_value[6]) - self.cost_other_value[1]) ** 2 +
            wphi * (self.X[2] - self.cost_other_value[2]) ** 2 +
            wv * (self.X[3] - self.cost_other_value[3]) ** 2 +
            wa * (self.X[4] - self.cost_other_value[4]) ** 2 +
            wdelta * (self.X[5] - self.cost_other_value[5]) ** 2 +
            wj * (self.U[0]) ** 2 +
            wddelta * (self.U[1]) ** 2
        )

        # terminal cost (without control effort)
        self.final_cost = (
            wX * (self.X[0] + lookahead_distance * cos(self.X[2] + alpha * self.cost_other_value[6]) - self.cost_other_value[0]) ** 2 +
            wY * (self.X[1] + lookahead_distance * sin(self.X[2] + alpha * self.cost_other_value[6]) - self.cost_other_value[1]) ** 2 +
            wphi * (self.X[2] - self.cost_other_value[2]) ** 2 +
            wv * (self.X[3] - self.cost_other_value[3]) ** 2 +
            wa * (self.X[4] - self.cost_other_value[4]) ** 2 +
            wdelta * (self.X[5] - self.cost_other_value[5]) ** 2
        )

        # residual-weighted (square-root) form for stage/terminal
        self.rw_path_cost = vertcat(
            sqrt(wX) * (self.X[0] + lookahead_distance * cos(self.X[2] + alpha * self.cost_other_value[6]) - self.cost_other_value[0]),
            sqrt(wY) * (self.X[1] + lookahead_distance * sin(self.X[2] + alpha * self.cost_other_value[6]) - self.cost_other_value[1]),
            sqrt(wphi) * (self.X[2] - self.cost_other_value[2]),
            sqrt(wv) * (self.X[3] - self.cost_other_value[3]),
            sqrt(wa) * (self.X[4] - self.cost_other_value[4]),
            sqrt(wdelta) * (self.X[5] - self.cost_other_value[5]),
            sqrt(wj) * self.U[0],
            sqrt(wddelta) * self.U[1],
        )

        self.rw_final_cost = vertcat(
            sqrt(wX) * (self.X[0] + lookahead_distance * cos(self.X[2] + alpha * self.cost_other_value[6]) - self.cost_other_value[0]),
            sqrt(wY) * (self.X[1] + lookahead_distance * sin(self.X[2] + alpha * self.cost_other_value[6]) - self.cost_other_value[1]),
            sqrt(wphi) * (self.X[2] - self.cost_other_value[2]),
            sqrt(wv) * (self.X[3] - self.cost_other_value[3]),
            sqrt(wa) * (self.X[4] - self.cost_other_value[4]),
            sqrt(wdelta) * (self.X[5] - self.cost_other_value[5]),
        )

    def initConstraints(self):
        """Box constraints on controls and selected states (v,a,delta)."""
        max_U = np.array([6, 2])
        min_U = np.array([-6, -2])
        max_X = np.array([30, 4, np.pi / 4])
        min_X = np.array([-1, -4, -np.pi / 4])

        path_inequ_Uub = self.U - max_U
        path_inequ_Ulb = -(self.U - min_U)
        path_inequ_Xub = self.X[[3, 4, 5]] - max_X
        path_inequ_Xlb = -(self.X[[3, 4, 5]] - min_X)

        self.path_inequ = vcat([path_inequ_Uub, path_inequ_Ulb, path_inequ_Xub, path_inequ_Xlb])


# -------------------------- Helper to build COC system --------------------------
def build_coc_from_env(env: Bicycle_Model, dt: float, gamma: float):
    """Discretize dynamics with forward Euler and build SafeSafePDP COC system."""
    env_dyn = env.X + dt * env.f

    coc = SafePDP.COCsys()
    coc.setAuxvarVariable(env.cost_auxvar)
    coc.setStateVariable(env.X)
    coc.setControlVariable(env.U)
    coc.setDyn(env_dyn)

    coc.setCostOtherValue(env.cost_other_value)
    coc.setPathCost(env.path_cost)
    coc.setFinalCost(env.final_cost)
    coc.setRwPathCost(env.rw_path_cost)
    coc.setRwFinalCost(env.rw_final_cost)

    coc.setPathInequCstr(env.path_inequ)
    coc.setBlankdX0()
    coc.convert2BarrierOC(gamma=gamma)
    return coc

def mpc_rollout(
    method_name: str,
    params,
    start_index: int,
    end_index: int,
    horizon: int,
    dt: float,
    vehicle_data: pd.DataFrame,
    rp,                 # RoutePredictor
    coc,                # SafeSafePDP.COCsys
    save_prefix: str = None
):
    """
    Rolling MPC-like rollout:
      - At each log index i: build reference horizon from route predictor
      - Solve OC with current state and 'params'
      - Apply the first control to get next state x_{k+1} (from solver's optimal trajectory)
      - Log one row and advance i <- i+1 with xk <- x_{k+1}

    Args:
        method_name: name tag ("FastDOC", "IDOC", "SafePDP", "Init", etc.)
        params:      auxiliary parameter vector for the OC cost (1D iterable/np.ndarray)
        start_index: starting row index in `vehicle_data`
        end_index:   last row index (exclusive) to roll over
        horizon:     planning horizon length
        dt:          discretization step (s)
        vehicle_data: DataFrame with columns ['x','y','phi','v','a','delta','ref_idx']
        rp:          RoutePredictor providing future_from_index(idx, N, dt)
        coc:         SafeSafePDP.COCsys already configured (dyn/cost/constraints/barrier)
        save_prefix: if not None, save CSV as f"{save_prefix}_{method_name}.csv"

    Returns:
        DataFrame with columns:
          ['index','x','y','phi','v','a','delta','u_j','u_ddelta','ref_idx']
    """
    base_cols = ['x', 'y', 'phi', 'v', 'delta']
    has_a = 'a' in vehicle_data.columns

    if not has_a:
        # 如果没有 a 列，就用有限差分算 a
        v = vehicle_data['v'].to_numpy()
        a = np.zeros_like(v)
        a[1:] = np.diff(v) / dt
        vehicle_data['a'] = a
        print("[info] 'a' not found in vehicle_csv, computed from velocity difference.")

    # 现在可以安全地读取所有列
    cols = ['x', 'y', 'phi', 'v', 'a', 'delta']
    i = int(start_index)
    xk = vehicle_data.loc[i, cols].to_numpy(float)


    rows = []
    while i < int(end_index):
        # 1) build reference horizon
        ref_idx = int(vehicle_data.loc[i, 'ref_idx'])
        ref = rp.future_from_index(idx=ref_idx, N=horizon, dt=dt)
        ref = align_yaw_to_init(ref, xk[2], phi_col=2)
        ref = np.c_[ref, np.full((len(ref), 1), xk[5])]

        # 2) solve OC
        traj = coc.solveBarrierOC(
            horizon=horizon,
            init_state=xk,
            cost_other_value=ref,
            auxvar_value=params
        )

        # 3) check solver and extract first control / next state
        if traj.get('control_traj_opt') is None or len(traj['control_traj_opt']) == 0:
            print(f"[{method_name}] solveBarrierOC failed at i={i}, stop.")
            break

        u0 = np.asarray(traj['control_traj_opt'][0], dtype=float).ravel()
        x1 = np.asarray(traj['state_traj_opt'][1],   dtype=float).ravel()

        # 4) log one row (index i with current state xk and applied control u0)
        rows.append([i, *xk, *u0, ref_idx])

        # 5) roll forward
        xk = x1
        i += 1

    df = pd.DataFrame(rows, columns=[
        'index','x','y','phi','v','a','delta','u_j','u_ddelta','ref_idx'
    ])

    if save_prefix:
        out_csv = f"{save_prefix}_{method_name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[{method_name}] saved: {out_csv}  (steps={len(df)})")

    return df

# ----------------------------- Training routine -----------------------------
def train_on_segment_df(
    roadname: str,
    vehicle_data: pd.DataFrame,
    rp,                      # RoutePredictor (already constructed)
    coc,                     # SafeSafePDP.COCsys (already constructed)
    start_index: int,
    end_index: int,
    horizon: int = 5,
    dt: float = 0.1,
    init_parameter = (16, 16, 4, 8, 2, 1, 0.5, 1, 14, 1),
    max_iter: int = 1500,
    lr: float = 1e-2,
    grad_protection_threshold: float = 1e5,
    seed: int = 100,
):
    """
    Train FastDOC / IDOC / SafePDP on a given segment using prebuilt data & COC.
    - vehicle_data: DataFrame with at least ['x','y','phi','v','delta','ref_idx'].
      If column 'a' is absent, it will be computed via finite difference.
    - rp: RoutePredictor
    - coc: configured SafeSafePDP.COCsys (dynamics/cost/constraints already set)
    Saves:
      - loss_curves_all_{roadname}.png
      - training_results_all_{roadname}.npz
    Returns a small summary dict.
    """
    # Ensure acceleration column exists
    if 'a' not in vehicle_data.columns:
        v = vehicle_data['v'].to_numpy()
        a = np.zeros_like(v)
        if len(v) > 1:
            a[1:] = np.diff(v) / float(dt)
        vehicle_data = vehicle_data.copy()
        vehicle_data['a'] = a

    np.random.seed(seed)
    init_parameter = np.asarray(init_parameter, dtype=float)

    current_parameter_FastDOC = init_parameter.copy()
    current_parameter_IDOC  = init_parameter.copy()
    current_parameter_SafePDP   = init_parameter.copy()

    loss_trace_FastDOC = []
    parameter_trace_FastDOC = np.empty((max_iter, coc.n_auxvar))
    loss_trace_IDOC = []
    parameter_trace_IDOC = np.empty((max_iter, coc.n_auxvar))
    loss_trace_SafePDP = []
    parameter_trace_SafePDP = np.empty((max_iter, coc.n_auxvar))

    previous_grad_FastDOC = 0
    previous_grad_IDOC  = 0
    previous_grad_SafePDP   = 0

    build_time_FastDOC = []
    solve_time_FastDOC = []
    build_time_IDOC  = []
    solve_time_IDOC = []
    build_time_SafePDP = []
    solve_time_SafePDP   = []
    total_count      = 0

    # Main training loop
    for k in range(max_iter):
        batch_loss_FastDOC = 0.0; batch_grad_FastDOC = 0.0
        batch_loss_IDOC  = 0.0; batch_grad_IDOC  = 0.0
        batch_loss_SafePDP   = 0.0; batch_grad_SafePDP   = 0.0

        for i in range(int(start_index), int(end_index)):
            demo = vehicle_data.iloc[i:i + horizon + 1]
            cols = ['x', 'y', 'phi', 'v', 'a', 'delta']
            states = demo.loc[:, cols].to_numpy(dtype=float)
            init_state = states[0, :]

            cost_other_value = rp.future_from_index(idx=int(demo.iloc[0]['ref_idx']), N=horizon, dt=float(dt))
            cost_other_value = align_yaw_to_init(cost_other_value, init_state[2], phi_col=2)
            cost_other_value = np.c_[cost_other_value, np.full((len(cost_other_value), 1), init_state[5])]

            # Solve OC for three pipelines
            traj_FastDOC = coc.solveBarrierOC(horizon=horizon, init_state=init_state,
                                            cost_other_value=cost_other_value,
                                            auxvar_value=current_parameter_FastDOC)
            traj_IDOC  = coc.solveBarrierOC(horizon=horizon, init_state=init_state,
                                            cost_other_value=cost_other_value,
                                            auxvar_value=current_parameter_IDOC)
            traj_SafePDP   = coc.solveBarrierOC(horizon=horizon, init_state=init_state,
                                            cost_other_value=cost_other_value,
                                            auxvar_value=current_parameter_SafePDP)

            if (traj_FastDOC.get('solver_success') is False or
                traj_IDOC.get('solver_success')  is False or
                traj_SafePDP.get('solver_success')   is False):
                continue

            # FastDOC
            t0 = time.perf_counter()
            auxsys_FastDOC = coc.barrier_oc.getAuxSys(
                state_traj_opt=traj_FastDOC['state_traj_opt'],
                control_traj_opt=traj_FastDOC['control_traj_opt'],
                costate_traj_opt=traj_FastDOC['costate_traj_opt'],
                cost_other_value=traj_FastDOC['cost_other_value'],
                auxvar_value=traj_FastDOC['auxvar_value'],
                GN = True
            )
            blocks_GN = FastDOC.build_blocks(auxsys_FastDOC)
            t1 = time.perf_counter()
            traj_deriv_FastDOC = FastDOC.explicit_solve(*blocks_GN, GN = True)
            loss_G, grad_G = SafePDP.Traj_L2_Loss(states, traj_FastDOC, traj_deriv_FastDOC)
            t2 = time.perf_counter()
            tFB = t1 - t0
            tFS = t2 - t1

            # IDOC
            t0 = time.perf_counter()
            auxsys_IDOC = coc.barrier_oc.getAuxSys(
                state_traj_opt=traj_IDOC['state_traj_opt'],
                control_traj_opt=traj_IDOC['control_traj_opt'],
                costate_traj_opt=traj_IDOC['costate_traj_opt'],
                cost_other_value=traj_IDOC['cost_other_value'],
                auxvar_value=traj_IDOC['auxvar_value'],
                GN = False
            )
            blocks_ID = FastDOC.build_blocks(auxsys_IDOC)
            t1 = time.perf_counter()
            traj_deriv_IDOC = FastDOC.explicit_solve(*blocks_ID, GN = False)
            loss_I, grad_I = SafePDP.Traj_L2_Loss(states, traj_IDOC, traj_deriv_IDOC)
            t2 = time.perf_counter()
            tIB = t1 - t0
            tIS = t2 - t1

            # SafePDP
            t0 = time.perf_counter()
            aux_sol_SafePDP, t1 = coc.auxSysBarrierOC(opt_sol=traj_SafePDP)
            loss_P, grad_P = SafePDP.Traj_L2_Loss(states, traj_SafePDP, aux_sol_SafePDP)
            t2 = time.perf_counter()
            tPB = t1 - t0
            tPS = t2 - t1

            # print("Backward Time:", tG, tI, tP)

            # Accumulate
            batch_loss_FastDOC += loss_G; batch_grad_FastDOC += grad_G
            batch_loss_IDOC  += loss_I; batch_grad_IDOC  += grad_I
            batch_loss_SafePDP   += loss_P; batch_grad_SafePDP   += grad_P

            build_time_FastDOC.append(tFB); build_time_IDOC.append(tIB); build_time_SafePDP.append(tPB)
            solve_time_FastDOC.append(tFS); solve_time_IDOC.append(tIS); solve_time_SafePDP.append(tPS)
            total_count      += 1

        # Gradient protection
        if norm_2(batch_grad_FastDOC) > grad_protection_threshold:
            batch_grad_FastDOC = previous_grad_FastDOC
        else:
            previous_grad_FastDOC = batch_grad_FastDOC

        if norm_2(batch_grad_IDOC) > grad_protection_threshold:
            batch_grad_IDOC = previous_grad_IDOC
        else:
            previous_grad_IDOC = batch_grad_IDOC

        if norm_2(batch_grad_SafePDP) > grad_protection_threshold:
            batch_grad_SafePDP = previous_grad_SafePDP
        else:
            previous_grad_SafePDP = batch_grad_SafePDP

        # Log traces
        loss_trace_FastDOC.append(batch_loss_FastDOC); parameter_trace_FastDOC[k] = current_parameter_FastDOC
        loss_trace_IDOC.append(batch_loss_IDOC);   parameter_trace_IDOC[k]  = current_parameter_IDOC
        loss_trace_SafePDP.append(batch_loss_SafePDP);     parameter_trace_SafePDP[k]   = current_parameter_SafePDP

        # Print progress
        print(
            f"iter #{k:4d}: "
            f"FastDOC: loss={batch_loss_FastDOC:.6f} | "
            f"IDOC: {batch_loss_IDOC:.6f} | "
            f"SafePDP: {batch_loss_SafePDP:.6f} | "
            f"lr={lr:.4f}"
        )

        # SGD step (elementwise floor at 20% of init)
        floor = 0.2 * np.array(init_parameter)
        current_parameter_FastDOC = np.maximum(current_parameter_FastDOC - lr * batch_grad_FastDOC, floor)
        current_parameter_IDOC  = np.maximum(current_parameter_IDOC  - lr * batch_grad_IDOC,  floor)
        current_parameter_SafePDP   = np.maximum(current_parameter_SafePDP   - lr * batch_grad_SafePDP,   floor)

    # === Create output directory ===
    outdir = "learning_result"
    os.makedirs(outdir, exist_ok=True)

    # === Plot & save ===
    iters = np.arange(len(loss_trace_FastDOC))
    loss_FastDOC = np.asarray(loss_trace_FastDOC, dtype=float)
    loss_IDOC = np.asarray(loss_trace_IDOC, dtype=float)
    loss_SafePDP = np.asarray(loss_trace_SafePDP, dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(iters, loss_FastDOC, label='Our Method', linewidth=2)
    plt.plot(iters, loss_IDOC, label='IDOC', linewidth=2)
    plt.plot(iters, loss_SafePDP, label='SafePDP', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (FastDOC vs IDOC vs SafePDP) - {roadname}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save inside learning_result/
    loss_png = os.path.join(outdir, f'loss_curves_all_{roadname}.png')
    plt.savefig(loss_png, dpi=150, bbox_inches='tight')
    plt.close()

    # === Time statistics ===
    build_time_FastDOC_arr = np.array(build_time_FastDOC)
    solve_time_FastDOC_arr = np.array(solve_time_FastDOC)
    build_time_IDOC_arr = np.array(build_time_IDOC)
    solve_time_IDOC_arr = np.array(solve_time_IDOC)
    build_time_SafePDP_arr = np.array(build_time_SafePDP)
    solve_time_SafePDP_arr = np.array(solve_time_SafePDP)

    den = max(1, total_count)
    avg_time_FastDOC_build = np.sum(build_time_FastDOC_arr) / den
    avg_time_IDOC_build = np.sum(build_time_IDOC_arr) / den
    avg_time_SafePDP_build = np.sum(build_time_SafePDP_arr) / den
    avg_time_FastDOC_solve = np.sum(solve_time_FastDOC_arr) / den
    avg_time_IDOC_solve = np.sum(solve_time_IDOC_arr) / den
    avg_time_SafePDP_solve = np.sum(solve_time_SafePDP_arr) / den

    var_time_FastDOC_build = np.var(build_time_FastDOC_arr)
    var_time_IDOC_build = np.var(build_time_IDOC_arr)
    var_time_SafePDP_build = np.var(build_time_SafePDP_arr)
    var_time_FastDOC_solve = np.var(solve_time_FastDOC_arr)
    var_time_IDOC_solve = np.var(solve_time_IDOC_arr)
    var_time_SafePDP_solve = np.var(solve_time_SafePDP_arr)

    print(
        f"Avg build time | FastDOC: {avg_time_FastDOC_build:.6e}s, IDOC: {avg_time_IDOC_build:.6e}s, SafePDP: {avg_time_SafePDP_build:.6e}s (over {total_count} samples)")
    print(
        f"Avg solve time | FastDOC: {avg_time_FastDOC_solve:.6e}s, IDOC: {avg_time_IDOC_solve:.6e}s, SafePDP: {avg_time_SafePDP_solve:.6e}s (over {total_count} samples)")

    # === Save training results to NPZ ===
    param_names = ['wX', 'wY', 'wphi', 'wv', 'wa', 'wdelta', 'wj', 'wddelta', 'lookahead_distance']
    parameter_trace_FastDOC = np.asarray(parameter_trace_FastDOC[:len(iters)])
    parameter_trace_IDOC = np.asarray(parameter_trace_IDOC[:len(iters)])
    parameter_trace_SafePDP = np.asarray(parameter_trace_SafePDP[:len(iters)])

    npz_path = os.path.join(outdir, f"training_results_all_{roadname}.npz")
    np.savez(
        npz_path,
        gamma=getattr(coc, "gamma", None), nn_seed=seed, lr=lr, dt=dt,
        loss_FastDOC=loss_FastDOC, loss_IDOC=loss_IDOC, loss_SafePDP=loss_SafePDP,
        params_FastDOC=parameter_trace_FastDOC,
        params_IDOC=parameter_trace_IDOC,
        params_SafePDP=parameter_trace_SafePDP,
        param_names=np.array(param_names, dtype=object),
        road_range=[start_index, end_index],
        build_time_avg_FastDOC=avg_time_FastDOC_build,
        build_time_avg_IDOC=avg_time_IDOC_build,
        build_time_avg_SafePDP=avg_time_SafePDP_build,
        solve_time_avg_FastDOC=avg_time_FastDOC_solve,
        solve_time_avg_IDOC=avg_time_IDOC_solve,
        solve_time_avg_SafePDP=avg_time_SafePDP_solve,
        build_time_var_FastDOC=var_time_FastDOC_build,
        build_time_var_IDOC=var_time_IDOC_build,
        build_time_var_SafePDP=var_time_SafePDP_build,
        solve_time_var_FastDOC=var_time_FastDOC_solve,
        solve_time_var_IDOC=var_time_IDOC_solve,
        solve_time_var_SafePDP=var_time_SafePDP_solve,
    )

    print(f"Saved PNG: {loss_png}")
    print(f"Saved NPZ: {npz_path}")

# ----------------------------- Example entry point -----------------------------
if __name__ == "__main__":
    # Build data once
    dt = 0.1
    vehicle_data = pd.read_csv("vehicle_log_straight.csv")
    rp = RoutePredictor("route_points.csv")

    # Build env/COC once
    env = Bicycle_Model()
    env.initDyn()
    env.initCost()
    env.initConstraints()
    dt = 0.1
    gamma = 1e-2
    coc = build_coc_from_env(env, dt=dt, gamma=gamma)

    init_parameter = (16, 16, 4, 8, 2, 1, 0.5, 1, 14, 1)

    # Train on straight
    train_on_segment_df(
        roadname="straight",
        vehicle_data=vehicle_data,
        rp=rp,
        coc=coc,
        start_index=100,
        end_index=180,
        horizon=5,
        dt=dt,
        init_parameter=init_parameter,
        max_iter=1000,
        lr=0.01,
        seed=100
    )

    vehicle_data = pd.read_csv("vehicle_log_curve.csv")

    # Train on straight
    train_on_segment_df(
        roadname="curve",
        vehicle_data=vehicle_data,
        rp=rp,
        coc=coc,
        start_index=0,
        end_index=180,
        horizon=5,
        dt=dt,
        init_parameter=init_parameter,
        max_iter=300,
        lr=0.001,
        seed=100
    )
