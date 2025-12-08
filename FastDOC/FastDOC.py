import time
from scipy.linalg import lapack
import numpy as np


class OptimizedCholeskySolver:
    def __init__(self):
        self.L = None

    def factorize(self, A):
        """Cholesky factorization using LAPACK"""
        if not np.allclose(A, A.T):
            raise ValueError("Matrix is not symmetric; Cholesky factorization cannot be performed.")
        A_copy = A.copy()

        if A_copy.dtype == np.float64:
            self.L, info = lapack.dpotrf(A_copy, lower=1, overwrite_a=1)
        elif A_copy.dtype == np.float32:
            self.L, info = lapack.spotrf(A_copy, lower=1, overwrite_a=1)
        else:
            raise ValueError("Unsupported data type; only float32 and float64 are supported.")

        if info != 0:
            raise ValueError(f"Cholesky factorization failed, LAPACK error code: {info}")

        return self

    def solve(self, b):
        if b.ndim == 1:
            b_2d = b.reshape(-1, 1)
        else:
            b_2d = b.copy()

        if self.L.dtype == np.float64:
            x, info = lapack.dpotrs(self.L, b_2d, lower=1, overwrite_b=1)
        else:
            x, info = lapack.spotrs(self.L, b_2d, lower=1, overwrite_b=1)

        if info != 0:
            raise ValueError(f"Cholesky solve failed, LAPACK error code: {info}")

        if b.ndim == 1:
            return x.squeeze()
        else:
            return x

    def solve_batch(self, A, b):
        self.factorize(A)
        x = self.solve(b)
        return x


def build_blocks(auxsys_OC):
    """
    This function takes the building blocks of the auxiliary COC system defined in the PDP paper
    and uses them to construct the blocks in our IDOC identities.

    Inputs:

    auxsys_COC object: Dictionary with values being Jacobian/Hessian blocks of the constraints/cost.

    Outputs: - H_t: List of first T blocks in H
             - H_T: Final state block in H
             - A: All blocks in A (lower diagonal) corresponding to init. state, dynamics + eq. + ineq. (r_{-1}, r_{0}, ... r{T-1})
             - B_t: First T blocks of B
             - B_T: Final block of B
             - C: All blocks of C except C_0 (C_1, ..., C_T). C_0 is just zeros
             - ns: number of states
             - nc: number of controls
             - T: Horizon
    """

    T = len(auxsys_OC['dynF'])
    ns = auxsys_OC['Hxx'][0].shape[0]
    nc = auxsys_OC['Huu'][0].shape[0]
    nz = auxsys_OC['Hxe'][0].shape[1]

    Hxx = np.stack(auxsys_OC['Hxx'], axis=0)
    Hxu = np.stack(auxsys_OC['Hxu'], axis=0)
    Huu = np.stack(auxsys_OC['Huu'], axis=0)
    H_t = np.block([[Hxx, Hxu], [Hxu.transpose(0, 2, 1), Huu]])
    H_T = auxsys_OC['hxx'][0]

    # A blocks
    dynFx = auxsys_OC['dynF']
    dynFu = auxsys_OC['dynG']

    A = -np.stack([np.block([dynFx[t], dynFu[t]])  for t in range(T)])

    # B blocks
    Hxe = np.stack(auxsys_OC['Hxe'], axis=0)
    Hue = np.stack(auxsys_OC['Hue'], axis=0)
    B_t = np.concatenate((Hxe, Hue), axis=1)
    B_T = auxsys_OC['hxe'][0]

    # C blocks
    dynE = auxsys_OC['dynE']
    C_0 = np.zeros((ns, nz))
    C = np.stack([C_0] + [-dynE[t] for t in range(T)])

    return H_t, H_T, A, B_t, B_T, C, ns, nc, T

def explicit_solve(H, H_T_block, A_blocks, B, B_T_block, C, ns, nc, T, GN = False, report_time = False):
    start_time = time.perf_counter()
    if not GN:
        # t1 = time.perf_counter()
        Hinv = np.linalg.inv(H)
        H_T_inv = np.linalg.inv(H_T_block)
        # t2 = time.perf_counter()

        # compute and cache (H^1-A^T) expression in Prop. 4.5 in DDN paper
        HinvAT_diag_blocks = Hinv[..., :ns]
        HinvAT_diag_T_block = H_T_inv
        HinvAT_upper_blocks = Hinv @ A_blocks.transpose(0, 2, 1)

        # compute AH^-1B - C in Prop. 4.5 in DDN paper
        AHinvB_C = HinvAT_diag_blocks.transpose(0, 2, 1) @ B - C[:-1, ...]
        AHinvB_C[1:, ...] += HinvAT_upper_blocks[:-1, ...].transpose(0, 2, 1) @ B[:-1, ...]
        AHinvB_C_T_block = HinvAT_diag_T_block.T @ B_T_block - C[-1, ...]
        AHinvB_C_T_block += HinvAT_upper_blocks[-1, ...].T @ B[-1, ...]

        # compute AH^-1AT expression in Prop. 4.5 in DDN paper
        AHAinvAT_diag = np.concatenate((Hinv[..., :ns, :ns], H_T_inv[None, ...]), axis=0)
        AHAinvAT_diag[1:, ...] += A_blocks @ HinvAT_upper_blocks
        AHAinvAT_upper = HinvAT_upper_blocks[:, :ns, :].copy()
        AHAinvAT_lower = A_blocks @ HinvAT_diag_blocks

        # t3 = time.perf_counter()

#         print("AH-1AT AND AH-1B-C:", time.perf_counter() - start_time)
        # use Thomas's algorithm for block tridiagonal matrices to solve for (AH^-1A^T)^-1(AH^-1B - C)
        Thomas_start = time.perf_counter()
        AHinvAT_AHinvB_C = [None] * T
        for t in range(1, T + 1):
            CR = np.linalg.solve(AHAinvAT_diag[t - 1], np.concatenate((AHAinvAT_upper[t - 1], AHinvB_C[t - 1]), axis=1))
            AHAinvAT_upper[t - 1], AHinvB_C[t - 1] = CR[:, :ns], CR[:, ns:]

            AHAinvAT_diag[t] -= AHAinvAT_lower[t - 1] @ AHAinvAT_upper[t - 1]
            if t == T:
                AHinvB_C_T_block = AHinvB_C_T_block - AHAinvAT_lower[t - 1] @ AHinvB_C[t - 1]
            else:
                AHinvB_C[t] -= AHAinvAT_lower[t - 1] @ AHinvB_C[t - 1]

        AHinvAT_AHinvB_C_T_block = np.linalg.solve(AHAinvAT_diag[T], AHinvB_C_T_block)
        AHinvAT_AHinvB_C[T - 1] = AHinvB_C[T - 1] - AHAinvAT_upper[T - 1] @ AHinvAT_AHinvB_C_T_block
        for t in reversed(range(T - 1)):  # backward recursion
            AHinvAT_AHinvB_C[t] = AHinvB_C[t] - AHAinvAT_upper[t] @ AHinvAT_AHinvB_C[t + 1]
        AHinvAT_AHinvB_C = np.stack(AHinvAT_AHinvB_C)

        # t4 = time.perf_counter()

        # multiply by H^-1AT to get H^-1AT(AH^-1A^T)^-1(AH^-1B - C) (left term, projection to constraint surface)
        left_term = HinvAT_diag_blocks @ AHinvAT_AHinvB_C
        left_term[:-1] += HinvAT_upper_blocks[:-1] @ AHinvAT_AHinvB_C[1:]
        left_term[-1] += HinvAT_upper_blocks[-1] @ AHinvAT_AHinvB_C_T_block
        left_term_T_block = HinvAT_diag_T_block @ AHinvAT_AHinvB_C_T_block

        # solve right term (gradient of unconstrained problem) H^-1B and subtract from left term
        right_term = Hinv @ B
        right_term_T_block = H_T_inv @ B_T_block

        combined = [grad_ for grad_ in left_term - right_term]
        combined += [left_term_T_block - right_term_T_block]
        dxdp_traj_vec = [comb[:ns, :] for comb in combined]
        dudp_traj_vec = [comb[ns:, :] for comb in combined[:-1]]

        # t5 = time.perf_counter()

        # if report_time:
        #     dur = {
        #         "H_inv": t2 - t1,
        #         "prepare": t3 - t2,
        #         "solve": t4 - t3,
        #         "combine": t5 - t4,
        #     }
        #     dur["total"] = t5 - t1
        #
        #     csv_path = "report_timing_fig.csv"
        #     header = ["run_at", "H_inv", "prepare", "solve", "combine", "total"]
        #     row = {"run_at": dt.datetime.now().isoformat(), **dur}
        #
        #     new_file = not os.path.exists(csv_path)
        #     with open(csv_path, "a", newline="") as f:
        #         w = csv.DictWriter(f, fieldnames=header)
        #         if new_file:
        #             w.writeheader()
        #         w.writerow(row)

        time_ = [k for k in range(T + 1)]
        sol_full = {'state_traj_opt': dxdp_traj_vec,
                    'control_traj_opt': dudp_traj_vec,
                    'time': time_}
    else:
        solver = OptimizedCholeskySolver()
        Hinv = [None] * T
        for t in range(T):
            Hinv[t] = solver.solve_batch(H[t], np.eye(ns+nc))
        H_T_inv = solver.solve_batch(H_T_block, np.eye(ns))
        Hinv = np.array(Hinv)

        # compute and cache (H^1-A^T) expression in Prop. 4.5 in DDN paper
        HinvAT_diag_blocks = Hinv[..., :ns]
        HinvAT_diag_T_block = H_T_inv
        HinvAT_upper_blocks = Hinv @ A_blocks.transpose(0, 2, 1)

        # compute AH^-1B - C in Prop. 4.5 in DDN paper
        AHinvB_C = HinvAT_diag_blocks.transpose(0, 2, 1) @ B - C[:-1, ...]
        AHinvB_C[1:, ...] += HinvAT_upper_blocks[:-1, ...].transpose(0, 2, 1) @ B[:-1, ...]
        AHinvB_C_T_block = HinvAT_diag_T_block.T @ B_T_block - C[-1, ...]
        AHinvB_C_T_block += HinvAT_upper_blocks[-1, ...].T @ B[-1, ...]

        # compute AH^-1AT expression in Prop. 4.5 in DDN paper
        AHAinvAT_diag = np.concatenate((Hinv[..., :ns, :ns], H_T_inv[None, ...]), axis=0)
        AHAinvAT_diag[1:, ...] += A_blocks @ HinvAT_upper_blocks
        AHAinvAT_upper = HinvAT_upper_blocks[:, :ns, :].copy()
        AHAinvAT_lower = A_blocks @ HinvAT_diag_blocks

        # use Thomas's algorithm for block tridiagonal matrices to solve for (AH^-1A^T)^-1(AH^-1B - C)
        Thomas_start = time.perf_counter()
        AHinvAT_AHinvB_C = [None] * T
        for t in range(1, T + 1):
            CR = solver.solve_batch(AHAinvAT_diag[t - 1], np.concatenate((AHAinvAT_upper[t - 1], AHinvB_C[t - 1]), axis=1))
            AHAinvAT_upper[t - 1], AHinvB_C[t - 1] = CR[:, :ns], CR[:, ns:]

            AHAinvAT_diag[t] -= AHAinvAT_lower[t - 1] @ AHAinvAT_upper[t - 1]
            if t == T:
                AHinvB_C_T_block = AHinvB_C_T_block - AHAinvAT_lower[t - 1] @ AHinvB_C[t - 1]
            else:
                AHinvB_C[t] -= AHAinvAT_lower[t - 1] @ AHinvB_C[t - 1]

        AHinvAT_AHinvB_C_T_block = solver.solve_batch(AHAinvAT_diag[T], AHinvB_C_T_block)
        AHinvAT_AHinvB_C[T - 1] = AHinvB_C[T - 1] - AHAinvAT_upper[T - 1] @ AHinvAT_AHinvB_C_T_block
        for t in reversed(range(T - 1)):  # backward recursion
            AHinvAT_AHinvB_C[t] = AHinvB_C[t] - AHAinvAT_upper[t] @ AHinvAT_AHinvB_C[t + 1]
        AHinvAT_AHinvB_C = np.stack(AHinvAT_AHinvB_C)


        # multiply by H^-1AT to get H^-1AT(AH^-1A^T)^-1(AH^-1B - C) (left term, projection to constraint surface)
        left_term = HinvAT_diag_blocks @ AHinvAT_AHinvB_C
        left_term[:-1] += HinvAT_upper_blocks[:-1] @ AHinvAT_AHinvB_C[1:]
        left_term[-1] += HinvAT_upper_blocks[-1] @ AHinvAT_AHinvB_C_T_block
        left_term_T_block = HinvAT_diag_T_block @ AHinvAT_AHinvB_C_T_block

        # solve right term (gradient of unconstrained problem) H^-1B and subtract from left term
        right_term = Hinv @ B
        right_term_T_block = H_T_inv @ B_T_block

        combined = [grad_ for grad_ in left_term - right_term]
        combined += [left_term_T_block - right_term_T_block]
        dxdp_traj_vec = [comb[:ns, :] for comb in combined]
        dudp_traj_vec = [comb[ns:, :] for comb in combined[:-1]]

        time_ = [k for k in range(T + 1)]
        sol_full = {'state_traj_opt': dxdp_traj_vec,
                    'control_traj_opt': dudp_traj_vec,
                    'time': time_}

    return sol_full