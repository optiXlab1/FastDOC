"""
# Module name: Pontryagin Differentiable Programming (PDP)
# Technical details can be found in the Arxiv Paper:
# Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework
# https://arxiv.org/abs/1912.12970

# If you want to use this modules or part of it in your academic work, please cite our paper:
# @article{jin2019pontryagin,
#   title={Pontryagin Differentiable Programming: An End-to-End Learning and Control Framework},
#   author={Jin, Wanxin and Wang, Zhaoran and Yang, Zhuoran and Mou, Shaoshuai},
#   journal={arXiv preprint arXiv:1912.12970},
#   year={2019}
# }

# Do NOT distribute without written permission from Wanxin Jin
# Do NOT use it for any commercial purpose

# Contact email: wanxinjin@gmail.com
# Last update: May 02, 2020
# Last update: Aug 11, 2020 (add the neural policy parameterization in OC mode)
"""

from casadi import *
import numpy

'''
# =============================================================================================================
# The OCSys class has multiple functionaries: 1) define an optimal control system, 2) solve the optimal control
# system, and 3) obtain the auxiliary control system.

# The standard form of the dynamics of an optimal control system is
# x_k+1= f（x_k, u_k, auxvar)
# The standard form of the cost function of an optimal control system is
# J = sum_0^(T-1) path_cost + final_cost,
# where path_cost = c(x, u, auxvar) and final_cost= h(x, auxvar).
# Note that in the above standard optimal control system form, "auxvar" is the parameter (which can be learned)
# If you don't need the parameter, e.g.m you just want to use this class to solve an optimal control problem,
# instead of learning the parameter, you can ignore setting this augment in your code.

# The procedure to use this class is fairly straightforward, just understand each method by looking at its name:
# Step 1: set state variable ----> setStateVariable
# Step 2: set control variable ----> setControlVariable
# Step 3: set parameter (if applicable) ----> setAuxvarVariable; otherwise you can ignore this step.
# Step 4: set dynamics equation----> setDyn
# Step 5: set path cost function ----> setPathCost
# Step 6: set final cost function -----> setFinalCost
# Step 7: solve the optimal control problem -----> ocSolver
# Step 8: differentiate the Pontryagin's maximum principle (if you have Step 3) -----> diffPMP
# Step 9: get the auxiliary control system (if have Step 3) ------> getAuxSys

# Note that if you are not wanting to learn the parameter in an optimal control system, you can ignore Step 3. 8. 9.
# Note that most of the notations used here are consistent with the notations defined in the PDP paper.
'''


class OCSys:

    def __init__(self, project_name="my optimal control system"):
        self.project_name = project_name

    def setAuxvarVariable(self, auxvar=None):
        if auxvar is None or auxvar.numel() == 0:
            self.auxvar = SX.sym('auxvar')
        else:
            self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.dyn = ode
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control, self.auxvar], [self.dyn])

    def setCostOtherValue(self, value):
        self.cost_other_value = value

    def setPathCost(self, path_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert path_cost.numel() == 1, "path_cost must be a scalar function"

        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.state, self.control, self.auxvar, self.cost_other_value], [self.path_cost])

    def setFinalCost(self, final_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert final_cost.numel() == 1, "final_cost must be a scalar function"

        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function('final_cost', [self.state, self.auxvar, self.cost_other_value], [self.final_cost])


    def setRwPathCost(self, rw_path_cost):
        self.rw_path_cost = rw_path_cost
        self.rw_path_cost_fn = casadi.Function('path_cost', [self.state, self.control, self.auxvar, self.cost_other_value],
                                            [self.rw_path_cost])

    def setRwFinalCost(self, rw_final_cost):
        self.rw_final_cost = rw_final_cost
        self.rw_final_cost_fn = casadi.Function('final_cost', [self.state, self.auxvar, self.cost_other_value],
                                             [self.rw_final_cost])

    def ocSolver(self, ini_state, horizon, cost_other_value, auxvar_value=1,
                 print_level=0, costate_option=0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost function first!"

        if type(ini_state) == numpy.ndarray:
            ini_state = ini_state.flatten().tolist()

        # Start with an empty NLP
        w, w0, lbw, ubw = [], [], [], []
        J, g, lbg, ubg = 0, [], [], []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w += [Xk];
        lbw += ini_state;
        ubw += ini_state;
        w0 += ini_state

        # Formulate the NLP
        for k in range(horizon):
            Uk = MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.control_lb
            ubw += self.control_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]

            Xnext = self.dyn_fn(Xk, Uk, auxvar_value)
            Ck = self.path_cost_fn(Xk, Uk, auxvar_value, cost_other_value[k, :])
            J = J + Ck

            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.state_lb
            ubw += self.state_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]

            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Final cost
        J = J + self.final_cost_fn(Xk, auxvar_value, cost_other_value[horizon, :])

        # Solve NLP
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes',
                'print_time': print_level, 'show_eval_warnings': False}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        sol_traj = numpy.concatenate((w_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        control_traj_opt = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])

        # Costates
        if costate_option == 0:
            costate_traj_opt = numpy.reshape(sol['lam_g'].full().flatten(), (-1, self.n_state))
        else:
            dfx_fun = casadi.Function('dfx', [self.state, self.control, self.auxvar],
                                      [jacobian(self.dyn, self.state)])
            dhx_fun = casadi.Function('dhx', [self.state, self.auxvar],
                                      [jacobian(self.final_cost, self.state)])
            dcx_fun = casadi.Function('dcx', [self.state, self.control, self.auxvar],
                                      [jacobian(self.path_cost, self.state)])
            costate_traj_opt = numpy.zeros((horizon, self.n_state))
            costate_traj_opt[-1, :] = dhx_fun(state_traj_opt[-1, :], auxvar_value)
            for k in range(horizon - 1, 0, -1):
                costate_traj_opt[k - 1, :] = (
                        dcx_fun(state_traj_opt[k, :], control_traj_opt[k, :], auxvar_value).full()
                        + numpy.dot(numpy.transpose(dfx_fun(state_traj_opt[k, :],
                                                            control_traj_opt[k, :],
                                                            auxvar_value).full()),
                                    costate_traj_opt[k, :])
                )

        stats = solver.stats()
        ok = bool(stats.get('success', False))
        status = stats.get('return_status', '')
        iters = stats.get('iter_count', None)
        if not ok:
            status, iters = False, None
            print("[ocSolver] Falling back to constant-control rollout (solver failed).")

            U_hold = numpy.array([0.5 * (lo + hi) for lo, hi in zip(self.control_lb, self.control_ub)], dtype=float)

            state_traj_opt = numpy.zeros((horizon + 1, self.n_state), dtype=float)
            control_traj_opt = numpy.tile(U_hold, (horizon, 1)).astype(float)
            time = numpy.arange(horizon + 1)

            xk = numpy.array(ini_state, dtype=float).ravel()
            state_traj_opt[0, :] = xk

            J_val = 0.0
            for k in range(horizon):
                uk = control_traj_opt[k, :]
                # dyn_fn 返回 casadi.DM，需要 .full().flatten()
                xnext = self.dyn_fn(xk, uk, auxvar_value).full().flatten()
                ck = self.path_cost_fn(xk, uk, auxvar_value, cost_other_value[k, :]).full().item()
                J_val += ck
                state_traj_opt[k + 1, :] = xnext
                xk = xnext
            J_val += self.final_cost_fn(state_traj_opt[-1, :], auxvar_value,
                                        cost_other_value[horizon, :]).full().item()

            costate_traj_opt = numpy.zeros((horizon, self.n_state), dtype=float)

            sol = {'f': numpy.array([[J_val]])}

        # ---------- 统一输出 ----------
        opt_sol = {
            "state_traj_opt": state_traj_opt,
            "control_traj_opt": control_traj_opt,
            "costate_traj_opt": costate_traj_opt,
            'auxvar_value': auxvar_value,
            "cost_other_value": cost_other_value,
            "time": time,
            "horizon": horizon,
            "cost": sol['f'].full() if hasattr(sol['f'], 'full') else sol['f'],
            "solver_success": ok,
        }
        if not ok:
            print(f"[IPOPT] Solve failed fallback used: {status} (iters={iters})")

        return opt_sol

    def diffPMP(self):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'path_cost'), "Define the running cost/reward function first!"
        assert hasattr(self, 'final_cost'), "Define the final cost/reward function first!"

        # Define the Hamiltonian function
        self.costate = casadi.SX.sym('lambda', self.state.numel())
        self.path_Hamil = self.path_cost + dot(self.dyn, self.costate)  # path Hamiltonian
        self.final_Hamil = self.final_cost  # final Hamiltonian

        # Differentiating dynamics; notations here are consistent with the PDP paper
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfe_fn = casadi.Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])

        # First-order derivative of path Hamiltonian
        self.dHx = jacobian(self.path_Hamil, self.state).T
        self.dHx_fn = casadi.Function('dHx', [self.state, self.control, self.costate, self.auxvar, self.cost_other_value], [self.dHx])
        self.dHu = jacobian(self.path_Hamil, self.control).T
        self.dHu_fn = casadi.Function('dHu', [self.state, self.control, self.costate, self.auxvar, self.cost_other_value], [self.dHu])

        self.dJx = jacobian(self.rw_path_cost, self.state).T
        self.dJx_fn = casadi.Function('dJx', [self.state, self.control, self.auxvar, self.cost_other_value], [self.dJx])
        self.dJu = jacobian(self.rw_path_cost, self.control).T
        self.dJu_fn = casadi.Function('dJu', [self.state, self.control, self.auxvar, self.cost_other_value], [self.dJu])

        # Second-order derivative of path Hamiltonian
        self.ddJxx = 2.0 * self.dJx @ self.dJx.T
        self.ddJxu = 2.0 * self.dJx @ self.dJu.T
        self.ddJux = 2.0 * self.dJu @ self.dJx.T
        self.ddJuu = 2.0 * self.dJu @ self.dJu.T

        self.ddJxx_fn = casadi.Function('ddJxx', [self.state, self.control, self.auxvar, self.cost_other_value], [self.ddJxx])
        self.ddJxu_fn = casadi.Function('ddJxu', [self.state, self.control, self.auxvar, self.cost_other_value], [self.ddJxu])
        self.ddJux_fn = casadi.Function('ddJux', [self.state, self.control, self.auxvar, self.cost_other_value], [self.ddJux])
        self.ddJuu_fn = casadi.Function('ddJuu', [self.state, self.control, self.auxvar, self.cost_other_value], [self.ddJuu])

        self.ddHxx = jacobian(self.dHx, self.state)
        self.ddHxx_fn = casadi.Function('ddHxx', [self.state, self.control, self.costate, self.auxvar, self.cost_other_value], [self.ddHxx])
        self.ddHxu = jacobian(self.dHx, self.control)
        self.ddHxu_fn = casadi.Function('ddHxu', [self.state, self.control, self.costate, self.auxvar, self.cost_other_value], [self.ddHxu])
        self.ddHxe = jacobian(self.dHx, self.auxvar)
        self.ddHxe_fn = casadi.Function('ddHxe', [self.state, self.control, self.costate, self.auxvar, self.cost_other_value], [self.ddHxe])
        self.ddHux = jacobian(self.dHu, self.state)
        self.ddHux_fn = casadi.Function('ddHux', [self.state, self.control, self.costate, self.auxvar, self.cost_other_value], [self.ddHux])
        self.ddHuu = jacobian(self.dHu, self.control)
        self.ddHuu_fn = casadi.Function('ddHuu', [self.state, self.control, self.costate, self.auxvar, self.cost_other_value], [self.ddHuu])
        self.ddHue = jacobian(self.dHu, self.auxvar)
        self.ddHue_fn = casadi.Function('ddHue', [self.state, self.control, self.costate, self.auxvar, self.cost_other_value], [self.ddHue])

        # First-order derivative of final Hamiltonian
        self.dhx = jacobian(self.final_Hamil, self.state).T
        self.dhx_fn = casadi.Function('dhx', [self.state, self.auxvar, self.cost_other_value], [self.dhx])

        self.dJhx = jacobian(self.rw_final_cost, self.state).T
        self.dJhx_fn = casadi.Function('dJhx', [self.state, self.auxvar, self.cost_other_value], [self.dJhx])

        # second order differential of path Hamiltonian
        self.ddhxx = jacobian(self.dhx, self.state)
        self.ddhxx_fn = casadi.Function('ddhxx', [self.state, self.auxvar, self.cost_other_value], [self.ddhxx])
        self.ddJhxx = 2.0 * self.dJhx @ self.dJhx.T
        self.ddJhxx_fn = casadi.Function('ddJhxx', [self.state, self.auxvar, self.cost_other_value], [self.ddJhxx])
        self.ddhxe = jacobian(self.dhx, self.auxvar)
        self.ddhxe_fn = casadi.Function('ddhxe', [self.state, self.auxvar, self.cost_other_value], [self.ddhxe])

    def getAuxSys(self, state_traj_opt, control_traj_opt, costate_traj_opt, cost_other_value, auxvar_value = 1,GN = False):
        statement = [hasattr(self, 'dfx_fn'), hasattr(self, 'dfu_fn'), hasattr(self, 'dfe_fn'),
                     hasattr(self, 'ddHxx_fn'), \
                     hasattr(self, 'ddHxu_fn'), hasattr(self, 'ddHxe_fn'), hasattr(self, 'ddHux_fn'),
                     hasattr(self, 'ddHuu_fn'), \
                     hasattr(self, 'ddHue_fn'), hasattr(self, 'ddhxx_fn'), hasattr(self, 'ddhxe_fn'), ]
        if not all(statement):
            self.diffPMP()

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the PDP paper.
        dynF, dynG, dynE = [], [], []
        matHxx, matHxu, matHxe, matHux, matHuu, matHue, mathxx, mathxe, matJxx, matJxu, matJux, matJuu = [], [], [], [], [], [], [], [], [], [], [], []

        # Solve the above coefficient matrices
        for t in range(numpy.size(control_traj_opt, 0)):
            curr_x = state_traj_opt[t, :]
            curr_u = control_traj_opt[t, :]
            cost_other = cost_other_value[t, :]
            next_lambda = costate_traj_opt[t, :]
            dynF += [self.dfx_fn(curr_x, curr_u, auxvar_value).full()]
            dynG += [self.dfu_fn(curr_x, curr_u, auxvar_value).full()]
            dynE += [self.dfe_fn(curr_x, curr_u, auxvar_value).full()]
            if GN:
                matHxx += [self.ddJxx_fn(curr_x, curr_u, auxvar_value, cost_other).full()]
                matHxu += [self.ddJxu_fn(curr_x, curr_u, auxvar_value, cost_other).full()]
                matHux += [self.ddJux_fn(curr_x, curr_u, auxvar_value, cost_other).full()]
                matHuu += [self.ddJuu_fn(curr_x, curr_u, auxvar_value, cost_other).full()]
                mathxx = [self.ddJhxx_fn(state_traj_opt[-1, :], auxvar_value, cost_other_value[-1, :]).full()]
            else:
                matHxx += [self.ddHxx_fn(curr_x, curr_u, next_lambda, auxvar_value, cost_other).full()]
                matHxu += [self.ddHxu_fn(curr_x, curr_u, next_lambda, auxvar_value, cost_other).full()]
                matHux += [self.ddHux_fn(curr_x, curr_u, next_lambda, auxvar_value, cost_other).full()]
                matHuu += [self.ddHuu_fn(curr_x, curr_u, next_lambda, auxvar_value, cost_other).full()]
                mathxx = [self.ddhxx_fn(state_traj_opt[-1, :], auxvar_value, cost_other_value[-1, :]).full()]
            matHxe += [self.ddHxe_fn(curr_x, curr_u, next_lambda, auxvar_value, cost_other).full()]
            matHue += [self.ddHue_fn(curr_x, curr_u, next_lambda, auxvar_value, cost_other).full()]

        mathxe = [self.ddhxe_fn(state_traj_opt[-1, :], auxvar_value, cost_other_value[-1,:]).full()]


        auxSys = {"dynF": dynF,
                  "dynG": dynG,
                  "dynE": dynE,
                  "Hxx": matHxx,
                  "Hxu": matHxu,
                  "Hxe": matHxe,
                  "Hux": matHux,
                  "Huu": matHuu,
                  "Hue": matHue,
                  "hxx": mathxx,
                  "hxe": mathxe
                  }
        return auxSys

'''
# =============================================================================================================
# The LQR class is mainly for solving (time-varying or time-invariant) LQR problems.
# The standard form of the dynamics in the LQR system is
# X_k+1=dynF_k*X_k+dynG_k*U_k+dynE_k,
# where matrices dynF_k, dynG_k, and dynE_k are system dynamics matrices you need to specify (maybe time-varying)
# The standard form of cost function for the LQR system is
# J=sum_0^(horizon-1) path_cost + final cost, where
# path_cost  = trace (1/2*X'*Hxx*X +1/2*U'*Huu*U + 1/2*X'*Hxu*U + 1/2*U'*Hux*X + Hue'*U + Hxe'*X)
# final_cost = trace (1/2*X'*hxx*X +hxe'*X)
# Here, Hxx, Huu, Hux, Hxu, Heu, Hex, hxx, hex are cost matrices you need to specify (maybe time-varying).
# Some of the above dynamics and cost matrices, by default, are zero (none) matrices
# Note that the variable X and variable U can be matrix variables.
# The above defined standard form is consistent with the auxiliary control system defined in the PDP paper
'''


class LQR:

    def __init__(self, project_name="LQR system"):
        self.project_name = project_name

    def setDyn(self, dynF, dynG, dynE=None):
        if type(dynF) is numpy.ndarray:
            self.dynF = [dynF]
            self.n_state = numpy.size(dynF, 0)
        elif type(dynF[0]) is numpy.ndarray:
            self.dynF = dynF
            self.n_state = numpy.size(dynF[0], 0)
        else:
            assert False, "Type of dynF matrix should be numpy.ndarray  or list of numpy.ndarray"

        if type(dynG) is numpy.ndarray:
            self.dynG = [dynG]
            self.n_control = numpy.size(dynG, 1)
        elif type(dynG[0]) is numpy.ndarray:
            self.dynG = dynG
            self.n_control = numpy.size(self.dynG[0], 1)
        else:
            assert False, "Type of dynG matrix should be numpy.ndarray  or list of numpy.ndarray"

        if dynE is not None:
            if type(dynE) is numpy.ndarray:
                self.dynE = [dynE]
                self.n_batch = numpy.size(dynE, 1)
            elif type(dynE[0]) is numpy.ndarray:
                self.dynE = dynE
                self.n_batch = numpy.size(dynE[0], 1)
            else:
                assert False, "Type of dynE matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.dynE = None
            self.n_batch = None

    def setPathCost(self, Hxx, Huu, Hxu=None, Hux=None, Hxe=None, Hue=None):

        if type(Hxx) is numpy.ndarray:
            self.Hxx = [Hxx]
        elif type(Hxx[0]) is numpy.ndarray:
            self.Hxx = Hxx
        else:
            assert False, "Type of path cost Hxx matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if type(Huu) is numpy.ndarray:
            self.Huu = [Huu]
        elif type(Huu[0]) is numpy.ndarray:
            self.Huu = Huu
        else:
            assert False, "Type of path cost Huu matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if Hxu is not None:
            if type(Hxu) is numpy.ndarray:
                self.Hxu = [Hxu]
            elif type(Hxu[0]) is numpy.ndarray:
                self.Hxu = Hxu
            else:
                assert False, "Type of path cost Hxu matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxu = None

        if Hux is not None:
            if type(Hux) is numpy.ndarray:
                self.Hux = [Hux]
            elif type(Hux[0]) is numpy.ndarray:
                self.Hux = Hux
            else:
                assert False, "Type of path cost Hux matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hux = None

        if Hxe is not None:
            if type(Hxe) is numpy.ndarray:
                self.Hxe = [Hxe]
            elif type(Hxe[0]) is numpy.ndarray:
                self.Hxe = Hxe
            else:
                assert False, "Type of path cost Hxe matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxe = None

        if Hue is not None:
            if type(Hue) is numpy.ndarray:
                self.Hue = [Hue]
            elif type(Hue[0]) is numpy.ndarray:
                self.Hue = Hue
            else:
                assert False, "Type of path cost Hue matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hue = None

    def setFinalCost(self, hxx, hxe=None):

        if type(hxx) is numpy.ndarray:
            self.hxx = [hxx]
        elif type(hxx[0]) is numpy.ndarray:
            self.hxx = hxx
        else:
            assert False, "Type of final cost hxx matrix should be numpy.ndarray or list of numpy.ndarray"

        if hxe is not None:
            if type(hxe) is numpy.ndarray:
                self.hxe = [hxe]
            elif type(hxe[0]) is numpy.ndarray:
                self.hxe = hxe
            else:
                assert False, "Type of final cost hxe matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.hxe = None

    def lqrSolver(self, ini_state, horizon):

        # Data pre-processing
        n_state = numpy.size(self.dynF[0], 1)
        if type(ini_state) is list:
            self.ini_x = numpy.array(ini_state, numpy.float64)
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        elif type(ini_state) is numpy.ndarray:
            self.ini_x = ini_state
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        else:
            assert False, "Initial state should be of numpy.ndarray type or list!"

        self.horizon = horizon

        if self.dynE is not None:
            assert self.n_batch == numpy.size(self.dynE[0],
                                              1), "Number of data batch is not consistent with column of dynE"

        # Check the time horizon
        if len(self.dynF) > 1 and len(self.dynF) != self.horizon:
            assert False, "time-varying dynF is not consistent with given horizon"
        elif len(self.dynF) == 1:
            F = self.horizon * self.dynF
        else:
            F = self.dynF

        if len(self.dynG) > 1 and len(self.dynG) != self.horizon:
            assert False, "time-varying dynG is not consistent with given horizon"
        elif len(self.dynG) == 1:
            G = self.horizon * self.dynG
        else:
            G = self.dynG

        if self.dynE is not None:
            if len(self.dynE) > 1 and len(self.dynE) != self.horizon:
                assert False, "time-varying dynE is not consistent with given horizon"
            elif len(self.dynE) == 1:
                E = self.horizon * self.dynE
            else:
                E = self.dynE
        else:
            E = self.horizon * [numpy.zeros(self.ini_x.shape)]

        if len(self.Hxx) > 1 and len(self.Hxx) != self.horizon:
            assert False, "time-varying Hxx is not consistent with given horizon"
        elif len(self.Hxx) == 1:
            Hxx = self.horizon * self.Hxx
        else:
            Hxx = self.Hxx

        if len(self.Huu) > 1 and len(self.Huu) != self.horizon:
            assert False, "time-varying Huu is not consistent with given horizon"
        elif len(self.Huu) == 1:
            Huu = self.horizon * self.Huu
        else:
            Huu = self.Huu

        hxx = self.hxx

        if self.hxe is None:
            hxe = [numpy.zeros(self.ini_x.shape)]

        if self.Hxu is None:
            Hxu = self.horizon * [numpy.zeros((self.n_state, self.n_control))]
        else:
            if len(self.Hxu) > 1 and len(self.Hxu) != self.horizon:
                assert False, "time-varying Hxu is not consistent with given horizon"
            elif len(self.Hxu) == 1:
                Hxu = self.horizon * self.Hxu
            else:
                Hxu = self.Hxu

        if self.Hux is None:  # Hux is the transpose of Hxu
            Hux = self.horizon * [numpy.zeros((self.n_control, self.n_state))]
        else:
            if len(self.Hux) > 1 and len(self.Hux) != self.horizon:
                assert False, "time-varying Hux is not consistent with given horizon"
            elif len(self.Hux) == 1:
                Hux = self.horizon * self.Hux
            else:
                Hux = self.Hux

        if self.Hxe is None:
            Hxe = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        else:
            if len(self.Hxe) > 1 and len(self.Hxe) != self.horizon:
                assert False, "time-varying Hxe is not consistent with given horizon"
            elif len(self.Hxe) == 1:
                Hxe = self.horizon * self.Hxe
            else:
                Hxe = self.Hxe

        if self.Hue is None:
            Hue = self.horizon * [numpy.zeros((self.n_control, self.n_batch))]
        else:
            if len(self.Hue) > 1 and len(self.Hue) != self.horizon:
                assert False, "time-varying Hue is not consistent with given horizon"
            elif len(self.Hue) == 1:
                Hue = self.horizon * self.Hue
            else:
                Hue = self.Hue

        # Solve the Riccati equations: the notations used here are consistent with Lemma 4.2 in the PDP paper
        I = numpy.eye(self.n_state)
        PP = self.horizon * [numpy.zeros((self.n_state, self.n_state))]
        WW = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        PP[-1] = self.hxx[0]
        WW[-1] = self.hxe[0]

        for t in range(self.horizon - 1, 0, -1):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            HxuinvHuu = numpy.matmul(Hxu[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            Q_t = Hxx[t] - numpy.matmul(HxuinvHuu, numpy.transpose(Hxu[t]))
            N_t = Hxe[t] - numpy.matmul(HxuinvHuu, Hue[t])

            temp_mat = numpy.matmul(numpy.transpose(A_t), numpy.linalg.inv(I + numpy.matmul(P_next, R_t)))
            P_curr = Q_t + numpy.matmul(temp_mat, numpy.matmul(P_next, A_t))
            W_curr = N_t + numpy.matmul(temp_mat, W_next + numpy.matmul(P_next, M_t))

            PP[t - 1] = P_curr
            WW[t - 1] = W_curr

            # # # debug
            # Kt = -np.linalg.inv(Huu[t]) @ (
            #         Hux[t] + G[t].T @ np.linalg.inv(np.eye(self.n_state) + P_next @ R_t) @ P_next @ A_t)
            # kt = -np.linalg.inv(Huu[t]) @ (
            #         Hue[t] + G[t].T @ np.linalg.inv(np.eye(self.n_state) + P_next @ R_t) @ (P_next @ M_t + W_next))
            # curr_Muu = Huu[t] + G[t].T @ P_next @ G[t]
            # inv_ZwTMuuZw = np.linalg.inv(curr_Muu)
            # curr_Mue = Hue[t] + G[t].T @ W_next
            # curr_Mux = Hux[t] + G[t].T @ P_next @ F[t]
            # # print('O', inv_ZwTMuuZw)
            # curr_k = -(inv_ZwTMuuZw @ curr_Mue)
            # print('O',  kt)

        # Compute the trajectory using the Raccti matrices obtained from the above: the notations used here are
        # consistent with the PDP paper in Lemma 4.2
        state_traj_opt = (self.horizon + 1) * [numpy.zeros((self.n_state, self.n_batch))]
        control_traj_opt = (self.horizon) * [numpy.zeros((self.n_control, self.n_batch))]
        costate_traj_opt = (self.horizon) * [numpy.zeros((self.n_state, self.n_batch))]
        state_traj_opt[0] = self.ini_x
        for t in range(self.horizon):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))

            x_t = state_traj_opt[t]
            u_t = -numpy.matmul(invHuu, numpy.matmul(numpy.transpose(Hxu[t]), x_t) + Hue[t]) \
                  - numpy.linalg.multi_dot([invHuu, numpy.transpose(G[t]), numpy.linalg.inv(I + numpy.dot(P_next, R_t)),
                                            (numpy.matmul(numpy.matmul(P_next, A_t), x_t) + numpy.matmul(P_next,
                                                                                                         M_t) + W_next)])

            x_next = numpy.matmul(F[t], x_t) + numpy.matmul(G[t], u_t) + E[t]
            lambda_next = numpy.matmul(P_next, x_next) + W_next

            state_traj_opt[t + 1] = x_next
            control_traj_opt[t] = u_t
            costate_traj_opt[t] = lambda_next
        time = [k for k in range(self.horizon + 1)]

        opt_sol = {'state_traj_opt': state_traj_opt,
                   'control_traj_opt': control_traj_opt,
                   'costate_traj_opt': costate_traj_opt,
                   'time': time}
        return opt_sol
