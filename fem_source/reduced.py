import time

from firedrake import *
import numpy as np
from scipy.optimize import minimize

from . import (
    L2_error, rel_L2_error, pointwise_rel_L2_error, InvScarResult,
    create_box_mesh, create_spaces, symmetry_bcs,
    make_forward_solver, regularization_functionals,
    load_ground_truth, apply_noise, apply_pressure_noise, make_observation_weight,
    save_solution_checkpoint, gather_alpha_plot_data,
)

__all__ = ["invscar"]


def invscar(seed=0, **params):

    # Geometry
    Nx_i = params.get('Nx_inv', 10)
    Ny_i = params.get('Ny_inv', 10)
    Nz_i = params.get('Nz_inv', 5)

    # Physics parameters
    lambda_ = Constant(params.get('lambda_', 650.0))
    mu = Constant(params.get('mu', 8.0))
    p_load_val = -1 * params.get('p_load', 10.0)

    # Inverse parameters
    noise_level = params.get('noise_level', 1e-2)
    noise_seed = params.get('noise_seed', 123)
    p_noise_level = params.get('p_noise_level', 0.0)

    J_regu = params.get('J_regu', 'H1')
    lam_reg = Constant(params.get('lam_reg', 1e-3))

    lower_bnd = params.get('lower_bnd', 0.0)
    upper_bnd = params.get('upper_bnd', np.inf)

    obs_n_elements = params.get('obs_n_elements', None)

    data_csv = params.get("data_csv", "linear_symcube_p10.h5")

    # Mesh and spaces
    mm_inv = create_box_mesh(Nx_i, Ny_i, Nz_i)
    V_i, Q_i = create_spaces(mm_inv)
    bcs = symmetry_bcs(V_i)

    # Functions
    ud = Function(V_i, name="displ_data")
    u_i = Function(V_i, name="displacement")
    p_i = Function(V_i, name="adjoint")
    alpha_i = Function(Q_i, name="alpha")
    v_i = TestFunction(V_i)

    # Load ground truth and apply noise
    mm_true, alpha_t, u_t = load_ground_truth(data_csv)
    ud.interpolate(u_t)
    rng = apply_noise(ud, bcs, noise_level, noise_seed)
    p_load_func = apply_pressure_noise(mm_inv, p_load_val, p_noise_level, rng)
    p_load = p_load_func if p_load_func is not None else Constant(p_load_val)

    u_i.interpolate(ud)
    alpha_i.interpolate(Constant(1.5))

    # Observation weight (DG0 element mask)
    if obs_n_elements is not None:
        w_obs = make_observation_weight(mm_inv, obs_n_elements, seed=seed)
    else:
        w_obs = Constant(1.0)

    # Forward solver
    fwd_solver, W, G = make_forward_solver(u_i, alpha_i, bcs, lambda_, mu, p_load)

    # Objective and regularization
    J_R = regularization_functionals()[J_regu]

    J_ful = lambda d: 0.5 * dot(d, d) * w_obs * dx

    J = J_ful(u_i - ud) + lam_reg * J_R(alpha_i)
    dJ = lam_reg * derivative(J_R(alpha_i), alpha_i) + derivative(action(G, p_i), alpha_i)

    # Adjoint
    dG = adjoint(derivative(G, u_i))
    La = -dot(u_i - ud, v_i) * w_obs * dx
    adj_prob = LinearVariationalProblem(dG, La, p_i, bcs)
    adj_solver = LinearVariationalSolver(adj_prob)

    # Histories
    J_fid_hist, J_reg_hist = [], []
    err_u_abs_hist, err_u_rel_hist = [], []
    err_alpha_pwrel_hist = []

    J_fid_form = J_ful(u_i - ud)
    J_reg_form = J_R(alpha_i)

    # MPI gather/scatter setup
    from mpi4py import MPI as _MPI
    comm = mm_inv.comm
    rank = comm.rank
    local_size = alpha_i.dat.data.shape[0]
    all_sizes = np.array(comm.allgather(local_size))
    offsets = np.concatenate([[0], np.cumsum(all_sizes[:-1])])
    global_size = int(all_sizes.sum())

    _EVAL_J, _EVAL_DJ, _RECORD, _FINALIZE, _DONE = 0, 1, 2, 3, 4

    def _scatter_alpha(xvec_global):
        local_x = np.empty(local_size)
        comm.Scatterv([xvec_global, all_sizes, offsets, _MPI.DOUBLE], local_x, root=0)
        alpha_i.dat.data[:] = local_x

    def _gather_grad():
        local_grad = assemble(dJ).dat.data_ro.copy()
        full_grad = np.empty(global_size) if rank == 0 else None
        comm.Gatherv(local_grad, [full_grad, all_sizes, offsets, _MPI.DOUBLE], root=0)
        return full_grad

    def _do_record():
        fwd_solver.solve()
        J_fid_hist.append(float(assemble(J_fid_form)))
        J_reg_hist.append(float(assemble(J_reg_form)))
        err_u_abs_hist.append(L2_error(u_i, u_t))
        err_u_rel_hist.append(rel_L2_error(u_i, u_t))
        err_alpha_pwrel_hist.append(pointwise_rel_L2_error(alpha_i, alpha_t))

    # Rank-0 callbacks used by scipy
    def Jfun(xvec):
        eval_count["j"] += 1
        comm.bcast(_EVAL_J, root=0)
        _scatter_alpha(xvec)
        fwd_solver.solve()
        return float(assemble(J))

    def dJfun(xvec):
        eval_count["dj"] += 1
        comm.bcast(_EVAL_DJ, root=0)
        _scatter_alpha(xvec)
        fwd_solver.solve()
        adj_solver.solve()
        return _gather_grad()

    eval_count = {"j": 0, "dj": 0}

    # Rank-0 monitoring
    def _print_eval(tag, val=None):
        if rank == 0:
            msg = f"[reduced] {tag}  (Jfun={eval_count['j']}, dJfun={eval_count['dj']})"
            if val is not None:
                msg += f"  J={val:.6e}"
            print(msg, flush=True)

    state = {"k": -1}
    def _callback(xvec):
        state["k"] += 1
        _print_eval(f"iter {state['k']}", J_fid_hist[-1] if J_fid_hist else None)
        if state["k"] % 50 == 0:
            comm.bcast(_RECORD, root=0)
            _scatter_alpha(xvec)
            _do_record()

    # Worker loop for non-root ranks
    def _worker_loop():
        while True:
            signal = comm.bcast(None, root=0)
            if signal == _DONE:
                break
            _scatter_alpha(None)
            if signal == _EVAL_J:
                fwd_solver.solve()
                assemble(J)
            elif signal == _EVAL_DJ:
                fwd_solver.solve()
                adj_solver.solve()
                local_grad = assemble(dJ).dat.data_ro.copy()
                comm.Gatherv(local_grad, [None, all_sizes, offsets, _MPI.DOUBLE], root=0)
            elif signal == _RECORD:
                _do_record()
            elif signal == _FINALIZE:
                fwd_solver.solve()

    # Initial error recording (all ranks)
    state["k"] += 1
    _do_record()

    bfgs_disp = params.get('bfgs_disp', False)
    t0 = time.perf_counter()

    # Gather initial x0 to rank 0
    x0_local = alpha_i.dat.data_ro.copy()
    x0_global = np.empty(global_size) if rank == 0 else None
    comm.Gatherv(x0_local, [x0_global, all_sizes, offsets, _MPI.DOUBLE], root=0)

    if rank == 0:
        lb = np.full(global_size, lower_bnd)
        ub = np.full(global_size, upper_bnd)
        bnds = np.array([lb, ub]).T

        res = minimize(Jfun, x0_global, jac=dJfun, tol=1e-10, bounds=bnds,
                       method='L-BFGS-B', callback=_callback,
                       options={'disp': bfgs_disp})

        # Scatter final solution to all ranks, then exit worker loops
        comm.bcast(_FINALIZE, root=0)
        _scatter_alpha(res.x)
        fwd_solver.solve()
        comm.bcast(_DONE, root=0)
    else:
        _worker_loop()
        res = None

    wall_time = time.perf_counter() - t0

    # Final metrics — forward solve already done, assemble/error are collective
    final_u_abs = L2_error(u_i, u_t)
    final_u_rel = rel_L2_error(u_i, u_t)
    final_alpha_pwrel = pointwise_rel_L2_error(alpha_i, alpha_t)

    J_fid = float(assemble(J_fid_form))
    J_reg = float(assemble(J_reg_form))

    # Build used_params dict (all effective values including defaults)
    used_params = {
        'Nx_inv': Nx_i, 'Ny_inv': Ny_i, 'Nz_inv': Nz_i,
        'lambda_': float(lambda_), 'mu': float(mu), 'p_load': params.get('p_load', 10.0),
        'noise_level': noise_level, 'noise_seed': noise_seed, 'p_noise_level': p_noise_level,
        'J_regu': J_regu, 'lam_reg': float(lam_reg),
        'lower_bnd': lower_bnd, 'upper_bnd': upper_bnd,
        'obs_n_elements': obs_n_elements,
        'data_csv': data_csv,
        'solver': 'reduced',
    }

    metrics = {
        'J_fid_hist': J_fid_hist,
        'J_reg_hist': J_reg_hist,
        'err_u_abs_hist': err_u_abs_hist,
        'err_u_rel_hist': err_u_rel_hist,
        'err_alpha_pwrel_hist': err_alpha_pwrel_hist,
        'J_fid_final': float(J_fid),
        'J_reg_final': float(J_reg),
        'err_u_abs_final': float(final_u_abs),
        'err_u_rel_final': float(final_u_rel),
        'err_alpha_pwrel_final': float(final_alpha_pwrel),
        'nit': getattr(res, 'nit', None) if res is not None else None,
        'nfev': getattr(res, 'nfev', None) if res is not None else None,
        'njev': getattr(res, 'njev', None) if res is not None else None,
        'wall_time': wall_time,
    }

    solution_file = save_solution_checkpoint(u_i, alpha_i)
    alpha_coords, alpha_vals = gather_alpha_plot_data(alpha_i)

    return InvScarResult(
        params=used_params,
        metrics=metrics,
        solution_file=solution_file,
        alpha_coords=alpha_coords,
        alpha_vals=alpha_vals,
    )


if __name__ == "__main__":
    invscar()
