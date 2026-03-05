import time

from firedrake import *
import numpy as np
from scipy.optimize import minimize

from fem_source import (
    L2_error, rel_L2_error, pointwise_rel_L2_error, InvScarResult,
    create_box_mesh, create_spaces, symmetry_bcs,
    make_forward_solver, regularization_functionals,
    load_ground_truth, apply_noise,
    save_solution_checkpoint,
)

__all__ = ["invscar"]


def invscar(**params):

    # Geometry
    Nx_i = params.get('Nx_inv', 10)
    Ny_i = params.get('Ny_inv', 10)
    Nz_i = params.get('Nz_inv', 5)

    # Physics parameters
    lambda_ = Constant(params.get('lambda_', 650.0))
    mu = Constant(params.get('mu', 8.0))
    p_load = Constant(-1 * params.get('p_load', 10.0))

    # Inverse parameters
    noise_level = params.get('noise_level', 1e-2)
    noise_seed = params.get('noise_seed', 123)

    J_regu = params.get('J_regu', 'H1')
    lam_reg = Constant(params.get('lam_reg', 1e-3))

    lower_bnd = params.get('lower_bnd', 0.0)
    upper_bnd = params.get('upper_bnd', np.inf)

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
    apply_noise(ud, bcs, noise_level, noise_seed)

    u_i.interpolate(ud)
    alpha_i.interpolate(Constant(1.5))

    # Forward solver
    fwd_solver, W, G = make_forward_solver(u_i, alpha_i, bcs, lambda_, mu, p_load)

    # Objective and regularization
    J_R = regularization_functionals()[J_regu]

    J_ful = lambda d: 0.5 * dot(d, d) * dx

    J = J_ful(u_i - ud) + lam_reg * J_R(alpha_i)
    dJ = lam_reg * derivative(J_R(alpha_i), alpha_i) + derivative(action(G, p_i), alpha_i)

    # Adjoint
    dG = adjoint(derivative(G, u_i))
    La = -dot(u_i - ud, v_i) * dx
    adj_prob = LinearVariationalProblem(dG, La, p_i, bcs)
    adj_solver = LinearVariationalSolver(adj_prob)

    # Histories
    J_fid_hist, J_reg_hist = [], []
    err_u_abs_hist, err_u_rel_hist = [], []
    err_alpha_pwrel_hist = []

    # Optimization
    x0 = alpha_i.dat.data_ro.copy()

    lb = np.full_like(x0, lower_bnd)
    ub = np.full_like(x0, upper_bnd)
    bnds = np.array([lb, ub]).T

    def Jfun(xvec):
        alpha_i.dat.data[:] = xvec
        fwd_solver.solve()
        return assemble(J)

    def dJfun(xvec):
        alpha_i.dat.data[:] = xvec
        fwd_solver.solve()
        adj_solver.solve()
        return assemble(dJ).dat.data_ro

    J_fid_form = J_ful(u_i - ud)
    J_reg_form = J_R(alpha_i)

    state = {"k": -1}
    def _record_errors():
        state["k"] += 1
        if state["k"] % 50 == 0:
            fwd_solver.solve()
            J_fid_hist.append(float(assemble(J_fid_form)))
            J_reg_hist.append(float(assemble(J_reg_form)))
            err_u_abs_hist.append(L2_error(u_i, u_t))
            err_u_rel_hist.append(rel_L2_error(u_i, u_t))
            err_alpha_pwrel_hist.append(pointwise_rel_L2_error(alpha_i, alpha_t))

    _record_errors()

    bfgs_disp = params.get('bfgs_disp', False)
    t0 = time.perf_counter()
    res = minimize(Jfun, x0, jac=dJfun, tol=1e-10, bounds=bnds,
                   method='L-BFGS-B',
                   callback=lambda xk: _record_errors(),
                   options={'disp': bfgs_disp})
    wall_time = time.perf_counter() - t0

    # Final metrics
    final_u_abs = L2_error(u_i, u_t)
    final_u_rel = rel_L2_error(u_i, u_t)
    final_alpha_pwrel = pointwise_rel_L2_error(alpha_i, alpha_t)

    J_fid = float(assemble(J_fid_form))
    J_reg = float(assemble(J_reg_form))

    # Build used_params dict (all effective values including defaults)
    used_params = {
        'Nx_inv': Nx_i, 'Ny_inv': Ny_i, 'Nz_inv': Nz_i,
        'lambda_': float(lambda_), 'mu': float(mu), 'p_load': float(p_load) * -1,
        'noise_level': noise_level, 'noise_seed': noise_seed,
        'J_regu': J_regu, 'lam_reg': float(lam_reg),
        'lower_bnd': lower_bnd, 'upper_bnd': upper_bnd,
        'data_csv': data_csv,
        'solver': 'reduced',
    }

    metrics = {
        'J_fid_hist': J_fid_hist,
        'J_reg_hist': J_reg_hist,
        'err_u_abs_hist': err_u_abs_hist,
        'err_u_rel_hist': err_u_rel_hist,
        'err_alpha_pwrel_hist': err_alpha_pwrel_hist,
        'J_fid_final': J_fid,
        'J_reg_final': J_reg,
        'err_u_abs_final': float(final_u_abs),
        'err_u_rel_final': float(final_u_rel),
        'err_alpha_pwrel_final': float(final_alpha_pwrel),
        'nit': getattr(res, 'nit', None),
        'nfev': getattr(res, 'nfev', None),
        'njev': getattr(res, 'njev', None),
        'wall_time': wall_time,
    }

    solution_file = save_solution_checkpoint(u_i, alpha_i, u_true=u_t, alpha_true=alpha_t)

    return InvScarResult(
        params=used_params,
        metrics=metrics,
        solution_file=solution_file,
    )


if __name__ == "__main__":
    invscar()
