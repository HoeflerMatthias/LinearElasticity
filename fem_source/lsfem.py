import time

from firedrake import *
import numpy as np
from scipy.optimize import minimize

from . import (
    L2_error, rel_L2_error, pointwise_rel_L2_error, InvScarResult,
    create_box_mesh, create_spaces, symmetry_bcs,
    regularization_functionals,
    load_ground_truth, apply_noise,
    save_solution_checkpoint,
)

__all__ = ["invscar"]



def invscar(**params):

    # Geometry
    Nx_i = params.get('Nx_inv', 10)
    Ny_i = params.get('Ny_inv', 10)
    Nz_i = params.get('Nz_inv', 5)

    u_degree = params.get('P', 1)

    # Physics parameters
    lambda_ = Constant(params.get('lambda_', 650.0))
    mu = Constant(params.get('mu', 8.0))
    p_load = Constant(-1 * params.get('p_load', 10.0))

    # Inverse parameters
    noise_level = params.get('noise_level', 1e-2)
    noise_seed = params.get('noise_seed', 123)

    J_regu = params.get('J_regu', 'H1')

    lam_dat = Constant(params.get('lam_dat', 1e7))
    lam_pde = Constant(params.get('lam_pde', 1e-1))
    lam_bcn = Constant(params.get('lam_bcn', 1e-1))
    lam_reg = Constant(params.get('lam_reg', 2e-4))
    lam_jump = Constant(params.get('lam_jump', 1e6))

    data_csv = params.get("data_csv", "linear_symcube_p10.h5")

    # Mesh and spaces
    mm_inv = create_box_mesh(Nx_i, Ny_i, Nz_i)
    dim = mm_inv.geometric_dimension()
    V_i, Q_i = create_spaces(mm_inv, u_degree=u_degree)

    W_i = V_i * Q_i

    # Boundary conditions
    bcs_i_v = symmetry_bcs(V_i)
    bcs_i_w = symmetry_bcs(W_i.sub(0))

    # Functions
    ud = Function(V_i, name="displ_data")

    w_i = Function(W_i)
    u_i, alpha_i = split(w_i)
    u_ifun, alpha_ifun = w_i.subfunctions

    psi_i = TestFunction(W_i)
    v_i, beta_i = split(psi_i)

    # Load ground truth and apply noise
    mm_true, alpha_t, u_t = load_ground_truth(data_csv)
    ud.interpolate(u_t)
    rng = apply_noise(ud, bcs_i_v, noise_level, noise_seed)

    alpha_ifun.dat.data[:] = rng.uniform(low=0.5, high=2.5, size=alpha_ifun.dat.data.shape)

    # LSFEM residual formulation
    n = FacetNormal(mm_inv)
    h = CellDiameter(mm_inv)
    I = Identity(dim)

    eps_i = sym(grad(u_i))
    jump_trac = jump(grad(u_i))

    e_const = 2 * mu * alpha_i * eps_i + lambda_ * tr(eps_i) * I
    e_PDE = -div(e_const)

    e_bcn_1 = e_const * n - p_load * as_vector((0, 0, 1))
    e_bcn_2 = e_const * n

    J_R = regularization_functionals()[J_regu]

    # Objective terms
    R_data = 0.5 * inner(u_i - ud, u_i - ud) * dx
    R_cont = 0.5 * avg(h) * inner(jump_trac, jump_trac) * dS
    R_PDE = 0.5 * inner(e_PDE, e_PDE) * dx
    R_bcn_1 = 0.5 * inner(e_bcn_1, e_bcn_1) * ds(6)
    R_bcn_2 = 0.5 * (inner(e_bcn_2, e_bcn_2) * ds(2) + inner(e_bcn_2, e_bcn_2) * ds(4))
    R_reg = J_R(alpha_i)

    J = lam_dat * R_data + lam_pde * R_PDE + lam_reg * R_reg + lam_bcn * R_bcn_1 + lam_bcn * R_bcn_2 + lam_jump * R_cont

    # First-order optimality (Newton on AAO)
    J_form = J
    dJ_form = derivative(J, w_i, psi_i)

    def check_terms():
        vals = {
            "data": float(assemble(R_data)),
            "cont": float(assemble(R_cont)),
            "pde": float(assemble(R_PDE)),
            "bcn1": float(assemble(R_bcn_1)),
            "bcn2": float(assemble(R_bcn_2)),
            "reg": float(assemble(R_reg))
        }
        msg = "[terms] " + " ".join(
            f"{k:>6} = {v:12.5e}" for k, v in vals.items()
        )
        print(msg)
        return vals

    # Solve inverse problem
    J_hist, term_hist = [], []
    err_u_abs_hist, err_u_rel_hist = [], []
    err_alpha_pwrel_hist = []

    state = {"k": -1}
    def _record_errors():
        state["k"] += 1
        if state["k"] % 50 == 0:
            terms = check_terms()
            J_total = float(assemble(J))
            e_u_abs = L2_error(u_ifun, u_t)
            e_u_rel = rel_L2_error(u_ifun, u_t)
            e_a_pwrel = pointwise_rel_L2_error(alpha_ifun, alpha_t)
            print(f"[error]   absL2u={e_u_abs:12.5e}  relL2u={e_u_rel:12.5e}  pwrelL2a={e_a_pwrel:12.5e}")
            J_hist.append(J_total)
            term_hist.append(terms)
            err_u_abs_hist.append(e_u_abs)
            err_u_rel_hist.append(e_u_rel)
            err_alpha_pwrel_hist.append(e_a_pwrel)

    # Optimization (alternating u / alpha BFGS)
    lb = np.full_like(w_i.dat.data_ro[1].flatten(), 0.0)
    ub = np.full_like(w_i.dat.data_ro[1].flatten(), np.inf)
    bnds = np.array([lb, ub]).T

    def objective(w_vec, u=True, alpha=True):
        if u:
            if alpha:
                u_ifun.dat.data[:] = w_vec[:u_ifun.dat.data.size].reshape(u_ifun.dat.data.shape)
                alpha_ifun.dat.data[:] = w_vec[u_ifun.dat.data.size:].reshape(alpha_ifun.dat.data.shape)
            if not alpha:
                u_ifun.dat.data[:] = w_vec.reshape(u_ifun.dat.data.shape)
            for bc in bcs_i_w:
                bc.apply(w_i)
        if not u and alpha:
            alpha_ifun.dat.data[:] = w_vec.reshape(alpha_ifun.dat.data.shape)
        return assemble(J_form, form_compiler_parameters={'quadrature_degree': 2 * u_degree})

    def gradient(w_vec, u=True, alpha=True):
        if u:
            if alpha:
                u_ifun.dat.data[:] = w_vec[:u_ifun.dat.data.size].reshape(u_ifun.dat.data.shape)
                alpha_ifun.dat.data[:] = w_vec[u_ifun.dat.data.size:].reshape(alpha_ifun.dat.data.shape)
            if not alpha:
                u_ifun.dat.data[:] = w_vec.reshape(u_ifun.dat.data.shape)
            for bc in bcs_i_w:
                bc.apply(w_i)
        if not u and alpha:
            alpha_ifun.dat.data[:] = w_vec.reshape(alpha_ifun.dat.data.shape)
        data = [d.flatten() for d in assemble(dJ_form, form_compiler_parameters={'quadrature_degree': 2 * u_degree}).dat.data_ro]
        if u:
            if alpha:
                data = np.concatenate(data)
            if not alpha:
                data = data[0]
        if not u and alpha:
            data = data[1]
        return data

    obj_u = lambda x: objective(x, True, False)
    gr_u = lambda x: gradient(x, True, False)

    obj_a = lambda x: objective(x, False, True)
    gr_a = lambda x: gradient(x, False, True)

    bfgs_disp = params.get('bfgs_disp', True)
    n_outer = params.get('n_outer', 15)
    iter_hist = []
    total_nfev = 0
    t0 = time.perf_counter()

    for i in range(n_outer):
        print(f"--- BFGS outer iteration {i} ---")
        print("--- u solve")
        x0 = w_i.dat.data_ro[0].flatten()

        _record_errors()
        result = minimize(obj_u, x0, jac=gr_u, method="L-BFGS-B", tol=1e-10,
                          callback=lambda xk: _record_errors(),
                          options={"disp": bfgs_disp, "gtol": 1e-10, 'ftol': 1e-10, "eps": 1e-10})
        print(result.message, result.nit)
        u_nit, u_nfev = result.nit, result.nfev
        w_vec = result.x
        u_ifun.dat.data[:] = w_vec.reshape(u_ifun.dat.data.shape)

        for bc in bcs_i_w:
            bc.apply(w_i)

        print("--- a solve")
        x0 = w_i.dat.data_ro[1].flatten()

        _record_errors()
        result = minimize(obj_a, x0, jac=gr_a, method="L-BFGS-B", tol=1e-12, bounds=bnds,
                          callback=lambda xk: _record_errors(),
                          options={"disp": bfgs_disp, "gtol": 1e-12, 'ftol': 1e-12, "eps": 1e-12})
        print(result.message, result.nit)
        a_nit, a_nfev = result.nit, result.nfev
        w_vec = result.x

        alpha_ifun.dat.data[:] = w_vec.reshape(alpha_ifun.dat.data.shape)

        total_nfev += u_nfev + a_nfev
        iter_hist.append({'u_nit': u_nit, 'u_nfev': u_nfev, 'a_nit': a_nit, 'a_nfev': a_nfev})

    wall_time = time.perf_counter() - t0

    # Final metrics
    final_terms = check_terms()
    J_final = float(assemble(J_form))
    final_u_abs = L2_error(u_ifun, u_t)
    final_u_rel = rel_L2_error(u_ifun, u_t)
    final_alpha_pwrel = pointwise_rel_L2_error(alpha_ifun, alpha_t)

    # Build used_params dict
    used_params = {
        'Nx_inv': Nx_i, 'Ny_inv': Ny_i, 'Nz_inv': Nz_i,
        'P': u_degree,
        'lambda_': float(lambda_), 'mu': float(mu), 'p_load': float(p_load) * -1,
        'noise_level': noise_level, 'noise_seed': noise_seed,
        'J_regu': J_regu, 'lam_reg': float(lam_reg),
        'lam_dat': float(lam_dat), 'lam_pde': float(lam_pde),
        'lam_bcn': float(lam_bcn), 'lam_jump': float(lam_jump),
        'data_csv': data_csv,
        'solver': 'lsfem',
    }

    metrics = {
        'J_fid_hist': J_hist,
        'term_hist': term_hist,
        'err_u_abs_hist': err_u_abs_hist,
        'err_u_rel_hist': err_u_rel_hist,
        'err_alpha_pwrel_hist': err_alpha_pwrel_hist,
        'J_fid_final': float(J_final),
        'J_reg_final': float(final_terms.get('reg', 0.0)),
        'err_u_abs_final': float(final_u_abs),
        'err_u_rel_final': float(final_u_rel),
        'err_alpha_pwrel_final': float(final_alpha_pwrel),
        'nit': n_outer,
        'nfev': total_nfev,
        'iter_hist': iter_hist,
        'wall_time': wall_time,
    }

    solution_file = save_solution_checkpoint(u_ifun, alpha_ifun, u_true=u_t, alpha_true=alpha_t)

    return InvScarResult(
        params=used_params,
        metrics=metrics,
        solution_file=solution_file,
    )


if __name__ == "__main__":
    invscar()
