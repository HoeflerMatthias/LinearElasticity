from firedrake import *
import numpy as np
from petsc4py import PETSc

from . import (
    L2_error, rel_L2_error, pointwise_rel_L2_error, InvScarResult,
    create_box_mesh, create_spaces, symmetry_bcs,
    solve_forward,
    load_ground_truth, apply_noise,
    save_solution_checkpoint,
)

__all__ = ["invscar"]


def invscar(**params):

    # Geometry
    Nx_i = params.get('Nx_inv', 40)
    Ny_i = params.get('Ny_inv', 40)
    Nz_i = params.get('Nz_inv', 20)

    # Physics parameters
    lambda_ = Constant(params.get('lambda_', 650.0))
    mu = Constant(params.get('mu', 8.0))
    p_load = Constant(-1 * params.get('p_load', 10.0))

    # Inverse parameters
    noise_level = params.get('noise_level', 1e-2)
    noise_seed = params.get('noise_seed', 123)

    lam_L2 = params.get('lam_L2', 0.0)
    lam_H1 = params.get('lam_H1', 1e-5)
    lam_TV = params.get('lam_TV', 0.0)
    eps_tv = params.get('eps_tv', 1e-1)

    data_csv = params.get("data_csv", "linear_symcube_p10.h5")

    # Mesh and spaces
    mm_inv = create_box_mesh(Nx_i, Ny_i, Nz_i)
    V_i, Q_i = create_spaces(mm_inv)

    W_i = V_i * Q_i * V_i

    # Boundary conditions
    bcs_i_v = symmetry_bcs(V_i)
    bcs_i_w = symmetry_bcs(W_i.sub(0)) + symmetry_bcs(W_i.sub(2))

    # Functions
    ud = Function(V_i, name="displ_data")

    w_i = Function(W_i)
    u_i, alpha_i, p_i = split(w_i)
    u_ifun, alpha_ifun, _ = w_i.subfunctions

    psi_i = TestFunction(W_i)
    v_i, beta_i, q_i = split(psi_i)

    # Load ground truth and apply noise
    mm_true, alpha_t, u_t = load_ground_truth(data_csv)
    ud.interpolate(u_t)
    rng = apply_noise(ud, bcs_i_v, noise_level, noise_seed)

    u_ifun.interpolate(ud)
    alpha_ifun.interpolate(Constant(1.5))

    # Forward-consistent initial u
    u_fwd0 = solve_forward(alpha_ifun, bcs_i_v, lambda_, mu, p_load, V=V_i)
    gamma = float(params.get('u0_blend_gamma', 0.2))
    u_ifun.assign(u_fwd0)
    u_ifun.dat.data[:] = (1 - gamma) * u_ifun.dat.data_ro + gamma * ud.dat.data_ro
    for bc in bcs_i_v:
        bc.apply(u_ifun)

    # KKT formulation
    def a(u, v, a):
        return (lambda_ * inner(div(u), div(v)) + 2 * a * mu * inner(sym(grad(u)), sym(grad(v)))) * dx

    def F(v):
        return dot(p_load * as_vector((0, 0, 1)), v) * ds(6)

    adjoint_eq = inner(u_i - ud, v_i) * dx + a(v_i, p_i, alpha_i)

    # Derivative-form regularization (specific to KKT)
    dR_L2 = lambda a, b: inner(a, b) * dx
    dR_H1 = lambda a, b: inner(grad(a), grad(b)) * dx
    dR_TV = lambda a, b: inner(grad(a), grad(b)) / sqrt(eps_tv + inner(grad(a), grad(a))) * dx

    reg_term = Constant(0) * inner(alpha_i, beta_i) * dx
    for w, dR in [(lam_L2, dR_L2), (lam_H1, dR_H1), (lam_TV, dR_TV)]:
        if w:
            reg_term = reg_term + Constant(w) * dR(alpha_i, beta_i)

    control = reg_term + 2 * mu * beta_i * inner(sym(grad(u_i)), sym(grad(p_i))) * dx

    state_eq = a(u_i, q_i, alpha_i) - F(q_i)

    J = adjoint_eq + control + state_eq

    # Objective evaluation for monitoring
    def compute_objective_terms(u_fun, alpha_fun):
        J_fid = 0.5 * assemble(inner(u_fun - ud, u_fun - ud) * dx(domain=mm_inv))
        J_reg = 0.0
        if lam_L2:
            J_reg += lam_L2 * 0.5 * assemble(alpha_fun ** 2 * dx(domain=mm_inv))
        if lam_H1:
            J_reg += lam_H1 * 0.5 * assemble(inner(grad(alpha_fun), grad(alpha_fun)) * dx(domain=mm_inv))
        if lam_TV:
            J_reg += lam_TV * assemble(sqrt(eps_tv + inner(grad(alpha_fun), grad(alpha_fun))) * dx(domain=mm_inv))
        return J_fid, J_reg

    # Build solver
    prob = NonlinearVariationalProblem(J, w_i, bcs=bcs_i_w,
                                       form_compiler_parameters={'quadrature_degree': 2})
    solver = NonlinearVariationalSolver(
        prob,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_monitor": None,
            "snes_linesearch_monitor": None,
            "snes_converged_reason": None,
            "ksp_monitor_true_residual": None,
            "ksp_converged_reason": None,
        },
    )

    # Monitor
    J_fid_hist, J_reg_hist = [], []
    err_u_abs_hist, err_u_rel_hist = [], []
    err_alpha_pwrel_hist = []

    def _monitor(snes, it, rnorm):
        J_fid, J_reg = compute_objective_terms(u_ifun, alpha_ifun)
        e_u_abs = L2_error(u_ifun, u_t)
        e_u_rel = rel_L2_error(u_ifun, u_t)
        e_a_pwrel = pointwise_rel_L2_error(alpha_ifun, alpha_t)
        J_fid_hist.append(float(J_fid))
        J_reg_hist.append(float(J_reg))
        err_u_abs_hist.append(e_u_abs)
        err_u_rel_hist.append(e_u_rel)
        err_alpha_pwrel_hist.append(e_a_pwrel)
        PETSc.Sys.Print(f"[it {it:02d}] ||F||={rnorm:8.2e}  J_fid={J_fid:10.4e}  J_reg={J_reg:10.4e}  "
                        f"pwrelL2(a)={e_a_pwrel:7.3e}  absL2(u)={e_u_abs:7.3e}  relL2(u)={e_u_rel:7.3e}")

    solver.snes.setMonitor(_monitor)

    stage = PETSc.Log.Stage("kkt_solve")
    stage.push()
    t0 = PETSc.Log.getTime()
    solver.solve()
    wall_time = PETSc.Log.getTime() - t0
    stage.pop()

    nit = solver.snes.getIterationNumber()
    nlinit = solver.snes.getLinearSolveIterations()

    # Final metrics
    J_fid, J_reg = compute_objective_terms(u_ifun, alpha_ifun)
    final_u_abs = L2_error(u_ifun, u_t)
    final_u_rel = rel_L2_error(u_ifun, u_t)
    final_alpha_pwrel = pointwise_rel_L2_error(alpha_ifun, alpha_t)

    # Build used_params dict
    used_params = {
        'Nx_inv': Nx_i, 'Ny_inv': Ny_i, 'Nz_inv': Nz_i,
        'lambda_': float(lambda_), 'mu': float(mu), 'p_load': float(p_load) * -1,
        'noise_level': noise_level, 'noise_seed': noise_seed,
        'lam_L2': lam_L2, 'lam_H1': lam_H1, 'lam_TV': lam_TV, 'eps_tv': eps_tv,
        'u0_blend_gamma': gamma,
        'data_csv': data_csv,
        'solver': 'kkt',
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
        'nit': nit,
        'nlinit': nlinit,
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
