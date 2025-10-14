from firedrake import *
import ufl
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import json
from types import SimpleNamespace
from petsc4py import PETSc

__all__ = ["invscar"]


def invscar(**params):
    """Solve the inverse scar problem (two-mesh setup with error tracking).

    Parameters
    ----------
        # Geometry
        Nx_true, Ny_true, Nz_true : ints, truth mesh divisions (default: 40,40,20)
        Nx_inv,  Ny_inv,  Nz_inv  : ints, inversion mesh divisions (default: 20,20,10)
        Lx, Ly, Lz                : floats, box size (default: 2.0, 2.0, 1.0)

        # Material / physics
        lambda_     : bulk modulus (Constant or float, default 650.0)
        mu          : shear modulus (Constant or float, default 8.0)
        p_load      : load magnitude (Constant or float, default 10.0)

        # Truth contractility
        alpha_gt    : callable `alpha_expr(x)` or UFL expr; default conditional(x[0] < x[1], 1, 2)

        # Data & objective
        noise_level : float (default 1e-2)
        J_fide      : {'full','bcs'} (default 'full')
        J_regu      : {'L2','H1','TV'} (default 'H1')
        lmbda       : regularization weight (Constant or float, default 4e-8)
        lower_bnd   : float lower bound (default 0)
        upper_bnd   : float upper bound (default +inf)

        # Solver / IO
        bfgs_disp   : bool (default False)
        ofile_name  : str or None for VTK (default: None -> auto-tag)
        show_plot   : bool (default False, saves figures regardless)

    Return
    ------
    A simple object `res` with fields:
        res.u, res.alpha                : final inversion fields (on inversion mesh)
        res.res                         : SciPy result object
        res.u_true, res.alpha_true      : truth fields (on truth mesh)
        res.alpha_true_on_inv, res.u_true_on_inv : truth fields sampled to inversion mesh
        res.J_fid, res.J_reg            : objective terms at optimum
        res.err_*_hist, res.err_*_final : error histories and finals (alpha L2/H1s, u L2)
        res.J_hist                      : objective history
        res.tag                         : filename tag encoding meshes and p_load
    """

    # Geometry
    Nx_t = params.get('Nx_true', 80)
    Ny_t = params.get('Ny_true', 80)
    Nz_t = params.get('Nz_true', 40)

    Nx_i = params.get('Nx_inv', 40)
    Ny_i = params.get('Ny_inv', 40)
    Nz_i = params.get('Nz_inv', 20)

    Lx = params.get('Lx', 2.0)
    Ly = params.get('Ly', 2.0)
    Lz = params.get('Lz', 1.0)

    # Physics parameters
    lambda_ = Constant(params.get('lambda_', 650.0))
    mu = Constant(params.get('mu', 8.0))
    p_load = Constant(-1 * params.get('p_load', 10.0))

    # Inverse parameters
    noise_level = params.get('noise_level', 1e-2)
    noise_seed = params.get('noise_seed', 123)

    J_regu = params.get('J_regu', 'H1')
    lam_reg = Constant(params.get('lam_reg', 1e-5))

    # File handling
    tag = params.get('run_name', None)
    if tag is None:
        tag = f"Ntrue{Nx_t}x{Ny_t}x{Nz_t}_Ninv{Nx_i}x{Ny_i}x{Nz_i}_pload{float(p_load)}_noise{float(noise_level)}"

    # output root and run directory
    out_root = Path(params.get('out_root', 'runs_kkt'))
    run_dir = out_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # persist run metadata
    meta = {
        'Nx_true': Nx_t, 'Ny_true': Ny_t, 'Nz_true': Nz_t,
        'Nx_inv': Nx_i, 'Ny_inv': Ny_i, 'Nz_inv': Nz_i,
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'lambda_': float(lambda_), 'mu': float(mu), 'p_load': float(p_load),
        'noise_level': noise_level,
        'noise_seed': noise_seed,
        'lam_reg': float(lam_reg)
    }

    with open(run_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Geometry
    mm_true = BoxMesh(Nx_t, Ny_t, Nz_t, Lx, Ly, Lz, hexahedral=False)
    mm_inv = BoxMesh(Nx_i, Ny_i, Nz_i, Lx, Ly, Lz, hexahedral=False)

    dim = mm_true.geometric_dimension()

    x_t = SpatialCoordinate(mm_true)
    x_i = SpatialCoordinate(mm_inv)

    # Inverse param
    alpha_expr_t = conditional(x_t[0] < x_t[1], 1.0, 2.0)

    # Spaces
    V_t = VectorFunctionSpace(mm_true, "P", 1)
    Q_t = FunctionSpace(mm_true, "P", 1)

    V_i = VectorFunctionSpace(mm_inv, "P", 1)  # u
    Q_i = FunctionSpace(mm_inv, "P", 1)  # alpha

    W_i = V_i * Q_i * V_i

    # Boundary conditions
    bcs_t = [
        DirichletBC(V_t.sub(0), Constant(0.0), 1),
        DirichletBC(V_t.sub(1), Constant(0.0), 3),
        DirichletBC(V_t.sub(2), Constant(0.0), 5),
    ]

    bcs_i_v = [
        DirichletBC(V_i.sub(0), Constant(0.0), 1),
        DirichletBC(V_i.sub(1), Constant(0.0), 3),
        DirichletBC(V_i.sub(2), Constant(0.0), 5),
    ]

    bcs_i_w = [
        DirichletBC(W_i.sub(0).sub(0), Constant(0.0), 1),  # u_x on facet 1
        DirichletBC(W_i.sub(0).sub(1), Constant(0.0), 3),  # u_y on facet 3
        DirichletBC(W_i.sub(0).sub(2), Constant(0.0), 5),  # u_z on facet 5

        DirichletBC(W_i.sub(2).sub(0), Constant(0.0), 1),  # u_x on facet 1
        DirichletBC(W_i.sub(2).sub(1), Constant(0.0), 3),  # u_y on facet 3
        DirichletBC(W_i.sub(2).sub(2), Constant(0.0), 5),  # u_z on facet 5
    ]

    # Functions
    alpha_t = Function(Q_t, name="alpha_true")
    alpha_t.interpolate(alpha_expr_t)

    u_t = Function(V_t, name="u_true")

    ud = Function(V_i, name="displ_data")

    w_i = Function(W_i)
    u_i, alpha_i, p_i = split(w_i)
    u_ifun, alpha_ifun, p_ifunc = w_i.subfunctions

    psi_i = TestFunction(W_i)
    v_i, beta_i, q_i = split(psi_i)

    # Model
    eps_t = sym(grad(u_t))

    W_t = (lambda_ / 2) * tr(eps_t) ** 2 * dx \
          + alpha_t * mu * inner(eps_t, eps_t) * dx \
          - dot(p_load * Constant((0.0, 0.0, 1.0)), u_t) * ds(6)

    G_t = derivative(W_t, u_t)

    # Ground-truth solution
    fwd_prob_t = NonlinearVariationalProblem(G_t, u_t, bcs_t, form_compiler_parameters={'quadrature_degree': 2})
    fwd_solver_t = NonlinearVariationalSolver(fwd_prob_t)
    fwd_solver_t.solve()

    # Transfer data to inversion mesh (nodal sampling)
    ud.interpolate(u_t)

    # Add noise and re-apply constrained DOFs
    sigma_u = np.max(np.abs(ud.dat.data_ro), axis=0) / 3

    rng = np.random.default_rng(noise_seed)
    ud.dat.data[:] += noise_level * sigma_u * rng.normal(size=ud.dat.data.shape)

    for bc in bcs_i_v:
        bc.apply(ud)

    # Create initial guess
    alpha_ifun.assign(1.5)
    # --- Forward-consistent u0 from PDE, then optional blend with data ---
    def solve_forward_u(alpha_guess):
        """Solve the forward equilibrium on the inversion mesh for a given alpha."""
        u0 = Function(V_i, name="u0")
        eps0 = sym(grad(u0))
        W0 = (lambda_ / 2) * tr(eps0) ** 2 * dx \
             + mu * alpha_guess * inner(eps0, eps0) * dx \
             - dot(p_load * as_vector((0, 0, 1)), u0) * ds(6)
        G0 = derivative(W0, u0)
        prob0 = NonlinearVariationalProblem(G0, u0, bcs_i_v, form_compiler_parameters={'quadrature_degree': 2})
        NonlinearVariationalSolver(prob0).solve()
        return u0

    u_fwd0 = solve_forward_u(alpha_ifun)
    gamma = float(params.get('u0_blend_gamma', 0.2))  # 0 → pure PDE; 1 → pure data
    u_ifun.assign(u_fwd0)
    # blend safely at DoF level to stay in the same space
    u_ifun.dat.data[:] = (1 - gamma) * u_ifun.dat.data_ro + gamma * ud.dat.data_ro

    for bc in bcs_i_v:  bc.apply(u_ifun)

    # Inverse Model
    def a(u, v, a):
        return (lambda_ * inner(div(u), div(v)) + 2 * a * mu * inner(sym(grad(u)), sym(grad(v)))) * dx

    def F(v):
        return dot(p_load * as_vector((0, 0, 1)), v) * ds(6)

    adjoint = inner(u_i - ud, v_i) * dx + a(v_i, p_i, alpha_i)

    dR_L2 = lambda a,b: inner(a,b) * dx
    dR_H1 = lambda a,b: inner(grad(a), grad(b)) * dx
    dR_TV = lambda a,b: inner(grad(a), grad(b)) / sqrt(1e-1 + inner(grad(a), grad(a))) * dx + 1e-2 * dR_L2(a,b)
    dJ_R = {'L2': dR_L2, 'H1': dR_H1, 'TV': dR_TV}[J_regu]

    control = lam_reg * dJ_R(alpha_i, beta_i) + 2 * mu * beta_i * inner(sym(grad(u_i)), sym(grad(p_i))) * dx

    state = a(u_i, q_i, alpha_i) - F(q_i)

    J = adjoint + control + state

    # Solve inverse problem
    # --- truth on inversion mesh for error metrics ---
    alpha_true_on_inv = Function(Q_i, name="alpha_true_on_inv")
    alpha_true_on_inv.interpolate(conditional(x_i[0] < x_i[1], 1.0, 2.0))
    u_true_on_inv = Function(V_i, name="u_true_on_inv")
    u_true_on_inv.interpolate(u_t)

    # --- metric accumulators ---
    J_hist, err_alpha_L2_hist, err_alpha_H1s_hist, err_u_L2_hist = [], [], [], []

    def compute_objective_terms(u_fun, alpha_fun):
        # data misfit
        J_fid = 0.5 * assemble(inner(u_fun - ud, u_fun - ud) * dx(domain=mm_inv))
        # regularizer, consistent with your choices above
        if J_regu == 'L2':
            J_reg = 0.5 * float(lam_reg) * assemble(alpha_fun ** 2 * dx(domain=mm_inv))
        elif J_regu == 'H1':
            J_reg = 0.5 * float(lam_reg) * assemble(inner(grad(alpha_fun), grad(alpha_fun)) * dx(domain=mm_inv))
        else:  # 'TV' (ε-TV + tiny L2, matching your dR_TV)
            eps_tv = 1e-1
            tv = assemble(sqrt(eps_tv + inner(grad(alpha_fun), grad(alpha_fun))) * dx(domain=mm_inv))
            l2 = 0.5 * assemble(alpha_fun ** 2 * dx(domain=mm_inv))
            J_reg = float(lam_reg) * tv + 1e-2 * l2
        return J_fid, J_reg

    def compute_errors(u_fun, alpha_fun):
        # relative L2(alpha)
        num = sqrt(assemble((alpha_fun - alpha_true_on_inv) ** 2 * dx(domain=mm_inv)))
        den = sqrt(assemble(alpha_true_on_inv ** 2 * dx(domain=mm_inv))) + 1e-16
        rel_L2_alpha = float(num / den)
        # relative H1 seminorm(alpha)
        num_h1 = sqrt(assemble(inner(grad(alpha_fun - alpha_true_on_inv),
                                     grad(alpha_fun - alpha_true_on_inv)) * dx(domain=mm_inv)))
        den_h1 = sqrt(assemble(inner(grad(alpha_true_on_inv), grad(alpha_true_on_inv)) * dx(domain=mm_inv))) + 1e-16
        rel_H1s_alpha = float(num_h1 / den_h1)
        # relative L2(u)
        num_u = sqrt(assemble(inner(u_fun - u_true_on_inv, u_fun - u_true_on_inv) * dx(domain=mm_inv)))
        den_u = sqrt(assemble(inner(u_true_on_inv, u_true_on_inv) * dx(domain=mm_inv))) + 1e-16
        rel_L2_u = float(num_u / den_u)
        return rel_L2_alpha, rel_H1s_alpha, rel_L2_u

    # Build problem/solver so we can hook a monitor
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

    # monitor: record J, errors per nonlinear iteration
    def _monitor(snes, it, rnorm):
        J_fid, J_reg = compute_objective_terms(u_ifun, alpha_ifun)
        J_total = float(J_fid + J_reg)
        rel_L2_a, rel_H1s_a, rel_L2_u = compute_errors(u_ifun, alpha_ifun)
        J_hist.append(J_total)
        err_alpha_L2_hist.append(rel_L2_a)
        err_alpha_H1s_hist.append(rel_H1s_a)
        err_u_L2_hist.append(rel_L2_u)
        PETSc.Sys.Print(f"[it {it:02d}] ||F||={rnorm:8.2e}  J={J_total:10.4e}  "
                        f"relL2(a)={rel_L2_a:7.3e}  relH1s(a)={rel_H1s_a:7.3e}  relL2(u)={rel_L2_u:7.3e}")

    solver.snes.setMonitor(_monitor)
    solver.solve()

    # final objective split
    J_fid, J_reg = compute_objective_terms(u_ifun, alpha_ifun)

    # final errors
    final_alpha_L2, final_alpha_H1s, final_u_L2 = compute_errors(u_ifun, alpha_ifun)
    final_alpha_L22 = final_alpha_L2  # keep the extra field name for compatibility

    # CSV export (per-iteration history)
    with open((run_dir / f"reconstruction_metrics_{tag}.csv").as_posix(), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "J", "rel_L2(alpha)", "rel_H1s(alpha)", "rel_L2(u)"])
        for k in range(len(J_hist)):
            w.writerow([k, J_hist[k], err_alpha_L2_hist[k], err_alpha_H1s_hist[k], err_u_L2_hist[k]])

    # Output handling
    ofile_name = str((run_dir / f"out.pvd").as_posix())

    ofile = VTKFile(ofile_name)
    ofile.write(u_ifun, alpha_ifun)

    class InvScarResult:
        pass

    result = InvScarResult()
    result.u = u_ifun
    result.alpha = alpha_ifun
    result.res = None
    result.u_true = u_t
    result.alpha_true = alpha_t
    result.u_true_on_inv = u_true_on_inv
    result.alpha_true_on_inv = alpha_true_on_inv
    result.J_fid = float(J_fid)
    result.J_reg = float(J_reg)

    result.err_alpha_L2_hist = err_alpha_L2_hist
    result.err_alpha_H1s_hist = err_alpha_H1s_hist
    result.err_u_L2_hist = err_u_L2_hist
    result.J_hist = J_hist
    result.err_alpha_L2_final = final_alpha_L2
    result.err_alpha_L2_final2 = final_alpha_L22
    result.err_alpha_H1s_final = final_alpha_H1s
    result.err_u_L2_final = final_u_L2

    result.tag = tag
    result.run_dir = str(run_dir)

    # --- optionally append to global summary CSV ---
    if params.get('append_global_summary', True):
        summary_path = out_root / 'summary.csv'
        header = [
            'tag', 'run_dir', 'Nx_true', 'Ny_true', 'Nz_true', 'Nx_inv', 'Ny_inv', 'Nz_inv',
            'lambda_', 'mu', 'p_load', 'J_fide', 'J_regu', 'lmbda', 'noise_level',
            'J_fid', 'J_reg',
            'rel_L2_alpha_final', 'rel_L2_alpha_final2', 'rel_H1s_alpha_final', 'rel_L2_u_final',
            'nit', 'nfev', 'njev', 'success'
        ]
        row = [
            tag, str(run_dir), Nx_t, Ny_t, Nz_t, Nx_i, Ny_i, Nz_i,
            float(lambda_), float(mu), float(p_load), params.get('J_fide', 'full'), params.get('J_regu', 'H1'),
            float(lam_reg), params.get('noise_level', 1e-2),
            float(J_fid), float(J_reg),
            float(final_alpha_L2), float(final_alpha_L22), float(final_alpha_H1s), float(final_u_L2),
            None, None, None, True
        ]
        write_header = not summary_path.exists()
        with open(summary_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)


if __name__ == "__main__":
    # quick test run
    invscar()