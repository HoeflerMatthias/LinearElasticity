from firedrake import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path
import json

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

    # -----------------------------
    # Geometry: two meshes
    # -----------------------------
    Nx_t = params.get('Nx_true', 80)
    Ny_t = params.get('Ny_true', 80)
    Nz_t = params.get('Nz_true', 40)

    Nx_i = params.get('Nx_inv', 20)
    Ny_i = params.get('Ny_inv', 20)
    Nz_i = params.get('Nz_inv', 10)

    Lx = params.get('Lx', 2.0)
    Ly = params.get('Ly', 2.0)
    Lz = params.get('Lz', 1.0)

    mm_true = BoxMesh(Nx_t, Ny_t, Nz_t, Lx, Ly, Lz, hexahedral=False)
    mm_inv  = BoxMesh(Nx_i, Ny_i, Nz_i, Lx, Ly, Lz, hexahedral=False)

    dim = mm_true.geometric_dimension()

    # Spaces
    V_t = VectorFunctionSpace(mm_true, "P", 1)
    Q_t = FunctionSpace(mm_true, "P", 1)

    V = VectorFunctionSpace(mm_inv,  "P", 1)
    Q = FunctionSpace(mm_inv,  "P", 1)

    # -----------------------------
    # Parameters & physics
    # -----------------------------
    lambda_ = params.get('lambda_', Constant(650.0))
    if not isinstance(lambda_, Constant):
        lambda_ = Constant(lambda_)
    mu = params.get('mu', Constant(8.0))
    if not isinstance(mu, Constant):
        mu = Constant(mu)

    p_load = params.get('p_load', Constant(10.0))
    if not isinstance(p_load, Constant):
        p_load = Constant(p_load)

    # -----------------------------
    # Filename tag
    # -----------------------------
    tag = params.get('run_name', None)
    if tag is None:
        tag = f"Ntrue{Nx_t}x{Ny_t}x{Nz_t}_Ninv{Nx_i}x{Ny_i}x{Nz_i}_pload{float(p_load)}_noise{float(noise_level)}"

    # output root and run directory
    out_root = Path(params.get('out_root', 'runs'))
    run_dir = out_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # persist run metadata
    meta = {
        'Nx_true': Nx_t, 'Ny_true': Ny_t, 'Nz_true': Nz_t,
        'Nx_inv': Nx_i, 'Ny_inv': Ny_i, 'Nz_inv': Nz_i,
        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
        'lambda_': float(lambda_), 'mu': float(mu), 'p_load': float(p_load),
        'J_fide': params.get('J_fide', 'full'),
        'J_regu': params.get('J_regu', 'H1'),
        'lmbda': float(params.get('lmbda', 4e-8)),
        'noise_level': params.get('noise_level', 1e-2),
    }
    with open(run_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # -----------------------------
    # Boundary conditions
    # -----------------------------
    bcs_t = [
        DirichletBC(V_t.sub(0), Constant(0.0), 1),
        DirichletBC(V_t.sub(1), Constant(0.0), 3),
        DirichletBC(V_t.sub(2), Constant(0.0), 5),
    ]
    bcs = [
        DirichletBC(V.sub(0), Constant(0.0), 1),
        DirichletBC(V.sub(1), Constant(0.0), 3),
        DirichletBC(V.sub(2), Constant(0.0), 5),
    ]

    # -----------------------------
    # Truth problem (synthetic data)
    # -----------------------------
    x_t = SpatialCoordinate(mm_true)

    alpha_gt = params.get('alpha_gt', None)
    if alpha_gt is None:
        alpha_expr_t = conditional(x_t[0] < x_t[1], 1.0, 2.0)
    else:
        # accept either callable(alpha_gt)(x) or UFL expr
        try:
            alpha_expr_t = alpha_gt(x_t)
        except Exception:
            alpha_expr_t = alpha_gt

    alpha_true = Function(Q_t, name="alpha_true")
    alpha_true.interpolate(alpha_expr_t)

    u_true = Function(V_t, name="u_true")
    eps_t  = sym(grad(u_true))

    W_t = (lambda_/2)*tr(eps_t)**2 * dx \
        + alpha_true*mu*inner(eps_t, eps_t)*dx \
        + dot(p_load*Constant((0.0, 0.0, 1.0)), u_true)*ds(6)

    G_t = derivative(W_t, u_true)
    fwd_prob_t   = NonlinearVariationalProblem(G_t, u_true, bcs_t,
                        form_compiler_parameters={'quadrature_degree': 2})
    fwd_solver_t = NonlinearVariationalSolver(fwd_prob_t)
    fwd_solver_t.solve()

    # -----------------------------
    # Transfer data to inversion mesh (nodal sampling)
    # -----------------------------
    ud = Function(V, name="displ_data")
    coords_i = mm_inv.coordinates.dat.data_ro
    vals_ud = ud.dat.data
    for k, X in enumerate(coords_i):
        vals_ud[k, :] = u_true.at(X, dont_raise=True)

    # Add noise and re-apply constrained DOFs
    noise_level = params.get('noise_level', 1e-2)
    sigma_u = np.max(np.abs(ud.dat.data_ro),axis=0)/3
    #import pdb; pdb.set_trace()
    # allow deterministic noise via seed
    noise_seed = params.get('noise_seed', None)
    rng = np.random.default_rng(noise_seed)
    ud.dat.data[:] += noise_level * sigma_u * rng.normal(size=ud.dat.data.shape)
    for bc in bcs:
        bc.apply(ud)

    # -----------------------------
    # Inversion problem on inversion mesh
    # -----------------------------
    x = SpatialCoordinate(mm_inv)

    u     = Function(V, name="displacement")
    p     = Function(V, name="adjoint")
    alpha = Function(Q, name="alpha")

    eps = sym(grad(u))
    W = (lambda_/2)*tr(eps)**2 * dx \
      + alpha*mu*inner(eps, eps)*dx \
      + dot(p_load*Constant((0.0, 0.0, 1.0)), u)*ds(6)

    G = derivative(W, u)
    fwd_prob   = NonlinearVariationalProblem(G, u, bcs,
                    form_compiler_parameters={'quadrature_degree': 2})
    fwd_solver = NonlinearVariationalSolver(fwd_prob)

    # Objective and regularization
    R_L2 = lambda g: 0.5*g**2*dx
    R_H1 = lambda g: 0.5*inner(grad(g), grad(g))*dx
    R_TV = lambda g: sqrt(1e-2 + inner(grad(g), grad(g)))*dx

    J_ful = lambda d: 0.5*dot(d, d)*dx
    J_bcs = lambda d: 0.5*dot(d, d)*ds

    J_fide = params.get('J_fide', 'full')
    J_regu = params.get('J_regu', 'H1')
    J_f = {'full': J_ful, 'bcs': J_bcs}[J_fide]
    J_R = {'L2': R_L2, 'H1': R_H1, 'TV': R_TV}[J_regu]

    lmbda = params.get('lmbda', Constant(4e-8))
    if not isinstance(lmbda, Constant):
        lmbda = Constant(lmbda)

    J  = J_f(u - ud) + lmbda * J_R(alpha)
    dJ = lmbda * derivative(J_R(alpha), alpha) + derivative(action(G, p), alpha)

    # Adjoint
    v  = TestFunction(V)
    dG = adjoint(derivative(G, u))
    La = -dot(u - ud, v) * ({'full': dx, 'bcs': ds}[J_fide])
    adj_prob   = LinearVariationalProblem(dG, La, p, bcs)
    adj_solver = LinearVariationalSolver(adj_prob)

    # -----------------------------
    # Error helpers and histories
    # -----------------------------
    def rel_L2_error_alpha(alpha_curr, alpha_ref):
        num = assemble((alpha_curr - alpha_ref)**2 * dx)
        den = assemble(alpha_ref**2 * dx)
        return np.sqrt(num) / (np.sqrt(den) + 1e-16)

    def rel_L2_error_alpha2(alpha_curr, alpha_ref):
        num = assemble(((alpha_curr - alpha_ref) / alpha_ref)**2 * dx)
        #den = assemble(alpha_ref**2 * dx)
        return np.sqrt(num)

    def rel_H1s_error_alpha(alpha_curr, alpha_ref):
        num = assemble(inner(grad(alpha_curr - alpha_ref), grad(alpha_curr - alpha_ref)) * dx)
        den = assemble(inner(grad(alpha_ref), grad(alpha_ref)) * dx)
        if den <= 1e-30:
            return np.sqrt(num)
        return np.sqrt(num) / np.sqrt(den)

    def rel_L2_error_u(u_curr, u_ref):
        num = assemble(dot(u_curr - u_ref, u_curr - u_ref) * dx)
        den = assemble(dot(u_ref, u_ref) * dx)
        return np.sqrt(num) / (np.sqrt(den) + 1e-16)

    # Build truth fields on inversion mesh for diagnostics
    alpha_true_on_inv = Function(Q, name="alpha_true_on_inv")
    # Reuse same analytic expr on inversion mesh
    try:
        alpha_expr_inv = alpha_gt(x)
    except Exception:
        alpha_expr_inv = conditional(x[0] < x[1], 1.0, 2.0) if alpha_gt is None else alpha_gt
    alpha_true_on_inv.interpolate(alpha_expr_inv)

    u_true_on_inv = Function(V, name="u_true_on_inv")
    vals_ui = u_true_on_inv.dat.data
    for k, X in enumerate(coords_i):
        vals_ui[k, :] = u_true.at(X, dont_raise=True)

    # Histories
    err_alpha_L2_hist  = []
    err_alpha_H1s_hist = []
    err_u_L2_hist      = []
    J_hist             = []

    # -----------------------------
    # Optimization
    # -----------------------------
    alpha.interpolate(Constant(1.5))
    x0 = alpha.dat.data_ro.copy()

    lower_bnd = params.get('lower_bnd', 0.0)
    upper_bnd = params.get('upper_bnd', np.inf)
    lb = np.full_like(x0, lower_bnd)
    ub = np.full_like(x0, upper_bnd)
    bnds = np.array([lb, ub]).T

    def Jfun(xvec):
        alpha.dat.data[:] = xvec
        fwd_solver.solve()
        return assemble(J)

    def dJfun(xvec, no_forward=True):
        alpha.dat.data[:] = xvec
        if not no_forward:
            fwd_solver.solve()
        adj_solver.solve()
        return assemble(dJ).dat.data_ro

    # Output handling
    ofile_name = params.get('ofile_name', None)
    if ofile_name is None:
        ofile_name = str((run_dir / f"out_{tag}.pvd").as_posix())

    ofile = VTKFile(ofile_name)

    def _record_errors_and_write():
        # ensure forward is consistent
        fwd_solver.solve()
        J_hist.append(assemble(J))
        err_alpha_L2_hist.append(rel_L2_error_alpha(alpha, alpha_true_on_inv))
        err_alpha_H1s_hist.append(rel_H1s_error_alpha(alpha, alpha_true_on_inv))
        err_u_L2_hist.append(rel_L2_error_u(u, u_true_on_inv))
        ofile.write(alpha, u)

    # initial metrics
    _record_errors_and_write()

    bfgs_disp = params.get('bfgs_disp', False)

    res = minimize(Jfun, x0, jac=dJfun, tol=1e-10, bounds=bnds,
                   method='L-BFGS-B',
                   callback=lambda xk: _record_errors_and_write(),
                   options={'disp': bfgs_disp})

    # final metrics
    final_alpha_L2  = rel_L2_error_alpha(alpha, alpha_true_on_inv)
    final_alpha_L22  = rel_L2_error_alpha2(alpha, alpha_true_on_inv)
    final_alpha_H1s = rel_H1s_error_alpha(alpha, alpha_true_on_inv)
    final_u_L2      = rel_L2_error_u(u, u_true_on_inv)

    # Objective components at optimum
    J_fid = assemble(J_f(u - ud))
    J_reg = assemble(J_R(alpha))

    # -----------------------------
    # Plots & CSV with tag
    # -----------------------------
    # Truth snapshot vs inversion snapshot
    # (save only; avoid plt.show for headless runs)
    # Truth
    pts_true  = mm_true.coordinates.dat.data_ro + u_true.dat.data_ro
    c_true = np.linalg.norm(u_true.dat.data_ro, axis=-1)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.axis('equal')
    scat = ax.scatter(pts_true[:, 0], pts_true[:, 1], pts_true[:, 2], c=c_true, cmap='viridis', s=5)
    ax2 = fig.add_subplot(122, projection='3d')
    scat2 = ax2.scatter(mm_true.coordinates.dat.data_ro[:, 0],
                        mm_true.coordinates.dat.data_ro[:, 1],
                        mm_true.coordinates.dat.data_ro[:, 2],
                        c=alpha_true.dat.data_ro, cmap='viridis', s=5)
    fig.colorbar(scat2, ax=ax2, shrink=0.5, aspect=5)
    fig.savefig((run_dir / f"gt_{tag}.png").as_posix(), dpi=180)
    plt.close(fig)

    # Obtained
    pts_inv  = mm_inv.coordinates.dat.data_ro + u.dat.data_ro
    c_inv = np.linalg.norm(u.dat.data_ro, axis=-1)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    ax.axis('equal')
    scat = ax.scatter(pts_inv[:, 0], pts_inv[:, 1], pts_inv[:, 2], c=c_inv, cmap='viridis', s=5)
    ax2 = fig.add_subplot(122, projection='3d')
    scat2 = ax2.scatter(mm_inv.coordinates.dat.data_ro[:, 0],
                        mm_inv.coordinates.dat.data_ro[:, 1],
                        mm_inv.coordinates.dat.data_ro[:, 2],
                        c=alpha.dat.data_ro, cmap='viridis', s=5)
    fig.colorbar(scat2, ax=ax2, shrink=0.5, aspect=5)
    fig.savefig((run_dir / f"obtained_{tag}.png").as_posix(), dpi=180)
    plt.close(fig)

    # Error history plot
    fig = plt.figure()
    plt.semilogy(J_hist, label='J')
    plt.semilogy(err_alpha_L2_hist, label='||alpha-alpha*||_L2/||alpha*||_L2')
    plt.semilogy(err_alpha_H1s_hist, label='|alpha-alpha*|_H1/|alpha*|_H1')
    plt.semilogy(err_u_L2_hist, label='||u-u*||_L2/||u*||_L2')
    plt.xlabel('Iteration (callback calls)')
    plt.legend()
    plt.tight_layout()
    plt.savefig((run_dir / f"error_history_{tag}.png").as_posix(), dpi=180)
    plt.close(fig)

    # CSV export
    with open((run_dir / f"reconstruction_metrics_{tag}.csv").as_posix(), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "J", "rel_L2(alpha)", "rel_H1s(alpha)", "rel_L2(u)"])
        for k in range(len(J_hist)):
            w.writerow([k, J_hist[k], err_alpha_L2_hist[k], err_alpha_H1s_hist[k], err_u_L2_hist[k]])

    # -----------------------------
    # Package result
    # -----------------------------
    class InvScarResult:
        pass

    result = InvScarResult()
    result.u = u
    result.alpha = alpha
    result.res = res
    result.u_true = u_true
    result.alpha_true = alpha_true
    result.u_true_on_inv = u_true_on_inv
    result.alpha_true_on_inv = alpha_true_on_inv
    result.J_fid = J_fid
    result.J_reg = J_reg

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
            'tag','run_dir','Nx_true','Ny_true','Nz_true','Nx_inv','Ny_inv','Nz_inv',
            'lambda_','mu','p_load','J_fide','J_regu','lmbda','noise_level',
            'J_fid','J_reg',
            'rel_L2_alpha_final','rel_L2_alpha_final2','rel_H1s_alpha_final','rel_L2_u_final',
            'nit','nfev','njev','success'
        ]
        row = [
            tag, str(run_dir), Nx_t, Ny_t, Nz_t, Nx_i, Ny_i, Nz_i,
            float(lambda_), float(mu), float(p_load), params.get('J_fide','full'), params.get('J_regu','H1'), float(lmbda), params.get('noise_level',1e-2),
            float(J_fid), float(J_reg),
            float(final_alpha_L2), float(final_alpha_L22), float(final_alpha_H1s), float(final_u_L2),
            getattr(res, 'nit', None), getattr(res, 'nfev', None), getattr(res, 'njev', None), getattr(res, 'success', None)
        ]
        write_header = not summary_path.exists()
        with open(summary_path, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)

    return result


if __name__ == "__main__":
    # quick test run
    invscar(J_fide='full', J_regu='H1')