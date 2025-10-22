from firedrake import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path
import json
from types import SimpleNamespace

__all__ = ["invscar"]

def L2_error(a, a_ref, rel=True):
    """
    Compute relative L2 error between 'a' and 'a_ref',
    interpolating 'a' onto the mesh of 'a_ref' if needed.
    """
    V_ref = a_ref.function_space()
    mesh_ref = V_ref.mesh()

    mesh_a = a.function_space().mesh()

    if mesh_a is not mesh_ref:
        a_fine = Function(V_ref)
        a_fine.interpolate(a)  # evaluate a on the fine mesh
    else:
        a_fine = a

    # --- 2. Assemble L2 norms on the reference (fine) mesh ---
    if rel:
        err = assemble(dot(a_fine - a_ref, a_fine - a_ref) / (dot(a_ref, a_ref) + 1e-12) * dx(domain=mesh_ref))
    else:
        err = assemble(dot(a_fine - a_ref, a_fine - a_ref) * dx(domain=mesh_ref))

    return float(np.sqrt(err))

def fmt(p):
    return (f"reg{p['J_regu']}_lam{float(p['lam_reg'])}_"
            f"Ninv{p['Nx_inv']}x{p['Ny_inv']}x{p['Nz_inv']}_"
            f"noise{p['noise_level']}"
    )

def invscar(**params):

    # Geometry
    Nx_i = params.get('Nx_inv', 10)
    Ny_i = params.get('Ny_inv', 10)
    Nz_i = params.get('Nz_inv', 5)

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
    lam_reg = Constant(params.get('lam_reg', 1e-3))

    lower_bnd = params.get('lower_bnd', 0.0)
    upper_bnd = params.get('upper_bnd', np.inf)

    data_csv = params.get("data_csv", "linear_symcube_p10.h5")

    # File handling
    tag = fmt({
        'J_regu': J_regu,
        'lam_reg': lam_reg,
        'Nx_inv': Nx_i,
        'Ny_inv': Ny_i,
        'Nz_inv': Nz_i,
        'noise_level': noise_level
    })

    # output root and run directory
    out_root = Path(params.get('out_root', 'runs_red'))
    run_dir = out_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # Geometry
    mm_inv = BoxMesh(Nx_i, Ny_i, Nz_i, Lx, Ly, Lz, hexahedral=False)

    dim = mm_inv.geometric_dimension()

    x_i = SpatialCoordinate(mm_inv)

    # Spaces
    V_i = VectorFunctionSpace(mm_inv, "P", 1)  # u
    Q_i = FunctionSpace(mm_inv, "P", 1)  # alpha

    # Boundary conditions
    bcs = [
        DirichletBC(V_i.sub(0), Constant(0.0), 1),
        DirichletBC(V_i.sub(1), Constant(0.0), 3),
        DirichletBC(V_i.sub(2), Constant(0.0), 5),
    ]

    # Functions
    ud = Function(V_i, name="displ_data")

    u_i = Function(V_i, name="displacement")
    p_i = Function(V_i, name="adjoint")
    alpha_i = Function(Q_i, name="alpha")

    v_i = TestFunction(V_i)

    # --- load forward/ground-truth data from CSV ---
    with CheckpointFile(data_csv, "r") as chk:
        mm_true = chk.load_mesh()
        alpha_t = chk.load_function(mm_true, name="alpha_true")
        u_t = chk.load_function(mm_true, name="u_true")

    ud.interpolate(u_t)

    sigma_u = np.max(np.abs(ud.dat.data_ro), axis=0) / 3

    rng = np.random.default_rng(noise_seed)
    ud.dat.data[:] += noise_level * sigma_u * rng.normal(size=ud.dat.data.shape)

    for bc in bcs:
        bc.apply(ud)

    u_i.interpolate(ud)
    alpha_i.interpolate(Constant(1.5))

    # -----------------------------
    # Inversion problem on inversion mesh
    # -----------------------------

    eps = sym(grad(u_i))
    W = (lambda_/2)*tr(eps)**2 * dx \
      + alpha_i*mu*inner(eps, eps)*dx \
      - dot(p_load*Constant((0.0, 0.0, 1.0)), u_i)*ds(6)

    G = derivative(W, u_i)
    fwd_prob   = NonlinearVariationalProblem(G, u_i, bcs, form_compiler_parameters={'quadrature_degree': 2})
    fwd_solver = NonlinearVariationalSolver(fwd_prob)

    # Objective and regularization
    R_L2 = lambda g: 0.5*g**2*dx
    R_H1 = lambda g: 0.5*inner(grad(g), grad(g))*dx
    R_TV = lambda g: sqrt(1e-2 + inner(grad(g), grad(g)))*dx

    J_ful = lambda d: 0.5*dot(d, d)*dx
    J_bcs = lambda d: 0.5*dot(d, d)*ds

    J_fide = params.get('J_fide', 'full')
    J_regu = params.get('J_regu', 'H1')

    J_R = {'L2': R_L2, 'H1': R_H1, 'TV': R_TV}[J_regu]

    J  = J_ful(u_i - ud) + lam_reg * J_R(alpha_i)
    dJ = lam_reg * derivative(J_R(alpha_i), alpha_i) + derivative(action(G, p_i), alpha_i)

    # Adjoint
    dG = adjoint(derivative(G, u_i))
    La = -dot(u_i - ud, v_i) * ({'full': dx, 'bcs': ds}[J_fide])
    adj_prob   = LinearVariationalProblem(dG, La, p_i, bcs)
    adj_solver = LinearVariationalSolver(adj_prob)

    # Histories
    err_alpha_L2_hist  = []
    err_u_L2_hist      = []
    J_hist             = []

    # -----------------------------
    # Optimization
    # -----------------------------
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

    # Output handling
    ofile_name = str((run_dir / f"out.pvd").as_posix())

    ofile = VTKFile(ofile_name)

    state = {"k": -1}
    def _record_errors_and_write():
        state["k"] += 1
        # ensure forward is consistent
        if state["k"] % 50 == 0:
            fwd_solver.solve()
            J_hist.append(assemble(J))
            err_alpha_L2_hist.append(L2_error(alpha_i, alpha_t))
            err_u_L2_hist.append(L2_error(u_i, u_t, rel=False))
            ofile.write(u_i, alpha_i, time=state["k"])

    # initial metrics
    _record_errors_and_write()

    bfgs_disp = params.get('bfgs_disp', False)
    res = minimize(Jfun, x0, jac=dJfun, tol=1e-10, bounds=bnds,
                   method='L-BFGS-B',
                   callback=lambda xk: _record_errors_and_write(),
                   options={'disp': bfgs_disp})

    # -----------------------------
    # Final metrics and splits
    # -----------------------------
    final_alpha_L2 = L2_error(alpha_i, alpha_t)
    final_u_L2 = L2_error(u_i, u_t)

    J_fid = float(assemble(J_ful(u_i - ud)))
    J_reg = float(assemble(J_R(alpha_i)))

    # -----------------------------
    # CSV export (same format as kkt.py)
    # -----------------------------
    with open((run_dir / f"reconstruction_metrics_{tag}.csv").as_posix(), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "J", "rel_L2(alpha)", "L2(u)"])
        for k in range(len(J_hist)):
            w.writerow([k, J_hist[k], err_alpha_L2_hist[k], err_u_L2_hist[k]])

    # -----------------------------
    # Package result (same fields as kkt.py)
    # -----------------------------
    class InvScarResult:
        pass

    result = InvScarResult()
    result.u = u_i
    result.alpha = alpha_i
    result.res = res
    result.u_true = u_t
    result.alpha_true = alpha_t
    result.J_fid = J_fid
    result.J_reg = J_reg

    result.err_alpha_L2_hist = err_alpha_L2_hist
    result.err_u_L2_hist = err_u_L2_hist
    result.J_hist = J_hist
    result.err_alpha_L2_final = final_alpha_L2
    result.err_u_L2_final = final_u_L2

    result.tag = tag
    result.run_dir = str(run_dir)

    # --- optionally append to global summary CSV ---
    if params.get('append_global_summary', True):
        summary_path = out_root / 'summary.csv'
        header = [
            'tag', 'run_dir', 'Nx_inv', 'Ny_inv', 'Nz_inv',
            'J_regu', 'lmbda', 'noise_level',
            'J_fid', 'J_reg',
            'rel_L2_alpha_final', 'L2_u_final',
            'nit', 'nfev', 'njev', 'success'
        ]
        row = [
            tag, str(run_dir), Nx_i, Ny_i, Nz_i,
            J_regu, float(lam_reg), noise_level,
            float(J_fid), float(J_reg),
            float(final_alpha_L2), float(final_u_L2),
            getattr(res, 'nit', None), getattr(res, 'nfev', None), getattr(res, 'njev', None),
            getattr(res, 'success', None)
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
    invscar()