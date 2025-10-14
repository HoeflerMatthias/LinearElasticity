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

    Nx_t = params.get('Nx_true', 40)
    Ny_t = params.get('Ny_true', 40)
    Nz_t = params.get('Nz_true', 20)

    Nx_i = params.get('Nx_inv', 20)
    Ny_i = params.get('Ny_inv', 20)
    Nz_i = params.get('Nz_inv', 10)

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

    lam_dat = Constant(params.get('lam_pde', 1e0))
    lam_pde = Constant(params.get('lam_pde', 1e0))
    lam_bcn = Constant(params.get('lam_bcn', 1e0))
    lam_reg = Constant(params.get('lam_reg', 1e-4))

    # File handling
    tag = params.get('run_name', None)
    if tag is None:
        tag = f"Ntrue{Nx_t}x{Ny_t}x{Nz_t}_Ninv{Nx_i}x{Ny_i}x{Nz_i}_pload{float(p_load)}_noise{float(noise_level)}"

    # output root and run directory
    out_root = Path(params.get('out_root', 'runs2'))
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
        'lam_pde': float(lam_pde), 'lam_bcn': float(lam_bcn), 'lam_reg': float(lam_reg)
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

    V_i = VectorFunctionSpace(mm_inv, "P", 3)  # u
    Q_i = FunctionSpace(mm_inv, "P", 1)  # alpha

    W_i = V_i * Q_i

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
    ]

    # Functions
    alpha_t = Function(Q_t, name="alpha_true")
    alpha_t.interpolate(alpha_expr_t)

    u_t = Function(V_t, name="u_true")

    ud = Function(V_i, name="displ_data")

    w_i = Function(W_i)
    u_i, alpha_i = split(w_i)
    u_ifun, alpha_ifun = w_i.subfunctions

    psi_i = TestFunction(W_i)
    v_i, beta_i = split(psi_i)

    # Model
    eps_t = sym(grad(u_t))

    W_t = (lambda_ / 2) * tr(eps_t) ** 2 * dx \
          + alpha_t * mu * inner(eps_t, eps_t) * dx \
          - dot(p_load * Constant((0.0, 0.0, 1.0)), u_t) * ds(6)

    G_t = derivative(W_t, u_t)

    # Ground-truth solution
    fwd_prob_t = NonlinearVariationalProblem(G_t, u_t, bcs_t, form_compiler_parameters={'quadrature_degree': 4})
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
    par = lambda a: ln(1 + exp(a))  # smooth approximation to max(0,a)
    alpha_ifun.assign(1.3)
    # --- Forward-consistent u0 from PDE, then optional blend with data ---
    def solve_forward_u(alpha_guess):
        """Solve the forward equilibrium on the inversion mesh for a given alpha."""
        u0 = Function(V_i, name="u0")
        eps0 = sym(grad(u0))
        W0 = (lambda_ / 2) * tr(eps0) ** 2 * dx \
             + mu * par(alpha_guess) * inner(eps0, eps0) * dx \
             - dot(p_load * as_vector((0, 0, 1)), u0) * ds(6)
        G0 = derivative(W0, u0)
        prob0 = NonlinearVariationalProblem(G0, u0, bcs_i_v, form_compiler_parameters={'quadrature_degree': 4})
        NonlinearVariationalSolver(prob0).solve()
        return u0

    u_fwd0 = solve_forward_u(alpha_ifun)
    gamma = float(params.get('u0_blend_gamma', 0.2))  # 0 → pure PDE; 1 → pure data
    u_ifun.assign(u_fwd0)
    # blend safely at DoF level to stay in the same space
    u_ifun.dat.data[:] = (1 - gamma) * u_ifun.dat.data_ro + gamma * ud.dat.data_ro

    for bc in bcs_i_v:  bc.apply(u_ifun)

    eps_i = sym(grad(u_i))
    I = Identity(dim)
    n = FacetNormal(mm_inv)
    # Objective terms

    h = CellDiameter(mm_inv)

    R_data = 0.5 * inner(u_i - ud, u_i - ud) * dx

    e_const = 2 * mu * par(alpha_i) * eps_i + lambda_ * tr(eps_i) * I
    jump_trac = jump(grad(u_i), n)#jump(dot(e_const, n))
    e_PDE = -div(e_const)
    R_PDE = 0.5 * h ** 2 * inner(e_PDE, e_PDE) * dx + 0.5 * avg(h) * inner(jump_trac, jump_trac) * dS

    e_bcn_1 = e_const * n - p_load * as_vector((0, 0, 1))
    R_bcn_1 = 0.5 * inner(e_bcn_1, e_bcn_1) * ds(6)

    e_bcn_2 = e_const * n
    R_bcn_2 = 0.5 * (inner(e_bcn_2, e_bcn_2) * ds(2) + inner(e_bcn_2, e_bcn_2) * ds(4))

    R_reg = 0.5 * inner(grad(par(alpha_i)), grad(par(alpha_i))) * dx

    J = lam_dat * R_data + lam_pde * R_PDE + lam_bcn * R_bcn_1 + lam_bcn * R_bcn_2 + lam_reg * R_reg

    def check_terms():
        vals = {
            "data": float(assemble(R_data)),
            "pde": float(assemble(R_PDE)),
            "bcn1": float(assemble(R_bcn_1)),
            "bcn2": float(assemble(R_bcn_2)),
            "reg": float(assemble(R_reg))
        }
        PETSc.Sys.Print("[init terms]", vals, comm=PETSc.COMM_WORLD)

    check_terms()

    # First-order optimality (Newton on AAO)
    dJ = derivative(J, w_i, psi_i)
    Jw = derivative(dJ, w_i, TrialFunction(W_i))

    # -----------------------------
    # Build solver with monitor for per-iteration outputs
    # -----------------------------
    solver_parameters = {
        "snes_type": "qn",
        "snes_qn_type": "lbfgs",
        "snes_qn_m": 20,
        "snes_qn_scale_type": "diagonal",
        "snes_linesearch_type": "cp",       # not "l2"
        #"snes_linesearch_damping": 0.4,
        "snes_rtol": 1e-10,
        "snes_atol": 0.0,
        "snes_max_it": 20000,
    }

    problem = NonlinearVariationalProblem(dJ, w_i, bcs=bcs_i_w, J=Jw, form_compiler_parameters={'quadrature_degree': 4})
    solver = NonlinearVariationalSolver(problem, solver_parameters=solver_parameters)

    # VTK writers (iteration index -> "time")
    u_file = File(str(run_dir / "u_iter.pvd"))
    alpha_file = File(str(run_dir / "alpha_iter.pvd"))

    # Histories
    J_hist, r_hist, alpha_err_hist = [], [], []

    # Sample alpha_true to inversion mesh for error tracking
    alpha_true_on_inv = Function(Q_i, name="alpha_true_on_inv")
    # nodal sampling (consistent with CG1 nodes)
    vals_at = alpha_true_on_inv.dat.data
    coords_i = mm_inv.coordinates.dat.data_ro
    for k, X in enumerate(coords_i):
        vals_at[k] = alpha_t.at(X, dont_raise=True)

    def snes_monitor(snes, it, rnorm):
        if it % 100 == 0:
            PETSc.Sys.Print(f"{it:4d} SNES Function norm {rnorm:.12e}")
            alpha_eff = Function(Q_i, name="alpha_eff")
            alpha_eff.interpolate(par(alpha_ifun))
            alpha_file.write(alpha_eff, time=it)
            r_hist.append(float(rnorm))
            J_hist.append(float(assemble(J)))
            u_file.write(u_ifun, time=it)
            e = Function(Q_i);
            e.assign(alpha_eff - alpha_true_on_inv)
            alpha_err_hist.append(float(sqrt(assemble(inner(e, e) * dx))))

    solver.snes.setMonitor(snes_monitor)

    # Solve
    solver.solve()
    rtol = 1e-8
    for s in range(5):
        lam_pde.assign(float(lam_pde) * 1e1)
        lam_bcn.assign(float(lam_bcn) * 1e1)
        #lam_dat.assign(float(lam_dat) * 1e1)

        # keep u feasible (optional but safe)
        for bc in bcs_i_v: bc.apply(u_ifun)

        # tighten SNES stopping criteria
        #solver.snes.setTolerances(rtol=rtol, max_it=2000)

        # solve from current w_i
        solver.solve()

        # optional: check/print reason
        PETSc.Sys.Print("stage done, reason:", solver.snes.getConvergedReason(), comm=PETSc.COMM_WORLD)
        check_terms()
        PETSc.Sys.syncFlush(comm=PETSc.COMM_WORLD)

    # Final field dumps
    File(str(run_dir / "u_final.pvd")).write(u_ifun)
    File(str(run_dir / "alpha_final.pvd")).write(alpha_ifun)

    # Hist plots
    try:
        plt.figure();
        plt.semilogy(r_hist, marker='o');
        plt.xlabel('Newton iter');
        plt.ylabel('||F||');
        plt.title('Nonlinear residual');
        plt.grid(True, which='both', ls=':')
        plt.tight_layout();
        plt.savefig(run_dir / "residual_history.png", dpi=150);
        plt.close()

        plt.figure();
        plt.semilogy(J_hist, marker='o');
        plt.xlabel('Newton iter');
        plt.ylabel('J(w)');
        plt.title('Objective history');
        plt.grid(True, which='both', ls=':')
        plt.tight_layout();
        plt.savefig(run_dir / "objective_history.png", dpi=150);
        plt.close()

        plt.figure();
        plt.semilogy(alpha_err_hist, marker='o');
        plt.xlabel('Newton iter');
        plt.ylabel('||alpha - alpha_true||_L2 (inv mesh)');
        plt.title('Alpha error');
        plt.grid(True, which='both', ls=':')
        plt.tight_layout();
        plt.savefig(run_dir / "alpha_error_history.png", dpi=150);
        plt.close()
    except Exception:
        pass  # plotting is optional

    # CSV export
    with open(run_dir / "history.csv", "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["iter", "residual_norm", "J", "alpha_L2_err"])
        nrow = max(len(J_hist), len(r_hist), len(alpha_err_hist))
        for k in range(nrow):
            rr = r_hist[k] if k < len(r_hist) else ""
            jj = J_hist[k] if k < len(J_hist) else ""
            ae = alpha_err_hist[k] if k < len(alpha_err_hist) else ""
            wtr.writerow([k, rr, jj, ae])

    # --- return a simple results object (as promised in the docstring)
    return SimpleNamespace(
        u=u_ifun, alpha=alpha_ifun, res=None,
        u_true=u_t, alpha_true=alpha_t,
        tag=tag, J_hist=J_hist
    )


if __name__ == "__main__":
    # quick test run
    invscar()