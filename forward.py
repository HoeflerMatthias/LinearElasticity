from firedrake import *
import numpy as np
import argparse
from pathlib import Path

from fem_source import (create_box_mesh, create_spaces, symmetry_bcs,
                        solve_forward)

# Output directory
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)


def mollify(f, eps, Q):
    """Helmholtz smoothing: solve (u - eps^2 * Laplacian(u)) = f.

    Returns a smoothed Function on function space Q.  Zero Neumann BCs
    are applied implicitly (natural BCs of the weak form).
    """
    u = Function(Q, name=f.name())
    v = TestFunction(Q)
    F = (u * v + eps**2 * inner(grad(u), grad(v)) - f * v) * dx
    solve(F == 0, u)
    return u


def generate_and_save(name, param_expr, mm, V, Q, bcs, lambda_, mu, p_load,
                      mollify_eps=0.0):
    """Solve forward problem and save HDF5 + VTK + CSV.

    If mollify_eps > 0, the parameter field is smoothed via Helmholtz
    mollification before solving the forward problem.
    """
    if mollify_eps > 0.0:
        name = f"{name}_mollified{mollify_eps:g}"

    param_true = Function(Q, name="alpha_true")
    param_true.interpolate(param_expr)

    if mollify_eps > 0.0:
        param_true = mollify(param_true, mollify_eps, Q)

    u_true = solve_forward(param_true, bcs, lambda_, mu, p_load,
                           V=V, name="u_true")

    # CSV alpha column always stores param * mu for consistent scaling
    csv_param_vals = param_true.dat.data_ro * float(mu)

    # HDF5 checkpoint (for FEM inverse solvers)
    h5_path = str(out_dir / f"{name}.h5")
    with CheckpointFile(h5_path, "w") as chk:
        chk.save_mesh(mm)
        chk.save_function(param_true)
        chk.save_function(u_true)

    # VTK for visualization
    vtk_path = str(out_dir / f"{name}.pvd")
    VTKFile(vtk_path).write(u_true, param_true)

    # CSV for PINNs
    coords = mm.coordinates.dat.data_ro
    u_vals = u_true.dat.data_ro

    eps = sym(grad(u_true))
    strain_data = np.zeros((coords.shape[0], 9))
    for idx, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]):
        f = Function(Q)
        f.project(eps[i, j])
        strain_data[:, idx] = f.dat.data_ro

    data = np.column_stack([coords, u_vals, csv_param_vals, strain_data])
    header = "x,y,z,ux,uy,uz,alpha,e_xx,e_xy,e_xz,e_yx,e_yy,e_yz,e_zx,e_zy,e_zz"
    csv_path = str(out_dir / f"{name}.csv")
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    print(f"[{name}]")
    print(f"  HDF5: {h5_path}")
    print(f"  VTK:  {vtk_path}")
    print(f"  CSV:  {csv_path}  ({coords.shape[0]} points)")


# ── CLI ───────────────────────────────────────────────────────

CASES = ["split", "single_inclusion", "two_inclusions"]

parser = argparse.ArgumentParser(description="Generate forward elasticity data.")
parser.add_argument("cases", nargs="*", default=CASES, choices=CASES,
                    help="Test cases to run (default: all)")
parser.add_argument("--mollify", type=float, default=0.0, metavar="EPS",
                    help="Helmholtz mollification length scale for the parameter field (default: 0 = no smoothing)")
args = parser.parse_args()

# ── Common setup ──────────────────────────────────────────────

Nx_t, Ny_t, Nz_t = 80, 80, 40

mm_true = create_box_mesh(Nx_t, Ny_t, Nz_t)
V_t, Q_t = create_spaces(mm_true)
bcs_t = symmetry_bcs(V_t)

lambda_ = Constant(650.0)
mu = Constant(8.0)
p_load = Constant(-10.0)

x_t = SpatialCoordinate(mm_true)


def make_inclusion_field(inclusions, bg=1.0):
    """Build a piecewise-constant field from a list of elliptical inclusions.

    Each entry must have keys cx, cy, a, b, val.
    """
    expr = Constant(bg)
    for inc in inclusions:
        inside = ((x_t[0] - inc["cx"]) / inc["a"])**2 \
               + ((x_t[1] - inc["cy"]) / inc["b"])**2 < 1.0
        expr = conditional(inside, inc["val"], expr)
    return expr


# ── Test case 1: isotropic diagonal split ────────────────────

if "split" in args.cases:
    generate_and_save("linear_symcube_p10",
                      conditional(x_t[0] < x_t[1], 1.0, 2.0),
                      mm_true, V_t, Q_t, bcs_t, lambda_, mu, p_load,
                      mollify_eps=args.mollify)

# ── Test case 2: isotropic single centred inclusion ──────────

if "single_inclusion" in args.cases:
    generate_and_save("linear_symcube_single_inclusion_p10",
                      make_inclusion_field([{"cx": 1.0, "cy": 1.0, "a": 0.7, "b": 0.7, "val": 2.0}]),
                      mm_true, V_t, Q_t, bcs_t, lambda_, mu, p_load,
                      mollify_eps=args.mollify)

# ── Test case 3: isotropic two inclusions ────────────────────

if "two_inclusions" in args.cases:
    generate_and_save("linear_symcube_two_inclusions_p10",
                      make_inclusion_field([
                          {"cx": 0.6, "cy": 0.6, "a": 0.5, "b": 0.5, "val": 2.00},
                          {"cx": 1.6, "cy": 1.6, "a": 0.3, "b": 0.3, "val": 3.00},
                          {"cx": 1.6, "cy": 1.3, "a": 0.2, "b": 0.6, "val": 3.00},
                          {"cx": 1.3, "cy": 1.6, "a": 0.6, "b": 0.2, "val": 3.00},
                      ]),
                      mm_true, V_t, Q_t, bcs_t, lambda_, mu, p_load,
                      mollify_eps=args.mollify)
