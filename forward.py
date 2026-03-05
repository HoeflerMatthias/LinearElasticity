from firedrake import *
import numpy as np
from pathlib import Path

from fem_source import create_box_mesh, create_spaces, symmetry_bcs, solve_forward

# Output directory
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)


def generate_and_save(name, alpha_expr, mm, V, Q, bcs, lambda_, mu, p_load):
    """Solve forward problem and save HDF5 + VTK + CSV."""

    alpha_true = Function(Q, name="alpha_true")
    alpha_true.interpolate(alpha_expr)

    u_true = solve_forward(alpha_true, bcs, lambda_, mu, p_load, V=V, name="u_true")

    # HDF5 checkpoint (for FEM inverse solvers)
    h5_path = str(out_dir / f"{name}.h5")
    with CheckpointFile(h5_path, "w") as chk:
        chk.save_mesh(mm)
        chk.save_function(alpha_true)
        chk.save_function(u_true)

    # VTK for visualization
    vtk_path = str(out_dir / f"{name}.pvd")
    VTKFile(vtk_path).write(u_true, alpha_true)

    # CSV for PINNs
    # alpha column stores alpha * mu (effective shear modulus)
    coords = mm.coordinates.dat.data_ro
    u_vals = u_true.dat.data_ro
    alpha_vals = alpha_true.dat.data_ro
    alpha_mu_vals = alpha_vals * float(mu)

    eps = sym(grad(u_true))
    strain_data = np.zeros((coords.shape[0], 9))
    for idx, (i, j) in enumerate([(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]):
        f = Function(Q)
        f.project(eps[i, j])
        strain_data[:, idx] = f.dat.data_ro

    data = np.column_stack([coords, u_vals, alpha_mu_vals, strain_data])
    header = "x,y,z,ux,uy,uz,alpha,e_xx,e_xy,e_xz,e_yx,e_yy,e_yz,e_zx,e_zy,e_zz"
    csv_path = str(out_dir / f"{name}.csv")
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")

    print(f"[{name}]")
    print(f"  HDF5: {h5_path}")
    print(f"  VTK:  {vtk_path}")
    print(f"  CSV:  {csv_path}  ({coords.shape[0]} points)")


# ── Common setup ──────────────────────────────────────────────

Nx_t, Ny_t, Nz_t = 80, 80, 40

mm_true = create_box_mesh(Nx_t, Ny_t, Nz_t)
V_t, Q_t = create_spaces(mm_true)
bcs_t = symmetry_bcs(V_t)

lambda_ = Constant(650.0)
mu = Constant(8.0)
p_load = Constant(-10.0)

x_t = SpatialCoordinate(mm_true)

# ── Test case 1: diagonal split ──────────────────────────────
# alpha = 1 where x < y, alpha = 2 otherwise

alpha_split = conditional(x_t[0] < x_t[1], 1.0, 2.0)

generate_and_save("linear_symcube_p10", alpha_split,
                  mm_true, V_t, Q_t, bcs_t, lambda_, mu, p_load)

# ── Test case 2: two cylindrical inclusions (along z-axis) ───
# Circular inclusion:  center (0.7, 1.0), radius 0.3, alpha = 2
# Elliptical inclusion: center (1.4, 1.0), semi-axes a=0.4, b=0.2, alpha = 2
# Background: alpha = 1

r_circ_sq = (x_t[0] - 0.7)**2 + (x_t[1] - 1.0)**2
in_circle = r_circ_sq < 0.3**2

r_ell_sq = ((x_t[0] - 1.4) / 0.4)**2 + ((x_t[1] - 1.0) / 0.2)**2
in_ellipse = r_ell_sq < 1.0

alpha_inclusion = conditional(Or(in_circle, in_ellipse), 2.0, 1.0)

generate_and_save("linear_symcube_inclusion_p10", alpha_inclusion,
                  mm_true, V_t, Q_t, bcs_t, lambda_, mu, p_load)
