#!/usr/bin/env python3

import argparse
import numpy as np

import odil
import matplotlib.pyplot as plt
from odil import printlog
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import colors as mcolors


# ---------------------- CSV loader (eval only) ----------------------
def load_data_csv_to_grid(domain, path, mu_scale=1.0, dtype=np.float32):
    """
    CSV columns required:
      x,y,z,ux,uy,uz,alpha,e_xx,e_xy,e_xz,e_yx,e_yy,e_yz,e_zx,e_zy,e_zz
    Returns:
      data_u  : (Nx,Ny,Nz,3)
      data_mu : (Nx,Ny,Nz)
      data_E  : (Nx,Ny,Nz,3,3)  # optional (not used in loss)
      mask    : (Nx,Ny,Nz) with 1 where data present, else 0
    """
    import numpy as _np

    arr = _np.genfromtxt(path, delimiter=",", names=True)
    names = [n.lower() for n in arr.dtype.names]
    need = lambda cols: all(c in names for c in cols)
    assert need(["x", "y", "z", "ux", "uy", "uz", "alpha"]), "CSV missing required columns"

    Nx, Ny, Nz = domain.cshape
    x1, y1, z1 = domain.points_1d()

    def nearest_idx(grid, q):
        idx = _np.searchsorted(grid, q)
        idx = _np.clip(idx, 1, len(grid) - 1)
        left = grid[idx - 1]; right = grid[idx]
        idx = _np.where(_np.abs(q - left) <= _np.abs(q - right), idx - 1, idx)
        return idx.astype(_np.int64)

    I = nearest_idx(x1, arr["x"])
    J = nearest_idx(y1, arr["y"])
    K = nearest_idx(z1, arr["z"])

    data_u = _np.zeros((Nx, Ny, Nz, 3), dtype=dtype)
    data_mu = _np.zeros((Nx, Ny, Nz), dtype=dtype)
    mask = _np.zeros((Nx, Ny, Nz), dtype=dtype)

    data_u[I, J, K, 0] = arr["ux"].astype(dtype)
    data_u[I, J, K, 1] = arr["uy"].astype(dtype)
    data_u[I, J, K, 2] = arr["uz"].astype(dtype)
    data_mu[I, J, K] = (arr["alpha"].astype(dtype) * mu_scale)
    mask[I, J, K] = 1

    data_E = None
    if need(["e_xx", "e_xy", "e_xz", "e_yx", "e_yy", "e_yz", "e_zx", "e_zy", "e_zz"]):
        data_E = _np.zeros((Nx, Ny, Nz, 3, 3), dtype=dtype)
        data_E[I, J, K, 0, 0] = arr["e_xx"]; data_E[I, J, K, 0, 1] = arr["e_xy"]; data_E[I, J, K, 0, 2] = arr["e_xz"]
        data_E[I, J, K, 1, 0] = arr["e_yx"]; data_E[I, J, K, 1, 1] = arr["e_yy"]; data_E[I, J, K, 1, 2] = arr["e_yz"]
        data_E[I, J, K, 2, 0] = arr["e_zx"]; data_E[I, J, K, 2, 1] = arr["e_zy"]; data_E[I, J, K, 2, 2] = arr["e_zz"]

    return data_u, data_mu, data_E, mask


# ---------------------- operator ----------------------
def operator_odil(ctx):
    extra = ctx.extra
    mod = ctx.mod
    args = extra.args

    dx, dy, dz = ctx.step()
    ix, iy, iz = ctx.indices()
    nx, ny, nz = ctx.size()

    # ----------------- helpers -----------------
    def as_field(A):
        """Broadcast scalars to grid-shaped fields (same shape as ux)."""
        ux0 = ctx.field("ux", 0, 0, 0)  # template shape (grid)
        try:
            is_scalar = (getattr(A, "shape", None) is None) or (A.shape == ())
        except Exception:
            is_scalar = True
        return mod.ones_like(ux0) * A if is_scalar else A

    def stencil3(key, frozen=False):
        if not args.keep_frozen:
            frozen = False
        return dict(
            c=ctx.field(key, 0, 0, 0, frozen=frozen),
            xm=ctx.field(key, -1, 0, 0, frozen=frozen),
            xp=ctx.field(key, +1, 0, 0, frozen=frozen),
            ym=ctx.field(key, 0, -1, 0, frozen=frozen),
            yp=ctx.field(key, 0, +1, 0, frozen=frozen),
            zm=ctx.field(key, 0, 0, -1, frozen=frozen),
            zp=ctx.field(key, 0, 0, +1, frozen=frozen),
        )

    def apply_dirichlet_minface(st, axis_min_mask):
        e = odil.core.extrap_quadh
        c = st["c"]
        if axis_min_mask == "x":
            st["xm"] = mod.where(ix == 0, e(st["xp"], c, 0), st["xm"])
        elif axis_min_mask == "y":
            st["ym"] = mod.where(iy == 0, e(st["yp"], c, 0), st["ym"])
        elif axis_min_mask == "z":
            st["zm"] = mod.where(iz == 0, e(st["zp"], c, 0), st["zm"])
        return st

    # centered in interior; forward at min; backward at max
    def grad_center_or_one_sided(Q, h, axis, i, n):
        Qp = mod.roll(Q, -1, axis=axis)
        Qm = mod.roll(Q, +1, axis=axis)
        cen = (Qp - Qm) / (2 * h)
        fwd = (Qp - Q) / h
        bwd = (Q - Qm) / h
        return mod.where(i == 0, fwd, mod.where(i == n - 1, bwd, cen))

    # face operators
    def grad_face_normal(Q, h, axis, side):
        return (Q - mod.roll(Q, +1, axis=axis)) / h if side == "min" \
            else (mod.roll(Q, -1, axis=axis) - Q) / h

    def grad_tangent_face(Q, h, axis, i, n):
        Qp = mod.roll(Q, -1, axis=axis)
        Qm = mod.roll(Q, +1, axis=axis)
        cen = (Qp - Qm) / (2 * h)
        fwd = (Qp - Q) / h
        bwd = (Q - Qm) / h
        return mod.where(i == 0, fwd, mod.where(i == n - 1, bwd, cen))

    def face_avg(A, axis, side):
        Ain = mod.roll(A, +1 if side == "min" else -1, axis=axis)
        return 0.5 * (A + Ain)

    # ----------------- displacement stencils -----------------

    ux = stencil3("ux"); uy = stencil3("uy"); uz = stencil3("uz")
    ux = apply_dirichlet_minface(ux, "x")
    uy = apply_dirichlet_minface(uy, "y")
    uz = apply_dirichlet_minface(uz, "z")

    def ddx(st): return (st["xp"] - st["xm"]) / (2 * dx)
    def ddy(st): return (st["yp"] - st["ym"]) / (2 * dy)
    def ddz(st): return (st["zp"] - st["zm"]) / (2 * dz)

    ux_x, ux_y, ux_z = ddx(ux), ddy(ux), ddz(ux)
    uy_x, uy_y, uy_z = ddx(uy), ddy(uy), ddz(uy)
    uz_x, uz_y, uz_z = ddx(uz), ddy(uz), ddz(uz)

    # ----------------- strain -----------------
    exx = ux_x; eyy = uy_y; ezz = uz_z
    exy = 0.5 * (ux_y + uy_x)
    exz = 0.5 * (ux_z + uz_x)
    eyz = 0.5 * (uy_z + uz_y)
    trE = exx + eyy + ezz

    # ----------------- Lamé parameters -----------------
    lam_const = ctx.cast(args.lambda_)
    lam = as_field(lam_const)

    if args.infer_mu:
        mu_r = ctx.field("mu_raw", 0, 0, 0)
        mu = mu_r * mu_r  # grid-shaped
    else:
        # Prefer μ from CSV where available; fallback to constant elsewhere
        if getattr(extra, "data_mu", None) is not None:
            mu_data = mod.constant(extra.data_mu)
            if getattr(extra, "data_mask", None) is not None:
                m = mod.constant(extra.data_mask)
                mu = m * mu_data + (1 - m) * as_field(ctx.cast(args.mu))
            else:
                mu = mu_data
        else:
            mu = as_field(ctx.cast(args.mu))

    # ----------------- nodal stresses -----------------
    sxx = lam * trE + 2.0 * mu * exx
    syy = lam * trE + 2.0 * mu * eyy
    szz = lam * trE + 2.0 * mu * ezz
    sxy = 2.0 * mu * exy
    sxz = 2.0 * mu * exz
    syz = 2.0 * mu * eyz

    # ----------------- div σ -----------------
    sxx_x = grad_center_or_one_sided(sxx, dx, 0, ix, nx)
    sxy_y = grad_center_or_one_sided(sxy, dy, 1, iy, ny)
    sxz_z = grad_center_or_one_sided(sxz, dz, 2, iz, nz)

    sxy_x = grad_center_or_one_sided(sxy, dx, 0, ix, nx)
    syy_y = grad_center_or_one_sided(syy, dy, 1, iy, ny)
    syz_z = grad_center_or_one_sided(syz, dz, 2, iz, nz)

    sxz_x = grad_center_or_one_sided(sxz, dx, 0, ix, nx)
    syz_y = grad_center_or_one_sided(syz, dy, 1, iy, ny)
    szz_z = grad_center_or_one_sided(szz, dz, 2, iz, nz)

    div_sigma_x = sxx_x + sxy_y + sxz_z
    div_sigma_y = sxy_x + syy_y + syz_z
    div_sigma_z = sxz_x + syz_y + szz_z

    # ----------------- PDE residual -----------------
    kpde = ctx.cast(getattr(args, "kpde", 1.0))
    vol = ctx.cast(dx * dy * dz)
    res = []
    res += [
        ("fx", -kpde * div_sigma_x * mod.sqrt(vol)),
        ("fy", -kpde * div_sigma_y * mod.sqrt(vol)),
        ("fz", -kpde * div_sigma_z * mod.sqrt(vol)),
    ]

    # ----------------- Neumann BCs on max faces -----------------
    kneum = ctx.cast(getattr(args, "kneum", 1.0))
    ax = ctx.cast(dy * dz); ay = ctx.cast(dx * dz); az = ctx.cast(dx * dy)
    mask_xp = (ix == nx - 1); mask_yp = (iy == ny - 1); mask_zp = (iz == nz - 1)

    lam_xF = face_avg(lam, 0, "max"); mu_xF = face_avg(mu, 0, "max")
    lam_yF = face_avg(lam, 1, "max"); mu_yF = face_avg(mu, 1, "max")
    lam_zF = face_avg(lam, 2, "max"); mu_zF = face_avg(mu, 2, "max")

    # ---- z = z_max ----
    ux_z_F = grad_face_normal(ctx.field("ux", 0, 0, 0), dz, 2, "max")
    uy_z_F = grad_face_normal(ctx.field("uy", 0, 0, 0), dz, 2, "max")
    uz_z_F = grad_face_normal(ctx.field("uz", 0, 0, 0), dz, 2, "max")

    uz_x_F = grad_tangent_face(ctx.field("uz", 0, 0, 0), dx, 0, ix, nx)
    uz_y_F = grad_tangent_face(ctx.field("uz", 0, 0, 0), dy, 1, iy, ny)
    ux_x_F = grad_tangent_face(ctx.field("ux", 0, 0, 0), dx, 0, ix, nx)
    uy_y_F = grad_tangent_face(ctx.field("uy", 0, 0, 0), dy, 1, iy, ny)

    exz_F = 0.5 * (ux_z_F + uz_x_F)
    eyz_F = 0.5 * (uy_z_F + uz_y_F)
    ezz_F = uz_z_F
    trE_F = ux_x_F + uy_y_F + ezz_F

    sxz_F = 2.0 * mu_zF * exz_F
    syz_F = 2.0 * mu_zF * eyz_F
    szz_F = lam_zF * trE_F + 2.0 * mu_zF * ezz_F

    p_load = ctx.cast(args.pressure)
    res += [
        ("neum_x_face_z", kneum * mod.where(mask_zp, sxz_F - 0.0, ctx.cast(0)) * mod.sqrt(az)),
        ("neum_y_face_z", kneum * mod.where(mask_zp, syz_F - 0.0, ctx.cast(0)) * mod.sqrt(az)),
        ("neum_z_face_z", kneum * mod.where(mask_zp, szz_F + p_load, ctx.cast(0)) * mod.sqrt(az)),
    ]

    # ---- y = y_max ----
    ux_y_F = grad_face_normal(ctx.field("ux", 0, 0, 0), dy, 1, "max")
    uy_y_F = grad_face_normal(ctx.field("uy", 0, 0, 0), dy, 1, "max")
    uz_y_F = grad_face_normal(ctx.field("uz", 0, 0, 0), dy, 1, "max")

    uy_x_F = grad_tangent_face(ctx.field("uy", 0, 0, 0), dx, 0, ix, nx)
    uy_z_F = grad_tangent_face(ctx.field("uy", 0, 0, 0), dz, 2, iz, nz)
    ux_x_Fy = grad_tangent_face(ctx.field("ux", 0, 0, 0), dx, 0, ix, nx)
    uz_z_Fy = grad_tangent_face(ctx.field("uz", 0, 0, 0), dz, 2, iz, nz)

    exy_F = 0.5 * (ux_y_F + uy_x_F)
    eyz_Fy = 0.5 * (uy_z_F + uz_y_F)
    eyy_F = uy_y_F
    trE_Fy = ux_x_Fy + eyy_F + uz_z_Fy

    sxy_F = 2.0 * mu_yF * exy_F
    syy_F = lam_yF * trE_Fy + 2.0 * mu_yF * eyy_F
    syz_Fy = 2.0 * mu_yF * eyz_Fy

    res += [
        ("neum_x_face_y", kneum * mod.where(mask_yp, sxy_F - 0.0, ctx.cast(0)) * mod.sqrt(ay)),
        ("neum_y_face_y", kneum * mod.where(mask_yp, syy_F - 0.0, ctx.cast(0)) * mod.sqrt(ay)),
        ("neum_z_face_y", kneum * mod.where(mask_yp, syz_Fy - 0.0, ctx.cast(0)) * mod.sqrt(ay)),
    ]

    # ---- x = x_max ----
    ux_x_F = grad_face_normal(ctx.field("ux", 0, 0, 0), dx, 0, "max")
    uy_x_F = grad_face_normal(ctx.field("uy", 0, 0, 0), dx, 0, "max")
    uz_x_F = grad_face_normal(ctx.field("uz", 0, 0, 0), dx, 0, "max")

    ux_y_Fx = grad_tangent_face(ctx.field("ux", 0, 0, 0), dy, 1, iy, ny)
    ux_z_Fx = grad_tangent_face(ctx.field("ux", 0, 0, 0), dz, 2, iz, nz)
    uy_y_Fx = grad_tangent_face(ctx.field("uy", 0, 0, 0), dy, 1, iy, ny)
    uz_z_Fx = grad_tangent_face(ctx.field("uz", 0, 0, 0), dz, 2, iz, nz)

    exx_F = ux_x_F
    exy_Fx = 0.5 * (ux_y_Fx + uy_x_F)
    exz_Fx = 0.5 * (ux_z_Fx + uz_x_F)
    trE_Fx = exx_F + uy_y_Fx + uz_z_Fx

    sxx_F = lam_xF * trE_Fx + 2.0 * mu_xF * exx_F
    sxy_Fx = 2.0 * mu_xF * exy_Fx
    sxz_Fx = 2.0 * mu_xF * exz_Fx

    res += [
        ("neum_x_face_x", kneum * mod.where(mask_xp, sxx_F - 0.0, ctx.cast(0)) * mod.sqrt(ax)),
        ("neum_y_face_x", kneum * mod.where(mask_xp, sxy_Fx - 0.0, ctx.cast(0)) * mod.sqrt(ax)),
        ("neum_z_face_x", kneum * mod.where(mask_xp, sxz_Fx - 0.0, ctx.cast(0)) * mod.sqrt(ax)),
    ]

    # ----------------- optional data misfit -----------------
    if getattr(extra, "data_u", None) is not None and getattr(extra, "data_mask", None) is not None and args.kdata_u:
        uxh = ctx.field("ux", 0, 0, 0)
        uyh = ctx.field("uy", 0, 0, 0)
        uzh = ctx.field("uz", 0, 0, 0)
        mU = mod.constant(extra.data_mask)
        dux = mod.constant(extra.data_u[..., 0])
        duy = mod.constant(extra.data_u[..., 1])
        duz = mod.constant(extra.data_u[..., 2])
        kU = ctx.cast(args.kdata_u)
        res += [
            ("data_u_x", kU * mU * (uxh - dux)),
            ("data_u_y", kU * mU * (uyh - duy)),
            ("data_u_z", kU * mU * (uzh - duz)),
        ]

    # ----------------- μ-regularization -----------------
    if args.infer_mu and (args.kmu_reg or args.kmu_prior):
        def nb_roll(Q, axis, i, n, sgn):
            Qr = mod.roll(Q, -sgn, axis=axis)
            clamp = (i == (n - 1)) if (sgn > 0) else (i == 0)
            return mod.where(clamp, Q, Qr)

        mu_xp = nb_roll(mu, 0, ix, nx, +1); mu_xm = nb_roll(mu, 0, ix, nx, -1)
        mu_yp = nb_roll(mu, 1, iy, ny, +1); mu_ym = nb_roll(mu, 1, iy, ny, -1)
        mu_zp = nb_roll(mu, 2, iz, nz, +1); mu_zm = nb_roll(mu, 2, iz, nz, -1)

        gmx = (mu_xp - mu_xm) / (2 * dx)
        gmy = (mu_yp - mu_ym) / (2 * dy)
        gmz = (mu_zp - mu_zm) / (2 * dz)

        res_reg = []
        if args.kmu_reg:
            if args.tv:
                eps = ctx.cast(1e-8)
                tv = mod.sqrt(gmx * gmx + gmy * gmy + gmz * gmz + eps)
                res_reg += [("mu_tv", args.kmu_reg * tv)]
            else:
                res_reg += [("mu_h1_x", args.kmu_reg * gmx),
                            ("mu_h1_y", args.kmu_reg * gmy),
                            ("mu_h1_z", args.kmu_reg * gmz)]
        if args.kmu_prior:
            mu0 = as_field(ctx.cast(args.mu))
            res_reg += [("mu_prior", args.kmu_prior * (mu - mu0))]
        res += res_reg

    return res


# ---------------------- imposed points helpers ----------------------
# (Keep helpers for compatibility, but we now derive imposed from CSV below.)
def get_imposed_indices(domain, args, iflat):
    iflat = np.array(iflat)
    rng = np.random.default_rng(getattr(args, "seed", 0))
    if args.imposed == "random":
        imp_i = iflat.flatten()
        nimp = min(args.nimp, np.prod(imp_i.size))
        perm = rng.permutation(imp_i)
        imp_i = perm[:nimp]
    elif args.imposed == "stripe":
        imp_i = iflat.flatten()
        z = np.array(domain.points("z")).flatten()
        imp_i = imp_i[np.abs(z[imp_i] - 0.5) < 1.0 / 6.0]
        nimp = min(args.nimp, np.prod(imp_i.size))
        perm = rng.permutation(imp_i)
        imp_i = perm[:nimp]
    elif args.imposed == "none":
        imp_i = []
    else:
        raise ValueError("Unknown imposed=" + args.imposed)
    return imp_i


def get_imposed_mask(args, domain):
    mod = domain.mod
    size = np.prod(domain.cshape)
    row = range(size)
    iflat = np.reshape(row, domain.cshape)
    imp_i = np.unique(get_imposed_indices(domain, args, iflat))
    mask = np.zeros(size, dtype=domain.dtype)
    if len(imp_i):
        mask[imp_i] = 1
        points = [mod.flatten(domain.points(i)) for i in range(domain.ndim)]
        points = np.array(points)[:, imp_i].T
    else:
        points = np.zeros((0, domain.ndim))
    return mask.reshape(domain.cshape), points, imp_i


# ---------------------- CLI ----------------------
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Grid
    parser.add_argument("--Nx", type=int, default=80, help="Grid size in x")
    parser.add_argument("--Ny", type=int, default=80, help="Grid size in y")
    parser.add_argument("--Nz", type=int, default=40, help="Grid size in z")

    # Material
    parser.add_argument("--lambda_", type=float, default=650.0, help="Lamé parameter λ (fixed)")
    parser.add_argument("--mu", type=float, default=12.0, help="Lamé parameter μ if not inferred")
    parser.add_argument("--infer_mu", type=int, default=1, help="If 1, infer μ as a grid field")

    # Data (evaluation + optional u-data fitting)
    parser.add_argument("--data_csv", type=str, default="./../linear_symcube_p10.csv",
                        help="CSV with columns: x,y,z,ux,uy,uz,alpha,e_*")
    parser.add_argument("--mu_scale", type=float, default=1.0,
                        help="mu_gt = alpha * mu_scale (evaluation only)")
    parser.add_argument("--kdata_u", type=float, default=5e6,
                        help="Weight for displacement data misfit (0 disables)")

    # Loss weights / BCs
    parser.add_argument("--kneum", type=float, default=7e5, help="Weight for traction BC residuals")
    parser.add_argument("--kpde", type=float, default=5e5, help="Weight for PDE residuals")

    parser.add_argument("--pressure", type=float, default=10.0,
                        help="Pressure magnitude on z_max, traction = -pressure * e3")

    # Imposed points control (now tied to CSV if not 'none')
    parser.add_argument("--imposed", type=str, choices=["random", "stripe", "none"],
                        default="none", help="If not 'none', imposed set is taken from CSV mask")
    parser.add_argument("--nimp", type=int, default=0, help="(Unused with CSV-imposed) Number of imposed points for random")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise on imposed values")

    # μ regularization
    parser.add_argument("--kmu_reg", type=float, default=1e-2,
                        help="Regularization weight on μ (H1 if --tv=0, TV if --tv=1)")
    parser.add_argument("--kmu_prior", type=float, default=0.0,
                        help="Quadratic prior toward --mu (0 disables)")

    parser.add_argument("--tv", type=int, default=0,
                        help="If 1: TV-like reg on μ; else H1 (L2-grad)")

    # Per-component Dirichlet on min faces
    parser.add_argument("--dirichlet_xmin", type=str, default="ux",
        help="Comma-separated components to clamp at x=0 (subset of ux,uy,uz or 'none')")
    parser.add_argument("--dirichlet_ymin", type=str, default="uy",
        help="Comma-separated components to clamp at y=0 (subset of ux,uy,uz or 'none')")
    parser.add_argument("--dirichlet_zmin", type=str, default="uz",
        help="Comma-separated components to clamp at z=0 (subset of ux,uy,uz or 'none')")

    # Visualization
    parser.add_argument("--viz_refdef", type=int, default=1,
                        help="If 1, render 3D reference(mu) and deformed(|u|) with PyVista/mpl")
    parser.add_argument("--def_scale", type=float, default=1.0,
                        help="Visual scale factor for displacement in deformed view")
    parser.add_argument("--viz_img_prefix", type=str, default="viz",
                        help="Prefix for screenshots: viz_ref_*.png / viz_def_*.png")
    parser.add_argument("--viz_edges", type=int, default=0,
                        help="(unused here)")
    parser.add_argument("--viz_cmap", type=str, default=None,
                        help="Optional colormap name")
    parser.add_argument("--export_vtk", type=int, default=0,
                        help="If 1, also export VTK grids (vti) for Paraview")
    parser.add_argument("--vtk_prefix", type=str, default="field",
                        help="Prefix for VTK files, e.g. field_ref_*.vti and field_def_*.vti")

    # ODIL common args
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir="out_elasticity")
    parser.set_defaults(linsolver="direct")
    parser.set_defaults(optimizer="lbfgs")
    parser.set_defaults(lr=0.001)
    parser.set_defaults(double=1)
    parser.set_defaults(multigrid=0)
    parser.set_defaults(plotext="png", plot_title=1)
    parser.set_defaults(plot_every=500, report_every=200, history_full=10, history_every=100, frames=10)
    parser.set_defaults(keep_frozen=1)
    parser.set_defaults(kwreg=0.0, kwregdecay=0)
    return parser.parse_args()


# ---------------------- plotting & metrics ----------------------

from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib import colors as mcolors

def _edges_from_centers_1d(c1d):
    # Uniform (or nearly). Build edges by extrapolating half a step at ends.
    if c1d.size == 1:
        dx = 1.0
    else:
        dx = float(c1d[1] - c1d[0])
    e0 = c1d[0] - 0.5 * dx
    eN = c1d[-1] + 0.5 * dx
    em = 0.5 * (c1d[:-1] + c1d[1:])
    return np.concatenate([[e0], em, [eN]])

def _corners_from_centers_grids(X, Y, Z):
    # Recover 1D center arrays, then make corner (edge) grids
    xc = X[:, 0, 0]
    yc = Y[0, :, 0]
    zc = Z[0, 0, :]
    xe = _edges_from_centers_1d(xc)
    ye = _edges_from_centers_1d(yc)
    ze = _edges_from_centers_1d(zc)
    Xe, Ye, Ze = np.meshgrid(xe, ye, ze, indexing="ij")
    return Xe, Ye, Ze

def _centers_to_corners(Uc: np.ndarray) -> np.ndarray:
    """
    Convert per-cell-center values Uc (Nx,Ny,Nz) to per-corner values (Nx+1,Ny+1,Nz+1)
    by averaging the 8 adjacent cell centers around each corner with edge padding.
    """
    # Pad one on BOTH sides with 'edge' so boundary corners use nearest cells
    P = np.pad(Uc, ((1, 1), (1, 1), (1, 1)), mode="edge")  # (Nx+2, Ny+2, Nz+2)
    # Average the 8 neighbors that meet at each corner
    Ue = (
        P[0:-1, 0:-1, 0:-1] +  # (i,   j,   k)
        P[1:  , 0:-1, 0:-1] +  # (i+1, j,   k)
        P[0:-1, 1:  , 0:-1] +  # (i,   j+1, k)
        P[0:-1, 0:-1, 1:  ] +  # (i,   j,   k+1)
        P[1:  , 1:  , 0:-1] +  # (i+1, j+1, k)
        P[1:  , 0:-1, 1:  ] +  # (i+1, j,   k+1)
        P[0:-1, 1:  , 1:  ] +  # (i,   j+1, k+1)
        P[1:  , 1:  , 1:  ]    # (i+1, j+1, k+1)
    ) / 8.0
    # Now Ue has shape (Nx+1, Ny+1, Nz+1)
    return Ue



def _wire_segments_from_corners(Xe, Ye, Ze, step):
    Nx1, Ny1, Nz1 = Xe.shape  # = (Nx+1, Ny+1, Nz+1)
    segs = []

    # lines parallel to x at every (j,k) corner line
    for j in range(0, Ny1, step):
        for k in range(0, Nz1, step):
            x = Xe[:, j, k]; y = Ye[:, j, k]; z = Ze[:, j, k]
            if x.size > 1:
                p0 = np.column_stack([x[:-1], y[:-1], z[:-1]])
                p1 = np.column_stack([x[1:],  y[1:],  z[1:]])
                segs.append(np.stack([p0, p1], axis=1))

    # lines parallel to y
    for i in range(0, Nx1, step):
        for k in range(0, Nz1, step):
            x = Xe[i, :, k]; y = Ye[i, :, k]; z = Ze[i, :, k]
            if y.size > 1:
                p0 = np.column_stack([x[:-1], y[:-1], z[:-1]])
                p1 = np.column_stack([x[1:],  y[1:],  z[1:]])
                segs.append(np.stack([p0, p1], axis=1))

    # lines parallel to z
    for i in range(0, Nx1, step):
        for j in range(0, Ny1, step):
            x = Xe[i, j, :]; y = Ye[i, j, :]; z = Ze[i, j, :]
            if z.size > 1:
                p0 = np.column_stack([x[:-1], y[:-1], z[:-1]])
                p1 = np.column_stack([x[1:],  y[1:],  z[1:]])
                segs.append(np.stack([p0, p1], axis=1))

    if not segs:
        return np.zeros((0, 2, 3), dtype=float)
    return np.concatenate(segs, axis=0)

def _cell_face_quads_and_vals(Xe, Ye, Ze, ValCell, cstep):
    # ValCell is per cell (Nx,Ny,Nz); Xe,Ye,Ze are corners (Nx+1,Ny+1,Nz+1)
    Nx = Xe.shape[0] - 1
    Ny = Ye.shape[1] - 1
    Nz = Ze.shape[2] - 1
    polys = []
    vals  = []

    def face_quads_i(i, j, k):
        # Return the 6 faces (as lists of 4 xyz points) of cell (i,j,k)
        # corners: (i,j,k) ... (i+1,j+1,k+1)
        # helper to grab a corner:
        def C(ii, jj, kk):
            return np.array([Xe[ii, jj, kk], Ye[ii, jj, kk], Ze[ii, jj, kk]])
        c000 = C(i,   j,   k  ); c100 = C(i+1, j,   k  )
        c010 = C(i,   j+1, k  ); c110 = C(i+1, j+1, k  )
        c001 = C(i,   j,   k+1); c101 = C(i+1, j,   k+1)
        c011 = C(i,   j+1, k+1); c111 = C(i+1, j+1, k+1)
        # 6 faces: order them so normals are consistent (any consistent order is fine)
        return [
            [c000, c100, c110, c010],  # z = k   (bottom)
            [c001, c011, c111, c101],  # z = k+1 (top)
            [c000, c010, c011, c001],  # x = i   (left)
            [c100, c101, c111, c110],  # x = i+1 (right)
            [c000, c001, c101, c100],  # y = j   (front)
            [c010, c110, c111, c011],  # y = j+1 (back)
        ]

    for i in range(0, Nx, cstep):
        for j in range(0, Ny, cstep):
            for k in range(0, Nz, cstep):
                faces = face_quads_i(i, j, k)
                polys.extend(faces)
                vals.extend([ValCell[i, j, k]] * 6)

    if not polys:
        return [], np.zeros((0,), dtype=float)
    return polys, np.asarray(vals, dtype=float)


def plot_ref_and_def(domain, ux, uy, uz, mu, def_scale=1.0,
                     prefix="viz", frame=0, sstep=4, color_by="mag", cmap="viridis",
                     cstep=2, cell_alpha=0.25, wire_lw=0.6):
    """
    Reference (corner-grid): grid lines + filled cells colored by μ
    Deformed  (corner-grid): grid lines + filled cells colored by displacement (‖u‖ by default)
    """
    # --------- center grids & fields (as before) ----------
    X, Y, Z = [np.array(a) for a in domain.points()]
    Ux, Uy, Uz = np.array(ux), np.array(uy), np.array(uz)

    # choose displacement scalar (per center)
    if color_by == "ux":
        scal_u = Ux; cbar_u = "u_x"
    elif color_by == "uy":
        scal_u = Uy; cbar_u = "u_y"
    elif color_by == "uz":
        scal_u = Uz; cbar_u = "u_z"
    else:
        scal_u = np.sqrt(Ux**2 + Uy**2 + Uz**2); cbar_u = "‖u‖"

    # --------- corners (edges) ----------
    Xe, Ye, Ze = _corners_from_centers_grids(X, Y, Z)

    # interpolate displacements to corners -> deformed corners
    Ux_e = _centers_to_corners(Ux)
    Uy_e = _centers_to_corners(Uy)
    Uz_e = _centers_to_corners(Uz)
    Xed = Xe + def_scale * Ux_e
    Yed = Ye + def_scale * Uy_e
    Zed = Ze + def_scale * Uz_e

    # --------- wire segments (corners) ----------
    seg_ref = _wire_segments_from_corners(Xe, Ye, Ze, step=sstep)
    seg_def = _wire_segments_from_corners(Xed, Yed, Zed, step=sstep)

    # --------- per-cell colors ----------
    mu_cell  = np.array(mu)          # (Nx,Ny,Nz)
    u_cell   = np.array(scal_u)      # (Nx,Ny,Nz)

    # --------- build cell faces ----------
    polys_ref, vals_ref = _cell_face_quads_and_vals(Xe, Ye, Ze, mu_cell, cstep=cstep)
    polys_def, vals_def = _cell_face_quads_and_vals(Xed, Yed, Zed, u_cell, cstep=cstep)

    # --------- Reference fig ----------
    vmin_mu = float(np.min(vals_ref)) if vals_ref.size else 0.0
    vmax_mu = float(np.max(vals_ref)) if vals_ref.size else 1.0
    if vmin_mu == vmax_mu: vmax_mu = vmin_mu + 1e-12
    norm_mu = mcolors.Normalize(vmin=vmin_mu, vmax=vmax_mu)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # filled cells
    pc_ref = Poly3DCollection(polys_ref, alpha=cell_alpha)
    pc_ref.set_cmap(cmap); pc_ref.set_norm(norm_mu)
    pc_ref.set_array(vals_ref)
    pc_ref.set_edgecolor("none")
    ax.add_collection3d(pc_ref)

    # wire
    lc_ref = Line3DCollection(seg_ref, linewidths=wire_lw, alpha=0.7, colors="k")
    ax.add_collection3d(lc_ref)

    ax.set_xlim(Xe.min(), Xe.max()); ax.set_ylim(Ye.min(), Ye.max()); ax.set_zlim(Ze.min(), Ze.max())
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Reference grid colored by μ (cells)")
    cbar_mu = fig.colorbar(pc_ref, ax=ax, shrink=0.6, pad=0.08); cbar_mu.set_label("μ")
    path_ref = f"{prefix}_ref_{frame:05d}.png"
    fig.savefig(path_ref, dpi=150, bbox_inches="tight"); plt.close(fig)

    # --------- Deformed fig ----------
    vmin_u = float(np.min(vals_def)) if vals_def.size else 0.0
    vmax_u = float(np.max(vals_def)) if vals_def.size else 1.0
    if vmin_u == vmax_u: vmax_u = vmin_u + 1e-12
    norm_u = mcolors.Normalize(vmin=vmin_u, vmax=vmax_u)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    pc_def = Poly3DCollection(polys_def, alpha=cell_alpha)
    pc_def.set_cmap(cmap); pc_def.set_norm(norm_u)
    pc_def.set_array(vals_def)
    pc_def.set_edgecolor("none")
    ax.add_collection3d(pc_def)

    lc_def = Line3DCollection(seg_def, linewidths=wire_lw, alpha=0.7, colors="k")
    ax.add_collection3d(lc_def)

    ax.set_box_aspect([Xed.ptp(), Yed.ptp(), Zed.ptp()])
    ax.set_xlim(Xed.min(), Xed.max()); ax.set_ylim(Yed.min(), Yed.max()); ax.set_zlim(Zed.min(), Zed.max())
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Deformed grid colored by {cbar_u} (scale={def_scale})")
    cbar = fig.colorbar(pc_def, ax=ax, shrink=0.6, pad=0.08); cbar.set_label(cbar_u)
    path_def = f"{prefix}_def_{frame:05d}.png"
    fig.savefig(path_def, dpi=150, bbox_inches="tight"); plt.close(fig)

    printlog(path_ref, path_def)



def plot_func(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    args = problem.extra.args

    ux = np.array(domain.field(state, "ux"))
    uy = np.array(domain.field(state, "uy"))
    uz = np.array(domain.field(state, "uz"))
    if "mu" in state.fields:
        mu = np.array(domain.field(state, "mu"))
    elif "mu_raw" in state.fields:
        mu_r = np.array(domain.field(state, "mu_raw"))
        mu = mu_r * mu_r
    else:
        # Prefer CSV μ for visualization if available
        if problem.extra.data_mu is not None:
            mu = problem.extra.data_mu
        else:
            mu = np.full(domain.cshape, args.mu, dtype=ux.dtype)

    plot_ref_and_def(domain, ux, uy, uz, mu,
                     def_scale=getattr(args, "def_scale", 1.0),
                     prefix=getattr(args, "viz_img_prefix", "viz"),
                     frame=frame,
                     sstep=4,
                     color_by="mag",
                     cmap=(args.viz_cmap or "viridis"))


def _rms(a, b, mask=None):
    if b is None:
        return None
    if mask is not None:
        w = mask.astype(a.dtype)
        if w.ndim < a.ndim:
            w = w[(...,) + (None,) * (a.ndim - w.ndim)]
        den = np.sum(w)
        if den == 0:
            return None
        return float(np.sqrt(np.sum(w * (a - b) ** 2) / den))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _rel(a, b, mask=None):
    if b is None:
        return None
    aa = a.astype(np.float64)
    bb = b.astype(np.float64)
    if mask is not None:
        w = mask.astype(np.float64)
        if w.ndim < aa.ndim:
            w = w[(...,) + (None,) * (aa.ndim - w.ndim)]
        num = np.sum(w * (aa - bb) ** 2)
        den = np.sum(w * (bb) ** 2)
    else:
        num = np.sum((aa - bb) ** 2)
        den = np.sum((bb) ** 2)
    if den <= 0:
        return None
    return float(np.sqrt(num / den))


def get_error(domain, extra, state, key):
    if key == "u":
        if extra.data_u is None:
            return None
        ux = np.array(domain.field(state, "ux"))
        uy = np.array(domain.field(state, "uy"))
        uz = np.array(domain.field(state, "uz"))
        u = np.stack([ux, uy, uz], axis=-1)
        m3 = None if extra.data_mask is None else extra.data_mask[..., None]
        return _rms(u, extra.data_u, mask=m3)

    elif key in ("mu", "mu_rel"):
        if extra.data_mu is None:
            return None
        if "mu" in state.fields:
            mu = np.array(domain.field(state, "mu"))
        elif "mu_raw" in state.fields:
            mu_r = np.array(domain.field(state, "mu_raw"))
            mu = mu_r * mu_r
        else:
            # When not inferring μ, compare CSV μ vs CSV μ (=> zero), but keep fallback
            mu = extra.data_mu if extra.data_mu is not None else np.full(domain.cshape, extra.args.mu, dtype=extra.data_mu.dtype)
        if key == "mu":
            return _rms(mu, extra.data_mu, mask=extra.data_mask)
        else:
            return _rel(mu, extra.data_mu, mask=extra.data_mask)

    return None


def history_func(problem, state, epoch, history, cbinfo):
    e_u = get_error(problem.domain, problem.extra, state, "u")
    e_mu = get_error(problem.domain, problem.extra, state, "mu")
    r_mu = get_error(problem.domain, problem.extra, state, "mu_rel")
    if e_u is not None: history.append("eval_rms_u", e_u)
    if e_mu is not None: history.append("eval_rms_mu", e_mu)
    if r_mu is not None: history.append("eval_rel_mu", r_mu)


def report_func(problem, state, epoch, cbinfo):
    e_u = get_error(problem.domain, problem.extra, state, "u")
    e_mu = get_error(problem.domain, problem.extra, state, "mu")
    r_mu = get_error(problem.domain, problem.extra, state, "mu_rel")
    msgs = []
    if e_u is not None: msgs.append(f"eval_rms_u={e_u:.4e}")
    if e_mu is not None: msgs.append(f"eval_rms_mu={e_mu:.4e}")
    if r_mu is not None: msgs.append(f"eval_rel_mu={r_mu:.4e}")
    if msgs: printlog(" | ".join(msgs))


# ---------------------- problem build ----------------------
def make_problem(args):
    dtype = np.float64 if args.double else np.float32
    domain = odil.Domain(
        cshape=(args.Nx, args.Ny, args.Nz),
        dimnames=("x", "y", "z"),
        lower=(0, 0, 0),
        upper=(2, 2, 1),
        multigrid=args.multigrid,
        dtype=dtype
    )
    if domain.multigrid:
        printlog("multigrid levels:", domain.mg_cshapes)

    mod = domain.mod

    # ----------------- Load CSV (for u, μ, optional E) -----------------
    data_u = data_mu = data_E = data_mask = None
    if args.data_csv is not None:
        data_u, data_mu, data_E, data_mask = load_data_csv_to_grid(
            domain, args.data_csv, mu_scale=args.mu_scale, dtype=dtype
        )
        covered = int(np.sum(data_mask)) if data_mask is not None else 0
        total = int(np.prod(domain.cshape))
        printlog(f"data coverage: {covered}/{total} = {covered / total:.2%}")

    # ----------------- Imposed derived solely from CSV -----------------
    if args.imposed != "none" and data_mask is not None:
        imp_mask = (data_mask > 0).astype(domain.dtype)
        imp_indices = np.flatnonzero(imp_mask.ravel())
        if imp_indices.size:
            points = [mod.flatten(domain.points(i)) for i in range(domain.ndim)]
            imp_points = np.array(points)[:, imp_indices].T
        else:
            imp_points = np.zeros((0, domain.ndim))
        imp_u = np.array(data_u, dtype=dtype) if data_u is not None else None
        if imp_u is not None and args.noise:
            rng = np.random.default_rng(getattr(args, "seed", 0))
            imp_u = imp_u + rng.normal(0, args.noise, size=imp_u.shape)
        imp_size = imp_points.shape[0]
    else:
        imp_mask = np.zeros(domain.cshape, dtype=domain.dtype)
        imp_points = np.zeros((0, domain.ndim))
        imp_indices = np.array([], dtype=np.int64)
        imp_u = None
        imp_size = 0

    # ----------------- Dirichlet component sets -----------------
    def _parse_dir(s):
        s = (s or "").strip().lower()
        if s in ("", "none"): return set()
        items = {t.strip() for t in s.split(",") if t.strip()}
        valid = {"ux", "uy", "uz"}
        unknown = items - valid
        if unknown:
            raise ValueError(f"Unknown components in Dirichlet list: {sorted(unknown)}")
        return items

    # ----------------- State -----------------
    state = odil.State()
    state.fields["ux"] = np.zeros(domain.cshape, dtype=dtype)
    state.fields["uy"] = np.zeros(domain.cshape, dtype=dtype)
    state.fields["uz"] = np.zeros(domain.cshape, dtype=dtype)
    if args.infer_mu:
        state.fields["mu_raw"] = np.full(domain.cshape, np.sqrt(args.mu), dtype=dtype)

    state = domain.init_state(state)

    # ----------------- Pack extras -----------------
    extra = argparse.Namespace()
    extra.args = args
    extra.imp_mask = imp_mask
    extra.imp_size = imp_size
    extra.imp_u = imp_u
    extra.imp_indices = imp_indices
    extra.imp_points = imp_points
    extra.data_u = data_u
    extra.data_mu = data_mu
    extra.data_E = data_E
    extra.data_mask = data_mask
    extra.epoch = mod.variable(domain.cast(0))

    problem = odil.Problem(operator_odil, domain, extra)
    return problem, state


# ---------------------- main ----------------------
def main():
    args = parse_args()
    odil.setup_outdir(args, relpath_args=["checkpoint"])
    problem, state = make_problem(args)
    callback = odil.make_callback(
        problem, args, plot_func=plot_func, history_func=history_func, report_func=report_func
    )
    odil.util.optimize(args, args.optimizer, problem, state, callback)

    # Post-optimization plot
    plot_func(problem, state, epoch=0, frame=99999)

    with open("done", "w") as f:
        pass


if __name__ == "__main__":
    main()
