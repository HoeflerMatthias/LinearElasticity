import odil
import numpy as np
import meshio
from pathlib import Path

from utils_odil import *

def param(a):
    return 8.0 * a * a

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    ux = np.array(domain.field(state, "ux"))
    uy = np.array(domain.field(state, "uy"))
    uz = np.array(domain.field(state, "uz"))
    u = np.stack([ux, uy, uz], axis=-1)

    if "mu" in state.fields:
        mu = np.array(domain.field(state, "mu"))
    elif "mu_raw" in state.fields:
        mu_r = np.array(domain.field(state, "mu_raw"))
        mu = param(mu_r)

    e_u = rmse(u, problem.extra.data_u)
    e_mu = rmse(mu, problem.extra.data_mu)

    history.append("eval_rms_u", e_u)
    history.append("eval_rms_mu", e_mu)

    return e_u, e_mu

def report_func(problem, state, epoch, cbinfo):
    e = history_func(problem, state, epoch, [], cbinfo)
    printlog(f"Epoch {epoch:4d}: rms_u = {e[0]:.6f}, rms_mu = {e[1]:.6f}")

def plot_func(problem, state, epoch, frame, cbinfo=None):
    domain = problem.domain
    args = problem.extra.args

    # --- nodal fields (same as your original) ---
    ux = np.array(domain.field(state, "ux"))
    uy = np.array(domain.field(state, "uy"))
    uz = np.array(domain.field(state, "uz"))
    if "mu" in state.fields:
        mu = np.array(domain.field(state, "mu"))
    elif "mu_raw" in state.fields:
        mu_r = np.array(domain.field(state, "mu_raw"))
        mu = param(mu_r)      # or 8*mu_r*mu_r if that’s your convention elsewhere

    # --- build structured coordinates from domain bounds/shape ---
    nx, ny, nz = domain.cshape                                      # (Nx,Ny,Nz)
    (lx, ly, lz), (uxb, uyb, uzb) = domain.lower, domain.upper      # bounds
    x = np.linspace(lx, uxb, nx); y = np.linspace(ly, uyb, ny); z = np.linspace(lz, uzb, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points_ijk_xyz = np.stack([X, Y, Z], axis=-1)

    # --- write VTU file (ParaView-ready) ---
    outdir = Path(getattr(args, "viz_dir", "viz"))
    prefix = getattr(args, "viz_img_prefix", "viz")
    vtu_path = outdir / f"{prefix}_{frame:05d}.vtu"
    write_vtu_structured(points_ijk_xyz, (nx, ny, nz), ux, uy, uz, mu, str(vtu_path))

def invscar(**params):
    # Geometry
    Nx_i = params.get('Nx_inv', 20)
    Ny_i = params.get('Ny_inv', 20)
    Nz_i = params.get('Nz_inv', 10)

    Lx = params.get('Lx', 2.0)
    Ly = params.get('Ly', 2.0)
    Lz = params.get('Lz', 1.0)

    # Physics parameters
    lambda_ = params.get('lambda_', 650.0)
    mu0 = params.get('mu', 8.0)
    p_load = -1 * params.get('p_load', 10.0)

    # Inverse parameters
    noise_level = params.get('noise_level', 1e-2)
    noise_seed = params.get('noise_seed', 123)

    lam_pde = params.get("lam_pde", 1.0)
    lam_bcn = params.get("lam_bcn", 1.0)
    lam_dat = params.get("lam_dat", 1.0)

    J_regu = params.get('J_regu', 'TV')
    lam_reg = params.get('lam_reg', 1e-3)

    # File handling
    tag = params.get('run_name', None)
    if tag is None:
        tag = f"Ninv{Nx_i}x{Ny_i}x{Nz_i}_pload{float(p_load)}_noise{float(noise_level)}"

    # output root and run directory
    out_root = Path(params.get('out_root', 'runs_odil'))
    out_dir = out_root / tag
    params['outdir'] = str(out_dir)
    params['viz_dir'] = str(out_dir / "viz")

    # Optimizer parameters
    params.setdefault('optimizer', 'adam')
    params.setdefault('linsolver', 'direct')
    params.setdefault('lr', 0.001)

    def operator(ctx):
        extra = ctx.extra
        mod = ctx.mod
        args = extra.args

        dx, dy, dz = ctx.step()
        ix, iy, iz = ctx.indices()
        nx, ny, nz = ctx.size()

        def stencil(key):
            return dict(
                c=ctx.field(key, 0, 0, 0),
                xm=ctx.field(key, -1, 0, 0, ),
                xp=ctx.field(key, +1, 0, 0),
                ym=ctx.field(key, 0, -1, 0),
                yp=ctx.field(key, 0, +1, 0),
                zm=ctx.field(key, 0, 0, -1),
                zp=ctx.field(key, 0, 0, +1),
            )

        def apply_dirichlet_minface(st, axis_min_mask):
            # Zero Dirichlet conditions, quadratic extrapolation.
            extrap = odil.core.extrap_quadh
            c = st["c"]
            if axis_min_mask == "x":
                st["xm"] = mod.where(ix == 0, extrap(st["xp"], c, 0), st["xm"])
            elif axis_min_mask == "y":
                st["ym"] = mod.where(iy == 0, extrap(st["yp"], c, 0), st["ym"])
            elif axis_min_mask == "z":
                st["zm"] = mod.where(iz == 0, extrap(st["zp"], c, 0), st["zm"])
            return st

        ux = apply_dirichlet_minface(stencil("ux"), "x")
        uy = apply_dirichlet_minface(stencil("uy"), "y")
        uz = apply_dirichlet_minface(stencil("uz"), "z")

        lam = mod.ones_like(ux["c"]) * ctx.cast(lambda_)

        mu_r = ctx.field("mu_raw", 0, 0, 0)
        mu = param(mu_r)  # already grid-shaped

        # ----------------- displacement stencils -----------------

        def ddx(st):
            return (st["xp"] - st["xm"]) / (2 * dx)

        def ddy(st):
            return (st["yp"] - st["ym"]) / (2 * dy)

        def ddz(st):
            return (st["zp"] - st["zm"]) / (2 * dz)

        ux_x, ux_y, ux_z = ddx(ux), ddy(ux), ddz(ux)
        uy_x, uy_y, uy_z = ddx(uy), ddy(uy), ddz(uy)
        uz_x, uz_y, uz_z = ddx(uz), ddy(uz), ddz(uz)

        # ----------------- strain -----------------
        exx = ux_x;
        eyy = uy_y;
        ezz = uz_z
        exy = 0.5 * (ux_y + uy_x)
        exz = 0.5 * (ux_z + uz_x)
        eyz = 0.5 * (uy_z + uz_y)
        trE = exx + eyy + ezz

        # ----------------- nodal stresses -----------------
        sxx = lam * trE + 2.0 * mu * exx
        syy = lam * trE + 2.0 * mu * eyy
        szz = lam * trE + 2.0 * mu * ezz
        sxy = 2.0 * mu * exy
        sxz = 2.0 * mu * exz
        syz = 2.0 * mu * eyz

        # ----------------- div σ -----------------

        # centered in interior; forward at min; backward at max
        def grad_center_or_one_sided(Q, h, axis, i, n):
            Qp = mod.roll(Q, -1, axis=axis)
            Qm = mod.roll(Q, +1, axis=axis)
            cen = (Qp - Qm) / (2 * h)
            fwd = (Qp - Q) / h
            bwd = (Q - Qm) / h
            return mod.where(i == 0, fwd, mod.where(i == n - 1, bwd, cen))

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
        vol = ctx.cast(dx * dy * dz)

        kpde = ctx.cast(lam_pde) * mod.sqrt(vol)
        res = [
            ("fx", -kpde * div_sigma_x),
            ("fy", -kpde * div_sigma_y),
            ("fz", -kpde * div_sigma_z),
        ]

        # ----------------- Neumann BCs on max faces -----------------

        ayz = ctx.cast(dy * dz)
        axz = ctx.cast(dx * dz)
        axy = ctx.cast(dx * dy)

        kneum_x = ctx.cast(lam_bcn) * mod.sqrt(ayz)
        kneum_y = ctx.cast(lam_bcn) * mod.sqrt(axz)
        kneum_z = ctx.cast(lam_bcn) * mod.sqrt(axy)

        mask_xp = (ix == nx - 1);
        mask_yp = (iy == ny - 1);
        mask_zp = (iz == nz - 1)

        # face-averaged Lamé parameters (now grid-shaped)
        lam_xF = face_avg(lam, 0, "max");
        mu_xF = face_avg(mu, 0, "max")
        lam_yF = face_avg(lam, 1, "max");
        mu_yF = face_avg(mu, 1, "max")
        lam_zF = face_avg(lam, 2, "max");
        mu_zF = face_avg(mu, 2, "max")

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

        res += [
            ("neum_x_face_z", kneum_z * mod.where(mask_zp, sxz_F, ctx.cast(0))),
            ("neum_y_face_z", kneum_z * mod.where(mask_zp, syz_F, ctx.cast(0))),
            ("neum_z_face_z", kneum_z * mod.where(mask_zp, szz_F - ctx.cast(p_load), ctx.cast(0))),
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
            ("neum_x_face_y", kneum_y * mod.where(mask_yp, sxy_F, ctx.cast(0))),
            ("neum_y_face_y", kneum_y * mod.where(mask_yp, syy_F, ctx.cast(0))),
            ("neum_z_face_y", kneum_y * mod.where(mask_yp, syz_Fy, ctx.cast(0))),
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
            ("neum_x_face_x", kneum_x * mod.where(mask_xp, sxx_F, ctx.cast(0))),
            ("neum_y_face_x", kneum_x * mod.where(mask_xp, sxy_Fx, ctx.cast(0))),
            ("neum_z_face_x", kneum_x * mod.where(mask_xp, sxz_Fx, ctx.cast(0))),
        ]

        # ----------------- data misfit -----------------
        uxh = ctx.field("ux", 0, 0, 0)
        uyh = ctx.field("uy", 0, 0, 0)
        uzh = ctx.field("uz", 0, 0, 0)
        dux = mod.constant(extra.data_u[..., 0])
        duy = mod.constant(extra.data_u[..., 1])
        duz = mod.constant(extra.data_u[..., 2])

        res += [
            ("data_u_x", ctx.cast(lam_dat) * (uxh - dux)),
            ("data_u_y", ctx.cast(lam_dat) * (uyh - duy)),
            ("data_u_z", ctx.cast(lam_dat) * (uzh - duz)),
        ]

        # ----------------- μ-regularization -----------------
        def nb_roll(Q, axis, i, n, sgn):
            Qr = mod.roll(Q, -sgn, axis=axis)
            clamp = (i == (n - 1)) if (sgn > 0) else (i == 0)
            return mod.where(clamp, Q, Qr)

        mu_xp = nb_roll(mu, 0, ix, nx, +1);
        mu_xm = nb_roll(mu, 0, ix, nx, -1)
        mu_yp = nb_roll(mu, 1, iy, ny, +1);
        mu_ym = nb_roll(mu, 1, iy, ny, -1)
        mu_zp = nb_roll(mu, 2, iz, nz, +1);
        mu_zm = nb_roll(mu, 2, iz, nz, -1)

        gmx = (mu_xp - mu_xm) / (2 * dx)
        gmy = (mu_yp - mu_ym) / (2 * dy)
        gmz = (mu_zp - mu_zm) / (2 * dz)

        if J_regu == 'TV':
            eps = ctx.cast(1e-8)
            tv = mod.sqrt(gmx * gmx + gmy * gmy + gmz * gmz + eps)
            res += [("mu_tv", ctx.cast(lam_reg) * tv)]
        elif J_regu == 'H1':
            res += [("mu_h1_x", ctx.cast(lam_reg) * gmx),
                        ("mu_h1_y", ctx.cast(lam_reg) * gmy),
                        ("mu_h1_z", ctx.cast(lam_reg) * gmz)]

        return res

    odil.setup_outdir(params, relpath_args=["checkpoint"])

    domain = odil.Domain(
        cshape=(Nx_i, Ny_i, Nz_i),
        dimnames=("x", "y", "z"),
        lower=(0, 0, 0),
        upper=(Lx, Ly, Lz),
        multigrid=False,
        dtype=np.float64
    )

    data_u, data_mu, data_E, data_mask = load_data_csv_to_grid(
        domain, args.data_csv, mu_scale=args.mu_scale, dtype=np.float64
    )
    covered = int(np.sum(data_mask)) if data_mask is not None else 0
    total = int(np.prod(domain.cshape))
    printlog(f"data coverage: {covered}/{total} = {covered / total:.2%}")

    # Pack extras once
    extra = argparse.Namespace()
    extra.args = args

    extra.data_u = data_u
    extra.data_mu = data_mu
    extra.data_E = data_E
    extra.data_mask = data_mask
    extra.epoch = mod.variable(domain.cast(0))

    # State
    state = odil.State()
    state.fields["ux"] = np.zeros(domain.cshape, dtype=np.float64)
    state.fields["uy"] = np.zeros(domain.cshape, dtype=np.float64)
    state.fields["uz"] = np.zeros(domain.cshape, dtype=np.float64)
    state.fields["mu_raw"] = np.full(domain.cshape, np.sqrt(1.5), dtype=np.float64)

    state = domain.init_state(state)
    problem = odil.Problem(operator, domain, extra)

    callback = odil.make_callback(
        problem, args, plot_func=plot_func, history_func=history_func, report_func=report_func
    )
    odil.util.optimize(args, args.optimizer, problem, state, callback)

if __name__ == "__main__":
    invscar()