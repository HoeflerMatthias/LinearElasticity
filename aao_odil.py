import odil
import numpy as np
import meshio
from types import SimpleNamespace
from pathlib import Path
import argparse
from odil import printlog
import tensorflow as tf
import mlflow
import pandas as pd
import os

from utils_odil import *

# from paper
def gradient(tensor, step, axis, final_op=False):
    """
    Compute the gradient of a tensor using a central difference scheme in the interior and a first-order
    scheme at the boundaries for 3D data.

    Parameters
    ----------
    tensor : tf.Tensor
        The tensor to differentiate. Shape: (nt, nx, ny, nz).
    step : float
        The step size.
    axis : int
        The axis along which to compute the gradient.
    final_op : bool
        If this gradient operation is the final operation. Default is False.

    Returns
    -------
    tf.Tensor
        The gradient of the input tensor.
    """
    tensor_before = tf.roll(tensor, shift=1, axis=axis)
    tensor_after = tf.roll(tensor, shift=-1, axis=axis)

    # Central difference in the interior of the domain
    gradient = (tensor_after - tensor_before) / (2 * step)

    def grad_left(f0,f1,f2,step):
        return (-3.0*f0 + 4.0*f1 - f2) / (2.0*step)
    def grad_right(fn,fnm1,fnm2,step):
        return (3.0*fn - 4.0*fnm1 + fnm2) / (2.0*step)

    if axis == 0:
        # Forward difference at the left boundary and backward difference at the right boundary
        #gradient_left = (tensor[1, :, :] - tensor[0, :, :]) / step
        #gradient_right = (tensor[-1, :, :] - tensor[-2, :, :]) / step
        gradient_left = grad_left(tensor[0, :, :], tensor[1, :, :], tensor[2, :, :], step)
        gradient_right = grad_right(tensor[-1, :, :], tensor[-2, :, :], tensor[-3, :, :], step)

        gradient = tf.concat([gradient_left[None, :, :], gradient[1:-1, :, :], gradient_right[None, :, :]],
                             axis=axis)

    elif axis == 1:
        # Forward difference at the top boundary and backward difference at the bottom boundary
        #gradient_top = (tensor[:, 1, :] - tensor[:, 0, :]) / step
        #gradient_bottom = (tensor[:, -1, :] - tensor[:, -2, :]) / step
        gradient_bottom = grad_left(tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :], step)
        gradient_top = grad_right(tensor[:, -1, :], tensor[:, -2, :], tensor[:, -3, :], step)

        gradient = tf.concat([gradient_bottom[:, None, :], gradient[:, 1:-1, :], gradient_top[:, None, :]],
                             axis=axis)

    elif axis == 2:
        # Forward difference at the front boundary and backward difference at the back boundary
        #gradient_front = (tensor[:, :, 1] - tensor[:, :, 0]) / step
        #gradient_back = (tensor[:, :, -1] - tensor[:, :, -2]) / step
        gradient_back = grad_left(tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2], step)
        gradient_front = grad_right(tensor[:, :, -1], tensor[:, :, -2], tensor[:, :, -3], step)

        gradient = tf.concat([gradient_back[:, :, None], gradient[:, :, 1:-1], gradient_front[:, :, None]],
                             axis=axis)

    return gradient

def compute_strain_tensor_lagrangian_full(u_x, u_y, u_z, domain):
    """
    Compute the full 3D Green-Lagrange strain tensor E given the x, y, and z components of the displacement field.

    Parameters
    ----------
    u_x : tf.Tensor
        The x-component of the displacement field. Shape: (nt, nx, ny, nz).
    u_y : tf.Tensor
        The y-component of the displacement field. Shape: (nt, nx, ny, nz).
    u_z : tf.Tensor
        The z-component of the displacement field. Shape: (nt, nx, ny, nz).

    Returns
    -------
    E : tf.Tensor
        The Green-Lagrange strain tensor. Shape: (3, 3, nt, nx, ny, nz).
    """
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')

    # Compute the spatial gradients of the displacement fields
    u_x_x = gradient(u_x, dx, axis=0)  # partial derivative of u_x w.r.t. x
    u_x_y = gradient(u_x, dy, axis=1)  # partial derivative of u_x w.r.t. y
    u_x_z = gradient(u_x, dz, axis=2)  # partial derivative of u_x w.r.t. z
    u_y_x = gradient(u_y, dx, axis=0)  # partial derivative of u_y w.r.t. x
    u_y_y = gradient(u_y, dy, axis=1)  # partial derivative of u_y w.r.t. y
    u_y_z = gradient(u_y, dz, axis=2)  # partial derivative of u_y w.r.t. z
    u_z_x = gradient(u_z, dx, axis=0)  # partial derivative of u_z w.r.t. x
    u_z_y = gradient(u_z, dy, axis=1)  # partial derivative of u_z w.r.t. y
    u_z_z = gradient(u_z, dz, axis=2)  # partial derivative of u_z w.r.t. z

    # Compute the Green-Lagrange strain tensor components
    E_xx = u_x_x
    E_yy = u_y_y
    E_zz = u_z_z
    E_xy = 0.5 * (u_x_y + u_y_x)
    E_xz = 0.5 * (u_x_z + u_z_x)
    E_yz = 0.5 * (u_y_z + u_z_y)

    # Combine strain tensor components into a single 6D array
    E = tf.stack([[E_xx, E_xy, E_xz], [E_xy, E_yy, E_yz], [E_xz, E_yz, E_zz]])

    return E

def compute_stress(E, lambda_, mu):
    # Compute trace of the strain tensor (sum of diagonal elements)
    trace_E = E[0, 0, ...] + E[1, 1, ...] + E[2, 2, ...]

    # Create identity tensor
    I = tf.eye(3, dtype=tf.float64)[..., tf.newaxis, tf.newaxis, tf.newaxis] # Shape (3,3,1,1,1)
    I = tf.broadcast_to(I, E.shape)  # Shape (3,3,nx,ny,nz)
    # Calculate the stress tensor components
    mue = tf.broadcast_to(mu, E.shape)
    sigma = lambda_ * trace_E * I + 2 * mue * E
    return sigma

def compute_div_stress(E, lambda_, mu, dx, dy, dz):

    E_xx_x = gradient(E[0,0,...], dx, axis=0)
    E_xx_y = gradient(E[0,0,...], dy, axis=1)
    E_xx_z = gradient(E[0,0,...], dz, axis=2)

    E_yy_x = gradient(E[1,1,...], dx, axis=0)
    E_yy_y = gradient(E[1,1,...], dy, axis=1)
    E_yy_z = gradient(E[1,1,...], dz, axis=2)

    E_zz_x = gradient(E[2,2,...], dx, axis=0)
    E_zz_y = gradient(E[2,2,...], dy, axis=1)
    E_zz_z = gradient(E[2,2,...], dz, axis=2)

    trE_x = E_xx_x + E_yy_x + E_zz_x
    trE_y = E_xx_y + E_yy_y + E_zz_y
    trE_z = E_xx_z + E_yy_z + E_zz_z

    div1 = lambda_ * tf.stack([trE_x, trE_y, trE_z], axis=0)

    mu_x = gradient(mu, dx, axis=0)
    mu_y = gradient(mu, dy, axis=1)
    mu_z = gradient(mu, dz, axis=2)

    E_xy_x = gradient(E[0,1,...], dx, axis=0)
    E_xy_y = gradient(E[0,1,...], dy, axis=1)
    E_xy_z = gradient(E[0,1,...], dz, axis=2)

    E_xz_x = gradient(E[0,2,...], dx, axis=0)
    E_xz_y = gradient(E[0,2,...], dy, axis=1)
    E_xz_z = gradient(E[0,2,...], dz, axis=2)

    E_yz_x = gradient(E[1,2,...], dx, axis=0)
    E_yz_y = gradient(E[1,2,...], dy, axis=1)
    E_yz_z = gradient(E[1,2,...], dz, axis=2)

    div2_x = 2 * (mu_x * E[0,0,...] + mu * E_xx_x + mu_y * E[0,1,...] + mu * E_xy_y + mu_z * E[0,2,...] + mu * E_xz_z)
    div2_y = 2 * (mu_x * E[1,0,...] + mu * E_xy_x + mu_y * E[1,1,...] + mu * E_yy_y + mu_z * E[1,2,...] + mu * E_yz_z)
    div2_z = 2 * (mu_x * E[2,0,...] + mu * E_xz_x + mu_y * E[2,1,...] + mu * E_yz_y + mu_z * E[2,2,...] + mu * E_zz_z)

    div2 = tf.stack([div2_x, div2_y, div2_z], axis=0)

    div = div1 + div2
    return div


def divergence_fd(tensor, dx, dy, dz):
    """
    Compute the divergence of a vector field in 3D using the gradient function.

    Parameters
    ----------
    tensor : tf.Tensor
        The vector field tensor with shape (3,3, nx, ny, nz).
    dx : float, optional
        The step size in the x-direction.
    dy : float, optional
        The step size in the y-direction.
    dz : float, optional
        The step size in the z-direction.

    Returns
    -------
    tf.Tensor
        The divergence of the input tensor.
    """

    txx = tensor[0,0,...]
    tyx = tensor[1,0,...]
    tzx = tensor[2,0,...]
    txy = tensor[0,1,...]
    tyy = tensor[1,1,...]
    tzy = tensor[2,1,...]
    txz = tensor[0,2,...]
    tyz = tensor[1,2,...]
    tzz = tensor[2,2,...]

    txx_x = gradient(txx, dx, axis=0)
    txy_y = gradient(txy, dy, axis=1)
    txz_z = gradient(txz, dz, axis=2)

    tyx_x = gradient(tyx, dx, axis=0)
    tyy_y = gradient(tyy, dy, axis=1)
    tyz_z = gradient(tyz, dz, axis=2)

    tzx_x = gradient(tzx, dx, axis=0)
    tzy_y = gradient(tzy, dy, axis=1)
    tzz_z = gradient(tzz, dz, axis=2)

    div = tf.stack([txx_x + txy_y + txz_z,
                    tyx_x + tyy_y + tyz_z,
                    tzx_x + tzy_y + tzz_z], axis=0)
    return div

def param(a):
    return 8.0 * a

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

    mu_t = problem.extra.data_mu

    e_u = rmse(u, problem.extra.data_u)
    e_mu = rmse(mu, mu_t)
    e_mu_rel = np.mean(((mu - mu_t) / mu_t) ** 2)

    if history is not None:
        history.append("eval_rms_u", e_u)
        history.append("eval_rms_mu", e_mu)
        history.append("eval_rel_mu", e_mu_rel)

    return e_u, e_mu, e_mu_rel

def report_func(problem, state, epoch, cbinfo):
    e = history_func(problem, state, epoch, None, cbinfo)
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
    Nx_i = params.get('Nx_inv', 21)
    Ny_i = params.get('Ny_inv', 21)
    Nz_i = params.get('Nz_inv', 11)

    Lx = params.get('Lx', 2.0)
    Ly = params.get('Ly', 2.0)
    Lz = params.get('Lz', 1.0)

    # Physics parameters
    lambda_ = params.get('lambda_', 650.0)
    mu0 = params.get('mu', 8.0)
    p_load = -1 * params.get('p_load', 10.0)

    params['mu_scale'] = mu0
    params['data_csv'] = params.get('data_csv', 'linear_symcube_p10.csv')

    # Inverse parameters
    noise_level = params.get('noise_level', 1e-2)
    noise_seed = params.get('noise_seed', 123)

    lam_pde = params.get("lam_pde", 1e2)
    lam_bcn = params.get("lam_bcn", 1e1)
    lam_dat = params.get("lam_dat", 0e0)

    J_regu = params.get('J_regu', 'TV')
    lam_reg = params.get('lam_reg', 1e-3)

    # File handling
    tag = params.get('run_name', None)
    if tag is None:
        tag = f"Ninv{Nx_i}x{Ny_i}x{Nz_i}_noise{float(noise_level)}"

    # output root and run directory
    out_root = Path(params.get('out_root', 'runs_odil'))
    out_dir = out_root / tag
    params['outdir'] = str(out_dir)
    params['viz_dir'] = "viz"

    params = SimpleNamespace(**params)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    params = parser.parse_args(namespace=params)

    # Optimizer parameters
    params.optimizer1 = 'adam'
    params.optimizer2 = 'lbfgs'
    params.linsolver = 'direct'
    params.multigrid = True
    params.lr = 0.01
    params.epochs1 = 1000
    params.epochs2 = 1000001


    params.plot_every = 5000

    def operator(ctx):
        extra = ctx.extra
        mod = ctx.mod
        args = extra.args

        dx, dy, dz = ctx.step()
        ix, iy, iz = ctx.indices()
        nx, ny, nz = ctx.size()

        ux = ctx.field("ux", 0, 0, 0)
        uy = ctx.field("uy", 0, 0, 0)
        uz = ctx.field("uz", 0, 0, 0)
        ux = mod.where(ix == 0, ctx.cast(0.0), ux)
        uy = mod.where(iy == 0, ctx.cast(0.0), uy)
        uz = mod.where(iz == 0, ctx.cast(0.0), uz)

        mu_r = ctx.field("mu_raw", 0, 0, 0)

        l = ctx.cast(lambda_)
        mu = param(mu_r)  # already grid-shaped
        mask = ix < iy
        mu = mod.where(mask, ctx.cast(8.0), ctx.cast(16.0))
        # ----------------- strain -----------------
        E = compute_strain_tensor_lagrangian_full(ux, uy, uz, ctx)

        # ----------------- nodal stresses -----------------
        sigma= compute_stress(E, lambda_, mu)

        # ----------------- div σ -----------------
        #div = divergence_fd(sigma, dx, dy, dz)
        div = compute_div_stress(E, lambda_, mu, dx, dy, dz)

        # ----------------- PDE residual -----------------
        vol = ctx.cast(dx * dy * dz)

        offset = 1
        interior = (ix >= offset) & (ix <= nx - 1 - offset) & (iy >= offset) & (iy <= ny - 1 - offset) & (iz >= offset) & (iz <= nz - 1 - offset)
        M = ctx.cast(interior)

        kpde = ctx.cast(lam_pde) * mod.sqrt(vol)
        res = [
            ("pde", -kpde * M * div),
        ]

        # ----------------- Neumann BCs on max faces -----------------

        ayz = ctx.cast(dy * dz)
        axz = ctx.cast(dx * dz)
        axy = ctx.cast(dx * dy)

        kneum_x = ctx.cast(lam_bcn) * mod.sqrt(ayz)
        kneum_y = ctx.cast(lam_bcn) * mod.sqrt(axz)
        kneum_z = ctx.cast(lam_bcn) * mod.sqrt(axy)

        mask_xp = (ix == nx - 1)
        mask_yp = (iy == ny - 1)
        mask_zp = (iz == nz - 1)

        p_vec = tf.reshape(tf.constant([0.0, 0.0, p_load], dtype=tf.float64), (1,3,1,1,1))

        res += [
            ("bc_z", kneum_z * mod.where(mask_zp, sigma[:,2,...] - p_vec, ctx.cast(0))),
            ("bc_y", kneum_y * mod.where(mask_yp, sigma[:,1,...], ctx.cast(0))),
            ("bc_x", kneum_x * mod.where(mask_xp, sigma[:,0, ...], ctx.cast(0)))
        ]

        # ----------------- data misfit -----------------

        u = tf.stack([ux, uy, uz], axis=-1)
        res += [
            ("data_u", ctx.cast(lam_dat) * (u - mod.constant(extra.data_u))),
            ("r_u", ctx.cast(1e-2) * u)
        ]



        # ----------------- μ-regularization -----------------

        gmx = gradient(mu, dx, axis=0)
        gmy = gradient(mu, dy, axis=1)
        gmz = gradient(mu, dz, axis=2)

        if J_regu == 'TV':
            eps = ctx.cast(1e-8)
            tv = mod.sqrt(gmx * gmx + gmy * gmy + gmz * gmz + eps)
            res += [("mu_tv", ctx.cast(lam_reg) * tv)]
        elif J_regu == 'H1':
            res += [("mu_h1_x", ctx.cast(lam_reg) * gmx),
                        ("mu_h1_y", ctx.cast(lam_reg) * gmy),
                        ("mu_h1_z", ctx.cast(lam_reg) * gmz)]

        return res

    domain = odil.Domain(
        cshape=(Nx_i, Ny_i, Nz_i),
        dimnames=("x", "y", "z"),
        lower=(0, 0, 0),
        upper=(Lx, Ly, Lz),
        multigrid=False,
        dtype=np.float64
    )
    mod = domain.mod

    data_u, data_mu, data_E, data_mask = load_data_csv_to_grid(
        domain, params.data_csv, mu_scale=1.0, dtype=np.float64,
        noise_level=noise_level, noise_seed=noise_seed
    )

    odil.setup_outdir(params, relpath_args=["checkpoint"])

    covered = int(np.sum(data_mask)) if data_mask is not None else 0
    total = int(np.prod(domain.cshape))
    printlog(f"data coverage: {covered}/{total} = {covered / total:.2%}")

    # Pack extras once
    extra = argparse.Namespace()
    extra.args = params

    extra.data_u = data_u
    extra.data_mu = data_mu
    extra.data_E = data_E
    extra.data_mask = data_mask
    extra.epoch = mod.variable(domain.cast(0))

    # State
    state = odil.State(
        fields = {"ux": None, "uy": None, "uz": None, "mu_raw": None}
    )
    state = domain.init_state(state)

    #state.fields["mu_raw"] = np.full(domain.cshape, 1.5, dtype=np.float64)

    state = domain.init_state(state)
    problem = odil.Problem(operator, domain, extra)

    callback = odil.make_callback(
        problem, params, plot_func=plot_func, history_func=history_func, report_func=report_func
    )

    def cb(state, epoch, pinfo):
        # 1) Enforce constraints in-place
        a = state.fields["mu_raw"].array
        a = np.clip(a, 8.0/8.0, 16.0/8.0)  # enforce a >= 0 if desired

        state.fields["mu_raw"].array = a
        # 2) Continue with your normal callback (history/plot/report)
        return callback(state, epoch, pinfo)

    params.epochs = params.epochs1
    odil.util.optimize(params, params.optimizer1, problem, state, cb)
    params.epochs = params.epochs2
    # Rename first training log
    if os.path.exists("train.csv"):
        os.rename("train.csv", "train_tmp.csv")
    callback = odil.make_callback(
        problem, params, plot_func=plot_func, history_func=history_func, report_func=report_func
    )
    odil.util.optimize(params, params.optimizer2, problem, state, cb)
    # Combine both training logs
    if os.path.exists("train_tmp.csv") and os.path.exists("train.csv"):
        df1 = pd.read_csv("train_tmp.csv")
        df2 = pd.read_csv("train.csv")

        # Shift LBFGS epochs to continue from the end of Adam
        last_epoch = df1["epoch"].iloc[-1] if "epoch" in df1.columns else len(df1)
        if "epoch" in df2.columns:
            df2["epoch"] += last_epoch + 1

        # Concatenate
        df_combined = pd.concat([df1, df2], ignore_index=True)

        # Save combined training log
        df_combined.to_csv("train.csv", index=False)

        # Optionally remove temporary file
        os.remove("train_tmp.csv")

    png_path, cols = plot_losses_from_csv(
        "train.csv",
        out_png="train_losses.png",
        columns=["eval_rms_u", "eval_rms_mu", "eval_rel_mu", "loss", "norm_pde", "norm_data_u", "norm_bc_z", "norm_bc_y", "norm_bc_x"],
        logy=True,
    )

    mlflow.set_tracking_uri("http://143.50.189.222:8080")  # 143.50.189.222
    mlflow.set_experiment("elasticity_aao_odil:0.0.0")
    mlflow.start_run()
    mlflow.log_params(vars(params))
    mlflow.log_artifact("train.csv")
    mlflow.log_artifact("train_losses.png")

    # Load train.csv and select the last row
    train_df = pd.read_csv("train.csv")
    last_row = train_df.iloc[-1]

    # Log each metric in that row
    for key, value in last_row.items():
        if np.issubdtype(type(value), np.number):
            mlflow.log_metric(key, float(value))
    mlflow.end_run()

if __name__ == "__main__":
    invscar()
