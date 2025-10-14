#!/usr/bin/env python3

import argparse
import numpy as np

import odil
import matplotlib.pyplot as plt
from odil import printlog


# ---------------------- CLI ----------------------
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data (evaluation + optional u-data fitting)
    parser.add_argument("--data_csv", type=str, default="./../linear_symcube_p10.csv",
                        help="CSV with columns: x,y,z,ux,uy,uz,alpha,e_*")
    parser.add_argument("--mu_scale", type=float, default=8.0,
                        help="mu_gt = alpha * mu_scale (evaluation only)")
    parser.add_argument("--kdata_u", type=float, default=5e6,
                        help="Weight for displacement data misfit (0 disables)")


    parser.add_argument("--viz_refdef", type=int, default=1,
                        help="If 1, render 3D reference(mu) and deformed(|u|) with PyVista")
    parser.add_argument("--def_scale", type=float, default=1.0,
                        help="Visual scale factor for displacement in deformed view")
    parser.add_argument("--viz_img_prefix", type=str, default="viz",
                        help="Prefix for screenshots: viz_ref_*.png / viz_def_*.png")
    parser.add_argument("--viz_edges", type=int, default=0,
                        help="If 1, show mesh edges in render")
    parser.add_argument("--viz_cmap", type=str, default=None,
                        help="Optional PyVista colormap name (e.g. 'viridis'). None=default.")
    parser.add_argument("--export_vtk", type=int, default=0,
                        help="If 1, also export VTK grids (vti) for Paraview")
    parser.add_argument("--vtk_prefix", type=str, default="field",
                        help="Prefix for VTK files, e.g. field_ref_*.vti and field_def_*.vti")

    # ODIL common args
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir="out_elasticity")
    parser.set_defaults(linsolver="direct")
    parser.set_defaults(optimizer="adam")
    parser.set_defaults(lr=0.001)
    parser.set_defaults(double=0)
    parser.set_defaults(multigrid=0)
    parser.set_defaults(plotext="png", plot_title=1)
    parser.set_defaults(plot_every=500, report_every=200, history_full=10, history_every=100, frames=10)
    parser.set_defaults(keep_frozen=1)
    parser.set_defaults(kwreg=0.0, kwregdecay=0)
    return parser.parse_args()


# ---------------------- plotting & metrics ----------------------

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def plot_ref_and_def(domain, ux, uy, uz, mu, def_scale=1.0,
                     prefix="viz", frame=0, sstep=4):
    """
    Save 3D scatter plots:
      - Reference grid colored by mu
      - Deformed grid colored by |u|
    Only subsamples every `sstep` points for readability.
    """
    # Coordinates
    X, Y, Z = [np.array(a) for a in domain.points()]
    umag = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)

    # Subsample for plotting
    Xs = X[::sstep, ::sstep, ::sstep].ravel()
    Ys = Y[::sstep, ::sstep, ::sstep].ravel()
    Zs = Z[::sstep, ::sstep, ::sstep].ravel()
    mu_s = mu[::sstep, ::sstep, ::sstep].ravel()

    Xd = (X + def_scale * ux)[::sstep, ::sstep, ::sstep].ravel()
    Yd = (Y + def_scale * uy)[::sstep, ::sstep, ::sstep].ravel()
    Zd = (Z + def_scale * uz)[::sstep, ::sstep, ::sstep].ravel()
    um_s = umag[::sstep, ::sstep, ::sstep].ravel()

    # --- Reference plot (mu colors) ---
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(Xs, Ys, Zs, c=mu_s, cmap="viridis", s=8, alpha=0.8)
    fig.colorbar(p, ax=ax, shrink=0.5, pad=0.1, label="μ")
    ax.set_title("Reference geometry (μ)")
    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.set_zlabel("z")
    path_ref = f"{prefix}_ref_{frame:05d}.png"
    fig.savefig(path_ref, dpi=150, bbox_inches="tight");
    plt.close(fig)

    # --- Deformed plot (‖u‖ colors) ---
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(Xd, Yd, Zd, c=um_s, cmap="plasma", s=8, alpha=0.8)
    fig.colorbar(p, ax=ax, shrink=0.5, pad=0.1, label="‖u‖")
    ax.set_title(f"Deformed geometry (‖u‖, scale={def_scale})")
    ax.set_xlabel("x");
    ax.set_ylabel("y");
    ax.set_zlabel("z")
    path_def = f"{prefix}_def_{frame:05d}.png"
    fig.savefig(path_def, dpi=150, bbox_inches="tight");
    plt.close(fig)

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
        mu = np.full(domain.cshape, args.mu, dtype=ux.dtype)

    plot_ref_and_def(domain, ux, uy, uz, mu,
                     def_scale=getattr(args, "def_scale", 1.0),
                     prefix=getattr(args, "viz_img_prefix", "viz"),
                     frame=frame,
                     sstep=4)  # reduce density for faster plots



