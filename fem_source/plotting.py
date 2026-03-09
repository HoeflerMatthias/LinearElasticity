"""Plotting utilities for FEM solution fields."""
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def plot_alpha_slice(solution_file, z_frac=0.5, tol_frac=0.05):
    """Plot alpha field on a z-slice from an HDF5 checkpoint.

    Returns path to a temp PNG (caller must clean up).
    """
    from firedrake import CheckpointFile

    with CheckpointFile(solution_file, "r") as chk:
        mesh = chk.load_mesh()
        alpha = chk.load_function(mesh, name="alpha")

    coords = alpha.function_space().mesh().coordinates.dat.data_ro
    vals = alpha.dat.data_ro

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    z_min, z_max = z.min(), z.max()
    Lz = z_max - z_min
    z_target = z_min + z_frac * Lz
    tol = tol_frac * Lz

    mask = np.abs(z - z_target) < tol
    if mask.sum() < 3:
        tol = 0.15 * Lz
        mask = np.abs(z - z_target) < tol

    x_s, y_s, v_s = x[mask], y[mask], vals[mask]

    fig, ax = plt.subplots(figsize=(5, 4))
    triang = tri.Triangulation(x_s, y_s)
    tcf = ax.tricontourf(triang, v_s, levels=32, cmap="RdYlGn_r")
    fig.colorbar(tcf, ax=ax, label=r"$\alpha$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"alpha (z = {z_target:.2f})")
    ax.set_aspect("equal")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="alpha_slice_", delete=False)
    tmp.close()
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tmp.name
