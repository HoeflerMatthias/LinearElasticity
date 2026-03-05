from firedrake import *
import numpy as np


def load_ground_truth(data_csv):
    """Load ground truth from HDF5 checkpoint. Returns (mesh, alpha_true, u_true)."""
    with CheckpointFile(data_csv, "r") as chk:
        mesh = chk.load_mesh()
        alpha_true = chk.load_function(mesh, name="alpha_true")
        u_true = chk.load_function(mesh, name="u_true")
    return mesh, alpha_true, u_true


def apply_noise(ud, bcs, noise_level, noise_seed):
    """Add noise to displacement data in-place and re-apply BCs. Returns rng."""
    sigma_u = np.max(np.abs(ud.dat.data_ro), axis=0) / 3
    rng = np.random.default_rng(noise_seed)
    ud.dat.data[:] += noise_level * sigma_u * rng.normal(size=ud.dat.data.shape)
    for bc in bcs:
        bc.apply(ud)
    return rng
