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


def make_observation_weight(mesh, n_elements, seed):
    """Create a DG0 indicator function selecting *n_elements* random cells.

    Returns a Function that is 1.0 on selected elements and 0.0 elsewhere.
    If *n_elements* >= total number of elements, all elements are selected.
    MPI-safe: selects from the global element count and maps to local indices.
    """
    DG0 = FunctionSpace(mesh, "DG", 0)
    w = Function(DG0, name="obs_weight")

    comm = mesh.comm
    local_count = w.dat.data.shape[0]
    all_counts = comm.allgather(local_count)
    global_total = sum(all_counts)

    if n_elements >= global_total:
        if comm.rank == 0:
            print(f"[obs_weight] n_elements={n_elements} >= total={global_total}, using all elements")
        w.dat.data[:] = 1.0
        return w

    # Every rank draws the same global indices (same seed → same RNG state)
    rng = np.random.default_rng(seed)
    chosen_global = rng.choice(global_total, size=n_elements, replace=False)

    # Map global indices to local
    offset = sum(all_counts[:comm.rank])
    local_chosen = chosen_global[(chosen_global >= offset) & (chosen_global < offset + local_count)] - offset
    w.dat.data[local_chosen] = 1.0

    if comm.rank == 0:
        print(f"[obs_weight] selected {n_elements}/{global_total} elements (seed={seed})")
    return w
