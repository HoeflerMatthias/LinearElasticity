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
    from mpi4py import MPI as _MPI
    comm = ud.function_space().mesh().comm
    local_max = np.max(np.abs(ud.dat.data_ro), axis=0)
    global_max = np.empty_like(local_max)
    comm.Allreduce(local_max, global_max, op=_MPI.MAX)
    sigma_u = global_max / 3
    rng = np.random.default_rng(noise_seed)
    ud.dat.data[:] += noise_level * sigma_u * rng.normal(size=ud.dat.data.shape)
    for bc in bcs:
        bc.apply(ud)
    return rng


def apply_pressure_noise(mesh, p_load_value, p_noise_level, rng):
    """Create a pointwise-noisy pressure load Function on face 6.

    Returns a CG1 scalar Function with base value + per-DOF noise on
    face 6.  Noise scaling: sigma = p_noise_level * |p_load_value| / 3,
    matching the displacement noise convention.

    Returns None when p_noise_level <= 0.
    """
    if p_noise_level <= 0.0:
        return None
    V = FunctionSpace(mesh, "CG", 1)
    p_func = Function(V, name="p_load")
    p_func.assign(Constant(p_load_value))
    bc = DirichletBC(V, 0, 6)
    nodes = bc.nodes
    local_size = p_func.dat.data.shape[0]
    local_nodes = nodes[nodes < local_size]
    sigma = p_noise_level * abs(p_load_value) / 3
    p_func.dat.data[local_nodes] += sigma * rng.normal(size=len(local_nodes))
    return p_func


def make_observation_weight(mesh, n_elements, seed):
    """Create a DG0 weight function selecting *n_elements* random cells.

    Returns a Function that is (total/n_elements) on selected elements and 0.0
    elsewhere, so that the spatial average of w equals 1.0 regardless of the
    number of selected elements.  This keeps the data-fidelity term properly
    scaled relative to the regularization.
    If *n_elements* >= total number of elements, all elements get weight 1.0.
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
    scale = global_total / n_elements
    offset = sum(all_counts[:comm.rank])
    local_chosen = chosen_global[(chosen_global >= offset) & (chosen_global < offset + local_count)] - offset
    w.dat.data[local_chosen] = scale

    if comm.rank == 0:
        print(f"[obs_weight] selected {n_elements}/{global_total} elements, weight={scale:.2f} (seed={seed})")
    return w
