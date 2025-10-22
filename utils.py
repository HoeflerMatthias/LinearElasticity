from pathlib import Path
import itertools
import csv
import time
import json
import os

def grid_product(grid_dict):
    """Expand a dict of lists into a list of param dicts."""
    keys = list(grid_dict.keys())
    for vals in itertools.product(*(grid_dict[k] for k in keys)):
        yield dict(zip(keys, vals))

def run_grid(out_root='runs', base_params=None, grid=None, run_name_fmt=None, append_global_summary=True):
    """Run a hyperparameter grid and collect a master CSV and JSON index.

    Parameters
    ----------
    out_root : str
        Root output folder. Each run goes into out_root/<tag>.
    base_params : dict or None
        Parameters passed to every run (e.g., geometry, physics). Overridden by grid-specific values.
    grid : dict
        Keys are parameter names, values are lists of values to sweep.
        Example: {'J_regu':['L2','H1'], 'lmbda':[1e-8,1e-7], 'Nx_inv':[16,24]}
    run_name_fmt : callable or str or None
        If callable(params)->str, use to build the run tag.
        If string, it will be formatted with params via .format(**params).
        If None, invscar will auto-generate tag from meshes and p_load.
    append_global_summary : bool
        Whether each run should append a line to out_root/summary.csv (invscar handles it).
    """
    Path(out_root).mkdir(parents=True, exist_ok=True)

    base_params = dict(base_params or {})
    grid = grid or {}

    # index file to collect per-run metadata beyond the CSV
    index_json = Path(out_root) / 'index.json'
    index = []
    if index_json.exists():
        try:
            index = json.loads(index_json.read_text())
        except Exception:
            index = []

    for sweep_params in grid_product(grid):
        # compose full params
        params = dict(base_params)
        params.update(sweep_params)
        params['out_root'] = out_root
        params['append_global_summary'] = append_global_summary

        # optional deterministic noise per run
        noise_seed = params.get('noise_seed', None)
        if noise_seed is None:
            # derive a seed from key hyperparams for reproducibility
            seed = hash(tuple(sorted((k, str(v)) for k, v in sweep_params.items()))) % (2**32)
            params['noise_seed'] = seed

        # craft run tag if requested
        if run_name_fmt is not None:
            if callable(run_name_fmt):
                tag = run_name_fmt(params)
            else:
                tag = str(run_name_fmt).format(**{k: params.get(k) for k in params})
            params['run_name'] = tag

        print("\n=== Running:", {k: params[k] for k in sorted(sweep_params.keys())})
        t0 = time.time()
        res = invscar(**params)
        t1 = time.time()

    print("\nAll runs complete. Global summary CSV at:", Path(out_root) / 'summary.csv')
    print("JSON index at:", index_json)

def load_data_csv_to_grid(domain, path, mu_scale=8.0, dtype=np.float32, noise_level=0.0, noise_seed=1234):
    """
    CSV columns required:
      x,y,z,ux,uy,uz,alpha,e_xx,e_xy,e_xz,e_yx,e_yy,e_yz,e_zx,e_zy,e_zz
    Returns:
      data_u  : (Nx,Ny,Nz,3)
      data_mu : (Nx,Ny,Nz)
      data_E  : (Nx,Ny,Nz,3,3)  # optional (not used in loss)
      mask    : (Nx,Ny,Nz) with 1 where data present, else 0
    """

    arr = _np.genfromtxt(path, delimiter=",", names=True)
    names = [n.lower() for n in arr.dtype.names]
    need = lambda cols: all(c in names for c in cols)
    assert need(["x", "y", "z", "ux", "uy", "uz", "alpha"]), "CSV missing required columns"

    Nx, Ny, Nz = domain.cshape
    x1, y1, z1 = domain.points_1d()

    if noise_level > 0.0:
        # Add noise and re-apply constrained DOFs
        u = np.stack([arr["ux"], arr["uy"], arr["uz"]], axis=-1).astype(dtype)
        sigma_u = np.max(np.abs(u), axis=0) / 3

        rng = np.random.default_rng(noise_seed)
        u += noise_level * sigma_u * rng.normal(size=u.shape)
        arr["ux"] = u[:, 0]
        arr["uy"] = u[:, 1]
        arr["uz"] = u[:, 2]

    def nearest_idx(grid, q):
        idx = _np.searchsorted(grid, q)
        idx = _np.clip(idx, 1, len(grid) - 1)
        left = grid[idx - 1];
        right = grid[idx]
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

    #ux = particles_to_field_3d_average(arr["ux"], arr["x"], arr["y"], arr["z"], domain)
    #uy = particles_to_field_3d_average(arr["ux"], arr["x"], arr["y"], arr["z"], domain)
    #uz = particles_to_field_3d_average(arr["ux"], arr["x"], arr["y"], arr["z"], domain)
    #data_u = tf.stack([ux, uy, uz], axis=-1)

    mask[I, J, K] = 1

    data_E = None
    if need(["e_xx", "e_xy", "e_xz", "e_yx", "e_yy", "e_yz", "e_zx", "e_zy", "e_zz"]):
        data_E = _np.zeros((Nx, Ny, Nz, 3, 3), dtype=dtype)
        data_E[I, J, K, 0, 0] = arr["e_xx"];
        data_E[I, J, K, 0, 1] = arr["e_xy"];
        data_E[I, J, K, 0, 2] = arr["e_xz"]
        data_E[I, J, K, 1, 0] = arr["e_yx"];
        data_E[I, J, K, 1, 1] = arr["e_yy"];
        data_E[I, J, K, 1, 2] = arr["e_yz"]
        data_E[I, J, K, 2, 0] = arr["e_zx"];
        data_E[I, J, K, 2, 1] = arr["e_zy"];
        data_E[I, J, K, 2, 2] = arr["e_zz"]

    return data_u, data_mu, data_E, mask