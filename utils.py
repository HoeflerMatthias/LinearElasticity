from pathlib import Path
import itertools
import csv
import time
import json
import os
import numpy as np

def grid_product(grid_dict):
    """Expand a dict of lists into a list of param dicts."""
    keys = list(grid_dict.keys())
    for vals in itertools.product(*(grid_dict[k] for k in keys)):
        yield dict(zip(keys, vals))

def run_grid(invscar, out_root='runs', base_params=None, grid=None, run_name_fmt=None, append_global_summary=True):
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

def L2_error(a, a_ref, rel=True):
    """
    Compute relative L2 error between 'a' and 'a_ref',
    interpolating 'a' onto the mesh of 'a_ref' if needed.
    """
    mesh_ref = a_ref.ufl_domain()

    # --- 1. Interpolate 'a' to the reference mesh if meshes differ ---
    if a.ufl_domain().id() != mesh_ref.id():
        V_ref = a_ref.function_space()
        # Interpolate expression of 'a' onto the finer mesh
        a_fine = Function(V_ref)
        a_fine.interpolate(a)   # evaluates a at fine-mesh quadrature points
    else:
        a_fine = a

    # --- 2. Assemble L2 norms on the reference (fine) mesh ---
    if rel:
        err = assemble(dot(a_fine - a_ref, a_fine - a_ref) / (dot(a_ref, a_ref) + 1e-12) * dx(domain=mesh_ref))
    else:
        err = assemble(dot(a_fine - a_ref, a_fine - a_ref) * dx(domain=mesh_ref))

    return float(np.sqrt(err))