from pathlib import Path
import itertools
import csv
import time
import json
import os

# Import your invscar from the same folder (or adjust PYTHONPATH)
from aao_odil import invscar


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


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
    ensure_dir(out_root)

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


if __name__ == '__main__':
    # Example sweep
    base = dict(
        # geometry (truth vs inversion)
        #Nx_true=80, Ny_true=80, Nz_true=40,
        Nx_inv=11,  Ny_inv=11,  Nz_inv=6,
        # physics
        lambda_=650.0, mu=8.0, p_load=10.0,
        # objective
        lam_pde=1e1, lam_bcn=1e1, lam_dat=1e1, lam_reg=1e0,
        noise_level=1e-2,
        data_csv='/app/linear_symcube_p10.csv',
    )

    grid = dict(
        J_regu=['H1', 'TV'], #'L2', 'H1',
        lam_pde=[1e2, 1e3],
        lam_bcn=[1e0],
        lam_dat=[1e3],
        lam_reg=[1e-3],
    )

    # optional: give human-friendly run names
    def fmt(p):
        return (f"reg{p['J_regu']}_lam{p['lam_reg']}_"
                f"Ninv{p['Nx_inv']}x{p['Ny_inv']}x{p['Nz_inv']}_"
                f"weight{p['lam_dat']}x{p['lam_pde']}x{p['lam_bcn']}_"
                f"noise{p['noise_level']}"
        )

    run_grid(out_root='/app/runs_odil', base_params=base, grid=grid, run_name_fmt=fmt)