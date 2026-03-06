"""Unified experiment runner for FEM inverse solvers (lsfem, kkt, reduced).

Usage:
    python run_experiments.py lsfem
    python run_experiments.py kkt
    python run_experiments.py reduced
"""
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from utils import grid_product

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8080")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))

SOLVERS = {
    'lsfem': {
        'experiment': 'lin_elast:lsfem',
        'import': 'fem_source.lsfem',
    },
    'kkt': {
        'experiment': 'lin_elast:kkt',
        'import': 'fem_source.kkt',
    },
    'reduced': {
        'experiment': 'lin_elast:reduced',
        'import': 'fem_source.reduced',
    },
}


def _run_one(args):
    """Worker: import Firedrake fresh in spawned process, solve, and log."""
    solver_name, params = args
    import importlib
    import mlflow
    from fem_source import log_result_to_mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(SOLVERS[solver_name]['experiment'])

    mod = importlib.import_module(SOLVERS[solver_name]['import'])
    print("\n=== Running:", {k: v for k, v in params.items() if k not in ('data_csv',)})
    result = mod.invscar(**params)
    log_result_to_mlflow(result)


def build_grid_lsfem(base):
    grid = dict(
        J_regu=['L2', 'H1', 'TV'],
        lam_reg=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        noise_level=[1e-2, 5e-2, 1e-1],
    )
    return [{**base, **sweep} for sweep in grid_product(grid)]


def build_grid_kkt(base):
    lam_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    noise_values = [1e-2, 5e-2, 1e-1]

    all_params = []
    for regu_key in ['lam_L2', 'lam_H1', 'lam_TV']:
        grid = dict(
            **{regu_key: lam_values},
            noise_level=noise_values,
        )
        other_keys = {'lam_L2': 0.0, 'lam_H1': 0.0, 'lam_TV': 0.0}
        del other_keys[regu_key]

        for sweep_params in grid_product(grid):
            all_params.append({**base, **other_keys, **sweep_params})
    return all_params


def build_grid_reduced(base):
    grid = dict(
        J_regu=['L2', 'H1', 'TV'],
        lam_reg=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        noise_level=[1e-2, 5e-2, 1e-1],
    )
    return [{**base, **sweep} for sweep in grid_product(grid)]


GRID_BUILDERS = {
    'lsfem': build_grid_lsfem,
    'kkt': build_grid_kkt,
    'reduced': build_grid_reduced,
}


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in SOLVERS:
        print(f"Usage: python {sys.argv[0]} {{{','.join(SOLVERS)}}}")
        sys.exit(1)

    solver_name = sys.argv[1]

    base = dict(
        Nx_inv=10, Ny_inv=10, Nz_inv=5,
        data_csv='/app/linear_symcube_p10.h5',
    )

    all_params = GRID_BUILDERS[solver_name](base)

    print(f"Running {len(all_params)} {solver_name} experiments with {MAX_WORKERS} workers")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as pool:
        list(pool.map(_run_one, [(solver_name, p) for p in all_params]))