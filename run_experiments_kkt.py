import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from utils import grid_product

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8080")
EXPERIMENT = "elasticity_kkt"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))


def _run_one(params):
    """Worker: import Firedrake fresh in spawned process, solve, and log."""
    import mlflow
    from kkt import invscar
    from fem_source import log_result_to_mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    print("\n=== Running:", {k: v for k, v in params.items() if k not in ('data_csv',)})
    result = invscar(**params)
    log_result_to_mlflow(result)


if __name__ == '__main__':
    base = dict(
        Nx_inv=10, Ny_inv=10, Nz_inv=5,
        data_csv='/app/linear_symcube_p10.h5',
    )

    lam_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    noise_values = [1e-2, 5e-2, 1e-1]

    # Build full parameter list
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

    print(f"Running {len(all_params)} experiments with {MAX_WORKERS} workers")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as pool:
        list(pool.map(_run_one, all_params))
