import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from utils import grid_product

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8080")
EXPERIMENT = "elasticity_lsfem"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))


def _run_one(params):
    """Worker: import Firedrake fresh in spawned process, solve, and log."""
    import mlflow
    from lsfem import invscar
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

    grid = dict(
        J_regu=['L2', 'H1', 'TV'],
        lam_reg=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        noise_level=[1e-2, 5e-2, 1e-1],
    )

    all_params = [{**base, **sweep} for sweep in grid_product(grid)]

    print(f"Running {len(all_params)} experiments with {MAX_WORKERS} workers")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as pool:
        list(pool.map(_run_one, all_params))