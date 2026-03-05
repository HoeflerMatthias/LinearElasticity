import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from utils import grid_product

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 4))


def _run_one(params):
    """Worker: import in spawned process and solve."""
    from aao_odil import invscar

    print("\n=== Running:", {k: v for k, v in params.items() if k not in ('data_csv',)})
    invscar(**params)


if __name__ == '__main__':
    base = dict(
        Nx_inv=11, Ny_inv=11, Nz_inv=6,
        lambda_=650.0, mu=8.0, p_load=10.0,
        lam_pde=1e1, lam_bcn=1e1, lam_dat=1e1, lam_reg=1e0,
        noise_level=1e-2,
        data_csv='/app/linear_symcube_p10.csv',
    )

    grid = dict(
        J_regu=['H1', 'TV'],
        lam_pde=[1e2, 1e3],
        lam_bcn=[1e0],
        lam_dat=[1e1],
        lam_reg=[1e-3],
    )

    all_params = [{**base, **sweep} for sweep in grid_product(grid)]

    print(f"Running {len(all_params)} experiments with {MAX_WORKERS} workers")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as pool:
        list(pool.map(_run_one, all_params))
