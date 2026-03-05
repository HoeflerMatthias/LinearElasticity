import itertools
import time


def grid_product(grid_dict):
    """Expand a dict of lists into a list of param dicts."""
    keys = list(grid_dict.keys())
    for vals in itertools.product(*(grid_dict[k] for k in keys)):
        yield dict(zip(keys, vals))


def run_grid(invscar, base_params=None, grid=None, log_fn=None):
    """Run a hyperparameter grid, optionally logging each result.

    Parameters
    ----------
    invscar : callable
        Solver function (**params) -> InvScarResult.
    base_params : dict or None
        Parameters passed to every run.
    grid : dict
        Keys are parameter names, values are lists of values to sweep.
    log_fn : callable or None
        If provided, called with each InvScarResult for logging (e.g. log_result_to_mlflow).
    """
    base_params = dict(base_params or {})
    grid = grid or {}

    results = []
    for sweep_params in grid_product(grid):
        params = {**base_params, **sweep_params}

        # Derive a deterministic noise seed if not set
        noise_seed = params.get('noise_seed', None)
        if noise_seed is None:
            seed = hash(tuple(sorted((k, str(v)) for k, v in sweep_params.items()))) % (2**32)
            params['noise_seed'] = seed

        print("\n=== Running:", {k: params[k] for k in sorted(sweep_params.keys())})
        t0 = time.time()
        result = invscar(**params)
        t1 = time.time()
        print(f"    Completed in {t1 - t0:.1f}s")

        if log_fn is not None:
            log_fn(result)

        results.append(result)

    print(f"\nAll {len(results)} runs complete.")
    return results
