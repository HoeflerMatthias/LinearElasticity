"""Ray Tune sweep for FEM inverse solvers (lsfem, kkt, reduced).

Usage:
    python run_experiments.py lsfem
    python run_experiments.py kkt
    python run_experiments.py reduced
"""
import os
import shutil
import sys
import tempfile

from ray import tune

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 64))

SOLVERS = {
    "lsfem": "lin_elast:lsfem",
    "kkt": "lin_elast:kkt",
    "reduced": "lin_elast:reduced",
}


# ── Objective ─────────────────────────────────────────────────────────────── #

def objective(config):
    solver_name = config.pop("_solver")
    experiment_name = SOLVERS[solver_name]

    import importlib
    from pinn_source.experiment_runner import ExperimentRunner
    from fem_source.io import log_fem_artifacts

    solver_module = f"fem_source.{solver_name}"

    def algorithm_fn(params, seed):
        mod = importlib.import_module(solver_module)
        result = mod.invscar(**params)
        # Flatten scalar metrics for ExperimentRunner logging
        metrics = {}
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        metrics[f"{k}.{sub_k}"] = float(sub_v)
        return {"metrics": metrics, "fem_result": result}

    def post_run_fn(result_dict):
        log_fem_artifacts(result_dict["fem_result"])

    runner = ExperimentRunner(
        params=config,
        algorithm_fn=algorithm_fn,
        experiment_name=experiment_name,
        hash_exclude_keys=["data_csv"],
        post_run_fn=post_run_fn,
    )
    runner.run(seed=0)


# ── Search spaces ─────────────────────────────────────────────────────────── #

BASE = dict(
    Nx_inv=10, Ny_inv=10, Nz_inv=5,
    data_csv="/app/linear_symcube_p10.h5",
)


def search_space_lsfem():
    return {
        **BASE,
        "_solver": "lsfem",
        "J_regu": tune.grid_search(["L2", "H1", "TV"]),
        "lam_reg": tune.grid_search([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        "noise_level": tune.grid_search([1e-2, 5e-2, 1e-1]),
    }


def search_space_kkt():
    """KKT sweeps one regularisation type at a time (others set to 0)."""
    spaces = []
    lam_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    for regu_key in ["lam_L2", "lam_H1", "lam_TV"]:
        others = {k: 0.0 for k in ["lam_L2", "lam_H1", "lam_TV"] if k != regu_key}
        spaces.append({
            **BASE,
            **others,
            "_solver": "kkt",
            regu_key: tune.grid_search(lam_values),
            "noise_level": tune.grid_search([1e-2, 5e-2, 1e-1]),
        })
    return spaces


def search_space_reduced():
    return {
        **BASE,
        "_solver": "reduced",
        "J_regu": tune.grid_search(["L2", "H1", "TV"]),
        "lam_reg": tune.grid_search([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        "noise_level": tune.grid_search([1e-2, 5e-2, 1e-1]),
    }


SEARCH_SPACES = {
    "lsfem": search_space_lsfem,
    "kkt": search_space_kkt,
    "reduced": search_space_reduced,
}


# ── Launcher ──────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SOLVERS:
        print(f"Usage: python {sys.argv[0]} {{{','.join(SOLVERS)}}}")
        sys.exit(1)

    solver_name = sys.argv[1]
    spaces = SEARCH_SPACES[solver_name]()

    objective_cpu = tune.with_resources(objective, {"cpu": 1})
    ray_storage = tempfile.mkdtemp(prefix="ray_fem_")

    def run_tuner(param_space):
        tuner = tune.Tuner(
            objective_cpu,
            tune_config=tune.TuneConfig(
                num_samples=1,
                max_concurrent_trials=MAX_WORKERS,
            ),
            param_space=param_space,
            run_config=tune.RunConfig(storage_path=ray_storage),
        )
        tuner.fit()

    # KKT returns a list of spaces (one per regularisation type)
    if isinstance(spaces, list):
        for i, space in enumerate(spaces):
            print(f"--- KKT sweep {i+1}/{len(spaces)} ---")
            run_tuner(space)
    else:
        run_tuner(spaces)

    shutil.rmtree(ray_storage, ignore_errors=True)
