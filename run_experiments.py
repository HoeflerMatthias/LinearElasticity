"""Run each FEM inverse solver once with its yaml configuration.

Usage:
    python run_experiments.py

Configurations are loaded from fem_source/config_{solver}.yaml.
"""
import importlib
import os

import yaml

from pinn_source.experiment_runner import ExperimentRunner
from fem_source.io import log_fem_artifacts_rank0

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "fem_source")

SOLVERS = {
    "lsfem": "lin_elast:lsfem",
    "kkt": "lin_elast:kkt",
    "reduced": "lin_elast:reduced",
}


def load_config(solver_name):
    path = os.path.join(CONFIG_DIR, f"config_{solver_name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def run_solver(solver_name):
    from firedrake import COMM_WORLD

    config = load_config(solver_name)
    seed = config.pop("seed", 0)
    solver_module = f"fem_source.{solver_name}"

    # All ranks must execute the solver for MPI parallelism
    params = dict(config)
    params.pop("seed", None)
    mod = importlib.import_module(solver_module)
    result = mod.invscar(seed=seed, **params)

    # Only rank 0 logs to MLflow
    if COMM_WORLD.rank == 0:
        experiment_name = SOLVERS[solver_name]
        metrics = {}
        for k, v in result.metrics.items():
            if isinstance(v, (int, float)):
                metrics[k] = float(v)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float)):
                        metrics[f"{k}.{sub_k}"] = float(sub_v)

        runner = ExperimentRunner(
            params=config,
            algorithm_fn=lambda p, s: {"metrics": metrics, "fem_result": result},
            experiment_name=experiment_name,
            post_run_fn=lambda rd: log_fem_artifacts_rank0(rd["fem_result"]),
        )
        runner.run(seed=seed)


if __name__ == "__main__":
    failed = []
    for name in SOLVERS:
        print(f"--- {name} ---")
        try:
            run_solver(name)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            failed.append(name)
    if failed:
        print(f"\nFailed solvers: {', '.join(failed)}")
        exit(1)
