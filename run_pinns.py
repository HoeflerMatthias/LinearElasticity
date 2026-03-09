#!/usr/bin/env python3
"""Ray Tune hyperparameter sweep for PINNs inverse elasticity.

Each trial loads the base YAML config, applies search-space overrides,
then loops over seeds — one MLflow run per (config, seed) pair.
Ray Tune handles GPU assignment and trial concurrency.
"""

import os

from ray import tune

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

# ── Trial constants (not part of the search space) ────────────────────────── #
SETUP_FILE = "pinn_source/config.yaml"
EXPERIMENT_NAME = "lin_elast:pinns"
SEEDS = [1, 2, 3]


# ── Objective ─────────────────────────────────────────────────────────────── #

def objective(config):
    from pinn_source.utils.file_utils import load_config
    from pinn_source.experiment_runner import ExperimentRunner
    from pinn_source.run import apply_overrides, make_algorithm_fn, make_post_run_fn

    params = load_config(SETUP_FILE)
    apply_overrides(params, config)

    runner = ExperimentRunner(
        params=params,
        algorithm_fn=make_algorithm_fn(),
        experiment_name=EXPERIMENT_NAME,
        post_run_fn=make_post_run_fn(),
    )

    for seed in SEEDS:
        runner.run(seed=seed)


# ── Search space & launcher ───────────────────────────────────────────────── #

if __name__ == "__main__":
    search_space = {
        "numData": tune.grid_search([500, 1000, 2000]),

        "net/layers": tune.grid_search([[32, 32, 32], [64, 64, 64]]),
        "inverse_params/mu/net/layers": tune.grid_search([[16, 16, 16], [32, 32, 32]]),

        "wPDE": tune.grid_search([1e3, 1e5]),
        "wFit": tune.grid_search([1e5, 1e6]),
        "wBCN": tune.grid_search([1e3, 1e4]),
    }

    objective_gpu = tune.with_resources(objective, {"gpu": 1})

    tuner = tune.Tuner(
        objective_gpu,
        tune_config=tune.TuneConfig(
            num_samples=1,
            max_concurrent_trials=4,
        ),
        param_space=search_space,
    )
    tuner.fit()
