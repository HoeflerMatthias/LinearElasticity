#!/usr/bin/env python3
"""Helpers for launching PINNs runs with ExperimentRunner + MLflow."""

import os


# ── Config overrides ──────────────────────────────────────────────────────── #

def _set_nested(obj, keys, value):
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        obj = obj[key]
    obj[keys[-1]] = value


def apply_overrides(params, overrides):
    """Apply flat key/value overrides to a nested params dict.

    Keys use ``/`` for nesting (e.g., ``"net/layers"``).
    """
    for key, value in overrides.items():
        parts = key.split("/") if "/" in key else [key]
        _set_nested(params, parts, value)


# ── Algorithm / post-run factories ────────────────────────────────────────── #

def make_algorithm_fn(solver_fn=None):
    """Create an ``algorithm_fn(params, seed)`` for ExperimentRunner.

    Uses a temporary directory with a simple filename since MLflow
    provides the permanent, organized storage.
    """
    if solver_fn is None:
        from pinn_source.solver import run_solver
        solver_fn = run_solver

    def algorithm_fn(params, seed):
        import tempfile
        tmpdir = tempfile.mkdtemp(prefix="pinn_run_")
        params['program']['base_dir'] = tmpdir
        for key in params['program']:
            if key != 'base_dir':
                os.makedirs(os.path.join(tmpdir, params['program'][key]),
                            exist_ok=True)
        return solver_fn(params, "run")
    return algorithm_fn


def make_post_run_fn():
    """Create a ``post_run_fn(result)`` that logs PINNs artifacts and cleans up."""
    def post_run_fn(result):
        from pinn_source.mlflow_logging import log_pinns_artifacts
        log_pinns_artifacts(
            loss_handler=result['loss_handler'],
            train_handler=result['train_handler'],
            base_dir=result.get('base_dir'),
            timings=result.get('timings'),
        )
        # Clean up temporary run directory
        import shutil
        import tempfile
        base_dir = result.get('base_dir')
        if base_dir and base_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(base_dir, ignore_errors=True)
    return post_run_fn
