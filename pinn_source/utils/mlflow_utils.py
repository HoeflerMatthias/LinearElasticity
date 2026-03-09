import logging
import os
import subprocess
import tempfile

import yaml


def set_mlflow_tracking_uri(uri: str) -> str:
    """Configure the MLflow tracking URI."""
    import mlflow
    mlflow.set_tracking_uri(uri)
    return uri


def get_or_create_experiment_id(experiment_name: str) -> str:
    """Retrieve (or create) the MLflow experiment ID for *experiment_name*."""
    import mlflow
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return mlflow.create_experiment(experiment_name)
    return experiment.experiment_id


def config_completed_in_mlflow(
    param_id: str, seed: int, experiment_id: str, status: str = "complete"
) -> bool:
    """Check if a completed run with this param_id + seed already exists."""
    import mlflow
    client = mlflow.tracking.MlflowClient()
    try:
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.param_id = '{param_id}' "
                f"and tags.status = '{status}' "
                f"and tags.seed = '{seed}'"
            ),
        )
        return len(runs) > 0
    except Exception as e:
        logging.warning(f"[mlflow] Failed to query existing runs: {e}")
        return False


def log_run_metadata(param_id: str, seed: int, params: dict) -> None:
    """Set tags and log the full config as a YAML artifact."""
    import mlflow

    mlflow.set_tag("param_id", param_id)
    mlflow.set_tag("seed", str(seed))

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        mlflow.set_tag("git_commit", commit)
        mlflow.set_tag("git_branch", branch)
    except Exception:
        mlflow.set_tag("git_commit", "unknown")
        mlflow.set_tag("git_branch", "unknown")

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "config.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(params, f, sort_keys=False, default_flow_style=False)
        mlflow.log_artifact(out_path)


def log_params_flat(params: dict, prefix: str = "") -> None:
    """Log all parameters (nested dict) to MLflow in flattened form."""
    import mlflow
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            log_params_flat(v, prefix=key)
        else:
            try:
                mlflow.log_param(key, str(v)[:500])
            except Exception:
                pass


def log_metrics_safely(result: dict) -> None:
    """Log scalar metrics from a result dict."""
    import mlflow
    if not isinstance(result, dict):
        return
    for k, v in result.items():
        if v is None:
            continue
        try:
            mlflow.log_metric(k, float(v))
        except (TypeError, ValueError):
            pass
