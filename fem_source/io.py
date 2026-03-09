import os
import tempfile

from firedrake import CheckpointFile


def save_solution_checkpoint(u, alpha):
    """Save computed solution to a temp HDF5 file. Returns the file path."""
    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    tmp.close()
    with CheckpointFile(tmp.name, "w") as chk:
        chk.save_mesh(u.function_space().mesh())
        chk.save_function(u, name="u")
        chk.save_function(alpha, name="alpha")
    return tmp.name


def log_fem_artifacts(result):
    """Log FEM-specific artifacts to the active MLflow run.

    Assumes an MLflow run is already active (opened by ExperimentRunner).
    Scalar metrics are handled by ExperimentRunner; this logs history
    trajectories as step-metrics and the solution checkpoint as artifact.
    """
    import mlflow

    _log_history_metrics(result.metrics)
    if result.solution_file:
        mlflow.log_artifact(result.solution_file)
        os.unlink(result.solution_file)


def log_result_to_mlflow(result):
    """Log an InvScarResult to MLflow (opens its own run).

    Kept for standalone use outside ExperimentRunner.
    """
    import mlflow

    with mlflow.start_run():
        mlflow.log_params(result.params)

        for k, val in result.metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(k, float(val))
            elif isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, (int, float)):
                        mlflow.log_metric(f"{k}.{sub_k}", float(sub_v))

        log_fem_artifacts(result)


def _log_history_metrics(metrics, batch_size=500):
    """Log history lists as MLflow step-metrics using batched API."""
    import mlflow
    from mlflow.entities import Metric
    import time

    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    timestamp = int(time.time() * 1000)

    batch = []
    for key, values in metrics.items():
        if not (key.endswith("_hist") and isinstance(values, list) and values):
            continue
        # Scalar histories (e.g. J_fid_hist, err_u_rel_hist)
        if not isinstance(values[0], dict):
            metric_key = key.removesuffix("_hist")
            for step, val in enumerate(values):
                if isinstance(val, (int, float)):
                    batch.append(Metric(metric_key, float(val), timestamp, step))

    for i in range(0, len(batch), batch_size):
        client.log_batch(run_id, metrics=batch[i:i + batch_size])
