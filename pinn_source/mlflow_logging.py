"""PINNs-specific MLflow artifact and metric logging.

All functions assume an MLflow run is already active (opened by ExperimentRunner).
"""

import json
import os


def log_pinns_artifacts(loss_handler, train_handler,
                        base_dir=None, timings=None):
    """Log PINNs-specific metrics and artifacts to the active MLflow run.

    Parameters
    ----------
    loss_handler : PINNLossHandler
        Has test_losses with capturable final metrics.
    train_handler : PINNTrainHandler
        Has filenames dict pointing to saved history/plot files.
    base_dir : str or None
        Run output directory; all files are logged recursively as artifacts.
    timings : dict[str, float] or None
        Per-phase wall times in seconds (keys: 'fit', 'physics', 'main').
    """
    import mlflow

    # -- Metrics: evaluate test losses -------------------------------------- #
    for loss in loss_handler.test_losses.get("main", []):
        try:
            val = float(loss.loss_base_call().numpy())
            mlflow.log_metric(f"test/{loss.name}", val)
        except Exception as e:
            print(f"[mlflow] skipping metric test/{loss.name}: {e}")

    # -- Timings ------------------------------------------------------------ #
    if timings:
        total = 0.0
        for phase, secs in timings.items():
            mlflow.log_metric(f"time/{phase}_s", secs)
            total += secs
        mlflow.log_metric("time/total_s", total)

    # -- Metrics: loss trajectories from history JSONs ---------------------- #
    for phase, paths in train_handler.filenames.items():
        json_path = paths.get('data')
        if json_path and os.path.isfile(json_path):
            _log_loss_trajectories(json_path, phase)

    # -- Artifacts: entire run directory (recursively) ---------------------- #
    if base_dir and os.path.isdir(base_dir):
        mlflow.log_artifacts(base_dir)


def _log_loss_trajectories(json_path, phase, batch_size=500):
    """Read a history JSON and log each loss trajectory as MLflow step-metrics."""
    import mlflow
    from mlflow.entities import Metric
    import time

    with open(json_path, 'r') as f:
        history = json.load(f)

    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    timestamp = int(time.time() * 1000)

    metrics = []
    for name, entry in history.get('losses', {}).items():
        vals = entry.get('log', [])
        iters = entry.get('iter', [])
        if not vals:
            continue
        metric_key = f"{phase}/{name}"
        for step, val in zip(iters, vals):
            metrics.append(Metric(metric_key, val, timestamp, step))

    for i in range(0, len(metrics), batch_size):
        client.log_batch(run_id, metrics=metrics[i:i + batch_size])
