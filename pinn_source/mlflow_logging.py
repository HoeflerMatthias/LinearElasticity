"""PINNs-specific MLflow artifact and metric logging.

All functions assume an MLflow run is already active (opened by ExperimentRunner).
"""

import os


def log_pinns_artifacts(filename, loss_handler, train_handler,
                        artifact_dirs=None, timings=None):
    """Log PINNs-specific metrics and artifacts to the active MLflow run.

    Parameters
    ----------
    filename : str
        Run filename identifier (from params_to_filename).
    loss_handler : PINNLossHandler
        Has test_losses with capturable final metrics.
    train_handler : PINNTrainHandler
        Has filenames dict pointing to saved history/plot files.
    artifact_dirs : list[str] or None
        Directories whose contents (matching *filename*) are logged as artifacts.
    timings : dict[str, float] or None
        Per-phase wall times in seconds (keys: 'fit', 'physics', 'main').
    """
    import mlflow

    mlflow.log_param("run_filename", filename)

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

    # -- Artifacts: history JSON and plot files ------------------------------ #
    for phase, paths in train_handler.filenames.items():
        for kind, path in paths.items():
            if path and os.path.isfile(path):
                mlflow.log_artifact(path, artifact_path=f"{phase}/{kind}")

    # -- Artifacts: post-processing outputs --------------------------------- #
    for d in (artifact_dirs or []):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if filename in f:
                    mlflow.log_artifact(os.path.join(d, f),
                                        artifact_path=os.path.basename(d))
