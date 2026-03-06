"""MLflow logging for PINNs training runs."""

import os


def _flatten_params(d, prefix=""):
    """Flatten a nested dict into dot-separated keys for mlflow.log_params."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten_params(v, key))
        elif isinstance(v, (list, tuple)):
            flat[key] = str(v)
        else:
            flat[key] = v
    return flat


def log_pinns_to_mlflow(params, filename, loss_handler, train_handler,
                        artifact_dirs=None):
    """Log a completed PINNs run to MLflow.

    Parameters
    ----------
    params : dict
        Full (nested) parameter dict.
    filename : str
        Run filename identifier (from params_to_filename).
    loss_handler : PINNLossHandler
        Has test_losses with capturable final metrics.
    train_handler : PINNTrainHandler
        Has filenames dict pointing to saved history/plot files.
    artifact_dirs : list[str] or None
        Additional directories whose contents are logged as artifacts
        (e.g. solution plots, field plots, saved models).
    """
    import mlflow

    with mlflow.start_run():
        # -- Parameters -------------------------------------------------- #
        flat_params = _flatten_params(params)
        # MLflow limits param values to 500 chars
        for k, v in flat_params.items():
            mlflow.log_param(k, str(v)[:500])

        mlflow.log_param("run_filename", filename)

        # -- Metrics: evaluate test losses ------------------------------- #
        for loss in loss_handler.test_losses.get("main", []):
            try:
                val = float(loss.loss_base_call().numpy())
                mlflow.log_metric(f"test/{loss.name}", val)
            except Exception as e:
                print(f"[mlflow] skipping metric test/{loss.name}: {e}")

        # -- Artifacts: history JSON and plot files ---------------------- #
        for phase, paths in train_handler.filenames.items():
            for kind, path in paths.items():
                if path and os.path.isfile(path):
                    mlflow.log_artifact(path, artifact_path=f"{phase}/{kind}")

        # -- Artifacts: post-processing outputs -------------------------- #
        for d in (artifact_dirs or []):
            if os.path.isdir(d):
                mlflow.log_artifacts(d, artifact_path=os.path.basename(d))