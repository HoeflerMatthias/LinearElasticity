import csv
import tempfile
from pathlib import Path

from firedrake import CheckpointFile


def save_solution_checkpoint(u, alpha, u_true=None, alpha_true=None):
    """Save solution to a temp HDF5 file. Returns the file path."""
    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    tmp.close()
    with CheckpointFile(tmp.name, "w") as chk:
        chk.save_mesh(u.function_space().mesh())
        chk.save_function(u, name="u")
        chk.save_function(alpha, name="alpha")
        if u_true is not None:
            chk.save_function(u_true, name="u_true")
        if alpha_true is not None:
            chk.save_function(alpha_true, name="alpha_true")
    return tmp.name


def log_result_to_mlflow(result):
    """Log an InvScarResult to MLflow (one run per call)."""
    import mlflow

    with mlflow.start_run():
        mlflow.log_params(result.params)

        # Log scalar final metrics
        for k, val in result.metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(k, float(val))
            elif isinstance(val, dict):
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, (int, float)):
                        mlflow.log_metric(f"{k}.{sub_k}", float(sub_v))

        # Log history arrays as a CSV artifact
        _log_history_artifact(result.metrics)

        # Log solution checkpoint
        if result.solution_file:
            mlflow.log_artifact(result.solution_file)


def _log_history_artifact(metrics):
    """Write history lists from metrics to a temp CSV and log as MLflow artifact."""
    import mlflow
    import json

    # Scalar histories → CSV
    scalar_keys = [
        k for k in metrics
        if k.endswith("_hist") and isinstance(metrics[k], list)
        and metrics[k] and not isinstance(metrics[k][0], dict)
    ]
    if scalar_keys:
        max_len = max(len(metrics[k]) for k in scalar_keys)
        if max_len:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, prefix="history_"
            )
            writer = csv.writer(tmp)
            writer.writerow(["iter"] + scalar_keys)
            for i in range(max_len):
                row = [i]
                for k in scalar_keys:
                    lst = metrics[k]
                    row.append(lst[i] if i < len(lst) else "")
                writer.writerow(row)
            tmp.close()
            mlflow.log_artifact(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)

    # Dict histories (term_hist, iter_hist) → JSON
    dict_keys = [
        k for k in metrics
        if k.endswith("_hist") and isinstance(metrics[k], list)
        and metrics[k] and isinstance(metrics[k][0], dict)
    ]
    for k in dict_keys:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=f"{k}_"
        )
        json.dump(metrics[k], tmp)
        tmp.close()
        mlflow.log_artifact(tmp.name)
        Path(tmp.name).unlink(missing_ok=True)
