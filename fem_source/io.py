import os
import tempfile

import numpy as np
from firedrake import CheckpointFile

from .common import InvScarResult


def gather_alpha_plot_data(alpha):
    """Gather alpha coordinates and values to rank 0 for plotting.

    Returns (coords, vals) on rank 0, (None, None) on other ranks.
    """
    comm = alpha.function_space().mesh().comm
    local_coords = alpha.function_space().mesh().coordinates.dat.data_ro
    local_vals = alpha.dat.data_ro

    all_coords = comm.gather(local_coords, root=0)
    all_vals = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        return np.concatenate(all_coords), np.concatenate(all_vals)
    return None, None


def save_solution_checkpoint(u, alpha):
    """Save computed solution to a temp HDF5 file. Returns the file path."""
    comm = u.function_space().mesh().comm
    if comm.rank == 0:
        tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        tmp.close()
        path = tmp.name
    else:
        path = None
    path = comm.bcast(path, root=0)
    with CheckpointFile(path, "w") as chk:
        chk.save_mesh(u.function_space().mesh())
        chk.save_function(u, name="u")
        chk.save_function(alpha, name="alpha")
    return path


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
        try:
            from .plotting import plot_alpha_slice
            png_path = plot_alpha_slice(result.solution_file)
            mlflow.log_artifact(png_path)
            os.unlink(png_path)
        except Exception as e:
            print(f"[warning] Alpha plot failed: {e}")
        os.unlink(result.solution_file)

def log_fem_artifacts_rank0(result):
    """Like log_fem_artifacts but safe to call on rank 0 only.

    Skips plot_alpha_slice which uses CheckpointFile (MPI-collective).
    """
    import mlflow

    _log_history_metrics(result.metrics)
    if result.solution_file:
        mlflow.log_artifact(result.solution_file)
        os.unlink(result.solution_file)

def log_fem_artifacts_rank0(result):
    """Like log_fem_artifacts but safe to call on rank 0 only.

    Uses pre-extracted alpha_coords/alpha_vals for plotting instead of
    re-reading the checkpoint via CheckpointFile (which is MPI-collective).
    """
    import mlflow

    _log_history_metrics(result.metrics)
    if result.solution_file:
        mlflow.log_artifact(result.solution_file)
        if result.alpha_coords is not None:
            try:
                from .plotting import plot_alpha_slice_from_arrays
                png_path = plot_alpha_slice_from_arrays(
                    result.alpha_coords, result.alpha_vals)
                mlflow.log_artifact(png_path)
                os.unlink(png_path)
            except Exception as e:
                print(f"[warning] Alpha plot failed: {e}")
        os.unlink(result.solution_file)


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


