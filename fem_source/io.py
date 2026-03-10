import json
import os
import subprocess
import sys
import tempfile

from firedrake import CheckpointFile

from .common import InvScarResult


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
        try:
            from .plotting import plot_alpha_slice
            png_path = plot_alpha_slice(result.solution_file)
            mlflow.log_artifact(png_path)
            os.unlink(png_path)
        except Exception as e:
            print(f"[warning] Alpha plot failed: {e}")
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


def run_solver_mpi(solver_name, params, seed, nprocs):
    """Run a FEM solver under mpirun and return an InvScarResult."""
    input_fd, input_path = tempfile.mkstemp(suffix=".json", prefix="fem_in_")
    output_path = input_path.replace(".json", "_out.json")

    try:
        with os.fdopen(input_fd, "w") as f:
            json.dump({**params, "seed": seed}, f)

        proc = subprocess.run(
            ["mpirun", "--oversubscribe", "-np", str(nprocs),
             sys.executable, "-m", "fem_source.mpi_worker",
             input_path, solver_name, output_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"MPI solver failed (exit {proc.returncode}):\n"
                f"stdout: {proc.stdout[-2000:]}\n"
                f"stderr: {proc.stderr[-2000:]}"
            )

        with open(output_path) as f:
            data = json.load(f)

        return InvScarResult(
            params=data["params"],
            metrics=data["metrics"],
            solution_file=data["solution_file"],
        )
    finally:
        for p in (input_path, output_path):
            if os.path.exists(p):
                os.unlink(p)
