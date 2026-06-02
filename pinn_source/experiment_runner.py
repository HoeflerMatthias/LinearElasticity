"""ExperimentRunner — wraps a single run with MLflow lifecycle and duplicate detection."""

import copy
import os
from typing import Callable, Dict, List, Optional, Any

from pinn_source.utils.file_utils import compute_config_hash
from pinn_source.utils.mlflow_utils import (
    set_mlflow_tracking_uri,
    get_or_create_experiment_id,
    config_completed_in_mlflow,
    log_run_metadata,
    log_params_flat,
    log_metrics_safely,
)


class ExperimentRunner:
    def __init__(
        self,
        params: dict,
        algorithm_fn: Callable[[dict, int], dict],
        experiment_name: str = "lin_elast:pinns",
        hash_exclude_keys: Optional[List[str]] = None,
        post_run_fn: Optional[Callable[[dict], None]] = None,
    ):
        self.params = params
        self.algorithm_fn = algorithm_fn
        self.post_run_fn = post_run_fn
        self.seed = None
        self.results = []
        self.hash_exclude_keys = hash_exclude_keys or ["program", "seed"]

        self.experiment_name = experiment_name

        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            self.mlflow_uri = set_mlflow_tracking_uri(mlflow_uri)
        else:
            import mlflow
            self.mlflow_uri = mlflow.get_tracking_uri()

        self.experiment_id = get_or_create_experiment_id(self.experiment_name)

    @property
    def param_id(self) -> str:
        return compute_config_hash(
            self.params, seed=None, exclude_keys=self.hash_exclude_keys
        )

    def run(self, seed: int) -> None:
        import mlflow

        self.seed = seed

        if config_completed_in_mlflow(self.param_id, self.seed, self.experiment_id):
            print(f"[SKIP] Run already exists for config {self.param_id[:8]}_seed{self.seed}")
            return

        run_name = f"{self.param_id[:8]}_seed{self.seed}"

        with mlflow.start_run(run_name=run_name) as run:
            try:
                log_run_metadata(
                    param_id=self.param_id,
                    seed=self.seed,
                    params=self.params,
                )
                log_params_flat({k: v for k, v in self.params.items()
                                 if k != "seed"})
                mlflow.log_param("seed", self.seed)

                params = copy.deepcopy(self.params)
                params["seed"] = self.seed

                result = self.algorithm_fn(params, self.seed)

                if self.post_run_fn and result:
                    self.post_run_fn(result)

                if isinstance(result, dict):
                    metrics = result.get("metrics", {})
                    log_metrics_safely(metrics)
                    self.results.append(metrics)

                mlflow.set_tag("status", "complete")

            except Exception as e:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error_message", str(e)[:500])
                raise

    def __repr__(self):
        return f"<ExperimentRunner param_id={self.param_id} seed={self.seed}>"
