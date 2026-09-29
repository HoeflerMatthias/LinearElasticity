# Context and Objective

This repository contains the code for the numerical experiment of the paper
**[On Parameter Identification in Three-Dimensional Elasticity and Discretisation with Physics-Informed Neural Networks](https://doi.org/10.1088/1361-6420/ae7f46)**.

Currently, there are three methods implemented:
1. PINN-based reconstruction of the all-at-once optimization problem
2. FEM-based reconstruction of a reduced optimization problem
3. FEM-based reconstruction of the all-at-once optimization problem (LSFEM and KKT)

Each method comes with a run script and a configuration for hyperparameter sweeping via Ray Tune.
The forward problem is generated in `forward.py`.

### Some notes before running the reconstruction
- The current setup logs results to an MLflow server. Make sure to have a running instance. Otherwise, the logging needs to be changed individually.
- The environments provided are based on Docker/Docker Compose. Make sure that your MLflow server is reachable within your docker container by setting the `extra_hosts` parameter in the docker-compose.yml
- The forward simulation from `forward.py` needs to be run prior to the reconstruction.

## 1. PINN-based reconstruction of the all-at-once optimization problem
This is the main method considered in the paper.

The solver code is in `pinn_source/` with configuration in `pinn_source/config.yaml`.
Hyperparameter sweeps are launched via `run_pinns.py`.

`Dockerfile.pinns` together with `requirements.pinns.txt` provides a suitable environment (TensorFlow GPU).

## 2. FEM-based reconstruction of a reduced optimization problem
The method is originally described in [1].

The solver code is in `fem_source/reduced.py`.
Hyperparameter sweeps are launched via `run_experiments.py reduced`.

`Dockerfile` together with `requirements.txt` provides a suitable environment (Firedrake).

## 3. FEM-based reconstruction of the all-at-once optimization problem
Two all-at-once FEM formulations are available:
- **LSFEM** (least-squares FEM): `fem_source/lsfem.py`
- **KKT** (Newton on the KKT system): `fem_source/kkt.py`

Hyperparameter sweeps are launched via `run_experiments.py lsfem` or `run_experiments.py kkt`.

`Dockerfile` together with `requirements.txt` provides a suitable environment (Firedrake).

### MPI parallelism for FEM solvers

The FEM solvers support MPI for multi-core parallelism. Firedrake distributes the mesh and linear algebra across ranks automatically:

```bash
# Run with 4 MPI ranks
docker compose run --rm firedrake mpiexec -n 4 python3 run_experiments.py

# Single-process (default, no MPI needed)
docker compose run firedrake python run_experiments.py
```

## Credits

This project builds on the following repositories:

- **[pezzus/invscar](https://github.com/pezzus/invscar)** by Pezzuto et al. — basis for the FEM inverse solvers (methods 2 and 3).
- **[nisaba](https://github.com/FrancescoRegazzoni/LDNets)** by Regazzoni et al. - basis for the PINNs optimisation framework

## References

[1] Pozzi, G., Ambrosi, D., & Pezzuto, S. (2024). Reconstruction of the local contractility of the cardiac muscle from deficient apparent kinematics. Journal of the Mechanics and Physics of Solids, 192, 105793.

[2] Regazzoni, F., Pagani, S., Salvador, M., Dede’, L., & Quarteroni, A. (2024). Learning the intrinsic dynamics of spatio-temporal processes through latent dynamics networks. Nature Communications, 15(1), 1834.

# Quick Setup
These steps are required to make the code run as is.

## Setup of MLflow
The repository uses MLflow to store results and organise runs. The `docker-compose.yml` comes with a service for starting a local MLflow server:
```
docker compose up mlflow-server-local
```

## Generating the Simulation Data
The `forward.py` regenerates the FEM simulation data which is then also used for the reconstruction. Use the `firedrake` service to regenerate, e.g.
```
docker compose run firedrake python3 forward.py split
```
The output is stored in `data`.