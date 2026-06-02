# Context and Objective

This repository contains the code for the numerical experiment of the paper
"On Parameter Identification in Three-Dimensional Elasticity and Discretisation with Physics-Informed Neural Networks".

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

## References

[1] G. Pozzi, D. Ambrosi, and S. Pezzuto, ‘Reconstruction of the local contractility of the cardiac muscle from deficient apparent kinematics’, Apr. 17, 2024, arXiv: arXiv:2404.11137. Accessed: May 28, 2024. [Online]. Available: http://arxiv.org/abs/2404.11137
