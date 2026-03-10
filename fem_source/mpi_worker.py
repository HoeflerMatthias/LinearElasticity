"""Standalone MPI worker for FEM solvers.

Usage: mpirun -np N python -m fem_source.mpi_worker input.json solver_name output.json
"""
import importlib
import json
import sys


def main():
    input_path, solver_name, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(input_path) as f:
        params = json.load(f)

    seed = params.pop("seed", 0)
    mod = importlib.import_module(f"fem_source.{solver_name}")
    result = mod.invscar(seed=seed, **params)

    # Only rank 0 writes output
    from firedrake import COMM_WORLD
    if COMM_WORLD.rank == 0:
        out = {
            "params": result.params,
            "metrics": result.metrics,
            "solution_file": result.solution_file,
        }
        with open(output_path, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
