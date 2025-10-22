from pathlib import Path
import itertools
import csv
import time
import json
import os

from utils import run_grid
from reduced import invscar

if __name__ == '__main__':
    base = dict(
        # geometry (truth vs inversion)
        #Nx_true=80, Ny_true=80, Nz_true=40,
        Nx_inv=10,  Ny_inv=10,  Nz_inv=5,
        data_csv='/app/linear_symcube_p10.h5',
    )

    grid = dict(
        J_regu=['L2', 'H1', 'TV']
        lam_reg=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        noise_level=[1e-2, 5e-2, 1e-1],
    )

    run_grid(invscar, out_root='/app/runs_red', base_params=base, grid=grid)