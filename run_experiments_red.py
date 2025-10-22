from pathlib import Path
import itertools
import csv
import time
import json
import os

from utils import run_grid
from reduced import invscar

def fmt(p):
    return (f"reg{p['J_regu']}_lam{p['lam_reg']}_"
            f"Ninv{p['Nx_inv']}x{p['Ny_inv']}x{p['Nz_inv']}_"
            f"weight{p['lam_dat']}x{p['lam_pde']}x{p['lam_bcn']}_"
            f"noise{p['noise_level']}"
    )

if __name__ == '__main__':
    base = dict(
        # geometry (truth vs inversion)
        #Nx_true=80, Ny_true=80, Nz_true=40,
        Nx_inv=10,  Ny_inv=10,  Nz_inv=5,
        # physics
        lambda_=650.0, mu=8.0, p_load=10.0,
        # objective
        lam_pde=1e1, lam_bcn=1e1, lam_dat=1e1, lam_reg=1e0,
        noise_level=1e-2,
        data_csv='/app/linear_symcube_p10.csv',
    )

    grid = dict(
        J_regu=['H1', 'TV'], #'L2', 'H1',
        lam_pde=[1e2, 1e3],
        lam_bcn=[1e0],
        lam_dat=[1e3],
        lam_reg=[1e-3],
    )

    run_grid(invscar, out_root='/app/runs_red', base_params=base, grid=grid, run_name_fmt=fmt)