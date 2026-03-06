#!/usr/bin/env python3

# %%
from pinn_source.run import setup_trial

if __name__ == '__main__':

    setup_file = 'pinn_source/config.json'

    #############################################################################
    # Directories and file structure
    #############################################################################

    base_dir = 'LinearElasticity_inclusion'

    #############################################################################
    # Parameter sweep
    #############################################################################

    config = {
        "numData": [500, 1000, 2000],

        "net/layers": [[32, 32, 32], [64, 64, 64]],
        "inverse_params/mu/net/layers": [[16, 16, 16], [32, 32, 32]],

        "wPDE": [1e3, 1e5],
        "wFit": [1e5, 1e6],
        "wBCN": [1e3, 1e4],

        "program/base_dir": [base_dir]
    }

    seeds = [3]

    keylist = [
        'seed', 'SNR', 'numData',
        'inverse_params/mu/net/layers', 'net/layers',
        'wPDE', 'wFit', 'wBCN',
    ]

    #############################################################################
    # Run
    #############################################################################
    from pinn_source.solver import run

    setup_trial(run, setup_file, config, seeds, keylist)