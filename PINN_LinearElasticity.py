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
    # Parameter
    #############################################################################

    config = {
        "numPDE": [8000],
        "numBCN": [1000],
        "numData": [1000],

        "inverse_params/mu/net/wT": [0.0],

        "wPDE": [1e4*1e1],
        "wFit": [1e6],
        "wT": [0.0],
        "wBCN": [1e4],

        "program/base_dir": [base_dir]
    }

    seeds = [3]

    keylist = [
        'seed', 'SNR', 'numPDE', 'numBCN', 'numData',  'inverse_params/mu/net/layers', 'net/layers'
    ]

    #############################################################################
    # Run
    #############################################################################
    from pinn_source.solver import run

    setup_trial(run, setup_file, config, seeds, keylist)