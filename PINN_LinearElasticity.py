#!/usr/bin/env python3

# %%
import argparse
import os
from pinn_source.run import setup_trial

#############################################################################
# Console arguments
#############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PINN',
        description='',
        epilog='')
    parser.add_argument('cuda', type=str)

    args = parser.parse_args()

    #############################################################################
    # Program settings
    #############################################################################
    parallel = False
    exception_handling = False
    num_processes = 1

    setup_file = 'pinn_source/config.json'

    if args.cuda != "-":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    #############################################################################
    # Directories and file structure
    #############################################################################

    base_dir = 'LinearElasticity_paper'

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
        'seed', 'SNR', 'numPDE', 'numBCN', 'numData', 'model/pressure', 'inverse_params/mu/net/layers', 'net/layers' 
    ]

    #############################################################################
    # Multi processing
    #############################################################################
    from pinn_source.solver import run

    setup_trial(run, setup_file, config, seeds, keylist, parallel=parallel, num_processes=num_processes,
                exception_handling=exception_handling)