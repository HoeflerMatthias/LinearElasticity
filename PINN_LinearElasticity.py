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
        description='Run PINNs inverse problem')
    parser.add_argument('cuda', type=str,
                        help='GPU ID(s): "0", "0,1,2" for multi-GPU, or "-" for CPU')

    args = parser.parse_args()

    #############################################################################
    # Program settings
    #############################################################################

    setup_file = 'pinn_source/config.json'

    # Parse GPU IDs
    if args.cuda == "-":
        gpus = None
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        gpus = [int(g) for g in args.cuda.split(',')]
        if len(gpus) == 1:
            # Single GPU: set env var before importing TF
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
            gpus = None

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

    setup_trial(run, setup_file, config, seeds, keylist, gpus=gpus)