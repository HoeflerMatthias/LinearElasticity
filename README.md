# Context and Objective

This repository contains the code for the numerical experiment of the paper
"".

Currently, there are four methods implemented:
1. PINN-based reconstruction of the all-at-once optimization problem
2. Finite Element Method (FEM)-based reconstruction of a reduced optimization problem
3. FEM-based reconstruction of the all-at-once optimization problem
4. Finite Difference Method (FDM)-based reconstruction of the all-at-once optimization problem

Each method comes with a run scipt and a configuration script for hyperparameter sweeping. 
The forward problem is generated in `forward.py`.

## 1. PINN-based reconstruction of the all-at-once optimization problem
This is the main method considered in the paper.

## 2. FEM-based reconstruction of a reduced optimization problem
The method is originally described in [1]
and code is based on the GitHub repository https://github.com/pezzus/invscar by Pezzuto et al.

The relevant code files are `reduced.py` and `run_experiments_red.py`.

A dockerfile `Dockerfile` together with a requirements file `requirements.txt` is provided to set up a suitable environment.

## 3. FEM-based reconstruction of the all-at-once optimization problem
This is the same as method 2, but the optimization problem is solved in an all-at-once fashion.

The relevant code files are `kkt.py` and `run_experiments_kkt.py`.

A dockerfile `Dockerfile` together with a requirements file `requirements.txt` is provided to set up a suitable environment.

## 4. FDM-based reconstruction of the all-at-once optimization problem
This method is originally described in [2, 3]
and code is based on the GitHub repositories https://github.com/cselab/odil and https://github.com/m1balcerak/PhysRegTumor. 
The latter one serves as a basis for the finite difference discretization.

The relevant code files are `aao_odil.py` and `run_experiments_odil.py`.

A dockerfile `Dockerfile.odil` together with a requirements file `requirements.odil.txt` is provided to set up a suitable environment.

[1] G. Pozzi, D. Ambrosi, and S. Pezzuto, ‘Reconstruction of the local contractility of the cardiac muscle from deficient apparent kinematics’, Apr. 17, 2024, arXiv: arXiv:2404.11137. Accessed: May 28, 2024. [Online]. Available: http://arxiv.org/abs/2404.11137
[2] P. Karnakov, S. Litvinov, and P. Koumoutsakos, ‘Solving inverse problems in physics by optimizing a discrete loss: Fast and accurate learning without neural networks’, PNAS Nexus, vol. 3, no. 1, p. pgae005, Dec. 2023, doi: 10.1093/pnasnexus/pgae005.
[3] M. Balcerak et al., ‘Physics-Regularized Multi-Modal Image Assimilation for Brain Tumor Localization’, 2024, arXiv. doi: 10.48550/ARXIV.2409.20409.
