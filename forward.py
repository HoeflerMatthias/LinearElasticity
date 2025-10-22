from firedrake import *
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import csv
import os
from pathlib import Path
import json

__all__ = ["invscar"]

Nx_t = 80
Ny_t = 80
Nz_t = 40

Lx = 2.0
Ly = 2.0
Lz = 1.0

mm_true = BoxMesh(Nx_t, Ny_t, Nz_t, Lx, Ly, Lz, hexahedral=False)

dim = mm_true.geometric_dimension()

# Spaces
V_t = VectorFunctionSpace(mm_true, "P", 1)
Q_t = FunctionSpace(mm_true, "P", 1)

# -----------------------------
# Parameters & physics
# -----------------------------
lambda_ = Constant(650.0)
mu = Constant(8.0)
p_load = Constant(-10.0)

bcs_t = [
    DirichletBC(V_t.sub(0), Constant(0.0), 1),
    DirichletBC(V_t.sub(1), Constant(0.0), 3),
    DirichletBC(V_t.sub(2), Constant(0.0), 5),
]

x_t = SpatialCoordinate(mm_true)

alpha_expr_t = conditional(x_t[0] < x_t[1], 1.0, 2.0)

alpha_true = Function(Q_t, name="alpha_true")
alpha_true.interpolate(alpha_expr_t)

u_true = Function(V_t, name="u_true")
eps_t  = sym(grad(u_true))

W_t = (lambda_/2)*tr(eps_t)**2 * dx \
    + alpha_true*mu*inner(eps_t, eps_t)*dx \
    - dot(p_load*Constant((0.0, 0.0, 1.0)), u_true)*ds(6)

G_t = derivative(W_t, u_true)
fwd_prob_t   = NonlinearVariationalProblem(G_t, u_true, bcs_t,
                    form_compiler_parameters={'quadrature_degree': 2})
fwd_solver_t = NonlinearVariationalSolver(fwd_prob_t)
fwd_solver_t.solve()

file = VTKFile("linear_symcube_p10.pvd")
file.write(u_true, alpha_true)

with CheckpointFile("linear_symcube_p10.h5", "w") as chk:
    chk.save_function(alpha_true)
    chk.save_function(u_true)