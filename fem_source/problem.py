from firedrake import *


def create_box_mesh(Nx, Ny, Nz, Lx=2.0, Ly=2.0, Lz=1.0):
    """Create a tetrahedral box mesh."""
    return BoxMesh(Nx, Ny, Nz, Lx, Ly, Lz, hexahedral=False)


def create_spaces(mesh, u_degree=1):
    """Create function spaces. Returns (V, Q) where V is vector, Q is scalar."""
    V = VectorFunctionSpace(mesh, "P", u_degree)
    Q = FunctionSpace(mesh, "P", 1)
    return V, Q


def symmetry_bcs(V_sub):
    """Create symmetry BCs on facets 1/3/5 for a vector (sub)space."""
    return [
        DirichletBC(V_sub.sub(0), Constant(0.0), 1),
        DirichletBC(V_sub.sub(1), Constant(0.0), 3),
        DirichletBC(V_sub.sub(2), Constant(0.0), 5),
    ]


def strain_energy(u, alpha, lambda_, mu, p_load):
    """Elastic energy + pressure loading on ds(6)."""
    eps = sym(grad(u))
    return ((lambda_ / 2) * tr(eps) ** 2 * dx
            + alpha * mu * inner(eps, eps) * dx
            - dot(p_load * as_vector((0, 0, 1)), u) * ds(6))


def make_forward_solver(u, alpha, bcs, lambda_, mu, p_load):
    """Create forward solver. Returns (solver, W_form, G_form)."""
    W_form = strain_energy(u, alpha, lambda_, mu, p_load)
    G_form = derivative(W_form, u)
    prob = NonlinearVariationalProblem(
        G_form, u, bcs,
        form_compiler_parameters={'quadrature_degree': 2}
    )
    solver = NonlinearVariationalSolver(prob)
    return solver, W_form, G_form


def solve_forward(alpha, bcs, lambda_, mu, p_load, V=None, u=None, name="u_fwd"):
    """One-shot forward solve. Returns the solution Function."""
    if u is None:
        u = Function(V, name=name)
    solver, _, _ = make_forward_solver(u, alpha, bcs, lambda_, mu, p_load)
    solver.solve()
    return u


def regularization_functionals():
    """Return dict of regularization functionals: alpha -> UFL form."""
    R_L2 = lambda alpha: 0.5 * inner(alpha, alpha) * dx
    R_H1 = lambda alpha: 0.5 * inner(grad(alpha), grad(alpha)) * dx
    R_TV = lambda alpha: sqrt(1e-2 + inner(grad(alpha), grad(alpha))) * dx
    return {'L2': R_L2, 'H1': R_H1, 'TV': R_TV}
