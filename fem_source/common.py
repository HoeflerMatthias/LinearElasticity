from firedrake import *
import numpy as np


def _to_ref_mesh(a, a_ref):
    """Interpolate 'a' onto the mesh of 'a_ref' if needed."""
    V_ref = a_ref.function_space()
    mesh_ref = V_ref.mesh()
    if a.function_space().mesh() is not mesh_ref:
        a_fine = Function(V_ref)
        a_fine.interpolate(a)
        return a_fine, mesh_ref
    return a, mesh_ref


def L2_error(a, a_ref):
    """Absolute L2 error: ||a - a_ref||."""
    a_fine, mesh = _to_ref_mesh(a, a_ref)
    err = assemble(dot(a_fine - a_ref, a_fine - a_ref) * dx(domain=mesh))
    return float(np.sqrt(err))


def rel_L2_error(a, a_ref):
    """Relative L2 error: ||a - a_ref|| / ||a_ref||."""
    a_fine, mesh = _to_ref_mesh(a, a_ref)
    err = assemble(dot(a_fine - a_ref, a_fine - a_ref) * dx(domain=mesh))
    norm = assemble(dot(a_ref, a_ref) * dx(domain=mesh))
    return float(np.sqrt(err / (norm + 1e-30)))


def pointwise_rel_L2_error(a, a_ref, eps=1e-12):
    """Pointwise relative L2 error: sqrt(int |a-a_ref|^2 / (|a_ref|^2+eps) dx)."""
    a_fine, mesh = _to_ref_mesh(a, a_ref)
    err = assemble(dot(a_fine - a_ref, a_fine - a_ref) / (dot(a_ref, a_ref) + eps) * dx(domain=mesh))
    return float(np.sqrt(err))


def fmt_tag(p):
    """Format a run tag string from parameter dict."""
    mesh = f"Ninv{p['Nx_inv']}x{p['Ny_inv']}x{p['Nz_inv']}"
    noise = f"noise{p['noise_level']}"
    if 'J_regu' in p:
        reg = f"reg{p['J_regu']}_lam{float(p['lam_reg'])}"
    else:
        parts = []
        for k in ['lam_L2', 'lam_H1', 'lam_TV']:
            if p.get(k, 0.0):
                parts.append(f"{k}={p[k]}")
        reg = "_".join(parts) if parts else "noreg"
    return f"{reg}_{mesh}_{noise}"


class InvScarResult:
    """Container for inverse problem results.

    Parameters
    ----------
    params : dict
        All input parameters used (with defaults filled in).
    metrics : dict
        Standardized metrics dict with keys:
        - J_fid_hist, J_reg_hist, err_u_abs_hist, err_u_rel_hist, err_alpha_pwrel_hist (lists)
        - J_fid_final, J_reg_final (floats)
        - err_u_abs_final, err_u_rel_final, err_alpha_pwrel_final (floats)
        - nit (int or None)
        - solver-specific extras allowed (e.g. term_hist)
    solution_file : str
        Path to a temp HDF5 checkpoint with final u and alpha.
    """
    def __init__(self, *, params, metrics, solution_file):
        self.params = params
        self.metrics = metrics
        self.solution_file = solution_file
