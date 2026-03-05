"""Continuum-mechanics helpers for linear elasticity PINNs.

Replaces ``nisaba.experimental.physics.tens_style.*``.
"""

import tensorflow as tf


def linear_elasticity_stress(tape, d, x, mu, lam, dim):
    """First Piola-Kirchhoff stress for linear elasticity.

    .. math::
        P = 2\\mu\\,\\varepsilon + \\lambda\\,\\mathrm{tr}(\\varepsilon)\\,I

    where :math:`\\varepsilon = \\tfrac12(\\nabla u + \\nabla u^T)`.

    Parameters
    ----------
    tape : tf.GradientTape (persistent)
        Must already watch *x* and have *d = model(x)* recorded.
    d : tf.Tensor, shape ``(n, dim)``
        Displacement field evaluated at *x*.
    x : tf.Tensor, shape ``(n, dim)``
        Spatial coordinates (watched by *tape*).
    mu, lam : tf.Tensor
        Lamé parameters — may be per-point ``(n, 1, 1)`` or scalar.
    dim : int
        Spatial dimension (typically 3).

    Returns
    -------
    tf.Tensor, shape ``(n, dim, dim)``
    """
    # Displacement gradient: grad_d[i][:,j] = dd_i/dx_j
    grad_d = [tape.gradient(d[:, i], x)[:, :dim] for i in range(dim)]

    # Strain tensor epsilon_ij = 0.5 * (dd_i/dx_j + dd_j/dx_i)
    # Trace
    tr_eps = sum(grad_d[i][:, i] for i in range(dim))

    mu_flat = tf.reshape(mu, [-1])    # (n,) or (1,)
    lam_flat = tf.reshape(lam, [-1])  # (1,) or scalar

    rows = []
    for i in range(dim):
        cols = []
        for j in range(dim):
            eps_ij = 0.5 * (grad_d[i][:, j] + grad_d[j][:, i])
            delta_ij = 1.0 if i == j else 0.0
            P_ij = 2.0 * mu_flat * eps_ij + lam_flat * tr_eps * delta_ij
            cols.append(P_ij)
        rows.append(tf.stack(cols, axis=-1))

    return tf.stack(rows, axis=-2)  # (n, dim, dim)


def divergence_tensor(tape, P, x, dim):
    """Divergence of a second-order tensor field.

    .. math::
        (\\operatorname{div} P)_i = \\sum_j \\frac{\\partial P_{ij}}{\\partial x_j}

    Parameters
    ----------
    tape : tf.GradientTape (persistent)
    P : tf.Tensor, shape ``(n, dim, dim)``
    x : tf.Tensor, shape ``(n, dim)``
    dim : int

    Returns
    -------
    tf.Tensor, shape ``(n, dim)``
    """
    div_components = []
    for i in range(dim):
        div_i = None
        for j in range(dim):
            dPij_dx = tape.gradient(P[:, i, j], x)[:, j]
            div_i = dPij_dx if div_i is None else div_i + dPij_dx
        div_components.append(div_i)

    return tf.stack(div_components, axis=-1)
