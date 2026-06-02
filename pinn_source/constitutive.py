import pinn_source.pinn_lib as ns
from pinn_source.pinn_lib import physics
import tensorflow as tf

#############################################################################
# Constitutive model
#############################################################################

def Piola(tape, x, model, dim, mu_model, mu_func, lam, r=None, mu_const=None):
    lam_t = tf.constant([[lam]], dtype=ns.config.get_dtype())
    d = model(x)

    if r is None:
        # Isotropic: network predicts spatially-varying mu
        mu = mu_func(mu_model(x[:, :dim]))
        mu = tf.expand_dims(mu, -1)
        P = physics.linear_elasticity_stress(tape, d, x, mu, lam_t, dim)
    else:
        # Anisotropic: mu is constant, network predicts t field
        t = mu_func(mu_model(x[:, :dim]))
        t = tf.expand_dims(t, -1)
        mu_c = tf.constant([[mu_const]], dtype=ns.config.get_dtype())
        P = physics.linear_elasticity_stress(tape, d, x, mu_c, lam_t, dim, r=r, t=t)

    return P

def PDE(x, model, dim, mu_model, mu_func, lam, body_force, r=None, mu_const=None):
    force = tf.convert_to_tensor(body_force, dtype=ns.config.get_dtype())

    n_pts = tf.shape(x)[0]
    force = tf.repeat([force], n_pts, axis=0)

    with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)

        P = Piola(tape, x, model, dim, mu_model, mu_func, lam, r=r, mu_const=mu_const)

        div_P = physics.divergence_tensor(tape, P, x, dim)

    return tf.add(-div_P, -force)

#############################################################################
# Boundary conditions
#############################################################################

def Dirichlet(x, model, vector, component = None):
    d = model(x)

    if callable(vector):
        vec = vector(x)
    else:
        vec = vector

    if component is not None:
        d = d[:, component]
        vec = vec[component]

    return d - vec

def TangentialNeumann(x, stress_tensor, normal_axis, tangential_axes):
    """Zero tangential traction on a symmetry face."""
    with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        P = stress_tensor(tape, x)

    return tf.stack([P[:, t, normal_axis] for t in tangential_axes], axis=-1)

def Neumann(x, stress_tensor, normal_axis, vector, component = None):

    with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        P = stress_tensor(tape, x)

    if callable(vector):
        vec = vector(x)
    else:
        vec = vector

    if component is not None:
        P = P[:, component, normal_axis]
    else:
        P = P[:, :, normal_axis]

    return P - vec
