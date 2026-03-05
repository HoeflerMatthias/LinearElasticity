import pinn_source.pinn_lib as ns
from pinn_source.pinn_lib import physics
import tensorflow as tf

#############################################################################
# Constitutive model
#############################################################################

def Piola(tape, x, model, dim, mu_model, mu_func, lam):
    mu = mu_model(x[:, :dim])
    mu = mu_func(mu)
    mu = tf.expand_dims(mu, -1)

    lam = tf.constant([[lam]], dtype=ns.config.get_dtype())
    d = model(x)

    P = physics.linear_elasticity_stress(tape, d, x, mu, lam, dim)

    return P

def PDE(x, model, dim, mu_model, mu_func, lam, body_force):
    force = tf.convert_to_tensor(body_force, dtype=ns.config.get_dtype())

    n_pts = tf.shape(x)[0]
    force = tf.repeat([force], n_pts, axis=0)

    with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)

        P = Piola(tape, x, model, dim, mu_model, mu_func, lam)

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
