"""Optimization machinery for PINN training.

Based on the MIT-licensed LDNets ``OptimizationProblem`` / ``VariablesStitcher``:

    F. Regazzoni et al., "Learning the intrinsic dynamics of spatio-temporal
    processes through Latent Dynamics Networks", Nature Communications (2024)
    15, 1834.  https://github.com/FrancescoRegazzoni/LDNets
"""

import tensorflow as tf
import numpy as np
import scipy.optimize


def _ensure_tf_variable(v):
    """Convert a Keras 3 Variable to its backing ``tf.Variable``.

    Keras 3 introduces its own ``keras.Variable`` type which is not accepted
    by ``tf.GradientTape.watch()``.  Accessing ``.value`` returns the
    underlying ``tf.Variable``.
    """
    if isinstance(v, tf.Variable):
        return v
    if hasattr(v, 'value'):          # keras.Variable
        return v.value
    return v


# --------------------------------------------------------------------------- #
# Variable stitcher (flat <-> structured)
# --------------------------------------------------------------------------- #

class VariablesStitcher:
    """Converts between a list of ``tf.Variable`` and a flat NumPy vector."""

    def __init__(self, variables):
        self.variables = variables
        self.shapes = [v.shape for v in variables]
        self.dtypes = [v.dtype for v in variables]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.total_size = sum(self.sizes)

    def get_values(self):
        return np.concatenate([v.numpy().flatten() for v in self.variables])

    def set_values(self, flat_values):
        offset = 0
        for v, shape, size in zip(self.variables, self.shapes, self.sizes):
            v.assign(tf.reshape(
                tf.cast(flat_values[offset:offset + size], v.dtype), shape
            ))
            offset += size

    def flatten_gradients(self, grads):
        parts = []
        for g, size in zip(grads, self.sizes):
            if g is None:
                parts.append(np.zeros(size, dtype=np.float64))
            else:
                parts.append(g.numpy().astype(np.float64).flatten())
        return np.concatenate(parts)

    # --- TF-tensor variants (no NumPy round-trips) ----------------------- #

    def get_values_tf(self):
        """Return all variables as a single flat ``tf.Tensor`` (float64)."""
        return tf.concat(
            [tf.cast(tf.reshape(v, [-1]), tf.float64) for v in self.variables],
            axis=0,
        )

    def set_values_tf(self, flat_tensor):
        """Assign from a flat ``tf.Tensor`` back into the variables."""
        offset = 0
        for v, shape, size in zip(self.variables, self.shapes, self.sizes):
            v.assign(tf.reshape(
                tf.cast(flat_tensor[offset:offset + size], v.dtype), shape
            ))
            offset += size

    def flatten_gradients_tf(self, grads):
        """Flatten gradients into a single ``tf.Tensor`` (float64)."""
        parts = []
        for g, size in zip(grads, self.sizes):
            if g is None:
                parts.append(tf.zeros([size], dtype=tf.float64))
            else:
                parts.append(tf.cast(tf.reshape(g, [-1]), tf.float64))
        return tf.concat(parts, axis=0)


# --------------------------------------------------------------------------- #
# Optimization problem
# --------------------------------------------------------------------------- #

class OptimizationProblem:
    """Container holding variables, losses, data, callbacks, and history.

    Parameters
    ----------
    variables : list[tf.Variable]
        Trainable variables for this optimisation phase.
    train_losses : list[Loss]
        Loss terms evaluated (and differentiated) during training.
    test_losses : list[Loss]
        Loss terms evaluated for monitoring only.
    callbacks : list[callable]
        ``callback(pb, itr, itr_round)`` called every training step.
    data : DataCollection or None
        Batched collocation / boundary data.
    """

    def __init__(self, variables, train_losses, test_losses=None,
                 callbacks=None, data=None):
        self.variables = [_ensure_tf_variable(v) for v in variables]
        self.losses = train_losses          # nisaba convention
        self.train_losses = train_losses
        self.test_losses = test_losses or []
        self.callbacks = callbacks or []
        self.data = data

        # ---- history ---------------------------------------------------- #
        self.history = {'losses': {}}
        for loss in self.train_losses:
            if loss.name not in self.history['losses']:
                self.history['losses'][loss.name] = {'log': []}
        for loss in self.test_losses:
            if loss.name not in self.history['losses']:
                self.history['losses'][loss.name] = {'log': []}

    def compile(self, optimizers=None):
        """Placeholder for compatibility — no tf.function tracing."""
        self._optimizers = optimizers or []

    # -- internal helpers ------------------------------------------------- #

    def _compute_total_loss(self, data):
        total = tf.constant(0.0, dtype=tf.float64)
        for loss in self.train_losses:
            total = total + loss(data)
        return total


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def minimize(pb, method, optimizer_or_name, num_epochs=1000, verbose=True):
    """Run *num_epochs* optimisation steps.

    Parameters
    ----------
    pb : OptimizationProblem
    method : ``'keras'`` | ``'scipy'`` | ``'tfp'``
    optimizer_or_name : tf.keras.optimizers.Optimizer or ``'BFGS'``
    num_epochs : int
    verbose : bool
        Print progress every 100 steps.
    """
    pb._verbose = verbose
    if method == 'keras':
        _minimize_keras(pb, optimizer_or_name, num_epochs)
    elif method == 'scipy':
        _minimize_scipy(pb, optimizer_or_name, num_epochs)
    elif method == 'tfp':
        _minimize_tfp_lbfgs(pb, num_epochs)
    else:
        raise ValueError(f"Unknown method '{method}'")


# --------------------------------------------------------------------------- #
# Keras (Adam, etc.)
# --------------------------------------------------------------------------- #

def _minimize_keras(pb, optimizer, num_epochs):
    trainable_vars = pb.variables
    uses_minibatch = (pb.data is not None
                      and any(ds.batch_size is not None for ds in pb.data.datasets))

    # For full-batch: pre-fetch once and trace with constant data
    if pb.data is not None:
        pb.data.advance()

    @tf.function
    def train_step():
        data = pb.data.current_batch if pb.data is not None else None
        with tf.GradientTape() as tape:
            total_loss = tf.constant(0.0, dtype=tf.float64)
            loss_vals = []
            for loss in pb.train_losses:
                val = loss(data)
                loss_vals.append(val)
                total_loss = total_loss + val

        grads = tape.gradient(total_loss, trainable_vars)
        grads_and_vars = [
            (g, v) for g, v in zip(grads, trainable_vars) if g is not None
        ]
        optimizer.apply_gradients(grads_and_vars)
        return total_loss, loss_vals

    has_callbacks = len(pb.callbacks) > 0

    for epoch in range(num_epochs):
        if uses_minibatch:
            pb.data.advance()

        total_loss, loss_vals = train_step()

        # Log train losses (every 100 steps to avoid GPU sync stalls)
        if epoch % 100 == 0:
            for loss, val in zip(pb.train_losses, loss_vals):
                pb.history['losses'][loss.name]['log'].append(
                    float(val.numpy())
                )
            if pb._verbose:
                print(f"  Adam  {epoch:>6d}/{num_epochs}  loss={total_loss.numpy():.6e}")

        # Callbacks
        if has_callbacks:
            for cb in pb.callbacks:
                cb(pb, epoch, epoch)


# --------------------------------------------------------------------------- #
# SciPy L-BFGS-B
# --------------------------------------------------------------------------- #

def _minimize_scipy(pb, method_name, num_epochs):
    stitcher = VariablesStitcher(pb.variables)

    # Full-batch data (set_batch_size(None) called before BFGS)
    if pb.data is not None:
        pb.data.advance()
    data = pb.data.current_batch if pb.data is not None else None

    itr = [0]

    def loss_and_grad(flat_params):
        stitcher.set_values(flat_params)

        with tf.GradientTape() as tape:
            total_loss = pb._compute_total_loss(data)

        grads = tape.gradient(total_loss, pb.variables)
        flat_grads = stitcher.flatten_gradients(grads)

        return total_loss.numpy().astype(np.float64), flat_grads

    def scipy_callback(xk):
        stitcher.set_values(xk)
        itr[0] += 1

        # Callbacks
        for cb in pb.callbacks:
            cb(pb, itr[0], itr[0])

        # Log train losses (every 100 steps to reduce overhead)
        if itr[0] % 100 == 0:
            total = 0.0
            for loss in pb.train_losses:
                val = float(loss(data).numpy())
                pb.history['losses'][loss.name]['log'].append(val)
                total += val
            if pb._verbose:
                print(f"  BFGS  {itr[0]:>6d}/{num_epochs}  loss={total:.6e}")

    x0 = stitcher.get_values().astype(np.float64)

    result = scipy.optimize.minimize(
        loss_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        callback=scipy_callback,
        options={
            'maxiter': num_epochs,
            'maxfun': num_epochs * 15,
            'ftol': 0.0,
            'gtol': 0.0,
        },
    )

    stitcher.set_values(result.x)
    if pb._verbose:
        print(f"  BFGS  done  ({result.message.decode() if isinstance(result.message, bytes) else result.message})")


# --------------------------------------------------------------------------- #
# TensorFlow Probability L-BFGS (GPU-native)
# --------------------------------------------------------------------------- #

def _minimize_tfp_lbfgs(pb, num_epochs):
    import tensorflow_probability as tfp

    stitcher = VariablesStitcher(pb.variables)

    # Full-batch data (set_batch_size(None) called before BFGS)
    if pb.data is not None:
        pb.data.advance()
    data = pb.data.current_batch if pb.data is not None else None

    @tf.function
    def value_and_gradients(flat_params):
        # Assign flat tensor into variables
        offset = 0
        for v, shape, size, dtype in zip(
            stitcher.variables, stitcher.shapes,
            stitcher.sizes, stitcher.dtypes
        ):
            v.assign(tf.reshape(
                tf.cast(flat_params[offset:offset + size], dtype), shape
            ))
            offset += size

        with tf.GradientTape() as tape:
            total_loss = tf.constant(0.0, dtype=tf.float64)
            for loss in pb.train_losses:
                total_loss = total_loss + loss(data)

        grads = tape.gradient(total_loss, pb.variables)
        flat_grads = tf.concat([
            tf.cast(tf.reshape(g, [-1]), tf.float64)
            if g is not None else tf.zeros([s], dtype=tf.float64)
            for g, s in zip(grads, stitcher.sizes)
        ], axis=0)

        return total_loss, flat_grads

    x0 = stitcher.get_values_tf()

    if pb._verbose:
        print(f"  TFP-BFGS  starting  ({num_epochs} max iterations, {stitcher.total_size} variables)")

    result = tfp.optimizer.lbfgs_minimize(
        value_and_gradients,
        initial_position=x0,
        max_iterations=num_epochs,
        tolerance=0.0,
        x_tolerance=0.0,
        f_relative_tolerance=0.0,
    )

    # Assign final values back
    stitcher.set_values_tf(result.position)

    # Log final losses
    for loss in pb.train_losses:
        val = loss(data)
        pb.history['losses'][loss.name]['log'].append(float(val.numpy()))

    # Run callbacks at completion
    for cb in pb.callbacks:
        cb(pb, num_epochs, num_epochs)

    converged = bool(result.converged.numpy())
    num_itr = int(result.num_iterations.numpy())
    final_loss = float(result.objective_value.numpy())
    if pb._verbose:
        print(f"  TFP-BFGS  done  iterations={num_itr}  loss={final_loss:.6e}  converged={converged}")
