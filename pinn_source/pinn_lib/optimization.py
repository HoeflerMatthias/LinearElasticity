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
    """Converts between a list of ``tf.Variable`` and a flat vector."""

    def __init__(self, variables):
        self.variables = variables
        self.shapes = [v.shape for v in variables]
        self.dtypes = [v.dtype for v in variables]
        self.sizes = [int(np.prod(s)) for s in self.shapes]
        self.total_size = sum(self.sizes)

    # --- NumPy variants (for scipy) --------------------------------------- #

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

    # --- TF-tensor variants (for TFP, no NumPy round-trips) --------------- #

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
        self.history = {'losses': {}, 'transitions': []}
        self.iteration_offset = 0  # cumulative offset across optimizer phases
        for loss in self.train_losses:
            if loss.name not in self.history['losses']:
                self.history['losses'][loss.name] = {'log': [], 'iter': []}
        for loss in self.test_losses:
            key = f"test/{loss.name}"
            if key not in self.history['losses']:
                self.history['losses'][key] = {'log': [], 'iter': []}

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

def _minimize_keras(pb, optimizer, num_epochs, log_frequency=100):
    pb.history['transitions'].append({'iter': pb.iteration_offset, 'method': 'Adam'})
    trainable_vars = pb.variables
    uses_minibatch = (pb.data is not None
                      and any(ds.batch_size is not None for ds in pb.data.datasets))

    # For full-batch: pre-fetch once and trace with constant data
    if pb.data is not None:
        pb.data.advance()

    # Determine how many steps to batch on GPU before returning to Python.
    # Must align with callback frequencies so they fire at the right time.
    callback_freqs = [getattr(cb, 'frequency', log_frequency)
                      for cb in pb.callbacks]
    # GCD of all frequencies gives the largest safe batch size
    from math import gcd
    from functools import reduce
    all_freqs = [log_frequency] + callback_freqs
    steps_per_call = reduce(gcd, all_freqs)

    @tf.function
    def train_n_steps(n):
        """Run n forward+backward+apply steps entirely on GPU."""
        data = pb.data.current_batch if pb.data is not None else None
        for _ in tf.range(n):
            with tf.GradientTape() as tape:
                total_loss = tf.constant(0.0, dtype=tf.float64)
                for loss in pb.train_losses:
                    total_loss = total_loss + loss(data)

            grads = tape.gradient(total_loss, trainable_vars)
            grads_and_vars = [
                (g, v) for g, v in zip(grads, trainable_vars) if g is not None
            ]
            optimizer.apply_gradients(grads_and_vars)
        # Return last-step loss values for logging
        loss_vals = [loss(data) for loss in pb.train_losses]
        return total_loss, loss_vals

    has_callbacks = len(pb.callbacks) > 0

    epoch = 0
    while epoch < num_epochs:
        if uses_minibatch:
            pb.data.advance()

        remaining = num_epochs - epoch
        n = min(steps_per_call, remaining)
        total_loss, loss_vals = train_n_steps(tf.constant(n, dtype=tf.int32))
        epoch += n

        # Log train + test losses to history
        if epoch % log_frequency == 0:
            global_itr = pb.iteration_offset + epoch
            for loss, val in zip(pb.train_losses, loss_vals):
                pb.history['losses'][loss.name]['log'].append(
                    float(val.numpy())
                )
                pb.history['losses'][loss.name]['iter'].append(global_itr)
            for loss in pb.test_losses:
                key = f"test/{loss.name}"
                pb.history['losses'][key]['log'].append(
                    float(loss.loss_base_call().numpy())
                )
                pb.history['losses'][key]['iter'].append(global_itr)
            if pb._verbose:
                print(f"  Adam  {epoch:>6d}/{num_epochs}  loss={total_loss.numpy():.6e}")

        # Callbacks
        if has_callbacks:
            for cb in pb.callbacks:
                cb(pb, epoch, epoch)

    pb.iteration_offset += num_epochs


# --------------------------------------------------------------------------- #
# SciPy L-BFGS-B (CPU line search, GPU forward/backward)
# --------------------------------------------------------------------------- #

def _minimize_scipy(pb, method_name, num_epochs):
    pb.history['transitions'].append({'iter': pb.iteration_offset, 'method': 'L-BFGS'})
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

        # Log train + test losses to history (every 100 steps to reduce overhead)
        if itr[0] % 100 == 0:
            global_itr = pb.iteration_offset + itr[0]
            total = 0.0
            for loss in pb.train_losses:
                val = float(loss(data).numpy())
                pb.history['losses'][loss.name]['log'].append(val)
                pb.history['losses'][loss.name]['iter'].append(global_itr)
                total += val
            for loss in pb.test_losses:
                key = f"test/{loss.name}"
                pb.history['losses'][key]['log'].append(
                    float(loss.loss_base_call().numpy())
                )
                pb.history['losses'][key]['iter'].append(global_itr)
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
    pb.iteration_offset += itr[0]
    if pb._verbose:
        print(f"  BFGS  done  ({result.message.decode() if isinstance(result.message, bytes) else result.message})")


# --------------------------------------------------------------------------- #
# TensorFlow Probability L-BFGS (GPU-native)
# --------------------------------------------------------------------------- #

def _minimize_tfp_lbfgs(pb, num_epochs, log_frequency=500):
    pb.history['transitions'].append({'iter': pb.iteration_offset, 'method': 'L-BFGS'})
    import tensorflow_probability as tfp

    stitcher = VariablesStitcher(pb.variables)

    # Full-batch data (set_batch_size(None) called before BFGS)
    if pb.data is not None:
        pb.data.advance()
    data = pb.data.current_batch if pb.data is not None else None

    @tf.function
    def value_and_gradients(flat_params):
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

    if pb._verbose:
        print(f"  TFP-BFGS  starting  ({num_epochs} max iterations, "
              f"{stitcher.total_size} variables, logging every {log_frequency})")

    # Run in chunks to log intermediate losses.  TFP L-BFGS has no Python
    # callback, so we re-enter periodically.  The Hessian approximation is
    # rebuilt from the last ~50 steps, so the cost of resetting is minimal.
    total_itr = 0
    remaining = num_epochs
    converged = False

    while remaining > 0 and not converged:
        chunk = min(log_frequency, remaining)

        result = tfp.optimizer.lbfgs_minimize(
            value_and_gradients,
            initial_position=stitcher.get_values_tf(),
            max_iterations=chunk,
            tolerance=0.0,
            x_tolerance=0.0,
            f_relative_tolerance=0.0,
        )

        stitcher.set_values_tf(result.position)
        chunk_itr = int(result.num_iterations.numpy())
        total_itr += chunk_itr
        remaining -= chunk_itr
        converged = bool(result.converged.numpy())

        # Log train + test losses to history
        global_itr = pb.iteration_offset + total_itr
        total_loss_val = 0.0
        for loss in pb.train_losses:
            val = float(loss(data).numpy())
            pb.history['losses'][loss.name]['log'].append(val)
            pb.history['losses'][loss.name]['iter'].append(global_itr)
            total_loss_val += val
        for loss in pb.test_losses:
            key = f"test/{loss.name}"
            pb.history['losses'][key]['log'].append(
                float(loss.loss_base_call().numpy())
            )
            pb.history['losses'][key]['iter'].append(global_itr)

        if pb._verbose:
            print(f"  TFP-BFGS  {total_itr:>6d}/{num_epochs}  loss={total_loss_val:.6e}")

        for cb in pb.callbacks:
            cb(pb, total_itr, total_itr)

        if chunk_itr < chunk:
            break

    pb.iteration_offset += total_itr

    if pb._verbose:
        final_loss = float(result.objective_value.numpy())
        print(f"  TFP-BFGS  done  iterations={total_itr}  loss={final_loss:.6e}  converged={converged}")
