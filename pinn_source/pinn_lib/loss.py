import inspect
import tensorflow as tf


def _accepts_arg(func):
    """Check whether *func* accepts a positional argument (beyond self)."""
    try:
        sig = inspect.signature(func)
        params = [p for p in sig.parameters.values()
                  if p.name != 'self']
        if not params:
            return False
        first = params[0]
        return first.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    except (ValueError, TypeError):
        return True  # fallback: assume it takes data


class Loss:
    """Base loss class.

    Parameters
    ----------
    name : str
        Identifier for this loss term.
    loss_func : callable
        Zero-arg (closure) or one-arg (receives batch data dict) function
        returning a scalar loss value.
    weight : float or tf.Tensor or tf.Variable
        Multiplicative weight applied to the loss.
    non_negative : bool
        Display hint — the raw loss is already non-negative.
    display_sqrt : bool
        Display hint — show the square root of the loss value.
    """

    def __init__(self, name, loss_func, weight=1.0, non_negative=False, display_sqrt=False):
        self.name = name
        self._loss_func = loss_func
        self._takes_data = _accepts_arg(loss_func) if loss_func is not None else False
        if isinstance(weight, (tf.Variable, tf.Tensor)):
            self.weight = weight
        else:
            self.weight = tf.constant(weight, dtype=tf.float64)
        self.non_negative = non_negative
        self.display_sqrt = display_sqrt

    def loss_base_call(self, data=None):
        if self._takes_data:
            return self._loss_func(data)
        return self._loss_func()

    def __call__(self, data=None):
        return self.weight * self.loss_base_call(data)


class LossMeanSquares(Loss):
    """Mean-squared-error loss.

    Parameters
    ----------
    name : str
        Identifier for this loss term.
    eval_roots : callable
        Zero-arg or one-arg function returning the "roots" (residuals)
        whose MSE is the loss value.
    weight : float or tf.Tensor or tf.Variable
        Multiplicative weight.
    normalization : float
        Divisor applied to the raw MSE (default 1.0).
    expected_shape : tuple or None
        Expected shape of the roots tensor (informational / for init).
    """

    def __init__(self, name, eval_roots, weight=1.0, normalization=1.0, expected_shape=None):
        super().__init__(name, None, weight)
        self._eval_roots = eval_roots
        self._roots_takes_data = _accepts_arg(eval_roots)
        self.normalization = normalization
        self.expected_shape = expected_shape
        self.dtype = tf.float64

    # -- dispatch helper -------------------------------------------------- #

    def _eval_roots_dispatch(self, data=None):
        if self._roots_takes_data:
            return self._eval_roots(data)
        return self._eval_roots()

    # -- core interface --------------------------------------------------- #

    def loss_base_call(self, data=None):
        roots = self._eval_roots_dispatch(data)
        return tf.reduce_mean(tf.square(roots)) / self.normalization

    def __call__(self, data=None):
        return self.weight * self.loss_base_call(data)

    # -- utilities used by HLoss / callbacks ------------------------------ #

    def normalized_values(self, data=None, normalization=None):
        abs_values = tf.math.abs(self._eval_roots_dispatch(data))
        if normalization is not None:
            max_value = normalization
        else:
            max_value = tf.reduce_max(abs_values)
        return abs_values / max_value

    def roots(self, data=None):
        roots = self._eval_roots_dispatch(data)
        n_squares = tf.cast(tf.reduce_prod(tf.shape(roots)), self.dtype)
        return tf.reshape(roots, (-1,)) / tf.sqrt(n_squares * self.normalization)
