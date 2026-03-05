import tensorflow as tf
import nisaba as ns
from Codebase.HLoss import HLoss

class PINNLossHandler:

    def __init__(self):

        self.test_losses = {
            'fit': [],
            'physics': [],
            'main': []
        }
        self.train_losses = {
            'fit': [],
            'physics': [],
            'main': []
        }

    def setup_weight_decay_loss(self, model, weight: float, phases: list[str] = ['main'], identifier: str = 'tikhonov'):

        loss_func = lambda: sum([tf.reduce_sum(tf.square(v)) for v in model.trainable_variables])
        loss = ns.Loss(identifier, loss_func, weight=weight, non_negative=True, display_sqrt=True)

        for phase in phases:
            self.train_losses[phase] += [loss]

    def make_losses_adaptive(self, names, phase: str):
        for loss in self.train_losses[phase]:
            if loss.name in names:
                loss.weight = tf.Variable(loss.weight.numpy(), trainable=False, dtype=ns.config.get_dtype())

#############################################################################
# Getter
#############################################################################

    def get_rba_losses(self, phase: str):
        losses = []
        for loss in self.train_losses[phase]:
            if isinstance(loss, HLoss):
                losses += [loss]

        return losses

    def get_loss_by_name(self, name: str, phase: str, learning_type: str = 'train'):
        if learning_type == 'train':
            for loss in self.train_losses[phase]:
                if loss.name == name:
                    return loss
        else:
            for loss in self.test_losses[phase]:
                if loss.name == name:
                    return loss


#############################################################################
# Setter
#############################################################################

    def add_loss(self, loss: ns.Loss, phase, learning_type):

        if learning_type == 'train':
            self.train_losses[phase] += [loss]
        else:
            self.test_losses[phase] += [loss]

#############################################################################
# Loss utilities
#############################################################################

    @staticmethod
    def gradient_penalty(x, model, weight_func = None, type: str = 'h1'):
        with ns.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(x)

            y = model(x)

            grad = tape.gradient(y, x)

            if weight_func is not None:
                weights = weight_func(x)
                grad = tf.math.multiply(weights, grad)

            inner = tf.math.multiply(grad, grad)

            if type == 'tv':
                res = tf.math.sqrt(1e-2 + inner)
            elif type == 'l1':
                res = tf.norm(grad, ord=1.000001, axis=0)
            elif type == 'h1':
                res = inner

        return tf.math.reduce_sum(res)

    @staticmethod
    def dice(y_true, y_pred):
        """ from tensorflow 2.16 """
        """Computes the Dice loss value between `y_true` and `y_pred`.

        Formula:
        ```python
        loss = 1 - (2 * sum(y_true * y_pred)) / (sum(y_true) + sum(y_pred))
        ```

        Args:
            y_true: tensor of true targets.
            y_pred: tensor of predicted targets.

        Returns:
            Dice loss value.
        """
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        inputs = tf.reshape(y_true, [-1])
        targets = tf.reshape(y_pred, [-1])

        intersection = tf.math.reduce_sum(inputs * targets)
        dice = tf.math.divide(
            2.0 * intersection,
            tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred) + tf.keras.backend.epsilon(),
        )

        return 1 - dice