import tensorflow as tf
import pinn_source.pinn_lib as ns
import matplotlib.pyplot as plt
import copy

from pinn_source.losses import PINNLossHandler, HLoss


class PINNTrainHandler:

    def __init__(self, loss_handler: PINNLossHandler, variables: dict):

        self.variables = variables
        self.loss_handler = loss_handler
        self.train_preparation_callback = None
        self.callbacks = []
        self.adam_callbacks = []
        self._pending_finalize = []

        self.filenames = {
            'fit': {
                'history': 'history_fit',
                'data': 'data_fit'
            },
            'physics': {
                'history': 'history_fit',
                'data': 'data_fit'
            },
            'main': {
                'history': 'history_fit',
                'data': 'data_fit'
            }
        }

    def set_train_preparation_callback(self, callback):
        self.train_preparation_callback = callback

#############################################################################
# Training
#############################################################################

    def train_fit(self, learning_rate_adam: float, epochs_adam: int, epochs_bfgs, bfgs_backend='scipy'):

        callbacks = [
            ns.utils.HistoryPlotCallback(gui=False,
                                         filename=self.filenames['fit']['history'],
                                         filename_history=self.filenames['fit']['data'], frequency=10)
        ]

        pb_fit = ns.OptimizationProblem(self.variables['fit'], self.loss_handler.train_losses['fit'], self.loss_handler.test_losses['fit'], callbacks=callbacks)

        if self.train_preparation_callback is not None:
            self.train_preparation_callback(pb_fit)

        pb_fit.compile(optimizers=['keras', 'scipy'])

        ns.minimize(pb_fit, 'keras', tf.keras.optimizers.Adam(learning_rate=learning_rate_adam), num_epochs=epochs_adam)
        ns.minimize(pb_fit, bfgs_backend, 'BFGS', num_epochs=epochs_bfgs)

        self._pending_finalize.append((callbacks[0], pb_fit))

    def train_physics(self, learning_rate_adam: float, epochs_adam: int, epochs_bfgs, data: ns.DataCollection = None, bfgs_backend='scipy'):

        callbacks = [
            ns.utils.HistoryPlotCallback(gui=False,
                                         filename=self.filenames['physics']['history'],
                                         filename_history=self.filenames['physics']['data'], frequency=10)
        ]

        pb = ns.OptimizationProblem(self.variables['physics'], self.loss_handler.train_losses['physics'],
                                        self.loss_handler.test_losses['physics'], callbacks=callbacks, data=data)

        if self.train_preparation_callback is not None:
            self.train_preparation_callback(pb)

        pb.compile(optimizers=['keras', 'scipy'])

        ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=learning_rate_adam), num_epochs=epochs_adam)

        if data is not None:
            data.set_batch_size(None)

        ns.minimize(pb, bfgs_backend, 'BFGS', num_epochs=epochs_bfgs)

        self._pending_finalize.append((callbacks[0], pb))

    def train_main(self, learning_rate_adam: float, epochs_adam: int, epochs_bfgs, data: ns.DataCollection = None, bfgs_backend='scipy'):

        callbacks = [
            ns.utils.HistoryPlotCallback(gui=False,
                                         filename=self.filenames['main']['history'],
                                         filename_history=self.filenames['main']['data'], frequency=10)
        ]
        callbacks += self.callbacks

        pb = ns.OptimizationProblem(self.variables['main'], self.loss_handler.train_losses['main'],
                                        self.loss_handler.test_losses['main'], callbacks=callbacks + self.adam_callbacks, data=data)

        if self.train_preparation_callback is not None:
            self.train_preparation_callback(pb)

        pb.compile(optimizers=['keras', 'scipy'])

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_adam,
            decay_steps=10000,
            decay_rate=0.9
        )
        ns.minimize(pb, 'keras', tf.keras.optimizers.Adam(learning_rate=lr_schedule), num_epochs=epochs_adam)

        pb.callbacks = callbacks

        if data is not None:
            data.set_batch_size(None)

        ns.minimize(pb, bfgs_backend, 'BFGS', num_epochs=epochs_bfgs)

        self._pending_finalize.append((callbacks[0], pb))

    def finalize_all(self):
        """Save history JSON and loss-curve plots for all phases."""
        for cb, pb in self._pending_finalize:
            cb.finalize(pb, block=False)
        self._pending_finalize.clear()
        plt.close('all')

#############################################################################
# Utilities
#############################################################################

    @staticmethod
    def convert_to_serializable(mydict):
        for key in mydict:
            if isinstance(mydict[key], dict):
                PINNTrainHandler.convert_to_serializable(mydict[key])
            elif tf.is_tensor(mydict[key]):
                mydict[key] = mydict[key].numpy().tolist()

    @staticmethod
    def add_parameters(pb: ns.OptimizationProblem, params: dict):
        tmp = copy.deepcopy(params)
        PINNTrainHandler.convert_to_serializable(tmp)
        pb.history['parameter'] = tmp

    @staticmethod
    def add_weight_history(pb: ns.OptimizationProblem):
        pb.history['weights'] = dict()
        for loss in pb.losses:
            pb.history['weights'][loss.name] = dict()
            pb.history['weights'][loss.name]['log'] = []

    @staticmethod
    def add_lagrange_history(pb: ns.OptimizationProblem):
        for loss in pb.losses:
            if isinstance(loss, HLoss):
                pb.history['losses'][loss.name]['log_h'] = []

    @staticmethod
    def save_models(models, filenames):
        for model, file in zip(models, filenames):
            model.save(file)

#############################################################################
# Callback utilities
#############################################################################

    @staticmethod
    def lagrange_callback(gamma, lambda_0, eta, hloss_names=[], frequency=200):

        def callback(pb, itr, itr_round):
            if itr % frequency == 0 and itr > 0:
                for loss in pb.losses:
                    if loss.name in hloss_names:
                        lambda_old = loss.lagrange_mul
                        lambda_new = gamma * lambda_old + eta * loss.normalized_values(pb.data.current_batch)

                        delta_lambda = (lambda_new - lambda_old) / (lambda_old + 1e-9)
                        delta_lambda_avg = tf.reduce_mean(tf.abs(delta_lambda))
                        pb.history['losses'][loss.name]['log_h'].append(delta_lambda_avg.numpy())

                        loss.lagrange_mul.assign(lambda_new + lambda_0)

        callback.frequency = frequency
        return callback

    @staticmethod
    def weight_adjustment_callback(alpha=0.5, frequency=10):

        def callback(pb, itr, itr_round):
            if itr % frequency == 0 and itr > 0:
                norms = []
                norm_sum = 0.
                for loss in pb.losses:
                    if isinstance(loss.weight, tf.Variable):
                        with tf.GradientTape(watch_accessed_variables=False) as tape:
                            tape.watch(pb.variables)
                            loss_value = loss.loss_base_call(pb.data.current_batch)
                            grads = tape.gradient(loss_value, pb.variables)
                            gradient_norm = tf.constant(0, dtype=ns.config.get_dtype())

                            for gradient in grads:
                                if gradient is not None:
                                    gradient_norm += tf.reduce_sum(tf.square(gradient))
                            gradient_norm = tf.sqrt(gradient_norm)
                            norms += [(gradient_norm, loss)]
                            norm_sum += gradient_norm

                for (gradient_norm, loss) in norms:
                    if gradient_norm > 0.:
                        loss.weight.assign(alpha * norm_sum / gradient_norm + (1 - alpha) * loss.weight)
                        pb.history['weights'][loss.name]['log'].append(loss.weight.numpy().item())

        callback.frequency = frequency
        return callback

    @staticmethod
    def model_save_callback(models, filenames, frequency=1000):

        def callback(pb, itr, itr_round):
            if itr % frequency == 0 and itr > 0:
                PINNTrainHandler.save_models(models, filenames)

        return callback