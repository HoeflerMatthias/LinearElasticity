import tensorflow as tf
import nisaba as ns
import matplotlib.pyplot as plt

from Codebase.PINNLossHandler import PINNLossHandler
from Codebase.PINNTrainHandler import PINNTrainHandler

class PINNCubeTrainHandler(PINNTrainHandler):

    def __init__(self, loss_handler: PINNLossHandler, variables: dict):
        super().__init__(loss_handler, variables)

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

    def train_fit(self, learning_rate_adam: float, epochs_adam: int, epochs_bfgs):
        ####################
        # Step 1: only fitting
        ####################

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
        ns.minimize(pb_fit, 'scipy', 'BFGS', num_epochs=epochs_bfgs)

        callbacks[0].finalize(pb_fit, block=False)
        plt.close('all')



    def train_physics(self, learning_rate_adam: float, epochs_adam: int, epochs_bfgs, data: ns.DataCollection = None):
        ####################
        # Step 2: only physics
        ####################

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

        ns.minimize(pb, 'scipy', 'BFGS', num_epochs=epochs_bfgs)

        callbacks[0].finalize(pb, block=False)
        plt.close('all')

    def train_main(self, learning_rate_adam: float, epochs_adam: int, epochs_bfgs, data: ns.DataCollection = None):
        ####################
        # Step 3: fitting + physics
        ####################

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

        # ns.minimize(pb, 'adahessian', ns.optimization.adahessian.AdaHessian(learning_rate=params['lr2']), num_epochs=params['bfgs3'])
        ns.minimize(pb, 'scipy', 'BFGS', num_epochs=epochs_bfgs)#, options = {'gtol': 1e-20})

        callbacks[0].finalize(pb, block=False)
        plt.close('all')
