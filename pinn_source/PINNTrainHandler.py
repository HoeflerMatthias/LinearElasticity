import tensorflow as tf
import nisaba as ns
import copy

from Codebase.PINNLossHandler import PINNLossHandler
from Codebase.HLoss import HLoss

class PINNTrainHandler:

    def __init__(self, loss_handler: PINNLossHandler, variables: dict):

        self.variables = variables
        self.loss_handler = loss_handler
        self.train_preparation_callback = None
        self.callbacks = []
        self.adam_callbacks = []

    def set_train_preparation_callback(self, callback):
        """
        Set the callback function that is called before the training starts.
        callback: function with signature (pb: ns.OptimizationProblem)
        """
        self.train_preparation_callback = callback

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
                # update HLosses
                for loss in pb.losses:
                    if loss.name in hloss_names:
                        # update Lagrange multiplier
                        lambda_old = loss.lagrange_mul  # tf.gather(loss.lagrange_mul, indices)
                        lambda_new = gamma * lambda_old + eta * loss.normalized_values(pb.data.current_batch)

                        delta_lambda = (lambda_new - lambda_old) / (lambda_old + 1e-9)
                        delta_lambda_avg = tf.reduce_mean(tf.abs(delta_lambda))
                        pb.history['losses'][loss.name]['log_h'].append(delta_lambda_avg.numpy())

                        loss.lagrange_mul.assign(lambda_new + lambda_0)

        return callback

    @staticmethod
    def curriculum_loss_callback(pde_weight, epsilon, frequency=10):

        def callback(pb, itr, itr_round):
            if itr % frequency == 0 and itr > 0:
                loss_values = [0. for _ in range(10)]
                for loss in pb.losses:
                    name = loss.name
                    if len(name) > 3 and name[:3] == 'PDE':
                        number = int(name[4])
                        loss_values[number] = loss.loss_base_call([])

                weights = [tf.constant(1., dtype=ns.config.get_dtype())]
                for i in range(1, 10):
                    weights.append(tf.math.exp(-epsilon * tf.reduce_mean(loss_values[0:i])))

                for loss in pb.losses:
                    name = loss.name
                    if len(name) > 3 and name[:3] == 'PDE':
                        number = int(name[4])
                        loss.weight = tf.expand_dims(pde_weight * weights[number], -1)

                        pb.history['weights'][loss.name]['log'].append(weights[number].numpy().item())

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
                            # print(loss_value)
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

        return callback

    @staticmethod
    def weight_multiplier_callback(loss_names, factor, frequency=10):

        def callback(pb, itr, itr_round):
            if itr % frequency == 0 and itr > 0:
                for loss in pb.losses:
                    if loss.name in loss_names:
                        loss.weight *= factor

        return callback

    @staticmethod
    def model_save_callback(models, filenames, frequency=1000):

        def callback(pb, itr, itr_round):
            if itr % frequency == 0 and itr > 0:
                PINNTrainHandler.save_models(models, filenames)

        return callback