import nisaba as ns
import tensorflow as tf

from Codebase.PINNCubeLossHandler import PINNCubeLossHandler
from Codebase.PINNLinearElasticityDataSet import PINNLinearElasticityDataSet
from Codebase.HLoss import HLoss

from Codebase.BSplineModel import SplineModel

import Codebase.constitutive as constitutive


class PINNLinearElasticityLossHandler(PINNCubeLossHandler):

    def __init__(self, dim = 3):
        super().__init__()
        self.dim = dim

    def setup_fixation_loss(self, displacement_model, dataset: PINNLinearElasticityDataSet, weight: float):

        x_data,_,num_batched = dataset.get_data('x_fixation', 'data', 'train')

        u_data,_,_ = dataset.get_data('fixation_displacement', 'data', 'train')

        if isinstance(displacement_model, SplineModel):
            loss_func = lambda: displacement_model(x_data, 'x_fixation') - u_data
        else:
            loss_func = lambda: displacement_model(x_data) - u_data

        loss = ns.LossMeanSquares('fixation', loss_func, weight=weight, expected_shape=(num_batched, self.dim))

        self.train_losses['main'] += [loss]


    def setup_boundary_losses(self, model, stress_model, dataset: PINNLinearElasticityDataSet, weights: dict[str, float], rba: list = None):
        
        boundary_models = {
            'nxminus': lambda x, model: constitutive.Dirichlet(x, model, dataset.boundary_data['nxminus'], component = 0),
            'nxplus':  lambda x, model: constitutive.Neumann(x, stress_model, 0, dataset.boundary_data['nxplus']),
            'nyminus': lambda x, model: constitutive.Dirichlet(x, model, dataset.boundary_data['nyminus'], component = 1),
            'nyplus':  lambda x, model: constitutive.Neumann(x, stress_model, 1, dataset.boundary_data['nyplus']),
            'nzminus': lambda x, model: constitutive.Dirichlet(x, model, dataset.boundary_data['nzminus'], component = 2),
            'nzplus':  lambda x, model: constitutive.Neumann(x, stress_model, 2, dataset.boundary_data['nzplus']),
        }

        losses, test_losses = [], []

        def create_loss(key, output_dim = 3):
            identifier = key

            data_train, _, num_batched = dataset.get_data(key, 'bc', 'train')
            data_test, _, _ = dataset.get_data(key, 'bc', 'test')

            if isinstance(model, SplineModel) and False:
                loss_func = lambda data: boundary_models[key](data['x_'+identifier], lambda d: model(d,'x_'+identifier))
                test_loss_func = lambda: boundary_models[key](data_test, lambda d: model(d,'x_'+identifier+'_test'))
            else:
                loss_func = lambda data: boundary_models[key](data['x_'+identifier], model)
                test_loss_func = lambda: boundary_models[key](data_test, model)

            exp_shape = (num_batched,) if output_dim == None else (num_batched, output_dim)
            
            if key in rba:
                loss = HLoss(identifier, loss_func, weight=weights[key], expected_shape=exp_shape)
            else:
                loss = ns.LossMeanSquares(identifier, loss_func, weight=weights[key], expected_shape=exp_shape)
            test_loss = ns.LossMeanSquares(identifier, test_loss_func)
            return loss, test_loss

        if weights['nxminus'] > 0.:
            loss, test_loss = create_loss('nxminus', None)
            losses += [loss]
            test_losses += [test_loss]

        if weights['nxplus'] > 0.:
            loss, test_loss = create_loss('nxplus', self.dim)
            losses += [loss]
            test_losses += [test_loss]

        if weights['nyminus'] > 0.:
            loss, test_loss = create_loss('nyminus', None)
            losses += [loss]
            test_losses += [test_loss]

        if weights['nyplus'] > 0.:
            loss, test_loss = create_loss('nyplus', self.dim)
            losses += [loss]
            test_losses += [test_loss]

        if weights['nzminus'] > 0.:
            loss, test_loss = create_loss('nzminus', None)
            losses += [loss]
            test_losses += [test_loss]

        if weights['nzplus'] > 0.:
            loss, test_loss = create_loss('nzplus', self.dim)
            losses += [loss]
            test_losses += [test_loss]


        self.train_losses['physics'] += losses
        self.test_losses['physics'] += test_losses

        self.train_losses['main'] += losses
        self.test_losses['main'] += test_losses

    def setup_gradient_penalty_loss(self, model, weight: float, identifier: str = 'gradient_penalty', data = None):

        if isinstance(model, SplineModel) and False:
            model_func = lambda x: model(x, identifier)
        else: 
            model_func = lambda x: model(x)
        
        weight_func = lambda x: tf.ones_like(x)
        loss_func = lambda x: self.gradient_penalty(x, model_func, weight_func, type = 'h1')

        if data is not None:
            loss = ns.Loss(identifier, lambda: loss_func(data), weight=weight, non_negative=True, display_sqrt=True)
        else:
            loss = ns.Loss(identifier, lambda data: loss_func(data['x_prior']), weight=weight, non_negative=True, display_sqrt=True)

        self.train_losses['fit'] += [loss]
        self.train_losses['main'] += [loss]

    def second_order_diff_penalty_3d(self, grid: tf.Tensor, weight: float, identifier: str = 'curv_penalty') -> tf.Tensor:
        """
        Computes the sum of squared second-order finite differences
        along each axis of a 3D tensor.
        """
        def penality():
            dx2 = grid[2:, :, :] - 2 * grid[1:-1, :, :] + grid[:-2, :, :]
            dy2 = grid[:, 2:, :] - 2 * grid[:, 1:-1, :] + grid[:, :-2, :]
            dz2 = grid[:, :, 2:] - 2 * grid[:, :, 1:-1] + grid[:, :, :-2]
        
            penalty = tf.reduce_sum(dx2 ** 2) + tf.reduce_sum(dy2 ** 2) + tf.reduce_sum(dz2 ** 2)
            return penalty

        loss = ns.Loss(identifier, lambda: penality(), weight=weight, non_negative=True, display_sqrt=True)
        self.train_losses['fit'] += [loss]
        self.train_losses['main'] += [loss]