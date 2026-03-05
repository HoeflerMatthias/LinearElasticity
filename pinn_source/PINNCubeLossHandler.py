import tensorflow as tf
import nisaba as ns
import numpy as np

from Codebase.PINNLossHandler import PINNLossHandler
from Codebase.PINNCubeDataSet import PINNCubeDataSet
from Codebase.HLoss import HLoss

from Codebase.BSplineModel import SplineModel

import Codebase.constitutive as constitutive


class PINNCubeLossHandler(PINNLossHandler):

    def __init__(self, dim = 3):
        super().__init__()
        self.dim = dim

    def setup_fit_losses(self, displacement_model, dataset: PINNCubeDataSet, weight: float, weight_fit: float):

        x_test,num,_ = dataset.get_data('x_displacement', 'data', 'test')
        x_data,_,num_batched = dataset.get_data('x_displacement', 'data', 'train')

        u_test,_,_ = dataset.get_data('displacement', 'data', 'test')
        u_data,_,_ = dataset.get_data('displacement', 'data', 'train')

        x_strain_test,num_strain,_ = dataset.get_data('x_strain', 'data', 'test')
        x_strain_data,_,num_batched_strain = dataset.get_data('x_strain', 'data', 'train')

        strain_test,_,_ = dataset.get_data('strain', 'data', 'test')
        strain_data,_,_ = dataset.get_data('strain', 'data', 'train')

        if isinstance(displacement_model, SplineModel):
            loss_func = lambda: displacement_model(x_data, 'x_displacement') - u_data
            test_loss_func = lambda: displacement_model(x_test, 'x_displacement_test') - u_test
        else:
            loss_func = lambda: displacement_model(x_data) - u_data
            test_loss_func = lambda: displacement_model(x_test) - u_test

        strain_loss = ns.LossMeanSquares('strain', lambda: constitutive.GL_strain(x_strain_data, displacement_model, self.dim) - strain_data, weight=weight, expected_shape=(num_batched_strain, self.dim, self.dim))
        strain_loss_fit = ns.LossMeanSquares('strain', lambda: constitutive.GL_strain(x_strain_data, displacement_model, self.dim) - strain_data, weight=weight_fit, expected_shape=(num_batched_strain, self.dim, self.dim))
        strain_loss_test = ns.LossMeanSquares('strain', lambda: constitutive.GL_strain(x_strain_test, displacement_model, self.dim) - strain_test, expected_shape=(num_strain, self.dim, self.dim))

        loss_fit = ns.LossMeanSquares('fit', loss_func, weight=weight_fit, expected_shape=(num_batched, self.dim))
        loss = ns.LossMeanSquares('fit', loss_func, weight=weight, expected_shape=(num_batched, self.dim))
        loss_test = ns.LossMeanSquares('fit', test_loss_func, expected_shape=(num, self.dim))

        self.train_losses['fit'] += [loss_fit]#, strain_loss_fit]
        self.test_losses['fit'] += [loss_test]#, strain_loss_test]

        self.train_losses['main'] += [loss]#, strain_loss]
        self.test_losses['main'] += [loss_test, strain_loss_test]

    def setup_pde_losses(self, pde_model, dataset: PINNCubeDataSet, weight: float, rba: bool = False):

        _,_,num_batched = dataset.get_data(0, 'pde', 'train')
        data_test, _,_ = dataset.get_data(0, 'pde', 'test')

        loss_func = lambda data: pde_model(data['x_PDE_vec'])
        test_loss_func = lambda: pde_model(data_test)

        if rba:
            loss = HLoss('PDE', loss_func, weight=weight, expected_shape=(num_batched, self.dim))
        else:
            loss = ns.LossMeanSquares('PDE', loss_func, weight=weight, expected_shape=(num_batched, self.dim))
        loss_test = ns.LossMeanSquares('PDE', test_loss_func, expected_shape=(None, self.dim))

        self.train_losses['physics'] += [loss]
        self.test_losses['physics'] += [loss_test]

        self.train_losses['main'] += [loss]
        self.test_losses['main'] += [loss_test]

    def setup_boundary_losses(self, displacement_model, stress_model, dataset: PINNCubeDataSet, weights: dict[str, float], spring_face = False, spring_constant = 1.0, rba: bool = False):

        # for some reason, loss creation does not work with for loops
        if spring_face:
            BC_nyminus = lambda x: constitutive.Robin(x, stress_model, displacement_model, spring_constant, 1)
        else:
            BC_nyminus = lambda x: constitutive.Dirichlet(x, displacement_model, dataset.boundary_data['nyminus'])

        boundary_models = {
            'nxminus': lambda x: constitutive.Neumann(x, stress_model, 0, dataset.boundary_data['nxminus']),
            'nxplus': lambda x: constitutive.Neumann(x, stress_model, 0, dataset.boundary_data['nxplus']),
            'nyminus': BC_nyminus,
            'nyplus': lambda x: constitutive.Neumann(x, stress_model, 1, dataset.boundary_data['nyplus']),
            'nzminus': lambda x: constitutive.Neumann(x, stress_model, 2, dataset.boundary_data['nzminus']),
            'nzplus': lambda x: constitutive.Neumann(x, stress_model, 2, dataset.boundary_data['nzplus'])
        }

        losses, test_losses = [], []

        def create_loss(key):
            identifier = key

            _, _, num_batched = dataset.get_data(key, 'bc', 'train')
            data_test, _, _ = dataset.get_data(key, 'bc', 'test')

            loss_func = lambda data: boundary_models[key](data['x_' + identifier])
            test_loss_func = lambda: boundary_models[key](data_test)
            if rba:
                loss = HLoss(identifier, loss_func, weight=weights[key], expected_shape=(num_batched, self.dim))
            else:
                loss = ns.LossMeanSquares(identifier, loss_func, weight=weights[key],
                                          expected_shape=(num_batched, self.dim))
            test_loss = ns.LossMeanSquares(identifier, test_loss_func, expected_shape=(None, self.dim))
            return loss, test_loss

        if weights['nxminus'] > 0.:
            loss, test_loss = create_loss('nxminus')
            losses += [loss]
            test_losses += [test_loss]

        if weights['nxplus'] > 0.:
            loss, test_loss = create_loss('nxplus')
            losses += [loss]
            test_losses += [test_loss]

        if weights['nyminus'] > 0. and False:
            loss, test_loss = create_loss('nyminus')
            losses += [loss]
            test_losses += [test_loss]

        if weights['nyplus'] > 0.:
            loss, test_loss = create_loss('nyplus')
            losses += [loss]
            test_losses += [test_loss]

        if weights['nzminus'] > 0.:
            loss, test_loss = create_loss('nzminus')
            losses += [loss]
            test_losses += [test_loss]

        if weights['nzplus'] > 0.:
            loss, test_loss = create_loss('nzplus')
            losses += [loss]
            test_losses += [test_loss]

        self.train_losses['physics'] += losses
        self.test_losses['physics'] += test_losses

        self.train_losses['main'] += losses
        self.test_losses['main'] += test_losses

    def setup_strain_losses(self, strain_model, dataset: PINNCubeDataSet, weight: float):

        x_test, num, _ = dataset.get_data('x_strain', 'data', 'test')
        x_data, _, num_batched = dataset.get_data('x_strain', 'data', 'train')

        u_test, _, _ = dataset.get_data('strain', 'data', 'test')
        u_data, _, _ = dataset.get_data('strain', 'data', 'train')

        loss_func = lambda: strain_model(x_data) - u_data
        test_loss_func = lambda: strain_model(x_test) - u_test

        loss = ns.LossMeanSquares('strain', loss_func, weight=weight, expected_shape=(num_batched, self.dim))
        loss_test = ns.LossMeanSquares('strain', test_loss_func, expected_shape=(num, self.dim))

        if weight > 0.:
            self.train_losses['main'] += [loss]

        self.test_losses['main'] += [loss_test]

    def setup_stress_losses(self, stress_model, dataset: PINNCubeDataSet, weight: float):

        x_test, num, _ = dataset.get_data('x_stress', 'data', 'test')
        x_data, _, num_batched = dataset.get_data('x_stress', 'data', 'train')

        u_test, _, _ = dataset.get_data('stress', 'data', 'test')
        u_data, _, _ = dataset.get_data('stress', 'data', 'train')

        loss_func = lambda: stress_model(x_data) - u_data
        test_loss_func = lambda: stress_model(x_test) - u_test

        loss = ns.LossMeanSquares('stress', loss_func, weight=weight, expected_shape=(num_batched, self.dim))
        loss_test = ns.LossMeanSquares('stress', test_loss_func, expected_shape=(num, self.dim))

        if weight > 0.:
            self.train_losses['main'] += [loss]

        self.test_losses['main'] += [loss_test]

    def setup_mesh_losses(self, displacement_model, pde_model, dataset: PINNCubeDataSet):

        x_test = dataset.data_handler.x_mesh
        u_test = dataset.data_handler.get_displacement_orig()[0]

        test_loss_func_fit = lambda: displacement_model(x_test) - u_test
        test_loss_func_pde = lambda: pde_model(x_test)

        loss_test_fit = ns.LossMeanSquares('fit_mesh', test_loss_func_fit, expected_shape=(None, self.dim))
        loss_test_pde = ns.LossMeanSquares('PDE_mesh', test_loss_func_pde, expected_shape=(None, self.dim))

        self.test_losses['main'] += [loss_test_fit, loss_test_pde]

    def setup_box_constraints(self, model, dataset: PINNCubeDataSet, weight: float, identifier: str = 'box', lower_bound=None, upper_bound=None):

        _,_, num_batched = dataset.get_data('x_prior', 'reg', 'train')

        if lower_bound is not None and weight > 0.:
            min_loss_func = lambda data: tf.nn.relu(-model(data['x_prior']) + lower_bound)
            min_loss = ns.LossMeanSquares(identifier + '_min', min_loss_func, weight = weight, expected_shape=(num_batched, 1))

            self.train_losses['main'] += [min_loss]

        if upper_bound is not None and weight > 0.:
            max_loss_func = lambda data: tf.nn.relu(model(data['x_prior']) - upper_bound)
            max_loss = ns.LossMeanSquares(identifier + '_max', max_loss_func, weight=weight, expected_shape=(num_batched, 1))

            self.train_losses['main'] += [max_loss]

    def setup_gradient_penalty_loss(self, model, weight: float, identifier: str = 'gradient_penalty'):

        weight_func = lambda x: tf.expand_dims((x[:,0]-5)/10,-1)
        loss_func = lambda x: PINNLossHandler.gradient_penalty(x, model, weight_func, type = 'h1')

        loss = ns.Loss(identifier, lambda data: loss_func(data['x_prior']), weight=weight, non_negative=True, display_sqrt=True)

        self.train_losses['main'] += [loss]

    def setup_prior_loss(self, model, weight: float, identifier: str = 'prior', prior_guess = 0.):

        loss_func = lambda data: model(data['x_prior']) - prior_guess
        loss = ns.LossMeanSquares(identifier, loss_func, weight=weight, expected_shape=(None, 1))

        self.train_losses['main'] += [loss]

    def setup_dice_loss(self, model, dataset: PINNCubeDataSet, threshold, identifier: str = 'dice'):

        b_0 = tf.constant(0, dtype=tf.double)
        b_1 = tf.constant(1, dtype=tf.double)

        x_test = dataset.data_handler.x_mesh

        mask_orig = tf.where(dataset.data_handler.tag_values > threshold, b_1, b_0)

        loss_func = lambda: PINNLossHandler.dice(mask_orig, tf.where(model(x_test) > threshold, b_1, b_0))

        loss = ns.Loss(identifier, loss_func, non_negative=True, display_sqrt=False)

        self.test_losses['physics'] += [loss]
        self.test_losses['main'] += [loss]

    def setup_relative_error_loss(self, model, dataset: PINNCubeDataSet, identifier: str = 'error'):

        x_test = dataset.data_handler.x_mesh
        values = dataset.data_handler.tag_values

        loss_func = lambda: (model(x_test) - values) / values
        loss = ns.LossMeanSquares(identifier, loss_func)

        self.test_losses['main'] += [loss]
        self.test_losses['physics'] += [loss]

    def setup_relative_region_error_losses(self, model, dataset: PINNCubeDataSet, identifier: str = 'error'):

        losses = []
        for tag in dataset.data_handler.tag_dict:
            region_mesh, _, region_values = dataset.data_handler.get_mesh_points_for_tag(tag)

            name = dataset.data_handler.get_name_for_tag(tag)

            losses += [
                ns.LossMeanSquares(name + '_' + identifier,
                                   lambda: (model(region_mesh) - region_values) / region_values)
            ]

        self.test_losses['main'] += losses