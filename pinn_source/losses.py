import tensorflow as tf
import pinn_source.pinn_lib as ns
import pinn_source.constitutive as constitutive


class HLoss(ns.loss.LossMeanSquares):
    def __init__(self, name, eval_roots, weight = 1.0, normalization = 1.0, expected_shape = None, full_size = None, dataset_key = None):

        if full_size is not None:
            self.lagrange_mul = tf.Variable(tf.zeros(full_size, dtype=ns.config.get_dtype()))
        else:
            self.lagrange_mul = tf.Variable(tf.zeros(expected_shape, dtype=ns.config.get_dtype()))

        self.full_size = full_size
        self.dataset_key = dataset_key

        super().__init__(name, eval_roots, weight, normalization, expected_shape)

    def loss_base_call(self, data):
        self.loss_values = self._eval_roots(data)
        if self.dataset_key is not None:
            data = data[self.dataset_key]

        if self.full_size is not None:
            lagrange_mul = tf.gather(self.lagrange_mul, tf.cast(data[:,-1], dtype=tf.int32))
        else:
            lagrange_mul = self.lagrange_mul

        val = tf.math.multiply(lagrange_mul, self.loss_values)
        loss = tf.reduce_mean(tf.square(val))
        return loss

    def normalized_values(self, data, normalization = None):
        abs_values = tf.math.abs(self._eval_roots(data))
        if normalization is not None:
            max_value = normalization
        else:
            max_value = tf.reduce_max(abs_values)
        return abs_values/max_value

    def roots(self, data):
        roots = self._eval_roots(data)
        n_squares = tf.cast(tf.reduce_prod(tf.shape(roots)), self.dtype)
        return tf.reshape(roots, (-1,)) / tf.sqrt( n_squares * self.normalization )


class PINNLossHandler:

    def __init__(self, dim = 3):

        self.dim = dim

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

#############################################################################
# Loss setup
#############################################################################

    def setup_fit_losses(self, displacement_model, dataset, weight: float, weight_fit: float):

        x_test,num,_ = dataset.get_data('x_displacement', 'data', 'test')
        x_data,_,num_batched = dataset.get_data('x_displacement', 'data', 'train')

        u_test,_,_ = dataset.get_data('displacement', 'data', 'test')
        u_data,_,_ = dataset.get_data('displacement', 'data', 'train')

        loss_func = lambda: displacement_model(x_data) - u_data
        test_loss_func = lambda: displacement_model(x_test) - u_test

        loss_fit = ns.LossMeanSquares('fit', loss_func, weight=weight_fit, expected_shape=(num_batched, self.dim))
        loss = ns.LossMeanSquares('fit', loss_func, weight=weight, expected_shape=(num_batched, self.dim))
        loss_test = ns.LossMeanSquares('fit', test_loss_func, expected_shape=(num, self.dim))

        self.train_losses['fit'] += [loss_fit]
        self.test_losses['fit'] += [loss_test]

        self.train_losses['main'] += [loss]
        self.test_losses['main'] += [loss_test]

    def setup_pde_losses(self, pde_model, dataset, weight: float, rba: bool = False):

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

    def setup_boundary_losses(self, model, stress_model, dataset, weights: dict[str, float], rba: list = None):

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

            loss_func = lambda data: boundary_models[key](data['x_'+identifier], model)
            test_loss_func = lambda: boundary_models[key](data_test, model)

            exp_shape = (num_batched,) if output_dim == None else (num_batched, output_dim)

            if key in rba:
                loss = HLoss(identifier, loss_func, weight=weights[key], expected_shape=exp_shape)
            else:
                loss = ns.LossMeanSquares(identifier, loss_func, weight=weights[key], expected_shape=exp_shape)
            test_loss = ns.LossMeanSquares(identifier, test_loss_func)
            return loss, test_loss

        bc_dims = {
            'nxminus': None, 'nxplus': self.dim,
            'nyminus': None, 'nyplus': self.dim,
            'nzminus': None, 'nzplus': self.dim,
        }
        for key, output_dim in bc_dims.items():
            if weights[key] > 0.:
                loss, test_loss = create_loss(key, output_dim)
                losses.append(loss)
                test_losses.append(test_loss)

        self.train_losses['physics'] += losses
        self.test_losses['physics'] += test_losses

        self.train_losses['main'] += losses
        self.test_losses['main'] += test_losses

    def setup_mesh_losses(self, displacement_model, pde_model, dataset):

        x_test = dataset.data_handler.x_mesh
        u_test = dataset.data_handler.get_displacement_orig()[0]

        test_loss_func_fit = lambda: displacement_model(x_test) - u_test
        test_loss_func_pde = lambda: pde_model(x_test)

        loss_test_fit = ns.LossMeanSquares('fit_mesh', test_loss_func_fit, expected_shape=(None, self.dim))
        loss_test_pde = ns.LossMeanSquares('PDE_mesh', test_loss_func_pde, expected_shape=(None, self.dim))

        self.test_losses['main'] += [loss_test_fit, loss_test_pde]

    def setup_weight_decay_loss(self, model, weight: float, phases: list[str] = ['main'], identifier: str = 'tikhonov'):

        loss_func = lambda: sum([tf.reduce_sum(tf.square(v)) for v in model.trainable_variables])
        loss = ns.Loss(identifier, loss_func, weight=weight, non_negative=True, display_sqrt=True)

        for phase in phases:
            self.train_losses[phase] += [loss]

    def setup_box_constraints(self, model, dataset, weight: float, identifier: str = 'box', lower_bound=None, upper_bound=None):

        _,_, num_batched = dataset.get_data('x_prior', 'reg', 'train')

        if lower_bound is not None and weight > 0.:
            min_loss_func = lambda data: tf.nn.relu(-model(data['x_prior']) + lower_bound)
            min_loss = ns.LossMeanSquares(identifier + '_min', min_loss_func, weight = weight, expected_shape=(num_batched, 1))

            self.train_losses['main'] += [min_loss]

        if upper_bound is not None and weight > 0.:
            max_loss_func = lambda data: tf.nn.relu(model(data['x_prior']) - upper_bound)
            max_loss = ns.LossMeanSquares(identifier + '_max', max_loss_func, weight=weight, expected_shape=(num_batched, 1))

            self.train_losses['main'] += [max_loss]

    def setup_dice_loss(self, model, dataset, threshold, identifier: str = 'dice'):

        b_0 = tf.constant(0, dtype=tf.double)
        b_1 = tf.constant(1, dtype=tf.double)

        x_test = dataset.data_handler.x_mesh

        mask_orig = tf.where(dataset.data_handler.tag_values > threshold, b_1, b_0)

        loss_func = lambda: PINNLossHandler.dice(mask_orig, tf.where(model(x_test) > threshold, b_1, b_0))

        loss = ns.Loss(identifier, loss_func, non_negative=True, display_sqrt=False)

        self.test_losses['physics'] += [loss]
        self.test_losses['main'] += [loss]

    def setup_relative_error_loss(self, model, dataset, identifier: str = 'error'):

        x_test = dataset.data_handler.x_mesh
        values = dataset.data_handler.tag_values

        loss_func = lambda: (model(x_test) - values) / values
        loss = ns.LossMeanSquares(identifier, loss_func)

        self.test_losses['main'] += [loss]
        self.test_losses['physics'] += [loss]

    def setup_relative_region_error_losses(self, model, dataset, identifier: str = 'error'):

        losses = []
        for tag in dataset.data_handler.tag_dict:
            region_mesh, _, region_values = dataset.data_handler.get_mesh_points_for_tag(tag)

            name = dataset.data_handler.get_name_for_tag(tag)

            losses += [
                ns.LossMeanSquares(name + '_' + identifier,
                                   lambda rm=region_mesh, rv=region_values: (model(rm) - rv) / rv)
            ]

        self.test_losses['main'] += losses

    def make_losses_adaptive(self, names, phase: str):
        for loss in self.train_losses[phase]:
            if loss.name in names:
                loss.weight = tf.Variable(loss.weight.numpy(), trainable=False, dtype=ns.config.get_dtype())

#############################################################################
# Getter / Setter
#############################################################################

    def get_loss_by_name(self, name: str, phase: str, learning_type: str = 'train'):
        if learning_type == 'train':
            for loss in self.train_losses[phase]:
                if loss.name == name:
                    return loss
        else:
            for loss in self.test_losses[phase]:
                if loss.name == name:
                    return loss

    def add_loss(self, loss: ns.Loss, phase, learning_type):

        if learning_type == 'train':
            self.train_losses[phase] += [loss]
        else:
            self.test_losses[phase] += [loss]

#############################################################################
# Loss utilities
#############################################################################

    @staticmethod
    def dice(y_true, y_pred):
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