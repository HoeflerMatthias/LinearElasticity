import tensorflow as tf
import pinn_source.pinn_lib as ns


def get_param_function(min_val, max_val, a=1.0):
    min_val = tf.expand_dims(tf.constant(min_val, dtype=ns.config.get_dtype()), -1)
    max_val = tf.expand_dims(tf.constant(max_val, dtype=ns.config.get_dtype()), -1)

    logit = lambda x: tf.math.log(tf.math.divide(x, 1 - x))

    param_func = lambda x: tf.math.add(tf.math.multiply(max_val - min_val, tf.math.sigmoid(a * x)), min_val)
    inv_param_func = lambda x: logit(tf.math.divide(x - min_val, max_val - min_val)) / a

    return param_func, inv_param_func


#############################################################################
# Main loop
#############################################################################
def run(args):
    import numpy as np
    from pinn_source.network import load_network, get_network
    from pinn_source.data_handler import FEMDataHandler
    from pinn_source.plotting import DataPlotter
    from pinn_source.dataset import PINNDataSet
    from pinn_source.losses import PINNLossHandler
    from pinn_source.training import PINNTrainHandler
    from pinn_source.post_processing import save_plots
    import pinn_source.constitutive
    import random
    import os

    params = args[0]
    filename = args[1]
    
    # Set seeds for reproducibility
    np.random.seed(params['seed'])
    np_random_generator = np.random.default_rng(params['seed'])
    tf.random.set_seed(params['seed'])
    tf_random_generator = tf.random.Generator.from_seed(params['seed'])
    os.environ['PYTHONHASHSEED'] = str(params['seed'])
    random.seed(params['seed'])
    tf.keras.utils.set_random_seed(params['seed'])
    
    params['model']['lam'] *= params['model']['scaling']
    params['model']['pressure'] *= params['model']['scaling']

    # prepare parameter functions
    param_func, inv_param_func = {}, {}
    for p in params['inverse_params']:
        config = params['inverse_params'][p]
        config['sigmoid_scalings'] = tf.cast([s * params['model']['scaling'] for s in config['sigmoid_scalings']],
                                             dtype=ns.config.get_dtype())

        f, finv = get_param_function(config['sigmoid_scalings'][0], config['sigmoid_scalings'][1])
        param_func[p] = f
        inv_param_func[p] = finv
    
    dim = 3

    #############################################################################
    # Initialization
    #############################################################################
    
    # Data
    ####################
    print("read data")
    data_handler = FEMDataHandler(data_file = params["data"])

    data_handler.read()
    data_handler.apply_noise(params['SNR'], dim, np_random_generator)

    dataset = PINNDataSet(data_handler, tf_random_generator, np_random_generator, ns.config.get_dtype(), params['model']['pressure'])
    dataset.set_batch_size_fraction(params['batch_size'])

    # Collocation points
    ####################
    print("initialize collocation points")
    num_BC_nyminus = params['numBCN']
    num_BC_nyplus = params['numBCN']

    num_BC_nxminus = params['numBCN']*2
    num_BC_nxplus = params['numBCN']*2

    num_BC_nzminus = params['numBCN']*4
    num_BC_nzplus = params['numBCN']*4

    dataset.set_bc_plane_corners()

    dataset.set_pde_points(params['numPDE'], 'train')
    dataset.set_pde_points(params['numPDE'], 'test')

    dataset.set_bc_points(num_BC_nxminus, num_BC_nxplus, num_BC_nyminus, num_BC_nyplus, num_BC_nzminus, num_BC_nzplus,
                          'train')
    dataset.set_bc_points(int(num_BC_nxminus/2), int(num_BC_nxplus/2), int(num_BC_nyminus/2), int(num_BC_nyplus/2), int(num_BC_nzminus/2), int(num_BC_nzplus/2),
                          'test')

    dataset.sample_pde_points()
    dataset.sample_bc_points()

    dataset.combine_collocation_points(['pde', 'bc'], 'train')

    collocation_points = []
    data, num, num_batched = dataset.get_data(0, 'pde', 'train')

    collocation_points += [ns.DataSet(data, 'x_PDE_vec', batch_size=num_batched)]

    for bc_key in ['nxminus', 'nxplus', 'nyminus', 'nyplus', 'nzminus', 'nzplus']:
        data, num, num_batched = dataset.get_data(bc_key, 'bc', 'train')

        collocation_points += [ns.DataSet(data, 'x_' + bc_key, batch_size=num_batched)]

    x_prior, _, _ = dataset.get_data('x_prior', 'reg', 'train')
    x_prior = ns.DataSet(x_prior, 'x_prior', batch_size=int(x_prior.shape[0] * params['batch_size']))
    collocation_points += [x_prior]
    collocation_points = ns.DataCollection(collocation_points)
    
    # Data points
    ####################
    print("initialize data")
    
    dataset.set_data_points(params['numData'], 'train')
    dataset.set_data_points(params['numTest'], 'test')

    dataset.sample_displacement_points()

    u_max_components = data_handler.get_max_displacement_components()
    u_max = np.max(u_max_components)

    data_handler.tag_values *= params['model']['scaling']
    data_handler.set_regions()

    # Region mu values (sorted ascending) from auto-detected regions
    region_mu_values = sorted(data_handler.tag_dict.values())
    num_regions = len(region_mu_values)
    region_mu_min = region_mu_values[0]
    region_mu_max = region_mu_values[-1]

    # Network definition
    #####################
    print("initialize network")

    # displacement model
    if params['net']['model_path']:
        model = load_network(params['net']['model_path'])
    else:
        input_mean = tf.math.pow(data_handler.mesh_sizes, 2)
        input_variance = tf.math.pow(data_handler.mesh_sizes, 2)
        if params['inverse_params']['mu']['on']:
            output_mean = np.mean(data_handler.get_displacement()[0], axis=0)
            output_variance = tf.math.pow(np.max(np.abs(data_handler.get_displacement()[0]), axis=0), 2)
        else:
            output_mean = [1., 1., 1.]
            output_variance = [1., 1., 1.]



        model = get_network(dim, input_mean, input_variance, dim, output_mean, output_variance, params['net'],
                            np_random_generator, name="Displacement")

        model.summary()
    
    # Inverse problem
    ####################

    print("initialize inverse params")
    param_models = {}

    inverse_param = 'mu'
    config = params['inverse_params'][inverse_param]
    if config['on']:

        # model parameter as field
        if config['net']['on']:
            scaling = tf.constant(1.0, dtype=ns.config.get_dtype())
            mean = tf.constant(0.0, dtype=ns.config.get_dtype())

            if config['net']["model_path"]:
                model_param = load_network(config['net']["model_path"])
            else:
                input_mean = np.mean(data_handler.x_mesh,axis=0)
                input_variance = tf.math.pow(data_handler.mesh_sizes, 2)
                output_mean = mean
                output_variance = tf.math.pow(scaling, 2)

                model_param = get_network(dim, input_mean, input_variance, 1, output_mean, output_variance,
                                          config['net'], np_random_generator, name=inverse_param)

            model_param.summary()
            param_models[inverse_param] = model_param

        # model parameter as scalar (two-region case)
        else:
            left = tf.Variable(inv_param_func[inverse_param](region_mu_min * config['initial_factor']),dtype=ns.config.get_dtype(), name="region_1")
            right = tf.Variable(inv_param_func[inverse_param](region_mu_max * config['initial_factor']),dtype=ns.config.get_dtype(), name="region_2")

            model_param = lambda x: tf.expand_dims(tf.where(x[:,0] < x[:,1], left, right),-1)
            param_models[inverse_param] = model_param
    
    # Loss definition
    ####################
    print("initialize losses")

    loss_handler = PINNLossHandler()

    # fit losses
    weight_FIT = params['wFit'] / (u_max ** 2)
    #if params['inverse_params']['mu']['on']:
    loss_handler.setup_fit_losses(model, dataset, weight_FIT, 1.0)

    # pde losses
    weight_PDE = params['wPDE'] / ((u_max / (np.max(data_handler.mesh_sizes) ** 2)) ** 2)
    mu_model = param_models['mu']
    mu_func = param_func['mu']
    lam = params['model']['lam']
    body_force = params['model']['body_force']

    pde_model = lambda x: pinn_source.constitutive.PDE(x, model, dim, mu_model, mu_func, lam, body_force)
    loss_handler.setup_pde_losses(pde_model, dataset, weight_PDE, rba = 'PDE' in params['RBA_params']['losses'])

    # boundary losses
    weight_BCNeumann = params['wBCN'] / ((u_max / (np.max(data_handler.mesh_sizes) ** 2)) ** 2)

    weight_BC = {
        'nxminus': weight_BCNeumann,
        'nxplus': weight_BCNeumann,
        'nyminus': weight_BCNeumann,
        'nyplus': weight_BCNeumann,
        'nzminus': weight_BCNeumann,
        'nzplus': weight_BCNeumann
    }

    stress_bc_model = lambda tape, x: pinn_source.constitutive.Piola(tape, x, model, dim, mu_model, mu_func, lam)
    loss_handler.setup_boundary_losses(model, stress_bc_model, dataset, weight_BC, rba = [key for key in params['RBA_params']['losses'] if key in weight_BC])

    if params['mesh_loss']:
        loss_handler.setup_mesh_losses(model, pde_model, dataset)

    # weight decay regularization
    if params['wT'] > 0.:
        loss_handler.setup_weight_decay_loss(model, params['wT'], phases = ['fit'])
    # inverse problem
    for inverse_param in params['inverse_params']:
        config = params['inverse_params'][inverse_param]
        if config['on']:
            param_lambda = lambda x: param_func[inverse_param](param_models[inverse_param](x))

            # dice test loss
            if config['net']['on'] and num_regions > 1:
                threshold = params['inverse_params'][inverse_param]['threshold'] * \
                            params['inverse_params'][inverse_param]['max_value']
                loss_handler.setup_dice_loss(param_lambda, dataset, threshold, identifier=inverse_param + '_dice')

            # region wise error losses
            if config['net']['on'] and num_regions > 1:
                loss_handler.setup_relative_region_error_losses(param_lambda, dataset,
                                                                identifier=inverse_param + '_error')

            # total error test loss
            if config['net']['on']:
                loss_handler.setup_relative_error_loss(param_lambda, dataset, identifier=inverse_param + '_error')

            # per-region relative error (scalar or network)
            if not config['net']['on'] and num_regions > 1:
                loss_handler.setup_relative_region_error_losses(param_lambda, dataset,
                                                                identifier=inverse_param + '_error')

            # box constraints
            lower_bound = region_mu_min * config['min_factor']
            upper_bound = region_mu_max * config['max_factor']

            loss_handler.setup_box_constraints(param_lambda, dataset, params['wM'], identifier=inverse_param,
                                               lower_bound=lower_bound, upper_bound=upper_bound)

            if params['wP'] > 0.:
                prior_scale = 2.0 * region_mu_max
                weight_Prior = params['wP'] / (prior_scale ** 2)
                prior_guess = sum(region_mu_values) / num_regions
                loss_handler.setup_prior_loss(param_lambda, weight_Prior, identifier=inverse_param + '_prior',
                                              prior_guess=prior_guess)

            if config['net']['on']:
                # weight decay regularization
                if config['net']['wT'] > 0.0:
                    loss_handler.setup_weight_decay_loss(param_models[inverse_param], config['net']['wT'], phases = ['main'], identifier = inverse_param + '_tikhonov')

                # gradient penalty
                if config['net']['wTV'] > 0.0:
                    param_model = lambda x: param_func[inverse_param](param_models[inverse_param](x))
                    loss_handler.setup_gradient_penalty_loss(param_model, config['net']['wTV'],
                                                             identifier=inverse_param + '_tv')

    if params['adapt']:
        loss_handler.make_losses_adaptive(['fit', 'PDE', 'nxminus', 'nxplus', 'nyminus', 'nyplus', 'nzminus', 'nzplus'], 'main')

    # Variable definition
    ####################
    phase_variables = {
        'fit': [],
        'physics': [],
        'main': []
    }

    model_variables = model.variables #[m.value for m in model.variables]
    phase_variables['fit'] += model_variables
    # phase_variables[1] += model_variables
    phase_variables['main'] += model_variables

    # Inverse problem
    has_inverse_problem = False
    for inverse_param in params['inverse_params']:
        config = params['inverse_params'][inverse_param]
        if config['on']:
            if config['net']['on']:
                param_variables = param_models[inverse_param].variables
                phase_variables['fit'] += param_variables
                phase_variables['physics'] += param_variables
                phase_variables['main'] += param_variables
            else:
                phase_variables['physics'] += [left, right]
                phase_variables['main'] += [left, right]
            has_inverse_problem = True
    
    #############################################################################
    # Training
    #############################################################################

    train_handler = PINNTrainHandler(loss_handler, phase_variables)
    data_plotter = DataPlotter()

    # prepare models and filenames for setting save points during training
    model_list = [model]
    filename_list = [params['program']['base_dir'] + '/' + params['program']['model_dir'] + '/' + filename + '.keras']

    for inverse_param in params['inverse_params']:
        config = params['inverse_params'][inverse_param]
        if config['on'] and config['net']['on']:
            model_list += [param_models[inverse_param]]
            filename_list += [params['program']['base_dir'] + '/' + params['program'][
                'model_dir'] + '/' + inverse_param + '_' + filename + '.keras']

    if 'mu' in param_models:
        param_lambda = lambda x: param_func['mu'](param_models['mu'](x))
    else:
        param_lambda = None

    # add weight histories and parameters to json output of optimization problem
    def train_preparation_callback(pb: ns.OptimizationProblem):
        train_handler.add_parameters(pb, params)
        train_handler.add_weight_history(pb)
        train_handler.add_lagrange_history(pb)

    train_handler.set_train_preparation_callback(train_preparation_callback)

    train_handler.filenames = {
        'fit': {
            'history': params['program']['base_dir'] + '/' + params['program'][
                'history_fit_dir'] + '/' + filename + '.png',
            'data': params['program']['base_dir'] + '/' + params['program']['data_fit_dir'] + '/' + filename + '.json'
        },
        'physics': {
            'history': params['program']['base_dir'] + '/' + params['program'][
                'history_physics_dir'] + '/' + filename + '.png',
            'data': params['program']['base_dir'] + '/' + params['program'][
                'data_physics_dir'] + '/' + filename + '.json'
        },
        'main': {
            'history': params['program']['base_dir'] + '/' + params['program']['history_dir'] + '/' + filename + '.png',
            'data': params['program']['base_dir'] + '/' + params['program']['data_dir'] + '/' + filename + '.json'
        }
    }
    
    bfgs_backend = params.get('bfgs_backend', 'scipy')

    import time as _time

    timings = {}

    print('start training')
    # Step 1: only fitting
    ####################
    if params['phases'][0]:
        t0 = _time.perf_counter()

        train_handler.train_fit(params['lr1'], params['adam1'], params['bfgs1'], bfgs_backend=bfgs_backend)

        timings['fit'] = _time.perf_counter() - t0

        data_plotter.animate_error_plot(data_handler, lambda x: model(x[:, :dim]),
                                        filename=params['program']['base_dir'] + '/' + params['program'][
                                            'solution_fit_dir'] + '/' + filename + '.png')

        data_plotter.plot_data(dataset,
                               filename=params['program']['base_dir'] + '/' + params['program'][
                                   'solution_sample_dir'] + '/' + filename + '.png')

    # Step 2: only physics
    ####################
    if params['phases'][1] and has_inverse_problem:
        t0 = _time.perf_counter()

        train_handler.train_physics(params['lr1'], params['adam1'], params['bfgs2'], data=collocation_points, bfgs_backend=bfgs_backend)

        timings['physics'] = _time.perf_counter() - t0

        physics_plot = params['program']['base_dir'] + '/' + params['program']['solution_fit_dir'] + '/P' + filename + '.png'
        save_plots(data_handler, model, param_lambda, physics_plot, params, plot_solution=True)

    # Step 3: fitting + physics
    ####################
    if params['phases'][2]:

        if 'RBA' in params and params['RBA']:
            train_handler.adam_callbacks += [train_handler.lagrange_callback(
                params['RBA_params']['gamma'],
                params['RBA_params']['lambda0'],
                params['RBA_params']['eta'],
                params['RBA_params']['losses'],
                frequency=params['RBA_params']['update_frequency']
            )]

        if 'adapt' in params and params['adapt']:
            train_handler.adam_callbacks += [
                train_handler.weight_adjustment_callback(alpha=params['adapt_alpha'], frequency=10)
            ]

        t0 = _time.perf_counter()

        train_handler.train_main(params['lr2'], params['adam3'], params['bfgs3'], data=collocation_points, bfgs_backend=bfgs_backend)

        timings['main'] = _time.perf_counter() - t0

    #############################################################################
    # Post-processing
    #############################################################################

        # model saving
        ####################
        train_handler.save_models(model_list, filename_list)

        # plotting
        ####################

        # plot rba weights for pde if applicable
        if 'PDE' in params['RBA_params']['losses']:
            data_plotter.plot_weights(data_handler,
                                      [loss_handler.get_loss_by_name('PDE', 'main', 'train')],
                                      [dataset.get_data(0, 'pde', 'train')[0]],
                                      filename=params['program']['base_dir'] + '/' + params['program'][
                                          'solution_weight_dir'] + '/pde_' + filename + '.png')
        # plot rba weights for bc if applicable
        boundary_names = dataset.get_labels('bc', 'train')
        losses = [loss_handler.get_loss_by_name(name, 'main', 'train') for name in params['RBA_params']['losses'] if name in boundary_names]
        if len(losses) > 0:
            points = [dataset.get_data(name, 'bc', 'train')[0][0] for name in params['RBA_params']['losses'] if
                      name in boundary_names]
            data_plotter.plot_weights(data_handler, losses, points,
                                      filename=params['program']['base_dir'] + '/' + params['program'][
                                          'solution_weight_dir'] + '/bc_' + filename + '.png')
        # plot rba weight for fit if applicable
        if 'fit' in params['RBA_params']['losses']:
            data_plotter.plot_weights(data_handler,
                [loss_handler.get_loss_by_name('fit', 'main', 'train')],
                [dataset.get_data('x_displacement', 'fit', 'train')[0]],
                filename=params['program']['base_dir'] + '/' + params['program'][
                    'solution_weight_dir'] + '/fit_' + filename + '.png')

        # last call to save callback
        save_plots(data_handler, model, param_lambda, filename, params, plot_solution=True)

    #############################################################################
    # MLflow logging
    #############################################################################
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        import mlflow
        from pinn_source.mlflow_logging import log_pinns_to_mlflow

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "lin_elast:pinns"))

        # Collect post-processing output directories to log as artifacts
        base = params['program']['base_dir']
        artifact_dirs = [
            os.path.join(base, params['program'][key])
            for key in ['solution_dir', 'solution_field_dir',
                        'solution_weight_dir', 'model_dir']
            if key in params['program']
        ]

        log_pinns_to_mlflow(params, filename, loss_handler, train_handler,
                            artifact_dirs=artifact_dirs, timings=timings)