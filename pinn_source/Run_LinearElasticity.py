#############################################################################
# Main loop
#############################################################################
def run(args):
    import tensorflow as tf
    import nisaba as ns

    import nisaba.experimental as nse
    import numpy as np
    from Codebase.network import load_network, get_network
    from Codebase.train_utilities import get_param_function
    from Codebase.PseudoFEMDataHandler import PseudoFEMDataHandler
    from Codebase.CubeDataPlotter import CubeDataPlotter
    from Codebase.Sampler import Sampler
    from Codebase.PINNLinearElasticityDataSet import PINNLinearElasticityDataSet
    from Codebase.PINNLinearElasticityLossHandler import PINNLinearElasticityLossHandler
    from Codebase.PINNCubeTrainHandler import PINNCubeTrainHandler
    from Codebase.post_processing_linear_elasticity import save_plots
    from Codebase.DistanceLayer import DistanceLayer
    from Codebase.BSplineModel import SplineModel
    import Codebase.constitutive
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
    
    def Piola(tape, x, model, dim, params, param_models):
        if params['inverse_params']['mu']['on']:
            mu = param_models['mu'](x[:, :dim])
            mu = param_func['mu'](mu)
            mu = tf.expand_dims(mu, -1)
        else:
            mu = tf.expand_dims(field_model(x),-1)
        
        lam = tf.constant([[params['model']['lam']]], dtype=ns.config.get_dtype())
        d = model(x)
        
        P = nse.physics.tens_style.linear_elasticity_stress(tape, d, x, mu, lam, dim)
        
        return P

    def PDE(x, model, dim, params, param_models):
        force = tf.convert_to_tensor(params['model']['body_force'], dtype=ns.config.get_dtype())

        # repeat force for all points
        n_pts = tf.shape(x)[0]
        force = tf.repeat([force], n_pts, axis=0)

        with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(x)

            P = Piola(tape, x, model, dim, params, param_models)

            # add everything
            div_P = nse.physics.tens_style.divergence_tensor(tape, P, x, dim)

        return tf.add(-div_P, - force)

    params['model']['regions']['healthy']['mu'] *= params['model']['scaling']
    params['model']['regions']['scar']['mu'] *= params['model']['scaling']
    params['model']['lam'] *= params['model']['scaling']
    params['model']['pressure'] *= params['model']['scaling']

    # prepare parameter functions
    param_func, inv_param_func = {}, {}
    for p in params['inverse_params']:
        config = params['inverse_params'][p]
        config['sigmoid_scalings'] = tf.cast([s * params['model']['scaling'] for s in config['sigmoid_scalings']],
                                             dtype=ns.config.get_dtype())

        if config['param_func'] == 'sigmoid':
            f, finv = get_param_function('sigmoid', min=config['sigmoid_scalings'][0],
                                         max=config['sigmoid_scalings'][1])
        else:
            f, finv = get_param_function(config['param_func'])
        param_func[p] = f
        inv_param_func[p] = finv
    
    dim = 3
    space_dim = 3

    #############################################################################
    # Initialization
    #############################################################################
    
    # Data
    ####################
    print("read data")
    sampler = Sampler(ns.config.get_dtype())
    data_handler = PseudoFEMDataHandler(data_file = params["data"])

    data_handler.read()
    data_handler.apply_noise(params['SNR'], space_dim, np_random_generator)

    if params['measurements'] == 'outside':
        data_handler.restrict_to_outside()
    elif params['measurements'] == 'inside':
        data_handler.restrict_to_inside()

    dataset = PINNLinearElasticityDataSet(sampler, data_handler, tf_random_generator, np_random_generator, ns.config.get_dtype(), params['model']['pressure'])
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

    dataset.sample_pde_points(params['samp'])
    dataset.sample_bc_points(params['samp'])

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
    
    dataset.set_data_points(params['numData'], 0, 1000, 'train')
    dataset.set_data_points(params['numTest'], 0, 1000, 'test')

    dataset.sample_displacement_points()
    dataset.sample_strain_points()

    u_max_components = data_handler.get_max_displacement_components()
    u_max = np.max(u_max_components)

    data_handler.tag_values *= params['model']['scaling']

    # Network definition
    #####################
    print("initialize network")

    # displacement model
    if params['net']['model_path']:
        model = load_network(params['net']['model_path'])
    elif params['net']['bspline']:
        
        num_ctrl = params['net']['num_ctrl']
        
        model = SplineModel(
            grid_shape=(num_ctrl,num_ctrl,num_ctrl,dim),
            min_norm = data_handler.min_dim,
            max_norm = data_handler.max_dim
        )
        model(tf.constant([[1.0,1.0,1.0]],dtype=tf.float64))
        #import pdb; pdb.set_trace()
        print("Total params of displacement network: " + str(tf.size(model.control_grid).numpy()))
    
    else:
        input_mean = tf.math.pow(data_handler.mesh_sizes, 2)
        input_variance = tf.math.pow(data_handler.mesh_sizes, 2)
        if params['inverse_params']['mu']['on']:
            output_mean = np.mean(data_handler.get_displacement()[0], axis=0)
            output_variance = tf.math.pow(np.max(np.abs(data_handler.get_displacement()[0]), axis=0), 2)
        else:
            output_mean = [1., 1., 1.]
            output_variance = [1., 1., 1.]



        boundary_functions = [
            #DistanceLayer.make_add_func(dataset.boundary_data['nyminus'], axis=None, output_dims=3),
            #DistanceLayer.make_add_func(0.0, 2, 3)
        ]
        
        model = get_network(dim, input_mean, input_variance, space_dim, output_mean, output_variance, params['net'],
                            np_random_generator, boundary_function = boundary_functions, name="Displacement")

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
            if config['net']['bspline']:
        
                num_ctrl = config['net']['num_ctrl']
                
                model_param = SplineModel(
                    grid_shape=(num_ctrl,num_ctrl,num_ctrl,1),
                    min_norm = data_handler.min_dim,
                    max_norm = data_handler.max_dim
                )
                model_param(tf.constant([[1.0,1.0,1.0]],dtype=tf.float64))
                param_models[inverse_param] = model_param
                
                print("Total params of shear modulus network: " + str(tf.size(model_param.control_grid).numpy()))
            else:
                if params['inverse_params'][inverse_param]['param_func'] == "sigmoid":
                    scaling = tf.constant(1.0, dtype=ns.config.get_dtype())
                    mean = tf.constant(0.0, dtype=ns.config.get_dtype())
                else:
                    scaling = tf.cast(inv_param_func[inverse_param](params['model']['regions']['healthy'][inverse_param]),
                                      dtype=ns.config.get_dtype())
                    mean = tf.constant(0.0, dtype=ns.config.get_dtype())  # tf.cast(avg, dtype=ns.config.get_dtype())
    
                if config['net']["model_path"]:
                    model_param = load_network(config['net']["model_path"])
                else:
    
                    input_mean = np.mean(data_handler.x_mesh,axis=0)
                    input_variance = tf.math.pow(data_handler.mesh_sizes, 2)
                    output_mean = mean
                    output_variance = tf.math.pow(scaling, 2)
    
                    model_param = get_network(space_dim, input_mean, input_variance, 1, output_mean, output_variance,
                                              config['net'], np_random_generator, name=inverse_param)
    
                    
                
                model_param.summary()
                param_models[inverse_param] = model_param

        # model parameter as scalar
        else:
            left = tf.Variable(inv_param_func[inverse_param](params['model']['regions']['healthy']['mu'] * config['initial_factor']),dtype=ns.config.get_dtype(), name="healthy")
            right = tf.Variable(inv_param_func[inverse_param](params['model']['regions']['scar']['mu'] * config['initial_factor']),dtype=ns.config.get_dtype(), name="scar")
            
            model_param = lambda x: tf.expand_dims(tf.where(x[:,0] < x[:,1], left, right),-1)
            param_models[inverse_param] = model_param
    
    # Loss definition
    ####################
    print("initialize losses")

    loss_handler = PINNLinearElasticityLossHandler()

    # fit losses
    weight_FIT = params['wFit'] / (u_max ** 2)
    #if params['inverse_params']['mu']['on']:
    loss_handler.setup_fit_losses(model, dataset, weight_FIT, 1.0)

    # pde losses
    weight_PDE = params['wPDE'] / ((u_max / (np.max(data_handler.mesh_sizes) ** 2)) ** 2)
    pde_model = lambda x: PDE(x, model, dim, params, param_models)
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

    stress_bc_model = lambda tape, x: Piola(tape, x, model, dim, params, param_models)
    loss_handler.setup_boundary_losses(model, stress_bc_model, dataset, weight_BC, rba = [key for key in params['RBA_params']['losses'] if key in weight_BC])

    if params['mesh_loss']:
        loss_handler.setup_mesh_losses(model, pde_model, dataset)

    # weight decay regularization
    if params['wT'] > 0.:
        #loss_handler.setup_weight_decay_loss(model, params['wT'], phases = ['fit'])
        #loss_handler.setup_gradient_penalty_loss(model, params['wTG'], identifier = 'gradient_penalty', data = tf.constant(data_handler.x_mesh, dtype=ns.config.get_dtype()))
        loss_handler.second_order_diff_penalty_3d(model.control_grid, params['wT']) 
    # inverse problem
    for inverse_param in params['inverse_params']:
        config = params['inverse_params'][inverse_param]
        if config['on']:
            param_lambda = lambda x: param_func[inverse_param](param_models[inverse_param](x))

            # dice test loss
            if config['net']['on'] and len(params['model']['regions'].keys()) > 1:
                threshold = params['inverse_params'][inverse_param]['threshold'] * \
                            params['inverse_params'][inverse_param]['max_value']
                loss_handler.setup_dice_loss(param_lambda, dataset, threshold, identifier=inverse_param + '_dice')

            # region wise error losses
            if config['net']['on'] and len(params['model']['regions'].keys()) > 1:
                loss_handler.setup_relative_region_error_losses(param_lambda, dataset,
                                                                identifier=inverse_param + '_error')

            # total error test loss
            if config['net']['on']:
                loss_handler.setup_relative_error_loss(param_lambda, dataset, identifier=inverse_param + '_error')

            # if parameter is only scalar
            if not config['net']['on']:
                # relative error w.r.t. healthy region
                loss_handler.add_loss(ns.Loss(inverse_param+"_healthy", lambda: tf.math.abs(param_func[inverse_param](
                    param_models[inverse_param](tf.constant([[0.0,1.0,0.0]], dtype=ns.config.get_dtype()))) - params['model']['regions']['healthy'][inverse_param]) /
                                                                     params['model']['regions']['healthy'][
                                                                         inverse_param]), 'main', 'test')
                # relative error w.r.t. scar region
                loss_handler.add_loss(ns.Loss(inverse_param+"_scar", lambda: tf.math.abs(param_func[inverse_param](
                    param_models[inverse_param](tf.constant([[1.0,0.0,0.0]], dtype=ns.config.get_dtype()))) - params['model']['regions']['scar'][inverse_param]) /
                                                                     params['model']['regions']['scar'][
                                                                         inverse_param]), 'main', 'test')

            # box constraints
            guess = params['model']['regions']['healthy'][inverse_param]
            lower_bound = guess * config['min_factor']
            upper_bound = guess * config['max_factor']

            loss_handler.setup_box_constraints(param_lambda, dataset, params['wM'], identifier=inverse_param,
                                               lower_bound=lower_bound, upper_bound=upper_bound)

            if params['wP'] > 0.:
                weight_Prior = (params['wP'] / ((32.0 * params['model']['scaling']) ** 2))
                prior_guess = params['model']['scaling'] * (16.0+8.0)/2.0
                loss_handler.setup_prior_loss(param_lambda, weight_Prior, identifier=inverse_param + '_prior',
                                              prior_guess=prior_guess)

            if config['net']['on'] and not config['net']['freeze']:
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
            if config['net']['on'] and not config['net']['freeze']:
                param_variables = param_models[inverse_param].variables #[m.value for m in param_models[inverse_param].variables]
                phase_variables['fit'] += param_variables
                phase_variables['physics'] += param_variables
                phase_variables['main'] += param_variables
            else:
                param_variables = [left,right]
                phase_variables['physics'] += param_variables
                phase_variables['main'] += param_variables
            has_inverse_problem = True
    
    #############################################################################
    # Training
    #############################################################################

    train_handler = PINNCubeTrainHandler(loss_handler, phase_variables)
    data_plotter = CubeDataPlotter()

    # prepare models and filenames for setting save points during training
    model_list = [model]
    filename_list = [params['program']['base_dir'] + '/' + params['program']['model_dir'] + '/' + filename + '.keras']

    for inverse_param in params['inverse_params']:
        config = params['inverse_params'][inverse_param]
        if config['on'] and config['net']['on']:
            model_list += [param_models[inverse_param]]
            filename_list += [params['program']['base_dir'] + '/' + params['program'][
                'model_dir'] + '/' + inverse_param + '_' + filename + '.keras']

    def plot_save_callback(pb, itr, itr_round, frequency=1000):
        if itr % frequency == 0 and itr > 0:
            #param_lambda = lambda x: field_model(x) 
            param_lambda = lambda x: param_func['mu'](param_models['mu'](x))
            save_plots(data_handler, model, param_lambda, filename, params, plot_solution=True)

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
    
    print('start training')
    # Step 1: only fitting
    ####################
    if params['phases'][0]:

        train_handler.train_fit(params['lr1'], params['adam1'], params['bfgs1'])

        data_plotter.animate_error_plot(data_handler, lambda x: model(x[:, :space_dim]),
                                        filename=params['program']['base_dir'] + '/' + params['program'][
                                            'solution_fit_dir'] + '/' + filename + '.png')

        data_plotter.plot_data(dataset,
                               filename=params['program']['base_dir'] + '/' + params['program'][
                                   'solution_sample_dir'] + '/' + filename + '.png')

    # Step 2: only physics
    ####################
    if params['phases'][1] and has_inverse_problem:
        train_handler.train_physics(params['lr1'], params['adam1'], params['bfgs2'], data=collocation_points)

        filename = params['program']['base_dir'] + '/' + params['program']['solution_fit_dir'] + '/P' + filename + '.png'
        save_plots(data_handler, model, param_lambda, filename, params, plot_solution=True)

    # Step 3: fitting + physics
    ####################
    if params['phases'][2]:

        if params['adapt']:
            train_handler.adam_callbacks += [train_handler.weight_adjustment_callback(alpha=params['adapt_alpha'])]

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

        train_handler.callbacks += [
            train_handler.model_save_callback(model_list, filename_list, frequency=200),
            lambda pb, itr, itr_round: plot_save_callback(pb, itr, itr_round, frequency=200)
        ]

        train_handler.train_main(params['lr2'], params['adam3'], params['bfgs3'], data=collocation_points)

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
            data_plotter.plot_weights(
                [loss_handler.get_loss_by_name('fit', 'main', 'train')],
                [dataset.get_data('x_displacement', 'fit', 'train')[0]],
                filename=params['program']['base_dir'] + '/' + params['program'][
                    'solution_weight_dir'] + '/fit_' + filename + '.png')

        # last call to save callback
        param_lambda = lambda x: param_func['mu'](param_models['mu'](x))
        save_plots(data_handler, model, param_lambda, filename, params, plot_solution=True)

        # plot faces
        #data_plotter.plot_faces(dataset, loss_handler, draw=False, block=False,
        #                        filename=params['program']['base_dir'] + '/' + params['program'][
        #                            'solution_bc_dir'] + '/' + filename + '.png')