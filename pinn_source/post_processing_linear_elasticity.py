import os
import numpy as np
import matplotlib.pyplot as plt
from Codebase.CubeDataPlotter import CubeDataPlotter
import nisaba as ns
import nisaba.experimental as nse
import tensorflow as tf


def save_plots(data_handler, model, model_field, filename, params, plot_solution=True, plot_field=True):

    space_dim = data_handler.mesh_dimension

    model_displacement = lambda x: model(x[:, :space_dim])

    dpi = 400

    data_plotter = CubeDataPlotter()

    if plot_solution:

        file = params['program']['base_dir'] + '/' + params['program']['solution_dir'] + '/' + filename + '.png'
        if os.path.exists(file):
            os.remove(file)
        data_plotter.animate_error_plot(data_handler, lambda x: model(x[:, :space_dim]),
                                        filename=params['program']['base_dir'] + '/' + params['program'][
                                            'solution_dir'] + '/' + filename + '.png')

        file = params['program']['base_dir'] + '/' + params['program']['solution_dir'] + '/s_' + filename + '.png'
        if os.path.exists(file):
            os.remove(file)

        def strain_model(x):
            x = tf.constant(x, dtype=ns.config.get_dtype())
            with ns.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(x)
                u = model(x)
                d = 3
                
                u_grad = nse.physics.tens_style.gradient_vector(tape, u, x, d)
                u_gradT = tf.transpose(u_grad, perm = (0, 2, 1))
            
            return np.expand_dims(np.linalg.norm(0.5 * (u_grad+u_gradT),axis=(1,2)),-1)
        
        strain_orig = data_handler.get_strain_orig()
        strain_orig = np.expand_dims(np.linalg.norm(strain_orig[0],axis=(1,2)),-1)
        data_plotter.plot_field(data_handler, strain_orig, strain_model, relative_error = True, vmin = None, vmax = None, binary = False, filename=params['program']['base_dir'] + '/' + params['program'][
                                            'solution_dir'] + '/s_' + filename + '.png')

    if plot_field:

        file = params['program']['base_dir'] + '/' + params['program']['solution_field_dir'] + '/' + filename + '.png'
        if os.path.exists(file):
            os.remove(file)

        if params['inverse_params']['mu']['param_func'] == "sigmoid":
            vmin = params['inverse_params']['mu']['sigmoid_scalings'][0]
            vmax = params['inverse_params']['mu']['sigmoid_scalings'][1]
        else:
            vmin = None
            vmax = None

        data_plotter.plot_field(data_handler, data_handler.tag_values, model_field, relative_error = True, vmin = vmin, vmax = vmax, binary = False, filename = file, dpi=400, scale=True)
