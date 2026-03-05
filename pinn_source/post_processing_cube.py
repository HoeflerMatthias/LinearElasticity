import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from Codebase.constitutive import GL_strain
from Codebase.PINNCubeDataSet import PINNCubeDataSet
from Codebase.DataPlotter import DataPlotter

def save_plots(data_handler, model, param_models, param_func,
                       filename,
                       params, space_dim,
                       plot_field=True, plot_slices=True, plot_strain=True, plot_solution=True):

    threshold = params['inverse_params']['Sa']['threshold'] * params['inverse_params']['Sa']['max_value']
    b_0 = tf.constant(0, dtype=tf.double)
    b_1 = tf.constant(1, dtype=tf.double)

    space_dim = data_handler.mesh_dimension
    dim = space_dim

    model_mask = lambda x: tf.where(model_field(x) > threshold, b_1, b_0)
    mask_orig = tf.where(data_handler.tag_values > threshold, b_1, b_0)
    model_field = lambda x: param_func['Sa'](param_models['Sa'](x))
    model_displacement = lambda x: model(x[:, :space_dim])
    model_strain = lambda x: GL_strain(x[:, :space_dim], model, dim)

    data_plotter = DataPlotter()
    dpi = 400

    if plot_field:

        file = params['program']['base_dir'] + '/' + params['program']['solution_field_dir'] + '/' + filename + '.png'
        if os.path.exists(file):
            os.remove(file)
        data_plotter.plot_field(data_handler, data_handler.tag_values, model_field, relative_error=True, filename=file)

        file = params['program']['base_dir'] + '/' + params['program']['solution_field_dir'] + '/b_' + filename + '.png'
        if os.path.exists(file):
            os.remove(file)
        data_plotter.plot_field(data_handler, mask_orig, model_mask, binary=True, filename=file)

    if plot_slices:

        min_val, max_val = np.inf, -np.inf
        binary_tag_dict = {}
        for region in params['model']['regions']:
            tag = params['model']['regions'][region]['tag']
            binary_tag_dict[tag] = 1 if region == 'healthy' else 0
            min_val = min(min_val, params['model']['regions'][region]['Sa'])
            max_val = max(max_val, params['model']['regions'][region]['Sa'])

        binary_values = data_handler.interpolate(binary_tag_dict)
        field_values = data_handler.interpolate()

        for region in params['model']['regions']:

            if 'center' in params['model']['regions'][region]:
                file = params['program']['base_dir'] + '/' + params['program'][
                    'solution_field_dir'] + '/' + region + filename + '.png'
                if os.path.exists(file):
                    os.remove(file)
                data_plotter.plot_slices(data_handler, model_field, field_values,
                                         params['model']['regions'][region]['center'],
                                         relative_error=True, filename=file, dpi=dpi)

                # binary plots
                file = params['program']['base_dir'] + '/' + params['program'][
                    'solution_field_dir'] + '/b' + region + filename + '.png'
                if os.path.exists(file):
                    os.remove(file)
                data_plotter.plot_slices(data_handler, model_mask, binary_values,
                                         params['model']['regions'][region]['center'],
                                         binary=True,
                                         filename=file, dpi=dpi)

                if plot_strain:
                    file = params['program']['base_dir'] + '/' + params['program'][
                        'solution_strain_dir'] + '/' + region + filename + '.png'
                    if os.path.exists(file):
                        os.remove(file)
                    data_plotter.plot_slices(data_handler, model_strain, data_handler.get_strain_orig()[0],
                                             params['model']['regions'][region]['center'], scalar_values=False,
                                             relative_error=True, filename=file, dpi=dpi)

    if plot_strain:
        file = params['program']['base_dir'] + '/' + params['program']['solution_strain_dir'] + filename + '.png'
        if os.path.exists(file):
            os.remove(file)
        data_plotter.animate_strain_error_plot(data_handler, model_strain,
                                               time_scale=data_handler.time_scale,
                                               filename=file)

    if plot_solution:

        file = params['program']['base_dir'] + '/' + params['program']['solution_dir'] + '/' + filename + '.png'
        if os.path.exists(file):
            os.remove(file)
        data_plotter.animate_error_plot(data_handler, model_displacement,
                                        time_scale=data_handler.time_scale,
                                        filename=file)


