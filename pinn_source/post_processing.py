import os
from pinn_source.plotting import DataPlotter


def save_plots(data_handler, model, model_field, filename, params, plot_solution=True, plot_field=True):

    space_dim = data_handler.mesh_dimension

    data_plotter = DataPlotter()

    if plot_solution:

        file = params['program']['base_dir'] + '/' + params['program']['solution_dir'] + '/' + filename + '.png'
        if os.path.exists(file):
            os.remove(file)
        data_plotter.animate_error_plot(data_handler, lambda x: model(x[:, :space_dim]),
                                        filename=params['program']['base_dir'] + '/' + params['program'][
                                            'solution_dir'] + '/' + filename + '.png')

    if plot_field:

        file = params['program']['base_dir'] + '/' + params['program']['solution_field_dir'] + '/' + filename + '.png'
        if os.path.exists(file):
            os.remove(file)

        vmin = params['inverse_params']['mu']['sigmoid_scalings'][0]
        vmax = params['inverse_params']['mu']['sigmoid_scalings'][1]

        data_plotter.plot_field(data_handler, data_handler.tag_values, model_field, relative_error = True, vmin = vmin, vmax = vmax, binary = False, filename = file, dpi=400, scale=True)
