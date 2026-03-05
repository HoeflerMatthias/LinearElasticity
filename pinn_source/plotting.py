import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.animation as animation
import math
import tensorflow as tf
from pinn_source.pinn_lib.config import get_dtype


class DataPlotter:

    def __init__(self):
        self.FONT_SIZE = 12

    def _finish(self, fig, title = None, filename = None, draw = False, block = False, dpi = 600):
        if title is not None:
            fig.suptitle(title)
        if filename is not None:
            plt.savefig(filename, dpi=dpi, transparent=False)
        if draw:
            plt.draw()
            plt.pause(1e-16)
        if block:
            plt.show(block=True)

        plt.close(fig)

    def _setup_ax(self, ax, min_dims, max_dims, dim = 3):
        if dim == 3:
            ax.set_xlim3d([min_dims[0], max_dims[0]])
            ax.set_ylim3d([min_dims[1], max_dims[1]])
            ax.set_zlim3d([min_dims[2], max_dims[2]])
            ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.3, alpha=0.2)
        else:
            ax.set_xlim([min_dims[0], max_dims[0]])
            ax.set_ylim([min_dims[1], max_dims[1]])

    def _get_colorbar_extend(self, vmin, vmax, min_value, max_value):
        extend = 'neither'
        if max_value > vmax:
            if min_value < vmin:
                extend = 'both'
            else:
                extend = 'max'
        elif min_value < vmin:
            extend = 'min'
        return extend

    def _setup_error_col_layout(self, axs_flat, min_dims, max_dims, title_add = "", binary=False, relative_error=False, dim = 3):
        titles = [
            "Ground Truth" + title_add,
            "PINN Solution",
            "Signed Error"
        ]
        if binary:
            titles[-1] = "Error"
        if relative_error:
            titles[-1] = "Relative Error"

        for i, ax in enumerate(axs_flat[:3]):
            ax.set_title(titles[i], fontsize=self.FONT_SIZE, pad=10.0)

        for ax in axs_flat:
            self._setup_ax(ax, min_dims, max_dims, dim = dim)

    def animate_error_plot(self, data_handler, model, time_scale = 1e0, time_unit = "ms", filename = "data_points.gif", fig = None, step_size = 1, dpi=400):

        x = data_handler.x_mesh
        x_displaced = data_handler.get_x_displaced() # can be fewer points than x_mesh

        tvalues = data_handler.get_times(time_scale)
        t0, t1, tcount = tvalues[0], tvalues[-1], len(tvalues)

        d_orig = data_handler.get_displacement_orig()
        d_data = [d for d,_,_ in data_handler.displacement]

        if 'low_resolution' in data_handler.submeshes_reference_conf:
            x = x[data_handler.submeshes_reference_conf['low_resolution']]
            x_displaced = [x_displaced[t][data_handler.submeshes_reference_conf['low_resolution']] for t in range(len(x_displaced))]
            d_orig = d_orig[:,data_handler.submeshes_reference_conf['low_resolution']]
            d_data = [d[data_handler.submeshes_reference_conf['low_resolution']] for d in d_data]

        d_model = []
        for t in tvalues:
            constant_t = tf.constant(t, shape=(x.shape[0], 1), dtype=x.dtype)
            xt = tf.concat([x, constant_t], axis=1)
            d_model.append(model(xt))
        d_model = np.array(d_model, dtype=np.double)
        d_error = d_orig - d_model

        x_displaced_model = x + d_model

        c_model = LA.norm(d_model, axis=-1)
        c_data = [LA.norm(dd, axis=-1) for dd in d_data]
        c_error = LA.norm(d_error, axis=-1)

        v = [
            (min([np.min(cd) for cd in c_data]), max([np.max(cd) for cd in c_data])),
            (np.min(c_model), np.max(c_model)),
            (np.min(c_error), np.max(c_error))
        ]

        norm_orig = matplotlib.colors.Normalize(vmin=v[0][0], vmax=v[0][1])
        norm_error = matplotlib.colors.Normalize(vmin=v[2][0], vmax=v[2][1])

        cmap_plot = matplotlib.colormaps.get_cmap('RdBu_r')
        cmap_error = matplotlib.colormaps.get_cmap('viridis')


        if fig == None:
            fig = plt.figure(figsize=(8,4))

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        axs = [ax1, ax2, ax3]
        titles = ["Data Source", "PINN Solution", "Error"]

        scat1 = ax1.scatter(x_displaced[0][:, 0], x_displaced[0][:, 1], x_displaced[0][:, 2], c=cmap_plot(norm_orig(c_data[0])), cmap=cmap_plot,
                            s=5, vmin = v[0][0], vmax = v[0][1])
        scat2 = ax2.scatter(x_displaced_model[0, :, 0], x_displaced_model[0, :, 1], x_displaced_model[0, :, 2], c=cmap_plot(norm_orig(c_model[0])), cmap=cmap_plot,
                            s=5, vmin = v[1][0], vmax = v[1][1])
        scat3 = ax3.scatter(x_displaced[0][:, 0], x_displaced[0][:, 1], x_displaced[0][:, 2], c=cmap_error(norm_error(c_error[0])), cmap=cmap_error,
                            s=5, vmin = v[2][0], vmax = v[2][1])
        scatters = [scat1, scat2, scat3]

        t = t0
        fac = [1.3,1.3,1.3]
        for i, (ax, title, scat, v_val) in enumerate(zip(axs, titles, scatters, v)):
            self._setup_ax(ax, data_handler.min_dim, [fac[i]*dm for dm in data_handler.max_dim])
            ax.set_title(title)
            scat.set_clim(v_val[0], v_val[1])
            ax.set_aspect('equal')
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.zaxis.set_ticklabels([])

        num_ticks, num_ticks_error = 3, 3
        ticks = np.linspace(v[0][0], v[0][1], num=num_ticks)
        ticks_error = np.linspace(v[2][0], v[2][1], num=num_ticks)

        tick_labels = ["{:.2f}".format(l) for l in ticks]
        ticks_error_labels = ["{:.2f}".format(l) for l in ticks_error]

        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_orig, cmap=cmap_plot), location='left', ticks = ticks,
                 ax=axs[0:2], orientation='vertical', shrink = 0.4, aspect = 12, pad = 0.1)

        cbar.ax.set_yticklabels(tick_labels)

        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_error, cmap=cmap_error), location='right', ticks = ticks_error,
                 ax=axs[-1], orientation='vertical', shrink = 0.4, aspect = 12, pad = 0.1)

        cbar.ax.set_yticklabels(ticks_error_labels)

        iterations = math.ceil((tcount)/step_size)
        print("Animation: %d of %d" % (0, iterations))


        def animate_scatters(iteration, scatters):
            n = (iteration)*step_size
            t = tvalues[n]

            print("Animation: %d of %d" % (iteration, iterations+1))

            scatters[0]._offsets3d = (x_displaced[n][:,0], x_displaced[n][:,1], x_displaced[n][:,2])

            scatters[1]._offsets3d = (x_displaced_model[n, :, 0], x_displaced_model[n, :, 1], x_displaced_model[n, :, 2])

            return scatters

        ani = animation.FuncAnimation(fig, animate_scatters, fargs=(scatters,),
                                      frames=iterations, blit=False)

        ani.save(filename, writer='pillow', fps=math.ceil(iterations/10), dpi=dpi)
        plt.close(fig)

    def plot_weights(self, data_handler, hloss, x, filename = "weights.png", fig = None):

        if fig == None:
            fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(projection='3d')

        vmax = max([np.max(tf.reduce_mean(l.lagrange_mul, -1)) for l in hloss])
        vmin = min([np.min(tf.reduce_mean(l.lagrange_mul, -1)) for l in hloss])

        for hloss_i, x_i in zip(hloss, x):
            weights = tf.reduce_mean(hloss_i.lagrange_mul, -1)
            scat = ax.scatter(x_i[:,0], x_i[:,1], x_i[:,2], c=weights, cmap='viridis', s=5)
            scat.set_clim(vmin, vmax)

        self._setup_ax(ax, data_handler.min_dim, data_handler.max_dim, )
        fig.colorbar(scat, ax=ax, shrink=0.6, aspect=8, pad=0.1)
        ax.set_title("Weight")

        self._finish(fig, title = ",".join([h.name for h in hloss]), filename = filename, draw = False, block = False, dpi = 200)

    def plot_field(self, data_handler, original_data, model, relative_error = False, vmin = None, vmax = None, binary = False, filename = "data_points_strain.gif", dpi=400, scale=False):

        x = data_handler.x_displaced_orig[0][0]
        if 'low_resolution' in data_handler.submeshes_reference_conf:
            x = x[data_handler.submeshes_reference_conf['low_resolution']]
            original_data = original_data[data_handler.submeshes_reference_conf['low_resolution']]

        field = tf.reshape(model(x), [x.shape[0],1])

        c_data_orig = original_data
        c_data_field = field

        if binary:
            c_data_error = field - c_data_orig
        else:
            if relative_error:
                c_data_error = np.abs(field - c_data_orig)/c_data_orig
            else:
                c_data_error = (field - c_data_orig)

        min_value_field = np.min(field)
        max_value_field = np.max(field)

        min_value = np.min(c_data_orig)
        max_value = np.max(c_data_orig)

        min_value_error = np.min(c_data_error)
        max_value_error = np.max(c_data_error)

        vmin = min(min_value,min_value_field) if vmin is None else vmin
        vmax = max(max_value,max_value_field) if vmax is None else vmax

        if vmin == vmax:
            vmin = min_value_field
            vmax = max_value_field

        vmax_error = max_value_error
        vmin_error = min_value_error

        if relative_error:
            abs_error = max(np.abs(max_value_error), np.abs(min_value_error))
            vmin_error = 0#-abs_error
            vmax_error = abs_error

        if binary:
            vmax_error = 1
            vmin_error = -1

        vdata = [
            [vmin, vmax],
            [vmin, vmax],
            [vmin_error, vmax_error]
        ]

        if binary:
            cmap_plot = matplotlib.colormaps.get_cmap('gray')
            cmap_error = matplotlib.colormaps.get_cmap('coolwarm')
        else:
            cmap_plot = matplotlib.colormaps.get_cmap('RdYlGn_r')
            cmap_error = matplotlib.colormaps.get_cmap('viridis')

        norm_orig = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        norm_field = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        norm_error = matplotlib.colors.Normalize(vmin=vmin_error, vmax=vmax_error)

        # error bar extension
        extend_v_field = self._get_colorbar_extend(vmin, vmax, min_value_field, max_value_field)
        extend_v_error = self._get_colorbar_extend(vmin_error, vmax_error, min_value_error, max_value_error)

        num_ticks, num_ticks_error = 3, 3
        if binary:
            num_ticks = 2
        ticks = np.linspace(vmin, vmax, num=num_ticks)
        ticks_error = np.linspace(vmin_error, vmax_error, num=num_ticks)

        scaling = 1e2 if scale else 1e0

        tick_labels = ["{:.2f}".format(scaling*l) for l in ticks]
        ticks_error_labels = ["{:.2f}".format(l) for l in ticks_error]

        cdata = [
            cmap_plot(norm_orig(c_data_orig)),
            cmap_plot(norm_field(c_data_field)),
            cmap_error(norm_error(c_data_error))
        ]

        # plotting
        fig,axs = plt.subplots(figsize=(8, 4), ncols=3, nrows=1, subplot_kw={'projection': '3d'})
        self._setup_error_col_layout(axs.flat,
                                     data_handler.min_dim, data_handler.max_dim,
                                     binary = binary, relative_error = relative_error)

        fac = [1.3,1.3,1.3]
        for i, ax in enumerate(axs):

            if binary:
                if i == 0:
                    cdata[i][:,0,3] = np.squeeze(1-c_data_orig, axis=-1)
                elif i == 1:
                    cdata[i][:,0,3] = np.squeeze(1-c_data_field, axis=-1)
                else:
                    cdata[i][:,0,3] = 0.3*np.squeeze(np.abs(c_data_error), axis=-1)

            sct = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=cdata[i], s=5)
            sct.set_clim(vdata[i][0], vdata[i][1])

            ax.grid(b=True, color='grey',
                 linestyle='-.', linewidth=0.3,
                 alpha=0.2)
            self._setup_ax(ax, data_handler.min_dim, [fac[i]*dm for dm in data_handler.max_dim])
            ax.set_aspect('equal')
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.zaxis.set_ticklabels([])

        fig.tight_layout()
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_orig, cmap=cmap_plot), location='left', ax=axs[0:2], orientation='vertical', shrink = 0.4, aspect = 12, pad = 0.1, ticks = ticks, extend = extend_v_field)

        cbar.ax.set_yticklabels(tick_labels)

        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_error, cmap=cmap_error), location='right',
                 ax=axs[-1], orientation='vertical', shrink = 0.4, aspect = 12, pad = 0.1, ticks = ticks_error, extend = extend_v_error)

        cbar.ax.set_yticklabels(ticks_error_labels)

        self._finish(fig, filename = filename, draw = False, block = False, dpi = 400)
        print('plot field completed')

    def plot_data(self, dataset, draw=True, block=False, filename=None):

        x_data, num, _ = dataset.get_data('x_displacement', 'data', 'train')
        d_data, _, _ = dataset.get_data('displacement', 'data', 'train')
        x_displaced = x_data + d_data

        x_mesh = dataset.data_handler.get_mesh()
        d_mesh = dataset.data_handler.get_displacement()[0]
        x_mesh_displaced = x_mesh + d_mesh

        d_data_norm = np.linalg.norm(d_data, axis=1)
        d_mesh_norm = np.linalg.norm(d_mesh, axis=1)

        fig = plt.figure(figsize=(16, 5))

        # Add x, y gridlines
        def plot_view(ax, angle=None):
            ax.grid(b=True, color='grey',
                    linestyle='-.', linewidth=0.3,
                    alpha=0.2)

            ax.scatter3D(x_mesh_displaced[:,0], x_mesh_displaced[:,1], x_mesh_displaced[:,2], c='black', s=2, alpha=0.01)
            sct1 = ax.scatter3D(x_displaced[:,0], x_displaced[:,1], x_displaced[:,2], c=d_data_norm, cmap='RdBu_r', s=5)

            sct1.set_clim(np.min(d_mesh_norm), np.max(d_mesh_norm))

            if angle is None:
                ax.set_zlim(np.min(x_mesh_displaced[:,2]), np.max(x_mesh_displaced[:,2]))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            if angle is not None:
                ax.view_init(elev=angle[0], azim=angle[1], roll=angle[2])

            return sct1

        ax = fig.add_subplot(1, 3, 1, projection='3d')
        sct1 = plot_view(ax)

        # YZ
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        ax.set_proj_type('ortho')
        sct1 = plot_view(ax, [0, 0, 0])
        ax.set_xticklabels([])
        ax.set_xlabel('')

        # XY
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        ax.set_proj_type('ortho')
        sct1 = plot_view(ax, [90, -90, 0])
        ax.set_zticklabels([])
        ax.set_zlabel('')

        fig.colorbar(sct1, ax=ax, shrink=0.6, aspect=8, pad=0.1)
        fig.suptitle('Data Points', fontsize=self.FONT_SIZE)

        self._finish(fig, title = None, filename = filename, draw = draw, block = block, dpi = 600)
        print('plot data points completed')