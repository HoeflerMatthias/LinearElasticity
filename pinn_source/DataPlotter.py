import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
from numpy import linalg as LA
from plot_settings import *
import matplotlib.animation as animation
import math
import tensorflow as tf
from nisaba.config import get_dtype

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
            #ax.set_xlabel('X')

            ax.set_ylim3d([min_dims[1], max_dims[1]])
            #ax.set_ylabel('Y')

            ax.set_zlim3d([min_dims[2], max_dims[2]])
            #ax.set_zlabel('Z')
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
        #titles = ["Data Source $|$ t=%1.0f" + time_unit, "PINN Solution $|$ t=%1.0f" + time_unit, "Error w.r.t. Ground Truth $|$ t=%1.0f" + time_unit]
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
            #fig.colorbar(scat, ax=ax, shrink=0.4, aspect=10, pad=0.1)
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
            
            #for ax, title in zip(axs, titles):
            #    ax.set_title(title % t)
            
            scatters[0]._offsets3d = (x_displaced[n][:,0], x_displaced[n][:,1], x_displaced[n][:,2])
            #scatters[0].set_array(c_data[n])
    
            scatters[1]._offsets3d = (x_displaced_model[n, :, 0], x_displaced_model[n, :, 1], x_displaced_model[n, :, 2])
            #scatters[1].set_array(c_model[n])
            
            #scatters[2].set_array(c_error[n])
    
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


    def animate_train_data(self, data_handler, model, time_scale = 1e0, time_unit = "ms", filename = "data_points.gif", fig = None, step_size = 1, full_data = False):

        if fig == None:
            fig = plt.figure(figsize=(12, 4))

        # list with time points with units (time_scale)
        tvalues = data_handler.get_times(time_scale)
        t0, t1, tcount = tvalues[0], tvalues[-1], len(tvalues)

        def find_data_for_time(t, data, t_dim):
            mask = tf.where(data[:,t_dim] == t)
            return tf.squeeze(mask)

        if full_data:
            times = data_handler.get_times(1e-3)
            x = data_handler.x_mesh
        else:
            x = data_handler.train_data['x_mesh']
            d = data_handler.train_data['displacement']
            d_model = model(x)
        
        x_displaced = data_handler.train_data['x_displaced']
        x_displaced_model = x[:,:3] + d_model

        c = LA.norm(d, axis=-1)
        c_model = LA.norm(d_model, axis=-1)
        c_error = LA.norm(d_model-d, axis=-1)

        v = [
            (np.min(c), np.max(c)),
            (np.min(c), np.max(c)),
            (np.min(c_error), np.max(c_error))
        ]
        
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        
        t = t0
        mask = find_data_for_time(t, x, 3)
        
        scat1 = ax1.scatter(tf.gather(x_displaced, mask)[:,0], tf.gather(x_displaced, mask)[:,1], tf.gather(x_displaced, mask)[:,2], c=tf.gather(c, mask), cmap='RdBu_r', s=5)
        scat2 = ax2.scatter(tf.gather(x_displaced_model, mask)[:,0], tf.gather(x_displaced_model, mask)[:,1], tf.gather(x_displaced_model, mask)[:,2], c=tf.gather(c_model, mask), cmap='RdBu_r', s=5)
        scat3 = ax3.scatter(tf.gather(x, mask)[:,0], tf.gather(x, mask)[:,1], tf.gather(x, mask)[:,2], c=tf.gather(c_error, mask),
                        cmap='viridis', s=5)

        scatters = [scat1, scat2, scat3]
        axs = [ax1, ax2, ax3]
        titles = ["Ground Truth | t=%1.0f" + time_unit, "PINN Solution | t=%1.0f" + time_unit, "Error | t=%1.0f" + time_unit]
        
        for ax, title, scat, v_val in zip(axs, titles, scatters, v):
            self._setup_ax(ax, data_handler.min_dim, data_handler.max_dim, )
            fig.colorbar(scat, ax=ax, shrink=0.6, aspect=8, pad=0.1)
            ax.set_title(title % t)
            scat.set_clim(v_val[0], v_val[1])

        iterations = math.ceil((tcount)/step_size)
        print("Animation: %d of %d" % (0, iterations))
        
        def animate_scatters(iteration, scatters):
            n = (iteration)*step_size
            t = tvalues[n]
            
            print("Animation: %d of %d" % (iteration, iterations+1))
    
            mask = find_data_for_time(t, x, 3)
            
    
            for ax, title in zip(axs, titles):
                ax.set_title(title % t)
            
            scatters[0]._offsets3d = (tf.gather(x_displaced, mask)[:,0], tf.gather(x_displaced, mask)[:,1], tf.gather(x_displaced, mask)[:,2])
            scatters[0].set_array(tf.gather(c, mask))
    
            scatters[1]._offsets3d = (tf.gather(x_displaced_model, mask)[:,0], tf.gather(x_displaced_model, mask)[:,1], tf.gather(x_displaced_model, mask)[:,2])
            scatters[1].set_array(tf.gather(c_model, mask))
    
            scatters[2]._offsets3d = (tf.gather(x, mask)[:,0], tf.gather(x, mask)[:,1], tf.gather(x, mask)[:,2])
            scatters[2].set_array(tf.gather(c_error, mask))
    
            return scatters
    
        ani = animation.FuncAnimation(fig, animate_scatters, fargs=(scatters,),
                                      frames=iterations, blit=False)
    
        ani.save(filename, writer='pillow', fps=math.ceil(iterations/10), dpi=300)
        
    
    def animate_slices(self, data_handler, slice_positions, space_scale, time_scale = 1e0, time_unit = "ms", filename = "data_points.gif", fig = None, step_size = 1):

        if fig == None:
            fig = plt.figure(figsize=(12, 4))

        tvalues = data_handler.get_times(time_scale)
        t0, t1, tcount = tvalues[0], tvalues[-1], len(tvalues)
        
        x_displaced = [x for x,_,_ in data_handler.x_displaced]
        displacement = [d for d,_,_ in data_handler.displacement]
        c = [LA.norm(d, axis=-1) for d in displacement]

        v = (min([np.min(ci) for ci in c]), max([np.max(ci) for ci in c]))
        
        def plot_panes(ax, x_grid, y_grid, z_grid, alpha):
            X, Z = np.meshgrid(x_grid, z_grid)
            
            def func2(data, m):
                return m*np.ones([len(data[0]), len(data[1])])

            for y in y_grid:
                Y = func2(np.array([X, Z]), y)
                ax.plot_surface(X, Y, Z, linewidth=1, antialiased=True, alpha=alpha, color="grey")

        x_grid = np.linspace(-space_scale, space_scale, 2)
        z_grid = np.linspace(-space_scale, space_scale, 2)
        y_grid = slice_positions
        
        alpha = 0.3

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        axs = [ax1, ax2, ax3]

        angles = [(25, -15, 0), (0, 0, 0), (90, -90, 0)]

        plot_panes(ax1, x_grid, y_grid, z_grid, alpha)

        scatters = []
        
        for ax, angle in zip(axs, angles):
            self._setup_ax(ax, data_handler.min_dim, data_handler.max_dim, )
            scat = ax.scatter(x_displaced[0][:,0], x_displaced[0][:,1], x_displaced[0][:,2], c=c[0], cmap='RdBu_r', s=5)
            scat.set_clim(v[0], v[1])
            ax.view_init(elev=angle[0], azim=angle[1], roll=angle[2])
            scatters.append(scat)
        
        fig.colorbar(scat, ax=ax, shrink=0.6, aspect=8, pad=0.1)

        t = tvalues[0]*time_scale
        title = "Available Data | t=%1.0f" + time_unit
        fig.suptitle(title % t,fontsize=L_FONT_SIZE)

        iterations = math.ceil((tcount)/step_size)
        print("Animation: %d of %d" % (0, iterations))
        
        def animate_scatters(iteration, scatters):
            n = (iteration)*step_size
            t = tvalues[n]*time_scale
            
            print("Animation: %d of %d" % (iteration, iterations+1))

            fig.suptitle(title % t,fontsize=L_FONT_SIZE)
            
            for scat in scatters:
                scat._offsets3d = (x_displaced[n][:,0],      x_displaced[n][:,1],      x_displaced[n][:,2])
                scat.set_array(c[n])
    
            return scatters
    
        ani = animation.FuncAnimation(fig, animate_scatters, fargs=(scatters,),
                                      frames=iterations, blit=False)
    
        ani.save(filename, writer='pillow', fps=math.ceil(iterations/10), dpi=300)
        
    def animate_sample_points(self, data_handler, time_scale = 1e0, time_unit = "ms", filename = "data_points.gif", fig = None, step_size = 1):

        if fig == None:
            fig = plt.figure(figsize=(8, 4))
        
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')

        tvalues = data_handler.get_times(time_scale)
        t0, t1, tcount = tvalues[0], tvalues[-1], len(tvalues)

        x_displaced = [x for x,_,_ in data_handler.x_displaced]
        x_displaced_orig = [x for x,_,_ in data_handler.x_displaced_orig]

        displacement = [d for d,_,_ in data_handler.displacement]
        displacement_orig = [d for d,_,_ in data_handler.displacement_orig]
        
        c = [LA.norm(d, axis=-1) for d in displacement]
        c_orig = [LA.norm(d, axis=-1) for d in displacement_orig]

        scat1 = ax1.scatter(x_displaced_orig[0][:,0], x_displaced_orig[0][:,1], x_displaced_orig[0][:,2], c=c_orig[0], cmap='RdBu_r', s=5)
        scat2 = ax2.scatter(x_displaced[0][:,0],      x_displaced[0][:,1],      x_displaced[0][:,2],      c=c[0], cmap='RdBu_r', s=5)

        axs = [ax1, ax2]
        scatters = [scat1, scat2]
        titles = ["Ground Truth | t=%1.0f" + time_unit, "Available Data | t=%1.0f" + time_unit]

        t = tvalues[0]*time_scale
        for ax, scat, title in zip(axs, scatters, titles):
            self._setup_ax(ax, data_handler.min_dim, data_handler.max_dim, )
            fig.colorbar(scat, ax=ax, shrink=0.6, aspect=8, pad=0.1)
            ax.set_title(title % t0)

        iterations = math.ceil((tcount)/step_size)
        print("Animation: %d of %d" % (0, iterations))
        
        def animate_scatters(iteration, scatters):
            n = (iteration)*step_size
            t = tvalues[n]*time_scale
            
            print("Animation: %d of %d" % (iteration, iterations+1))

            for ax, scat, title in zip(axs, scatters, titles):
                ax.set_title(title % t)

            
            scatters[0]._offsets3d = (x_displaced_orig[n][:,0], x_displaced_orig[n][:,1], x_displaced_orig[n][:,2])
            scatters[0].set_array(c_orig[n])
    
            scatters[1]._offsets3d = (x_displaced[n][:,0],      x_displaced[n][:,1],      x_displaced[n][:,2])
            scatters[1].set_array(c[n])
    
            return scatters
    
        ani = animation.FuncAnimation(fig, animate_scatters, fargs=([scat1, scat2],),
                                      frames=iterations, blit=False)
    
        ani.save(filename, writer='pillow', fps=math.ceil(iterations/10), dpi=300)

    def animate_strain_error_plot(self, data_handler, model, time_scale = 1e0, time_unit = "ms", filename = "data_points_strain.gif", fig = None, step_size = 1, dpi=400):
        
        x = data_handler.x_mesh
        strain = data_handler.get_strain_orig() # TODO: can be fewer points than x_mesh

        print("strain plot: x|strain", x.shape, strain.shape)
        
        tvalues = data_handler.get_times(time_scale)
        t0, t1, tcount = tvalues[0], tvalues[-1], len(tvalues)
        
        strain_orig = data_handler.get_strain_orig()
        
        strain_model = []
        for t in tvalues:
            constant_t = tf.constant(t, shape=(x.shape[0], 1), dtype=x.dtype)
            xt = tf.concat([x, constant_t], axis=1)
            strain_model.append(model(xt))
        strain_model = np.array(strain_model, dtype=np.double)
        strain_error = strain_orig - strain_model
        
        c_model = LA.norm(strain_model, axis=-1)
        c_data = [LA.norm(dd, axis=-1) for dd in strain]
        c_error = LA.norm(strain_error, axis=-1)
        
        min_v = min(min([np.min(cd) for cd in c_data]), np.min(c_model))
        max_v = max(max([np.max(cd) for cd in c_data]), np.max(c_model))
        
        v = [
            (min_v, max_v),#(min([np.min(cd) for cd in c_data]), max([np.max(cd) for cd in c_data])),
            (min_v, max_v),#(np.min(c_model), np.max(c_model)),
            (np.min(c_error), np.max(c_error))
        ]
            

        if fig == None:
            fig = plt.figure(figsize=(12, 7))
        
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        axs = [ax1, ax2, ax3]
        titles = ["Data Source $|$ t=%1.0f" + time_unit, "PINN Solution $|$ t=%1.0f" + time_unit, "Error w.r.t. Ground Truth $|$ t=%1.0f" + time_unit]

        scat1 = ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c=c_data[0],  cmap='RdBu_r', s=5, vmin = v[0][0], vmax = v[0][1])
        scat2 = ax2.scatter(x[:, 0], x[:, 1], x[:, 2], c=c_model[0], cmap='RdBu_r', s=5, vmin = v[1][0], vmax = v[1][1])
        scat3 = ax3.scatter(x[:, 0], x[:, 1], x[:, 2], c=c_error[0], cmap = 'viridis', s=5, vmin = v[2][0], vmax = v[2][1])
        scatters = [scat1, scat2, scat3]

        t = t0
        for ax, title, scat, v_val in zip(axs, titles, scatters, v):
            self._setup_ax(ax, data_handler.min_dim, data_handler.max_dim, )
            fig.colorbar(scat, ax=ax, shrink=0.4, aspect=10, pad=0.1)
            ax.set_title(title % t)
            scat.set_clim(v_val[0], v_val[1])

        iterations = math.ceil((tcount)/step_size)

        #self._finish(filename = filename)

        print("Animation: %d of %d" % (0, iterations))
        
        
        def animate_scatters(iteration, scatters):
            n = (iteration)*step_size
            t = tvalues[n]
            
            print("Animation: %d of %d" % (iteration, iterations+1))
            
            for ax, title in zip(axs, titles):
                ax.set_title(title % t)
            
            scatters[0].set_array(c_data[n])
            scatters[1].set_array(c_model[n])
            scatters[2].set_array(c_error[n])
    
            return scatters
    
        ani = animation.FuncAnimation(fig, animate_scatters, fargs=(scatters,),
                                      frames=iterations, blit=False)
    
        ani.save(filename, writer='pillow', fps=math.ceil(iterations/10), dpi=dpi)
        plt.close(fig)
        

    def plot_slices(self, data_handler, model_field, true_values, midpoint, scalar_values = True, relative_error = False, binary = False,
                          filename = "data_points_strain.gif", dpi=400):
        
        rows, cols = 3,3
        
        index_to_label = {
            0: 'X',
            1: 'Y',
            2: 'Z'
        }
        
        x = data_handler.x_mesh
        x_min = data_handler.min_dim
        x_max = data_handler.max_dim
        
        field = model_field(tf.constant(x, dtype=get_dtype())).numpy()
        error = field - true_values
        if not scalar_values:
            field = np.linalg.norm(field, axis=-1)
            true_values = np.linalg.norm(true_values, axis=-1)
            error = np.linalg.norm(error, axis=-1)
        
        if relative_error:
            error /= true_values
        
        min_value, max_value = np.min(true_values), np.max(true_values)
        min_value_field, max_value_field = np.min(field), np.max(field)
        min_value_error, max_value_error = np.min(error), np.max(error)

        c_data_orig, c_data_field, c_data_error = [], [], []
        x_slices = []
        
        # get field values for each slice
        for row in range(rows):
            search_index = np.argmin(np.abs(x[:,row] - midpoint[row]))
            
            search_value = x[search_index,row]
            mask = x[:,row] == search_value
            x_slice = x[mask]
            x_slices += [x_slice]
            
            c_data_orig += [true_values[mask]]
            c_data_field += [field[mask]]
            c_data_error += [error[mask]]

        # setup colors
        if binary:
            cmap_plot = 'gray'
            cmap_error = 'coolwarm'
        else:
            cmap_plot = 'RdYlGn_r'
            cmap_error = 'seismic'#'viridis'
        
        vmin = min(min_value, min_value_field)
        vmax = max(max_value, max_value_field)

        if relative_error:
            vmax_error = 0.5
            vmin_error = -0.5
        else:
            vmax_error = max_value_error
            vmin_error = min_value_error
        
        if binary:
            vmax_error = 1.0
            vmin_error = -1.0
        
        vdata = [
            [vmin, vmax],
            [vmin, vmax],
            [vmin_error, vmax_error]
        ]

        cmap_plot = matplotlib.colormaps.get_cmap(cmap_plot)
        cmap_error = matplotlib.colormaps.get_cmap(cmap_error)
        
        norm_orig = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        norm_field = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        norm_error = matplotlib.colors.Normalize(vmin=vmin_error, vmax=vmax_error)

        # error bar extension
        extend_v_field = self._get_colorbar_extend(vmin, vmax, min_value_field, max_value_field)
        extend_v_error = self._get_colorbar_extend(vmin_error, vmax_error, min_value_error, max_value_error)
        
        num_ticks = 3
        if binary:
            ticks = [vmin, vmax]
            tick_labels = ["{:.2f}".format(vmin), "{:.2f}".format(vmax)]

            ticks_error = np.linspace(-1.0, 1.0, num=num_ticks)
            ticks_error_labels = ["{:.2f}".format(l) for l in ticks_error]
        else:
            avg = 0.5 * (min_value + max_value)
            #ticks = [vmin, min_val, avg, max_val, vmax] 
            ticks = np.linspace(vmin, vmax, num=num_ticks)
            #tick_labels = ["{:.2f}".format(vmin), "scar", "{:.2f}".format(avg), "healthy", "{:.2f}".format(vmax)]
            tick_labels = ["{:.2f}".format(l) for l in ticks]

            ticks_error = [vmin_error, 0.0, vmax_error]
            ticks_error_labels = ["{:.2f}".format(l) for l in ticks_error]

        ticks_plot = []
        for i in range(3):
            ticks_plot += [[x_min[i], midpoint[i], x_max[i]]]
        
        # plotting
        fig,axs = plt.subplots(figsize=(8, 6), ncols=3, nrows=3, constrained_layout=True)
        self._setup_error_col_layout(axs.flat, 
                                     data_handler.min_dim, data_handler.max_dim, 
                                     title_add = " sliced at \n[" + str(midpoint[0]) + "," + str(midpoint[1]) + "," + str(midpoint[2]) + "]",
                                    binary = binary, relative_error = relative_error, dim = 2)
        
        for row in range(rows):
            cdata = [
                cmap_plot(norm_orig(c_data_orig[row])),
                cmap_plot(norm_field(c_data_field[row])),
                cmap_error(norm_error(c_data_error[row]))
            ]
            
            for col in range(cols):
                
                ax = axs[row,col]
                sct = ax.scatter(x_slices[row][:, (row+1) % 3], x_slices[row][:, (row+2) % 3], c=cdata[col], s=5)
                sct.set_clim(vdata[col][0], vdata[col][1])
        
                ax.set_xlim([x_min[row], x_max[row]])
                ax.set_xlabel(index_to_label[(row+1) % 3], labelpad=.0)
                ax.set_xticks(ticks_plot[(row+1) % 3])
        
                ax.set_ylim([x_min[row], x_max[row]])
                ax.set_ylabel(index_to_label[(row+2) % 3], labelpad=.0)
                ax.set_yticks(ticks_plot[(row+2) % 3])

                ax.tick_params(axis='both', which='major', labelsize=self.FONT_SIZE*0.75)
        
                ax.grid(visible=True, color='grey',
                     linestyle='-.', linewidth=0.3,
                     alpha=0.2)
        
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_orig, cmap=cmap_plot), location='left',
                 ax=axs[:,0], orientation='vertical', shrink = 0.6, aspect = 16, pad = 0.15, ticks = ticks, extend = extend_v_field)
        cbar.ax.set_yticklabels(tick_labels)
        
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_error, cmap=cmap_error), location='right',
                 ax=axs[:,-1], orientation='vertical', shrink = 0.6, aspect = 16, pad = 0.15, ticks = ticks_error, extend = extend_v_error)
        cbar.ax.set_yticklabels(ticks_error_labels)
        
        self._finish(fig, filename = filename, draw = False, block = False, dpi = 400)
        print('plot field completed')

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