import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from plot_settings import *
import matplotlib.animation as animation
import math
import tensorflow as tf

def train_test_split(data, num_train, num_test, np_random_generator, ordered = False):
    if ordered:
        data_return = data[:num_train]
        data_test = data[-num_test:]
        idx_data = np.arange(num_train)
        idx_test = np.arange(num_train, num_test)
    else:
        # choose again random indices from 0...num_data+num_test-1
        idx = np.arange(num_train + num_test)
        np_random_generator.shuffle(idx)
        idx_data = np.sort(idx[:num_train])
        idx_test = np.sort(idx[-num_test:])
    
        if isinstance(data, list):
            data_test = []
            data_return = []
            for d in data:
                data_test.append(tf.gather(d, indices=idx_test))
                data_return.append(tf.gather(d, indices=idx_data))
            
        else:
            data_test = tf.gather(data, indices=idx_test)
            data_return = tf.gather(data, indices=idx_data)

    return data_return, data_test, idx_data, idx_test



def convert_to_serializable(mydict):
    for key in mydict:
        if isinstance(mydict[key], dict):
            convert_to_serializable(mydict[key])
        elif tf.is_tensor(mydict[key]):
            mydict[key] = mydict[key].numpy().tolist()


def plot_face(x_data, d_data, N_data, model, axis1, axis2, vmin=0, vmax=5, draw=False, block=False, title=None,
              filename=None, ):
    # if ax is None:
    #   _,  ax = plt.subplots() # ax = plt.axes(projection = "3d") #

    fig = plt.figure(figsize=(3, 3))

    data_x = np.reshape(x_data[:, 0], N_data)
    data_y = np.reshape(x_data[:, 1], N_data)
    data_z = np.reshape(x_data[:, 2], N_data)

    grid = tf.stack([data_x, data_y, data_z], axis=-1)
    d = model(grid).numpy()
    print('max(disp)')
    print(np.max(d))

    dist_x = np.reshape(x_data[:, 0] + d[:, 0], N_data)
    dist_y = np.reshape(x_data[:, 1] + d[:, 1], N_data)
    dist_z = np.reshape(x_data[:, 2] + d[:, 2], N_data)
    norm_dist = LA.norm(d - d_data, axis=1)

    xmin, xmax = plt.xlim()

    sct2 = plt.scatter(np.reshape(x_data[:, axis1], N_data), np.reshape(x_data[:, axis2], N_data), c=norm_dist,
                       cmap='RdBu_r', s=5)
    # sct2 = ax.scatter3D(dist_x, dist_y, dist_z, c = norm_dist, vmin=0, vmax=20, cmap = 'viridis', s = 5)
    fig.colorbar(sct2, shrink=0.6, aspect=8, pad=0.1)
    sct2.set_clim(vmin, vmax)

    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename, dpi=600)
    if draw:
        plt.draw()
        plt.pause(1e-16)
    if block:
        plt.show(block=True)
    print('plot solution completed')


def plot_data(x_data, d_data, N_data, x_data_tot, d_data_tot, N_data_tot, scale_factor, vmin = 0, vmax = 5, draw = True, block = False, title = None, filename = None):

   
    #if ax is None:
     #   _,  ax = plt.subplots() # ax = plt.axes(projection = "3d") #
     
    fig = plt.figure(figsize = (16, 5))

    #grid_x, grid_y, grid_z = np.meshgrid(np.linspace(x0, Wx, nx),  np.linspace(y0, Wy, ny) ,  np.linspace(z0, Wz, nz) )
    
    # Add x, y gridlines
    def plot_view(ax, angle = None): 
        ax.grid(b = True, color ='grey',
            	linestyle ='-.', linewidth = 0.3,
            	alpha = 0.2)
    
        data_x_tot = np.reshape(x_data_tot[:,0], N_data_tot)
        data_y_tot = np.reshape(x_data_tot[:,1], N_data_tot)
        data_z_tot = np.reshape(x_data_tot[:,2], N_data_tot)
    
        dist_x_tot = np.reshape(x_data_tot[:,0] + d_data_tot[:,0] , N_data_tot)
        dist_y_tot = np.reshape(x_data_tot[:,1] + d_data_tot[:,1] , N_data_tot)
        dist_z_tot = np.reshape(x_data_tot[:,2] + d_data_tot[:,2] , N_data_tot)
        disp_init_tot = LA.norm(d_data_tot, axis = 1)
        #disp_init = np.zeros(N_data)
        
        ax.scatter3D(dist_x_tot, dist_y_tot, dist_z_tot,  c = 'black', s = 2, alpha=0.01)
    
        data_x = np.reshape(x_data[:,0], N_data)
        data_y = np.reshape(x_data[:,1], N_data)
        data_z = np.reshape(x_data[:,2], N_data)
    
        dist_x = np.reshape(x_data[:,0] + d_data[:,0] , N_data)
        dist_y = np.reshape(x_data[:,1] + d_data[:,1] , N_data)
        dist_z = np.reshape(x_data[:,2] + d_data[:,2] , N_data)
        disp_init = LA.norm(d_data, axis = 1)
        #disp_init = np.zeros(N_data)
        
        sct1 = ax.scatter3D(dist_x, dist_y, dist_z,  c = disp_init, cmap = 'RdBu_r', s = 5)
    
        
        sct1.set_clim(vmin,vmax)
           
        xmin, xmax = plt.xlim()
        if angle is None:
            ax.set_zlim(np.min(data_z_tot)* scale_factor, np.max(data_z_tot) * scale_factor)

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

    #XY
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.set_proj_type('ortho')
    sct1 = plot_view(ax, [90, -90, 0])
    ax.set_zticklabels([])
    ax.set_zlabel('')
    
    fig.colorbar(sct1, ax = ax, shrink = 0.6, aspect = 8, pad = 0.1)
    fig.suptitle('Data Points',fontsize=L_FONT_SIZE)  

    if filename is not None:
        plt.savefig(filename, dpi=600 )
    if draw:
        plt.draw()
        plt.pause(1e-16)
    if block:
        plt.show(block = True) 
    print('plot data points completed')

def animate_time_dependent_scatter(fig, x_total_orig, u_total_orig, x_data_orig, x_data_total, t_data_total, model,
                                   vmin=0, vmax=None,
                                   time_resolution=5, scale=0.2, filename='filename.gif'):
    
    constant_t_orig = tf.constant(t_data_total[0], shape=(x_total_orig.shape[0], 1), dtype=x_total_orig.dtype)
    constant_t = tf.constant(t_data_total[0], shape=(x_data_total.shape[0], 1), dtype=x_data_total.dtype)

    xt_total_orig = tf.concat([x_total_orig, constant_t_orig], axis=1)
    xt_total = tf.concat([x_data_total, constant_t], axis=1)

    u_data_model_total = model(xt_total_orig)
    u_data_orig_total = x_data_orig - x_data_total
    
    u_data_model = model(xt_total)
    u_data_orig = x_data_orig - x_data_total

    x_data_model = x_data_total + u_data_model
    x_data_error = u_data_orig[0] - u_data_model

    c_data_orig = np.array([LA.norm(u_t, axis=-1) for u_t in u_data_orig[:]], dtype=float)

    if vmax is None:
        vmax = np.max(c_data_orig)

    c_data_model = LA.norm(u_data_model, axis=-1)
    c_data_error = LA.norm(x_data_error, axis=-1)

    vmin_error = 0
    vmax_error = vmax*0.2

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    scat1 = ax1.scatter(x_data_orig[0, :, 0], x_data_orig[0, :, 1], x_data_orig[0, :, 2], c=c_data_orig[0],
                        cmap='RdBu_r', s=5)
    scat2 = ax2.scatter(x_data_model[:, 0], x_data_model[:, 1], x_data_model[:, 2], c=c_data_model, cmap='RdBu_r', s=5)
    scat3 = ax3.scatter(x_data_total[:, 0], x_data_total[:, 1], x_data_total[:, 2], c=c_data_error, cmap = 'viridis', s=5, vmin = vmin_error, vmax = vmax_error)
    
    scat1.set_clim(vmin, vmax)
    scat2.set_clim(vmin, vmax)
    scat3.set_clim(vmin_error, vmax_error)

    fig.colorbar(scat1, ax=ax1, shrink=0.6, aspect=8, pad=0.1)
    fig.colorbar(scat2, ax=ax2, shrink=0.6, aspect=8, pad=0.1)
    fig.colorbar(scat3, ax=ax3, shrink=0.6, aspect=8, pad=0.1)

    ax1.set_xlim3d([-scale, scale])
    ax2.set_xlim3d([-scale, scale])
    ax3.set_xlim3d([-scale, scale])
    ax1.set_xlabel('X')
    ax2.set_xlabel('X')
    ax3.set_xlabel('X')

    ax1.set_ylim3d([-scale, scale])
    ax2.set_ylim3d([-scale, scale])
    ax3.set_ylim3d([-scale, scale])
    ax1.set_ylabel('Y')
    ax2.set_ylabel('Y')
    ax3.set_ylabel('Y')

    ax1.set_zlim3d([-scale, scale])
    ax2.set_zlim3d([-scale, scale])
    ax3.set_zlim3d([-scale, scale])
    ax1.set_zlabel('Z')
    ax2.set_zlabel('Z')
    ax3.set_zlabel('Z')

    ax1.set_title("Ground Truth | t=%1.4f" % t_data_total[0])
    ax2.set_title("Model | t=0")
    ax3.set_title("Error | t=0")

    ax1.grid(b=True, color='grey',
             linestyle='-.', linewidth=0.3,
             alpha=0.2)
    ax2.grid(b=True, color='grey',
             linestyle='-.', linewidth=0.3,
             alpha=0.2)
    ax3.grid(b=True, color='grey',
             linestyle='-.', linewidth=0.3,
             alpha=0.2)

    time_steps = x_data_orig.shape[0]

    iterations = math.floor((time_steps - 1) / time_resolution)

    scatters = [scat1, scat2, scat3]

    def animate_scatters(iteration, scatters):
        t = time_resolution * iteration
        print("Animation: %d of %d" % (iteration, iterations))

        ax1.set_title("Ground Truth | t=%1.4f" % (t_data_total[t]))
        ax2.set_title("PINN Solution | t=%1.4f" % (t_data_total[t]))
        ax3.set_title("Error | t=%1.4f" % (t_data_total[t]))

        constant_t = tf.constant(t_data_total[t], shape=(x_data_total.shape[0], 1), dtype=x_data_total.dtype)
        xt_total = tf.concat([x_data_total, constant_t], axis=1)

        u_data_model = model(xt_total)
        x_data_model = x_data_total + u_data_model
        x_data_error = u_data_orig[t] - u_data_model

        c_data_model = LA.norm(u_data_model, axis=-1)
        c_data_error = LA.norm(x_data_error, axis=-1)

        scatters[0]._offsets3d = (x_data_orig[t, :, 0], x_data_orig[t, :, 1], x_data_orig[t, :, 2])
        scatters[0].set_array(c_data_orig[t])

        scatters[1]._offsets3d = (x_data_model[:, 0], x_data_model[:, 1], x_data_model[:, 2])
        scatters[1].set_array(c_data_model)

        scatters[2]._offsets3d = (x_data_total[:, 0], x_data_total[:, 1], x_data_total[:, 2])
        scatters[2].set_array(c_data_error)

        return scatters

    ani = animation.FuncAnimation(fig, animate_scatters, fargs=(scatters,),
                                  frames=iterations, blit=False)

    ani.save(filename, writer='pillow', fps=math.ceil(iterations/10), dpi=300)