import numpy as np
from carputils.carpio import igb
import os
import csv
import tensorflow as tf
import nisaba as ns
from Codebase.myutility import train_test_split

def read_pts_file(filename):
    """
    read coordinates/points from file
    """
    with open(filename, "r") as f_pts:
        npts = int(f_pts.readline().strip())
        data = f_pts.read().replace('\n', ' ')
        data = np.fromstring(data, dtype=np.double, sep=' ')/1000 # um to mm (CARP mesh)
        data = np.reshape(data, (npts, 3))
        return data

def read_tens_txt_file(filename): #adapted from  carputils/carpio/txt.py
    """
    read tensor from file .txt
    """
    with open(filename, "r") as f_txt:
        data = f_txt.read().replace('\n', ' ')
        data = np.fromstring(data, dtype=np.double, sep=' ')
        print(data.shape[0]/9)
        npts = int(round(data.shape[0]/9))
        data = np.reshape(data, (npts, 9))
    
    return data   

def read_tens_file(filename):
    """
    read tensor from file .igb
    """
    space_dims = 3

    igb_file = igb.IGBFile(filename)

    header = igb_file.header()
    data = igb_file.data()

    num_tsteps = header.get('t')
    num_traces = header.get('x')

    npts = data.shape[0]
    if npts % (space_dims**2) != 0:
        raise Error('File '+ filename+' not compatible with space dimensions.')
    npts = int(round(npts/(space_dims**2)))
    if npts % num_tsteps != 0:
        raise Error('File '+ filename+' not compatible with time steps.')
    
    data = np.reshape(data, (num_tsteps, num_traces, space_dims**2))
    
    return data, header

def read_tdat_file(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        data = [[np.double(x.strip()) for x in row] for row in reader]

    return data

def get_final_displacement(folder_base, tn = None):
    """
    get final displacement of a simulation
    """
    dynpt = os.path.join(folder_base, "x.dynpt")
    displ = igb.IGBFile(dynpt)
    header = displ.header()
    u_data = displ.data()
    num_traces = header.get('x')
    num_tsteps = header.get('t')
    num_comp = 0
    if header.get('type') == 'vec3f':
        num_comp = 3
    else:
        print("Wrong number of components in displ file")
        return np.array([])  # return empty array
    # Check if simulation failed
    if len(u_data) < num_tsteps*num_traces*num_comp:
        return np.array([])  # return empty array
    if tn is not None:
        u_data = u_data.reshape(num_tsteps, num_traces * num_comp)
        final_pts = u_data[tn]/1000 # um to mm (CARP solution)
        final_pts = final_pts.reshape(num_traces, num_comp)
    else:
        u_data = u_data.reshape(num_tsteps, num_traces, num_comp)
        final_pts = u_data / 1000  # um to mm (CARP solution)

    return final_pts

def get_displacement_data(data_path, data_path_tdat, t0, t1, num_data, num_test, SNR, np_random_generator, slice_axis = None, slice_resolution = None, slice_tolerance = None, dx_old = None, dx_new = None):
    space_dims = 3

    # file reading
    ################
    # space points shape (p_num, 3)
    x_data_tot_orig = read_pts_file(data_path + '/block.pts')
    # final displacement shape (t_num, p_num, 3)
    x_displ_orig = get_final_displacement(data_path)

    d_data_tot_orig = x_displ_orig - x_data_tot_orig
    
    if not dx_old is None and not dx_new is None:
        min_lengths = np.min(x_data_tot_orig, axis=0)
        max_lengths = np.max(x_data_tot_orig, axis=0)
        lengths = max_lengths - min_lengths
        for t in range(d_data_tot_orig.shape[0]):
            d_data_tot_orig[t] = get_avg_data(lengths[0], lengths[1], lengths[2], d_data_tot_orig[t], dx_old, dx_new, space_dims)

    d_data_tot = add_noise(d_data_tot_orig, SNR, space_dims, np_random_generator)
    
    # list of time points with length t_num
    tdat = read_tdat_file(data_path_tdat)

    # extract time
    time_orig = np.array([row[0] for row in tdat])
    lower_tidx = tf.where(tf.greater_equal(time_orig, t0)).numpy()[0][0]
    upper_tidx = tf.where(tf.less_equal(time_orig, t1)).numpy()[-1][0]

    # time = time_orig[lower_tidx:upper_tidx+1]

    # compute data displacement
    ################

    # sampling
    # n_t_data_tot = len(time)
    n_x_data_tot = x_displ_orig.shape[1]
    n = num_data + num_test

    random_idx_t = np_random_generator.choice(np.arange(lower_tidx, upper_tidx + 1), n)
    random_idx_x = np_random_generator.choice(n_x_data_tot, n)
    random_idx_xt = np.column_stack((random_idx_x, random_idx_t))
    random_idx_xt = random_idx_xt[random_idx_xt[:, 1].argsort()]

    # data
    t_data = np.array(time_orig[random_idx_xt[:, 1]])
    x_data = x_data_tot_orig[random_idx_xt[:, 0]]

    # extract displacement
    d_data = d_data_tot[random_idx_xt[:, 1], random_idx_xt[:, 0]]
    
    d_data_plot = d_data_tot_orig[lower_tidx:upper_tidx + 1]
    d_data_plot_corrupted = d_data_tot[lower_tidx:upper_tidx + 1]

    # data formatting
    ################
    # merge space and time, shape (n, 4)
    xt_data = np.column_stack((x_data, t_data))

    # processing

    # split into test and train
    # choose again random indices from 0...num_data+num_test-1
    [xt_data, displ_data], [xt_test, displ_test] = train_test_split([xt_data, d_data], num_data, num_test, np_random_generator)
    
    xt_test = tf.convert_to_tensor(xt_test, dtype=ns.config.get_dtype())
    xt_data = tf.convert_to_tensor(xt_data, dtype=ns.config.get_dtype())

    displ_data = tf.convert_to_tensor(displ_data, dtype=ns.config.get_dtype())
    displ_test = tf.convert_to_tensor(displ_test, dtype=ns.config.get_dtype())

    return xt_test, displ_test, xt_data, displ_data, x_data_tot_orig, d_data_plot, x_data_tot_orig, d_data_plot_corrupted 

def get_displacement_data_at_timepoint(data_path, tn, num_data, num_test, SNR, np_random_generator, slice_axis = None, slice_resolution = None, slice_tolerance = None, dx_old = None, dx_new = None):
    space_dims = 3
    x_data_tot_orig = read_pts_file(data_path + '/block.pts')
    x_displ_orig = get_final_displacement(data_path)[tn]

    d_data_tot_orig = x_displ_orig - x_data_tot_orig

    if not dx_old is None and not dx_new is None:
        min_lengths = np.min(x_data_tot_orig, axis=0)
        max_lengths = np.max(x_data_tot_orig, axis=0)
        lengths = max_lengths - min_lengths
        d_data_tot_orig = get_avg_data(lengths[0], lengths[1], lengths[2], d_data_tot_orig, dx_old, dx_new, space_dims)
    
    d_data_tot = add_noise(d_data_tot_orig, SNR, space_dims, np_random_generator)

    if not slice_axis is None and not slice_resolution is None and not slice_tolerance is None:
        x_displ, mask = restrict_to_slices(x_displ_orig, slice_axis, slice_resolution, slice_tolerance)
        x_data_tot = x_data_tot_orig[mask]
        d_data_tot = d_data_tot[mask]
    else:
        x_data_tot = x_data_tot_orig
    
    n_data_tot = x_data_tot.shape[0]

    random_idx = np_random_generator.choice(n_data_tot, num_data + num_test)
    x_data = tf.convert_to_tensor(x_data_tot[random_idx[:num_data]], dtype=ns.config.get_dtype())
    d_data = tf.convert_to_tensor(d_data_tot[random_idx[:num_data]], dtype=ns.config.get_dtype())
    x_test = tf.convert_to_tensor(x_data_tot[random_idx[-num_test:]], dtype=ns.config.get_dtype())
    d_test = tf.convert_to_tensor(d_data_tot[random_idx[-num_test:]], dtype=ns.config.get_dtype())

    return x_test, d_test, x_data, d_data, x_data_tot_orig, d_data_tot_orig, x_data_tot, d_data_tot

def get_matrix_data_at_timepoint(data_path, matrix_data_file, tn, num_data, num_test, SNR, np_random_generator):
    #import pdb; pdb.set_trace()
    space_dims = 3
    x_data_tot_orig = read_pts_file(data_path + '/block.pts')
    if matrix_data_file.split('.')[-1] == "txt":
        matrix_data_tot_orig = read_tens_txt_file(data_path + '/' + matrix_data_file)
    else:
        matrix_data_tot_orig, header = read_tens_file(data_path + '/' + matrix_data_file)
    #print(matrix_data_tot)
    matrix_data_tot_orig = matrix_data_tot_orig[tn]
    #print(matrix_data_tot.shape)
    
    n_data_tot = matrix_data_tot_orig.shape[0]

    if n_data_tot != x_data_tot_orig.shape[0]:
        raise Error('File ' + matrix_data_file + ' not compatible with mesh.')

    matrix_data_tot_orig = add_noise(matrix_data_tot_orig, SNR, space_dims, np_random_generator)
    
    #x_data_tot, mask = restrict_to_interior(x_data_tot_orig)
    x_data_tot = x_data_tot_orig
    matrix_data_tot = matrix_data_tot_orig
    
    exclude_pts = 0 # remove first 5 points on base (singularity points)
    n_data_tot = matrix_data_tot.shape[0]
    
    random_idx = np_random_generator.choice(np.arange(exclude_pts,n_data_tot),  (num_data + num_test))
    
    matrix_data = tf.convert_to_tensor(matrix_data_tot[random_idx[:num_data]], dtype=ns.config.get_dtype())
    matrix_test = tf.convert_to_tensor(matrix_data_tot[random_idx[-num_test:]], dtype=ns.config.get_dtype())

    x_data = tf.convert_to_tensor(x_data_tot[random_idx[:num_data]], dtype=ns.config.get_dtype())
    x_test = tf.convert_to_tensor(x_data_tot[random_idx[-num_test:]], dtype=ns.config.get_dtype())
    
    return x_test, matrix_test, x_data, matrix_data, x_data_tot, matrix_data_tot

def get_Sa_data(data_path_tdat, t0, t1, num_data, num_test, SNR, np_random_generator):
    tdat = read_tdat_file(data_path_tdat)

    # extract data
    time_orig = np.array([row[0] for row in tdat])
    Sa_orig = np.array([row[1] for row in tdat])

    lower_tidx = tf.where(tf.greater_equal(time_orig, t0)).numpy()[0][0]
    upper_tidx = tf.where(tf.less_equal(time_orig, t1)).numpy()[-1][0]

    max_Sa_data = tf.reduce_max(Sa_orig, axis=0)
    if SNR > 0.0:
        sigma = SNR * max_Sa_data
        noise = np_random_generator.normal(0, sigma, Sa_orig.shape)
        Sa_orig = Sa_orig + noise
    
    # sampling
    n = num_data + num_test

    random_idx = np_random_generator.choice(np.arange(lower_tidx, upper_tidx + 1), n)

    # data
    t_data = np.array(time_orig[random_idx])
    Sa_data = np.array(Sa_orig[random_idx])

    t_data_test = tf.convert_to_tensor(t_data[-num_test:], dtype=ns.config.get_dtype())
    t_data      = tf.convert_to_tensor(t_data[:num_data], dtype=ns.config.get_dtype())

    Sa_data_test = tf.expand_dims(tf.convert_to_tensor(Sa_data[-num_test:], dtype=ns.config.get_dtype()), -1)
    Sa_data      = tf.expand_dims(tf.convert_to_tensor(Sa_data[:num_data], dtype=ns.config.get_dtype()), -1)

    return t_data_test, Sa_data_test, t_data, Sa_data, time_orig, Sa_orig
    
def add_noise(data, SNR, space_dims, np_random_generator):
    if SNR > 0.0:
        max_data = np.max(data)
        sigma = SNR * max_data / space_dims
        noise = np_random_generator.normal(0, sigma, data.shape)
        data_noise = data + noise
    else:
        data_noise = data
    
    return data_noise

def restrict_to_slices(data, slice_axis, resolution, tolerance):
    #import pdb; pdb.set_trace()
    
    mask = np.minimum(np.abs(data[:, slice_axis] % resolution), np.abs(resolution - data[:, slice_axis] % resolution)) <= tolerance
    #mask = np.in1d(x_displ[:, slice_axis], np.asarray(slice_axis_values))
    return data[mask], mask

def restrict_to_interior(data):
    mask1 = np.all(np.less(np.min(data, axis=0), data), axis=-1)
    mask2 = np.all(np.less(data, np.max(data, axis=0)), axis=-1)
    mask = mask1 & mask2
    return data[mask], mask

def get_avg_data(L, W, H, data_pts, dx_old, dx_new, dim):
    #Size of the cubic subregion
    #subregion_size = 0.4
    #No. of subregions in each dimension
    x_dim = int(L / dx_old) + 1
    y_dim = int(W / dx_old) + 1
    z_dim = int(H / dx_old) + 1

    data_pts = data_pts.reshape(x_dim, y_dim, z_dim, dim, order = 'F') # column-major order

    subregions_per_dimx = int(L / dx_new)
    subregions_per_dimy = int(W / dx_new)
    subregions_per_dimz = int(H /  dx_new)

    factor = int(dx_new/dx_old)
    
    avg_data = np.zeros((x_dim, y_dim,z_dim, dim))
    # Calculate average data in subregions

    for i in range(subregions_per_dimx):
        for j in range(subregions_per_dimy):
            for k in range(subregions_per_dimz):
    
                x_start = i * factor
                x_end = (i + 1) * factor
                y_start = j * factor
                y_end = (j + 1) * factor
                z_start = k * factor
                z_end = (k + 1) * factor
                
                subregion = data_pts[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1, :]
                #import pdb; pdb.set_trace()
                avg_subregion = np.array([[[np.mean(subregion, axis=(0, 1, 2) )]]])
                avg_data[x_start:x_end +1, y_start:y_end+1, z_start:z_end+1, :] = avg_subregion
                #import pdb; pdb.set_trace()
            
    avg_data = avg_data.reshape(x_dim * y_dim * z_dim, dim, order = 'F')
    print(avg_data)
    
    return avg_data


