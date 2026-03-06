import numpy as np
import tensorflow as tf
import pinn_source.pinn_lib as ns
import copy
from pinn_source.network import load_network


def _train_test_split(data, num_train, num_test, np_random_generator, ordered=False):
    if ordered:
        data_return = data[:num_train]
        data_test = data[-num_test:]
        idx_data = np.arange(num_train)
        idx_test = np.arange(num_train, num_test)
    else:
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


class FEMDataHandler:

    def __init__(self, data_file: str = None):
        self.data_file = data_file

#############################################################################
# Data reading
#############################################################################

    def read(self, model_file: str = None):

        data = np.loadtxt(self.data_file, delimiter=",", skiprows=1, dtype=np.double)
        indices = np.lexsort((data[:,0], data[:,1], data[:,2]))
        # sort mesh to simplify calculations later on
        data = data[indices]

        # read mesh
        self.x_mesh = data[:,:3]

        # calculate mesh characteristics
        self.min_dim = np.min(self.x_mesh, axis=0)
        self.max_dim = np.max(self.x_mesh, axis=0)
        self.mesh_sizes = self.max_dim - self.min_dim
        self.mesh_dimension = self.x_mesh.shape[-1]

        # assuming regular rectangular grid !
        self.points_per_dimension = [np.unique(self.x_mesh[:, i]).shape[0] for i in range(self.mesh_dimension)]

        # read displaced configuration as list [timepoint: (coordinates, mesh indices, time index)]
        if model_file is not None:
            self.model = load_network(model_file)
            self.model_file = model_file
            self.x_displaced_orig = [(self.model(self.x_mesh).numpy(), range(self.x_mesh.shape[0]), 0)]
        else:
            self.x_displaced_orig = [(data[:,:3]+data[:,3:6], range(self.x_mesh.shape[0]), 0)]
            self.tag_values = np.expand_dims(data[:,6],-1)

        self.x_displaced_indices = [(np.arange(0, self.x_mesh.shape[0]), 0)]
        self.x_displaced = copy.deepcopy(self.x_displaced_orig)

        # calculate displacement
        self.calc_displacement(recalc_orig=True)

        # submeshes in reference configuration
        self.submeshes_reference_conf = {}

        self.slice('nxminus', 0, self.min_dim[0])
        self.slice('nxplus', 0, self.max_dim[0])

        self.slice('nyminus', 1, self.min_dim[1])
        self.slice('nyplus', 1, self.max_dim[1])

        self.slice('nzminus', 2, self.min_dim[2])
        self.slice('nzplus', 2, self.max_dim[2])

        self.submeshes_reference_conf['low_resolution'] = np.arange(0, self.x_mesh.shape[0], step=4)

    def set_regions(self):
        """Auto-detect piecewise-constant regions from tag_values."""
        vals = self.tag_values.flatten()
        unique_vals = np.unique(np.round(vals, decimals=6))

        # Assign each point to nearest unique value
        distances = np.abs(vals[:, None] - unique_vals[None, :])
        nearest = np.argmin(distances, axis=1)

        self.tag_data = nearest + 1  # 1-based tags
        self.tag_dict = {i + 1: float(v) for i, v in enumerate(unique_vals)}
        self.name_dict = {i + 1: f"region_{i + 1}" for i in range(len(unique_vals))}

#############################################################################
# Getter
#############################################################################

    def get_times(self, time_scale):
        return [tidx*time_scale for _, _, tidx in self.x_displaced]

    def get_mesh(self):
        return [self.x_mesh[idx] for x, idx, tidx in self.x_displaced]

    def get_mesh_points_for_tag(self, tag):
        mask = (self.tag_data == tag).flatten()
        return self.x_mesh[mask], mask, self.tag_values[mask]

    def get_name_for_tag(self, tag):
        if tag in self.name_dict:
            return self.name_dict[tag]
        return tag

    def get_displacement(self):
        displacement = [d for d,_,_ in self.displacement]
        return np.array(displacement, dtype=np.double)

    def get_displacement_orig(self):
        displacement = [d for d,_,_ in self.displacement_orig]
        return np.array(displacement, dtype=np.double)

    def get_x_displaced(self):
        return [x for x,_,_ in self.x_displaced]

    def get_max_displacement_components(self):
        d_max = np.array([d for d,_ in self.max_displacement], dtype=np.double)
        return np.max(d_max, axis=0)

#############################################################################
# Sampling
#############################################################################

    def get_random_indices(self, num_train, num_test, np_random_generator):
        total_indices = self.x_displaced_indices

        spacepoints_per_time = [len(idx) for idx, _ in total_indices]
        total_points = sum(spacepoints_per_time)

        n = num_train + num_test

        random_idx = np_random_generator.choice(np.arange(total_points), n, replace=False)

        def idx_to_point(idx):
            t_idx = 0
            offset = 0
            while offset <= idx:
                offset += spacepoints_per_time[t_idx]
                t_idx += 1

            t_idx -= 1
            offset -= spacepoints_per_time[t_idx]

            return idx - offset, t_idx

        # get relative index for total_indices
        indices = [idx_to_point(idx) for idx in random_idx]

        # convert to mesh index
        indices = [(total_indices[tn][0][x],tn) for x,tn in indices]

        indices.sort(key=lambda i: i[0])

        return np.array(indices)

    def get_random(self, num_train, num_test, time_scale, np_random_generator):

        indices = self.get_random_indices(num_train, num_test, np_random_generator)
        n = num_train + num_test

        x_displaced = [(self.x_displaced[t][0][x], self.x_displaced[t][1][x], self.x_displaced[t][2]*time_scale) for x,t in indices]

        x_mesh = [np.append(self.x_mesh[xidx], t) for _, xidx, t in x_displaced]
        displacement = [x_displaced[i][0] - x_mesh[i][:3] for i in range(n)]

        x_displaced_tensor = tf.constant([x[0] for x in x_displaced], dtype=ns.config.get_dtype())
        x_mesh_tensor = tf.constant(x_mesh, dtype=ns.config.get_dtype())
        displacement_tensor = tf.constant(displacement, dtype=ns.config.get_dtype())

        data = [x_mesh_tensor, x_displaced_tensor, displacement_tensor]

        data_return, data_test, idx_data, idx_test = _train_test_split(data, num_train, num_test, np_random_generator)

        self.train_data = {
            'x_mesh': data_return[0],
            'x_displaced': data_return[1],
            'displacement': data_return[2],
            'indices': indices[idx_data]
        }

        self.test_data = {
            'x_mesh': data_test[0],
            'x_displaced': data_test[1],
            'displacement': data_test[2],
            'indices': indices[idx_test]
        }

        return data_test[0], data_test[1], data_test[2], data_return[0], data_return[1], data_return[2]

#############################################################################
# Utilities
#############################################################################

    def apply_noise(self, C, dim, np_random_generator):
        if C > 0.0:

            max_d = self.max_displacement[0][0]
            for d, _ in self.max_displacement:
                for i in range(d.shape[-1]):
                    max_d[i] = max(abs(d[i]), max_d[i])
            sigma = C * max_d / dim

            self.x_displaced = self._apply_noise(sigma, self.x_displaced, np_random_generator)

            self.calc_displacement()

    def _apply_noise(self, sigma, data, np_random_generator):
        for i, (x, _, _) in enumerate(data):
            noise = np_random_generator.normal(0, sigma, x.shape)

            data[i] = (data[i][0] + noise, data[i][1], data[i][2])

        return data

    def calc_displacement(self, recalc_orig = False):
        self.displacement = [(x - self.x_mesh[idx], idx, tidx) for x, idx, tidx in self.x_displaced]
        self.max_displacement = [(np.max(d, axis=0), tidx) for d, _, tidx in self.displacement]

        if recalc_orig:
            self.displacement_orig = copy.deepcopy(self.displacement)
            self.max_displacement_orig = copy.deepcopy(self.max_displacement)

    def slice(self, name, axis, value):
        self.mesh_sizes = self.max_dim - self.min_dim
        self.mesh_dimension = self.x_mesh.shape[-1]
        h = np.min([self.mesh_sizes[i] / (self.points_per_dimension[i] - 1) for i in range(self.mesh_dimension)])

        mask = np.abs(self.x_mesh[:,axis] - value) < h*1e-3
        indices = np.arange(0, self.x_mesh.shape[0])
        self.submeshes_reference_conf[name] = indices[mask]