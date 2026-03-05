import numpy as np
from carputils.carpio import igb
import csv
import tensorflow as tf
import nisaba as ns
import warnings
import copy
from Codebase.myutility import train_test_split
from functools import reduce

class FEMDataHandler:

    def __init__(self, displacement_file, mesh_file, time_scale = 2.0):
        """
        initialise
        """
        self.displacement_file = displacement_file
        self.mesh_file = mesh_file
        self.has_strain = False
        self.has_stress = False
        self.time_scale = time_scale

#############################################################################
# Data reading
#############################################################################
        
    def read(self, strain_file = None, stress_file = None, tag_data_file = None):
        """
        read basic file data (mesh, displacement) and calculate basic characteristics
        """

        # read mesh
        self.x_mesh = self.read_pts_file(self.mesh_file)

        # calculate mesh characteristics
        self.min_dim = np.min(self.x_mesh, axis=0)
        self.max_dim = np.max(self.x_mesh, axis=0)
        self.mesh_sizes = self.max_dim - self.min_dim
        self.mesh_dimension = self.x_mesh.shape[-1]

        # assuming regular rectangular grid !
        self.points_per_dimension = [np.unique(self.x_mesh[:,i]).shape[0] for i in range(self.mesh_dimension)]

        # read displaced configuration as list [timepoint: (coordinates, mesh indices, time index)]
        self.x_displaced_orig, self.x_displaced_indices = self.read_dynpt_file(self.displacement_file, dim = self.mesh_dimension)
        self.x_displaced = copy.deepcopy(self.x_displaced_orig)

        # calculate displacement
        self.calc_displacement(recalc_orig = True)
        
        # submeshes in reference configuration are more memory friendly since they have to be stored only once
        self.submeshes_reference_conf = {}
        # submeshes in deformed configuration can vary through time and hence have to be stored for each time step
        self.submeshes_deformed_conf = {}

        if strain_file is not None:
            self.strain_file = strain_file
            if strain_file[-3:] == "txt":
                self.strain_orig = self.read_tens_txt_file(strain_file, dim = self.mesh_dimension)
                self.strain_orig = [(self.strain_orig, np.arange(self.strain_orig.shape[0]), 100)]
            else:
                self.strain_orig = self.read_tens_file(strain_file, dim = self.mesh_dimension)
            self.strain = copy.deepcopy(self.strain_orig)
            self.has_strain = True
            
        if stress_file is not None:
            self.stress_file = stress_file
            if stress_file[-3:] == "txt":
                self.stress_orig = self.read_tens_txt_file(stress_file, dim = self.mesh_dimension)
                self.stress_orig = [(self.stress_orig, np.arange(self.stress_orig.shape[0]), 100)]
            else:
                self.stress_orig = self.read_tens_file(stress_file, dim = self.mesh_dimension)
            self.stress = copy.deepcopy(self.stress_orig)
            self.has_stress = True

        if tag_data_file is not None:
            self.tag_data = np.array(self.read_tdat_file(tag_data_file))
        else:
            self.tag_data = np.ones(self.x_mesh.shape[0])
    
    def read_axis_data(self, fiber_file, sheet_file):
        self.fiber_directions = self.read_vec_file(fiber_file, dim = self.mesh_dimension)
        self.sheet_directions = self.read_vec_file(sheet_file, dim = self.mesh_dimension)
        self.normal_directions = np.cross(self.fiber_directions, self.sheet_directions)

        self.tf_fiber_directions = tf.constant(self.fiber_directions, dtype=ns.config.get_dtype())
        self.tf_sheet_directions = tf.constant(self.sheet_directions, dtype=ns.config.get_dtype())
        self.tf_normal_directions = tf.constant(self.normal_directions, dtype=ns.config.get_dtype())

#############################################################################
# Setter
#############################################################################

    def set_submesh(self, mesh_file, configuration = "reference", name = None):
        mesh = self.read_vtx_file(mesh_file)
        if name is None:
            name = len(self.submeshes)
        
        if configuration == "reference":
            self.submeshes_reference_conf[name] = mesh
        elif configuration == "deformed":
            timepoints = len(self.x_displaced_indices)
            meshes = [mesh for _ in range(timepoints)]
            self.submeshes_deformed_conf[name] = meshes

    def set_submesh_complement(self, reference_submesh_names = [], name = None):
        if name is None:
            name = len(self.submeshes)
        indices = self.get_union_indices_reference_conf(reference_submesh_names)
        total_indices = np.arange(self.x_mesh.shape[0])
        complement = np.delete(total_indices, indices, axis=0)
        
        self.submeshes_reference_conf[name] = complement
    
    def set_tag_dict(self, tag_dict, name_dict = None):
        self.tag_dict = tag_dict
        self.name_dict = name_dict
    
    def interpolate(self, tag_dict = None):
        if tag_dict is None:
            tag_dict = self.tag_dict
            
        lower_tags = np.floor(self.tag_data).astype(int).flatten()
        upper_tags = np.ceil(self.tag_data).astype(int).flatten()

        lam = self.tag_data.flatten() - lower_tags

        lower_values = [tag_dict[i] for i in lower_tags]
        upper_values = [tag_dict[i] for i in upper_tags]
        diff_values = [upper_values[i] - lower_values[i] for i in range(len(self.tag_data))]
        
        values = [lower_values[i] + lam[i] * diff_values[i] for i in range(len(self.tag_data))]
        
        self.tag_values = np.expand_dims(values, axis=-1)
        return self.tag_values

#############################################################################
# Getter
#############################################################################

    # Mesh
    ####################

    def get_times(self, time_scale):
        return [tidx*time_scale for _, _, tidx in self.x_displaced]

    def get_mesh(self, time_scale = None, reference_submeshes = []):
        
        if time_scale is not None:
            x_d = [np.append(x,t) for _, xidx, t in self.x_displaced for x in self.x_mesh[xidx]]
        else:
            x_d = [self.x_mesh[idx] for x, idx, tidx in self.x_displaced]
        
        return x_d   
    
    def get_mesh_points_for_tag(self, tag):
        mask = (self.tag_data == tag).flatten()
        return self.x_mesh[mask], mask, self.tag_values[mask]

    def get_name_for_tag(self, tag):
        if tag in self.name_dict:
            name = self.name_dict[tag]
        else:
            name = tag

        return name

    def get_submesh_names(self):
        reference_conf = self.submeshes_reference_conf.keys()
        deformed_conf = self.submeshes_deformed_conf.keys()
        return reference_conf, deformed_conf

    # Displacement
    ####################
    
    def get_displacement(self):
        displacement = [d for d,_,_ in self.displacement]
        return np.array(displacement, dtype=np.double)

    def get_displacement_orig(self):
        displacement = [d for d,_,_ in self.displacement_orig]
        return np.array(displacement, dtype=np.double)
    
    def get_x_displaced(self):
        x_d = [x for x,_,_ in self.x_displaced]
        return x_d

    def get_max_displacement_components(self):
        d_max = np.array([d for d,_ in self.max_displacement], dtype=np.double)
        return np.max(d_max, axis=0)
    
    # Strain
    ####################
    
    def get_strain(self):
        strain = [s for s,_,_ in self.strain]
        return np.array(strain, dtype=np.double)
    
    def get_strain_orig(self):
        strain = [s for s,_,_ in self.strain_orig]
        return np.array(strain, dtype=np.double)

    def get_max_strain(self):
        s_max = np.array([np.max(s) for s,_,_ in self.strain], dtype=np.double)
        return np.max(s_max)
    
    # Stress
    ####################
    
    def get_stress(self):
        stress = [s for s,_,_ in self.stress]
        return np.array(stress, dtype=np.double)
    
    def get_stress_orig(self):
        stress = [s for s,_,_ in self.stress_orig]
        return np.array(stress, dtype=np.double)

    def get_max_stress(self):
        s_max = np.array([np.max(s) for s,_,_ in self.stress], dtype=np.double)
        return np.max(s_max)
    
    # Sampling
    ####################
    
    def get_random_indices(self, num_train, num_test, time_scale, np_random_generator, reference_submesh_names = [], deformed_submesh_names = []):
        
        if len(reference_submesh_names) == 0 and len(deformed_submesh_names) == 0:
            total_indices = self.x_displaced_indices
        else:
            if len(reference_submesh_names) > 0:
                reference_submesh_idx = self.get_filtered_indices_reference_conf(reference_submesh_names)
            else:
                reference_submesh_idx = np.arange(self.x_mesh.shape[0])
            
            if len(deformed_submesh_names) > 0:
                deformed_submesh_idx = self.get_filtered_indices_deformed_conf(deformed_submesh_names)
            else:
                deformed_submesh_idx = self.x_displaced_indices
            
            total_indices = [(np.intersect1d(reference_submesh_idx, def_idx, assume_unique=True), tidx) for def_idx, tidx in deformed_submesh_idx]
        
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
            
            x_idx = idx - offset

            return x_idx, t_idx

        # get relative index for total_indices
        indices = [idx_to_point(idx) for idx in random_idx]

        # convert to mesh index
        indices = [(total_indices[tn][0][x],tn) for x,tn in indices]
        
        indices.sort(key=lambda i: i[0])

        #import pdb; pdb.set_trace()
        return np.array(indices)
    
    def get_random(self, num_train, num_test, time_scale, np_random_generator, reference_submesh_names = [], deformed_submesh_names = []):
        
        indices = self.get_random_indices(num_train, num_test, time_scale, np_random_generator, reference_submesh_names, deformed_submesh_names)
        n = num_train + num_test
        
        x_displaced = [(self.x_displaced[t][0][x], self.x_displaced[t][1][x], self.x_displaced[t][2]*time_scale) for x,t in indices]
        
        x_mesh = [np.append(self.x_mesh[xidx], t) for _, xidx, t in x_displaced]
        displacement = [x_displaced[i][0] - x_mesh[i][:3] for i in range(n)]
        
        x_displaced_tensor = tf.constant([x[0] for x in x_displaced], dtype=ns.config.get_dtype())
        x_mesh_tensor = tf.constant(x_mesh, dtype=ns.config.get_dtype())
        displacement_tensor = tf.constant(displacement, dtype=ns.config.get_dtype())

        data = [x_mesh_tensor, x_displaced_tensor, displacement_tensor]
        
        data_return, data_test, idx_data, idx_test = train_test_split(data, num_train, num_test, np_random_generator)
        
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

    def get_random_strain(self, num_train, num_test, time_scale, np_random_generator, reference_submesh_names = [], deformed_submesh_names = []):
        
        indices = self.get_random_indices(num_train, num_test, time_scale, np_random_generator, reference_submesh_names, deformed_submesh_names)
        n = num_train + num_test
        
        strain = [(self.strain[t][0][x], self.strain[t][1][x], self.strain[t][2]*time_scale) for x,t in indices]
        
        x_mesh = [np.append(self.x_mesh[xidx], t) for _, xidx, t in strain]
        
        strain_tensor = tf.constant([x[0] for x in strain], dtype=ns.config.get_dtype())
        x_mesh_tensor = tf.constant(x_mesh, dtype=ns.config.get_dtype())
        
        data = [x_mesh_tensor, strain_tensor]

        data_return, data_test, idx_data, idx_test = train_test_split(data, num_train, num_test, np_random_generator)

        self.strain_train_data = {
            'x_mesh': data_return[0],
            'strain': data_return[1]
        }

        self.strain_test_data = {
            'x_mesh': data_test[0],
            'strain': data_test[1]
        }
        
        return data_test[0], data_test[1], data_return[0], data_return[1]
    
    def get_random_stress(self, num_train, num_test, time_scale, np_random_generator, reference_submesh_names = [], deformed_submesh_names = []):
        
        indices = self.get_random_indices(num_train, num_test, time_scale, np_random_generator, reference_submesh_names, deformed_submesh_names)
        n = num_train + num_test
        
        stress = [(self.stress[t][0][x], self.stress[t][1][x], self.stress[t][2]*time_scale) for x,t in indices]
        
        x_mesh = [np.append(self.x_mesh[xidx], t) for _, xidx, t in stress]
        
        stress_tensor = tf.constant([x[0] for x in stress], dtype=ns.config.get_dtype())
        x_mesh_tensor = tf.constant(x_mesh, dtype=ns.config.get_dtype())
        
        data = [x_mesh_tensor, stress_tensor]

        data_return, data_test, idx_data, idx_test = train_test_split(data, num_train, num_test, np_random_generator)

        self.stress_train_data = {
            'x_mesh': data_return[0],
            'stress': data_return[1],
            'indices': indices[idx_data]
        }

        self.stress_test_data = {
            'x_mesh': data_test[0],
            'stress': data_test[1],
            'indices': indices[idx_test]
        }
        
        return data_test[0], data_test[1], data_return[0], data_return[1]

    # Sa
    ####################

    def get_Sa_healthy(self, tn, filename: str):
        data = tf.constant(self.read_tdat_file(filename), dtype=ns.config.get_dtype())

        t = data[tn][0]
        Sa_healthy = data[tn][1]

        return Sa_healthy, t


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

            if self.has_strain:

                max_strain_data = [(np.max(d, axis=0), tidx) for d, _, tidx in self.strain]
                max_strain = max_strain_data[0][0]
                for d, _ in max_strain_data:
                    for i in range(d.shape[-1]):
                        max_strain[i] = max(abs(d[i]), max_strain[i])
                sigma = C * max_strain / (dim**2)
                self.strain = self._apply_noise(sigma, self.strain, np_random_generator)
                
            if self.has_stress:

                max_stress_data = [(np.max(d, axis=0), tidx) for d, _, tidx in self.stress]
                max_stress = max_stress_data[0][0]
                for d, _ in max_stress_data:
                    for i in range(d.shape[-1]):
                        max_stress[i] = max(abs(d[i]), max_stress[i])
                sigma = C * max_stress / (dim**2)
                self.stress = self._apply_noise(sigma, self.stress, np_random_generator)

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
    
    def get_filtered_indices_reference_conf(self, submesh_names):
        submeshes = [self.submeshes_reference_conf[name] for name in submesh_names]
        
        return reduce(lambda x,y : np.intersect1d(x,y, assume_unique=True), (submeshes))

    def get_union_indices_reference_conf(self, submesh_names):
        submeshes = [self.submeshes_reference_conf[name] for name in submesh_names]
        
        return reduce(np.union1d, (submeshes))
    
    def get_filtered_indices_deformed_conf(self, submesh_names):
        submeshes = []
        for i, (_, tidx) in enumerate(self.x_displaced_indices):
            submeshes_i = [self.submeshes_deformed_conf[name][i] for name in submesh_names]
            mask = reduce(lambda x,y : np.intersect1d(x,y, assume_unique=True), (submeshes_i))
            submeshes += [(mask, tidx)]
        
        return submeshes
    
#############################################################################
# Filter
#############################################################################
    
    def apply_time_filter(self, tn):
        """
        filters data set according to list of time indices (not time points) tn
        """
        if isinstance(tn, range):
            tn_list = list(tn)
        elif isinstance(tn, list):
            tn_list = tn
        else:
            tn_list = [tn]
        
        self.x_displaced, self.x_displaced_orig = self._apply_time_filter(copy.deepcopy(tn_list), self.x_displaced, self.x_displaced_orig)
        
        if self.has_strain:
            self.strain, self.strain_orig = self._apply_time_filter(copy.deepcopy(tn_list), self.strain, self.strain_orig)
        
        if self.has_stress:
            self.stress, self.stress_orig = self._apply_time_filter(copy.deepcopy(tn_list), self.stress, self.stress_orig)
        
        new_indices = []
        for i, (_, tidx) in enumerate(self.x_displaced_indices):
            if tidx in tn_list:
                tn_list.remove(tidx)
                new_indices += [self.x_displaced_indices[i]]
        self.x_displaced_indices = new_indices
        
        self.calc_displacement(recalc_orig = True)

    def _apply_time_filter(self, tn_list, data, data_orig = None):
        
        filtered, filtered_orig = [], []
        
        for i, (_, _, tidx) in enumerate(data):
            if tidx in tn_list:
                tn_list.remove(tidx)
                filtered += [data[i]]
                if data_orig is not None:
                    filtered_orig += [data_orig[i]]

        if len(tn_list) > 0:
            warnings.warn(f'FEMDataHandler.apply_time_filter: not all indices present. [{",".join([str(t) for t in tn_list])}] not contained in data set.')
        
        return filtered, filtered_orig

    def discard_first(self, n):
        indices = np.arange(0, self.x_mesh.shape[0])
        
        self.submeshes_reference_conf['discard_first'] = indices[n:]
    
    def apply_slicing(self, slice_axis, resolution, tolerance):
        
        #for i, (space_points, mesh_indices, tidx) in enumerate(self.x_displaced):
        #    pts, idx = self.slice_space_domain(space_points, mesh_indices, slice_axis, resolution, tolerance)
        #    self.x_displaced[i] = (pts, idx, tidx)
        
        slices = []
        for i, (mesh_indices, tidx) in enumerate(self.x_displaced_indices):
            space_points = self.x_displaced[i][0]
            _, idx = self.slice_space_domain(space_points, mesh_indices, slice_axis, resolution, tolerance)
            slices += [(idx, tidx)]
            
        self.submeshes_deformed_conf['slices'] = slices

    def apply_averaging(self, avg_factor):

        for i, (space_points, mesh_indices, tidx) in enumerate(self.x_displaced):
            pts = self.avg_space_domain(space_points, avg_factor, self.points_per_dimension)
            self.x_displaced[i] = (pts, mesh_indices, tidx)
        
        self.calc_displacement()
    
    def slice_space_domain(self, space_points, mesh_indices, slice_axis, resolution, tolerance):
        mask = np.minimum(np.abs(space_points[:, slice_axis] % resolution), np.abs(resolution - space_points[:, slice_axis] % resolution)) <= tolerance
        #mask = np.in1d(x_displ[:, slice_axis], np.asarray(slice_axis_values))
        return space_points[mask], mesh_indices[mask]

    def avg_space_domain(self, space_points, avg_factor, points_per_dim):

        mesh_dim = 3
        space_points = space_points.reshape(points_per_dim + [mesh_dim], order = 'F')

        subregions = [int(ppd / avg_factor) for ppd in points_per_dim]

        avg_data = np.zeros(points_per_dim + [mesh_dim])

        # Calculate average data in subregions
        # TODO: only for 3 dimensions possible
        for i in range(subregions[0]):
            for j in range(subregions[1]):
                for k in range(subregions[2]):
        
                    x_start = i * avg_factor
                    x_end = (i + 1) * avg_factor
                    y_start = j * avg_factor
                    y_end = (j + 1) * avg_factor
                    z_start = k * avg_factor
                    z_end = (k + 1) * avg_factor
                    
                    subregion = space_points[x_start:x_end+1, y_start:y_end+1, z_start:z_end+1, :]
                    avg_subregion = np.array([[[np.mean(subregion, axis=(0, 1, 2) )]]])
                    avg_data[x_start:x_end +1, y_start:y_end+1, z_start:z_end+1, :] = avg_subregion

        avg_data = avg_data.reshape(points_per_dim[0] * points_per_dim[1] * points_per_dim[2], mesh_dim, order = 'F')
        return avg_data
    
    def restrict_to_outside(self):
        mask = np.any(self.min_dim == self.x_mesh, axis = 1) | np.any(self.x_mesh == self.max_dim, axis = 1)
        indices = np.arange(0, self.x_mesh.shape[0])
        outside = indices[mask]
        
        self.submeshes_reference_conf['outside'] = outside
    
    def restrict_to_inside(self):
        mask = np.all(self.min_dim < self.x_mesh, axis = 1) & np.all(self.x_mesh < self.max_dim, axis = 1)
        indices = np.arange(0, self.x_mesh.shape[0])
        inside = indices[mask]
        
        self.submeshes_reference_conf['inside'] = inside

    def slice(self, name, axis, value):
        self.mesh_sizes = self.max_dim - self.min_dim
        self.mesh_dimension = self.x_mesh.shape[-1]
        h = np.min([self.mesh_sizes[i] / (self.points_per_dimension[i] - 1) for i in range(self.mesh_dimension)])
        
        mask = np.abs(self.x_mesh[:,axis] - value) < h*1e-3
        indices = np.arange(0, self.x_mesh.shape[0])
        slice = indices[mask]

        self.submeshes_reference_conf[name] = slice
        
#############################################################################
# File IO
#############################################################################

    def read_vtx_file(self, filename): #adapted from carputils/carpio/txt.py
        """
        Read data from .vtx file

        Returns:
            data ... 1D numpy array of
            domain ... 'intra' 

        """
        with open(filename, "r") as fp:
            num = int(fp.readline().strip())
            domain = fp.readline().strip()
            data = fp.readlines()
            data = np.asarray(data, dtype=int)
        return data
    
    def read_pts_file(self, file, dim = 3, scaling = 1e-3):
        """
        read coordinates/points from file
        """
        with open(file, "r") as f_pts:
            npts = int(f_pts.readline().strip())
            data = f_pts.read().replace('\n', ' ')
            data = np.fromstring(data, dtype=np.double, sep=' ')*scaling # um to mm (CARP mesh)
            data = np.reshape(data, (npts, dim))
            return data

    def read_vec_file(self, filename, dim = 3): #adapted from  carputils/carpio/txt.py
        """
        read coordinates/points of fibers or sheets from file .vec
        """
        with open(filename, "r") as f_vec:
            data = f_vec.read().replace('\n', ' ')
            data = np.fromstring(data, dtype=np.double, sep=' ')
            npts = int(round(data.shape[0]/dim))
            data = np.reshape(data, (npts, dim))
            return data
        
    def read_tens_txt_file(self, file, dim = 3): #adapted from  carputils/carpio/txt.py
        """
        read tensor from file .txt
        """
        with open(file, "r") as f_txt:
            data = f_txt.read().replace('\n', ' ')
            data = np.fromstring(data, dtype=np.double, sep=' ')
            print(data.shape[0]/(dim**2))
            npts = int(round(data.shape[0]/(dim**2)))
            data = np.reshape(data, (npts, (dim**2)))
        
        return data

    def read_tens_file(self, file, dim = 3):
        """
        read tensor from file .igb
        """
    
        igb_file = igb.IGBFile(file)
    
        header = igb_file.header()
        data = igb_file.data()
    
        num_tsteps = header.get('t')
        num_traces = header.get('x')
    
        npts = data.shape[0]
        
        if npts % (dim**2) != 0:
            raise Error('File '+ file+' not compatible with space dimensions.')
        npts = int(round(npts/(dim**2)))
        if npts % num_tsteps != 0:
            raise Error('File '+ file+' not compatible with time steps.')
        
        data = np.reshape(data, (num_tsteps, num_traces, dim**2))
        
        #transform into list for each time step, add mesh index and time index
        time_list = [(data[i], np.arange(0, num_traces), i) for i in range(num_tsteps)]
        
        return time_list

    def read_dynpt_file(self, file, dim = 3, scaling = 1e-3):
        """
        get final displacement of a simulation
        """
        displ = igb.IGBFile(file)
        header = displ.header()
        u_data = displ.data()
        num_traces = header.get('x')
        num_tsteps = header.get('t')
        num_comp = 0
        if header.get('type') == 'vec3f':
            num_comp = dim
        else:
            print("Wrong number of components in displ file")
            return np.array([])  # return empty array
            
        # Check if simulation failed
        if len(u_data) < num_tsteps*num_traces*num_comp:
            return np.array([])  # return empty array
        u_data = u_data.reshape(num_tsteps, num_traces, num_comp)
        final_pts = u_data*scaling  # um to mm (CARP solution)

        #transform into list for each time step, add mesh index and time index
        # TODO: separate x_displaced and mesh indices
        time_list = [(final_pts[i], np.arange(0, num_traces), i) for i in range(num_tsteps)]
        
        mesh_indices = [(np.arange(0, num_traces), t) for t in range(num_tsteps)]
        
        return time_list, mesh_indices

    def read_tdat_file(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            data = [[np.double(x.strip()) for x in row] for row in reader]
    
        return data
