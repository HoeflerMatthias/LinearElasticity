import numpy as np
import copy
from Codebase.network import load_network
from Codebase.FEMDataHandler import FEMDataHandler

class PseudoFEMDataHandler(FEMDataHandler):

    def __init__(self, data_file: str = None):
        self.data_file = data_file
        super().__init__("","")

    def read(self, bulk_modulus: float = 650.0, model_file: str = None):

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

            self.tag_data = 1 + (self.x_mesh[:,0] < self.x_mesh[:,1])
            self.tag_dict = {1: 8.0, 2: 16.0}
            self.name_dict = {1: "healthy", 2: "scar"}

            # read strain
            order = [7,8,9,10,11,12,13,14,15]
            strain = data[:,order].reshape(-1,3,3)
            self.strain_orig = [(strain, range(self.x_mesh.shape[0]), 0)]
            
            # read stress
            stress = bulk_modulus * np.expand_dims(np.expand_dims(np.trace(strain, axis1=1,axis2=2),-1),-1) * np.eye(self.mesh_dimension) + 2 * np.expand_dims(self.tag_values,-1) * strain
            self.stress_orig = [(stress, range(self.x_mesh.shape[0]), 0)]
        
        self.x_displaced_indices = [(np.arange(0, self.x_mesh.shape[0]), 0)]
        self.x_displaced = copy.deepcopy(self.x_displaced_orig)
        self.strain = copy.deepcopy(self.strain_orig)
        self.stress = copy.deepcopy(self.stress_orig)
        
        # calculate displacement
        self.calc_displacement(recalc_orig=True)
        
        # submeshes in reference configuration are more memory friendly since they have to be stored only once
        self.submeshes_reference_conf = {}
        # submeshes in deformed configuration can vary through time and hence have to be stored for each time step
        self.submeshes_deformed_conf = {}

        self.slice('nxminus', 0, self.min_dim[0])
        self.slice('nxplus', 0, self.max_dim[0])

        self.slice('nyminus', 1, self.min_dim[1])
        self.slice('nyplus', 1, self.max_dim[1])

        self.slice('nzminus', 2, self.min_dim[2])
        self.slice('nzplus', 2, self.max_dim[2])

        self.submeshes_reference_conf['low_resolution'] = np.arange(0, self.x_mesh.shape[0],step=4)