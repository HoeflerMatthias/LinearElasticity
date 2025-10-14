import numpy as np
import meshio
from pathlib import Path

def load_data_csv_to_grid(domain, path, mu_scale=8.0, dtype=np.float32):
    """
    CSV columns required:
      x,y,z,ux,uy,uz,alpha,e_xx,e_xy,e_xz,e_yx,e_yy,e_yz,e_zx,e_zy,e_zz
    Returns:
      data_u  : (Nx,Ny,Nz,3)
      data_mu : (Nx,Ny,Nz)
      data_E  : (Nx,Ny,Nz,3,3)  # optional (not used in loss)
      mask    : (Nx,Ny,Nz) with 1 where data present, else 0
    """
    import numpy as _np

    arr = _np.genfromtxt(path, delimiter=",", names=True)
    names = [n.lower() for n in arr.dtype.names]
    need = lambda cols: all(c in names for c in cols)
    assert need(["x", "y", "z", "ux", "uy", "uz", "alpha"]), "CSV missing required columns"

    Nx, Ny, Nz = domain.cshape
    x1, y1, z1 = domain.points_1d()

    def nearest_idx(grid, q):
        idx = _np.searchsorted(grid, q)
        idx = _np.clip(idx, 1, len(grid) - 1)
        left = grid[idx - 1];
        right = grid[idx]
        idx = _np.where(_np.abs(q - left) <= _np.abs(q - right), idx - 1, idx)
        return idx.astype(_np.int64)

    I = nearest_idx(x1, arr["x"])
    J = nearest_idx(y1, arr["y"])
    K = nearest_idx(z1, arr["z"])

    data_u = _np.zeros((Nx, Ny, Nz, 3), dtype=dtype)
    data_mu = _np.zeros((Nx, Ny, Nz), dtype=dtype)
    mask = _np.zeros((Nx, Ny, Nz), dtype=dtype)

    data_u[I, J, K, 0] = arr["ux"].astype(dtype)
    data_u[I, J, K, 1] = arr["uy"].astype(dtype)
    data_u[I, J, K, 2] = arr["uz"].astype(dtype)
    data_mu[I, J, K] = (arr["alpha"].astype(dtype) * mu_scale)
    mask[I, J, K] = 1

    data_E = None
    if need(["e_xx", "e_xy", "e_xz", "e_yx", "e_yy", "e_yz", "e_zx", "e_zy", "e_zz"]):
        data_E = _np.zeros((Nx, Ny, Nz, 3, 3), dtype=dtype)
        data_E[I, J, K, 0, 0] = arr["e_xx"];
        data_E[I, J, K, 0, 1] = arr["e_xy"];
        data_E[I, J, K, 0, 2] = arr["e_xz"]
        data_E[I, J, K, 1, 0] = arr["e_yx"];
        data_E[I, J, K, 1, 1] = arr["e_yy"];
        data_E[I, J, K, 1, 2] = arr["e_yz"]
        data_E[I, J, K, 2, 0] = arr["e_zx"];
        data_E[I, J, K, 2, 1] = arr["e_zy"];
        data_E[I, J, K, 2, 2] = arr["e_zz"]

    return data_u, data_mu, data_E, mask

def _grid_connectivity_hexa(nx, ny, nz):
    """8-node hex connectivity for a structured (nx,ny,nz) *node* grid."""
    def idx(i, j, k):
        return i + nx * (j + ny * k)

    cells = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                v0 = idx(i,   j,   k); v1 = idx(i+1, j,   k)
                v2 = idx(i+1, j+1, k); v3 = idx(i,   j+1, k)
                v4 = idx(i,   j,   k+1); v5 = idx(i+1, j,   k+1)
                v6 = idx(i+1, j+1, k+1); v7 = idx(i,   j+1, k+1)
                cells.append([v0, v1, v2, v3, v4, v5, v6, v7])
    return np.asarray(cells, dtype=np.int64)

def write_vtu_structured(points_ijk_xyz, cshape, ux, uy, uz, mu, filename):
    """
    points_ijk_xyz: (nx,ny,nz,3) node coords (consistent with your field layout)
    cshape:         (nx,ny,nz)
    ux,uy,uz,mu:    nodal arrays, shape = cshape
    filename:       path ending with .vtu
    """
    nx, ny, nz = cshape
    # Flatten in the SAME order as your fields (C-order)
    pts = points_ijk_xyz.reshape(nx*ny*nz, 3, order="C")
    U = np.stack([ux, uy, uz], axis=-1).reshape(nx*ny*nz, 3, order="C")
    MU = mu.reshape(nx*ny*nz, order="C")

    hexas = _grid_connectivity_hexa(nx, ny, nz)
    mesh = meshio.Mesh(points=pts, cells=[("hexahedron", hexas)],
                       point_data={"u": U, "mu": MU})
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    mesh.write(filename)