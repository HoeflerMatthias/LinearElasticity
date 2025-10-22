import numpy as np
import meshio
from pathlib import Path
import tensorflow as tf

def particles_to_field_3d_average(up, tx, ty, tz, domain):
    dx = domain.step('x')
    dy = domain.step('y')
    dz = domain.step('z')
    nx = domain.size('x')
    ny = domain.size('y')
    nz = domain.size('z')

    # Clip the trajectories within the domain boundaries
    tx = tf.clip_by_value(tx, domain.lower[0], domain.upper[0])
    ty = tf.clip_by_value(ty, domain.lower[1], domain.upper[1])
    tz = tf.clip_by_value(tz, domain.lower[2], domain.upper[2])

    # Calculate nearest grid indices
    ix = tf.cast(tf.round((tx - domain.lower[0]) / dx), tf.int32)
    iy = tf.cast(tf.round((ty - domain.lower[1]) / dy), tf.int32)
    iz = tf.cast(tf.round((tz - domain.lower[2]) / dz), tf.int32)

    # Ensure indices are within bounds
    ix = tf.clip_by_value(ix, 0, nx - 1)
    iy = tf.clip_by_value(iy, 0, ny - 1)
    iz = tf.clip_by_value(iz, 0, nz - 1)

    # Flatten the indices to 1D to simplify aggregation
    flat_indices = tf.stack([ix, iy, iz], axis=-1)

    # Initialize the output grid
    u = tf.zeros((nx, ny, nz), dtype=up.dtype)

    # Handling multiple particles mapped to the same grid point by averaging their contributions
    # Step 1: Create a tensor of ones to use for counting particle contributions per grid cell
    ones = tf.ones_like(up)

    # Step 2: Aggregate contributions and counts
    u = tf.tensor_scatter_nd_add(u, flat_indices, up)
    counts = tf.tensor_scatter_nd_add(tf.zeros((nx, ny, nz), dtype=up.dtype), flat_indices, ones)

    # Avoid division by zero by setting zero counts to one (or handle differently as needed)
    counts = tf.where(counts == 0, tf.ones_like(counts), counts)

    # Step 3: Compute the average by dividing the sum of contributions by the count
    u_average = u / counts

    return u_average

def _grid_connectivity_hexa(nx, ny, nz):
    """8-node hex connectivity for a structured (nx,ny,nz) *node* grid, i-fastest."""
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
    points_ijk_xyz: (nx,ny,nz,3) with axes (i,j,k) from meshgrid(..., indexing="ij")
    cshape:         (nx,ny,nz)
    ux,uy,uz,mu:    nodal arrays, shape = cshape
    """
    nx, ny, nz = cshape

    # Use order="F" so i is fastest (matches idx above)
    pts = points_ijk_xyz.reshape(nx*ny*nz, 3, order="F")
    U   = np.stack([ux, uy, uz], axis=-1).reshape(nx*ny*nz, 3, order="F")
    MU  = mu.reshape(nx*ny*nz, order="F")

    hexas = _grid_connectivity_hexa(nx, ny, nz)
    mesh  = meshio.Mesh(points=pts, cells=[("hexahedron", hexas)],
                        point_data={"u": U, "mu": MU})
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    mesh.write(filename)

def plot_losses_from_csv(csv_path, out_png=None, columns=None, logy=True):
    """
    csv_path : path to train.csv
    out_png  : output png path (defaults to csv_path with .png)
    columns  : list of column names to plot; if None -> auto-select numeric losses
    logy     : plot with log-scale on y
    smooth   : None (no smoothing), or dict like {"method": "ema", "alpha": 0.2} or {"method":"rolling","window":21}
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # --- pick x-axis (epoch/step) ---
    epoch_candidates = ["epoch", "step", "iter", "iteration", "global_step"]
    xcol = next((c for c in epoch_candidates if c in df.columns), None)
    if xcol is None:
        # fallback: use row index as "epoch"
        x = np.arange(len(df))
        x_label = "Index"
    else:
        x = df[xcol].to_numpy()
        x_label = xcol

    # --- pick y-columns (losses / metrics) ---
    if columns is None:
        # heuristics: numeric columns except xcol; prefer names containing "loss" or "rms"
        num_cols = [c for c in df.columns if c != xcol and np.issubdtype(df[c].dtype, np.number)]
        prefer = [c for c in num_cols if ("loss" in c.lower()) or ("rms" in c.lower())]
        ycols = prefer if prefer else num_cols
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in CSV: {missing}")
        ycols = columns

    if not ycols:
        raise ValueError("No numeric loss/metric columns found to plot.")

    # --- make plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in ycols:
        ax.plot(x, df[c].to_numpy(), label=c)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Loss / Error")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(ncol=1)
    fig.tight_layout()

    # --- save ---
    if out_png is None:
        out_png = csv_path.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return str(out_png), ycols

def load_data_csv_to_grid(domain, path, mu_scale=8.0, dtype=np.float32, noise_level=0.0, noise_seed=1234):
    """
    CSV columns required:
      x,y,z,ux,uy,uz,alpha,e_xx,e_xy,e_xz,e_yx,e_yy,e_yz,e_zx,e_zy,e_zz
    Returns:
      data_u  : (Nx,Ny,Nz,3)
      data_mu : (Nx,Ny,Nz)
      data_E  : (Nx,Ny,Nz,3,3)  # optional (not used in loss)
      mask    : (Nx,Ny,Nz) with 1 where data present, else 0
    """

    arr = np.genfromtxt(path, delimiter=",", names=True)
    names = [n.lower() for n in arr.dtype.names]
    need = lambda cols: all(c in names for c in cols)
    assert need(["x", "y", "z", "ux", "uy", "uz", "alpha"]), "CSV missing required columns"

    Nx, Ny, Nz = domain.cshape
    x1, y1, z1 = domain.points_1d()

    if noise_level > 0.0:
        # Add noise and re-apply constrained DOFs
        u = np.stack([arr["ux"], arr["uy"], arr["uz"]], axis=-1).astype(dtype)
        sigma_u = np.max(np.abs(u), axis=0) / 3

        rng = np.random.default_rng(noise_seed)
        u += noise_level * sigma_u * rng.normal(size=u.shape)
        arr["ux"] = u[:, 0]
        arr["uy"] = u[:, 1]
        arr["uz"] = u[:, 2]

    def nearest_idx(grid, q):
        idx = np.searchsorted(grid, q)
        idx = np.clip(idx, 1, len(grid) - 1)
        left = grid[idx - 1];
        right = grid[idx]
        idx = np.where(np.abs(q - left) <= np.abs(q - right), idx - 1, idx)
        return idx.astype(np.int64)

    I = nearest_idx(x1, arr["x"])
    J = nearest_idx(y1, arr["y"])
    K = nearest_idx(z1, arr["z"])

    data_u = np.zeros((Nx, Ny, Nz, 3), dtype=dtype)
    data_mu = np.zeros((Nx, Ny, Nz), dtype=dtype)
    mask = np.zeros((Nx, Ny, Nz), dtype=dtype)

    data_u[I, J, K, 0] = arr["ux"].astype(dtype)
    data_u[I, J, K, 1] = arr["uy"].astype(dtype)
    data_u[I, J, K, 2] = arr["uz"].astype(dtype)
    data_mu[I, J, K] = (arr["alpha"].astype(dtype) * mu_scale)

    #ux = particles_to_field_3d_average(arr["ux"], arr["x"], arr["y"], arr["z"], domain)
    #uy = particles_to_field_3d_average(arr["ux"], arr["x"], arr["y"], arr["z"], domain)
    #uz = particles_to_field_3d_average(arr["ux"], arr["x"], arr["y"], arr["z"], domain)
    #data_u = tf.stack([ux, uy, uz], axis=-1)

    mask[I, J, K] = 1

    data_E = None
    if need(["e_xx", "e_xy", "e_xz", "e_yx", "e_yy", "e_yz", "e_zx", "e_zy", "e_zz"]):
        data_E = np.zeros((Nx, Ny, Nz, 3, 3), dtype=dtype)
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