import numpy as np
from scipy.optimize import minimize
import pyvista as pv
import pandas as pd

def sym_grad(u, spacing=(1.0, 1.0, 1.0)):
    """
    Symmetric gradient ε(u) for u with shape [3, Nx, Ny, Nz].
    spacing = (hx, hy, hz).
    Returns eps with shape [3, 3, Nx, Ny, Nz], where eps[i,j] = 1/2 (∂_j u_i + ∂_i u_j).
    """
    assert u.ndim == 4 and u.shape[0] == 3, "u must be [3, Nx, Ny, Nz]"
    hx, hy, hz = spacing

    # ∂_x u_i, ∂_y u_i, ∂_z u_i for each i
    grads = []
    for i in range(3):
        # np.gradient order: axis 0->x, 1->y, 2->z for the spatial axes
        du_dx, du_dy, du_dz = np.gradient(u[i], hx, hy, hz, edge_order=2)
        grads.append((du_dx, du_dy, du_dz))

    # assemble ε_ij = 1/2 (∂_j u_i + ∂_i u_j)
    eps = np.empty((3, 3) + u.shape[1:], dtype=u.dtype)
    for i in range(3):
        for j in range(3):
            # grads[i][j] is ∂_j u_i ; grads[j][i] is ∂_i u_j
            eps[i, j] = 0.5 * (grads[i][j] + grads[j][i])
    return eps

def stress_isotropic(eps, mu, lam):
    """
    eps: [3,3,Nx,Ny,Nz] small-strain tensor
    mu:  [1,Nx,Ny,Nz] or broadcastable to [Nx,Ny,Nz]
    lam: scalar or broadcastable to [Nx,Ny,Nz]
    returns sigma with shape [3,3,Nx,Ny,Nz]
    """
    assert eps.shape[0] == 3 and eps.shape[1] == 3, "eps must be [3,3,Nx,Ny,Nz]"
    # Broadcast parameters to [Nx,Ny,Nz]
    mu  = np.asarray(mu).reshape(-1, *eps.shape[2:])
    mu  = mu[0]  # [Nx,Ny,Nz]
    lam = np.asarray(lam)
    if lam.ndim == 0:
        lam = np.broadcast_to(lam, eps.shape[2:])
    elif lam.ndim == 4 and lam.shape[0] == 1:
        lam = lam[0]
    elif lam.ndim == 3:
        pass
    else:
        raise ValueError("lam must be scalar or broadcastable to [Nx,Ny,Nz]")

    # trace(eps) -> [Nx,Ny,Nz]
    tr_eps = eps[0,0] + eps[1,1] + eps[2,2]

    sigma = np.empty_like(eps)
    # diagonal parts: 2*mu*eps_ii + lam*tr(eps)
    for i in range(3):
        sigma[i, i] = 2.0 * mu * eps[i, i] + lam * tr_eps
    # off-diagonals: 2*mu*eps_ij
    sigma[0,1] = 2.0 * mu * eps[0,1]; sigma[1,0] = sigma[0,1]
    sigma[0,2] = 2.0 * mu * eps[0,2]; sigma[2,0] = sigma[0,2]
    sigma[1,2] = 2.0 * mu * eps[1,2]; sigma[2,1] = sigma[1,2]
    return sigma

def div_sigma(sig, spacing=(1.0,1.0,1.0), b=None):
    """
    sig: [3,3,Nx,Ny,Nz] (σ_ij)
    returns r_p with shape [3,Nx,Ny,Nz], r_p[i] = - sum_j ∂_j σ_ij - b_i
    """
    hx, hy, hz = spacing
    rp = np.zeros((3,) + sig.shape[2:], dtype=sig.dtype)
    for i in range(3):
        d = []
        # ∂_x σ_{i0}, ∂_y σ_{i1}, ∂_z σ_{i2}
        d.append(np.gradient(sig[i,0], hx, axis=0, edge_order=2))
        d.append(np.gradient(sig[i,1], hy, axis=1, edge_order=2))
        d.append(np.gradient(sig[i,2], hz, axis=2, edge_order=2))
        rp[i] = -(d[0] + d[1] + d[2])
    if b is not None:
        rp = rp - b
    return rp

class DirichletBC:
    def __init__(self, mask, value):
        # mask, value: shape [3,Nx,Ny,Nz], mask dtype=bool
        self.mask  = mask
        self.value = value

def clamp_dirichlet(u, dbc: DirichletBC):
    if dbc is None: return u
    u = u.copy()
    u[dbc.mask] = dbc.value[dbc.mask]
    return u

class NeumannFace:
    def __init__(self, axis, side, traction, weight=1.0, outward_sign=+1.0):
        self.axis = axis      # 0=x,1=y,2=z
        self.side = side      # 0 (min face) or -1 (max face)
        self.traction = traction  # shape [3, ... face shape ...]
        self.weight = float(weight)
        # outward_sign flips normal on the min face if needed
        self.outward_sign = float(outward_sign)

def traction_on_face(sig, axis, side):
    if axis == 0:
        ix = 0 if side == 0 else -1
        # σ_{i0} restricted to x-face ix
        return sig[:, 0, ix, :, :]
    elif axis == 1:
        iy = 0 if side == 0 else -1
        # σ_{i1} restricted to y-face iy
        return sig[:, 1, :, iy, :]
    elif axis == 2:
        iz = 0 if side == 0 else -1
        # σ_{i2} restricted to z-face iz
        return sig[:, 2, :, :, iz]
    else:
        raise ValueError("axis must be 0, 1, or 2")

def pack(u, mu):
    return np.concatenate([u.ravel(), mu.ravel()])

def unpack(x, shape_u, shape_mu):
    nu = np.prod(shape_u)
    u = x[:nu].reshape(shape_u)
    mu = x[nu:].reshape(shape_mu)
    return u, mu

def residual_interior(u, mu, lam, spacing):
    eps = sym_grad(u, spacing)
    sig = stress_isotropic(eps, mu, lam)
    rp  = div_sigma(sig, spacing)  # [3,Nx,Ny,Nz]
    return rp, eps, sig

def loss_mixed(u, mu, lam, spacing, dbc: DirichletBC=None, nfaces=None):
    # clamp Dirichlet before computing residuals
    u = clamp_dirichlet(u, dbc)
    rp, eps, sig = residual_interior(u, mu, lam, spacing)
    J = 0.5*np.sum(rp**2)

    if nfaces:
        for f in nfaces:
            t = traction_on_face(sig, f.axis, f.side, f.outward_sign)
            tres = t - f.traction
            J += 0.5 * f.weight * np.sum(tres**2)
    return J, (rp, eps, sig)


def L_apply(p, mu, lam, spacing):
    eps_p = sym_grad(p, spacing)
    sig_p = stress_isotropic(eps_p, mu, lam)
    return div_sigma(sig_p, spacing)

def grad_mu_interior(u, mu, lam, spacing, rp):
    hx, hy, hz = spacing
    grad_rp = np.empty((3,3)+rp.shape[1:], dtype=rp.dtype)
    grad_rp[0,0] = np.gradient(rp[0], hx, axis=0, edge_order=2)
    grad_rp[0,1] = np.gradient(rp[0], hy, axis=1, edge_order=2)
    grad_rp[0,2] = np.gradient(rp[0], hz, axis=2, edge_order=2)
    grad_rp[1,0] = np.gradient(rp[1], hx, axis=0, edge_order=2)
    grad_rp[1,1] = np.gradient(rp[1], hy, axis=1, edge_order=2)
    grad_rp[1,2] = np.gradient(rp[1], hz, axis=2, edge_order=2)
    grad_rp[2,0] = np.gradient(rp[2], hx, axis=0, edge_order=2)
    grad_rp[2,1] = np.gradient(rp[2], hy, axis=1, edge_order=2)
    grad_rp[2,2] = np.gradient(rp[2], hz, axis=2, edge_order=2)
    eps = sym_grad(u, spacing)
    g_mu = 2.0 * np.sum(eps * grad_rp, axis=(0,1))
    return g_mu[None, ...]

def add_neumann_gradients(gu, gmu, mu, lam, spacing, eps, sig, nfaces):
    if not nfaces: return gu, gmu
    # helper: divergence of a 2-tensor field
    def div_eps(A, spacing):
        hx, hy, hz = spacing
        out = np.zeros((3,)+A.shape[2:], dtype=A.dtype)
        # out[i] = -sum_j ∂_j A[i,j]
        out[0] = -(np.gradient(A[0,0], hx, axis=0, edge_order=2) +
                   np.gradient(A[0,1], hy, axis=1, edge_order=2) +
                   np.gradient(A[0,2], hz, axis=2, edge_order=2))
        out[1] = -(np.gradient(A[1,0], hx, axis=0, edge_order=2) +
                   np.gradient(A[1,1], hy, axis=1, edge_order=2) +
                   np.gradient(A[1,2], hz, axis=2, edge_order=2))
        out[2] = -(np.gradient(A[2,0], hx, axis=0, edge_order=2) +
                   np.gradient(A[2,1], hy, axis=1, edge_order=2) +
                   np.gradient(A[2,2], hz, axis=2, edge_order=2))
        return out

    for f in nfaces:
        # traction residual on the face
        t = traction_on_face(sig, f.axis, f.side, f.outward_sign)  # [3, face...]
        tres = t - f.traction                                      # same shape
        # scatter into a_sigma with only column "axis" active at that face
        a_sig = np.zeros_like(sig)
        sl = [slice(None), slice(None), slice(None), slice(None), slice(None)]  # [i,j,x,y,z]
        sl[2+f.axis] = 0 if f.side == 0 else -1
        # put α_N * tres into column j=axis
        a_sig[:, f.axis][tuple(sl[2:])] = f.weight * tres

        # backprop through σ -> ε
        mu_field  = mu.reshape(sig.shape[2:])
        lam_field = np.broadcast_to(lam, sig.shape[2:])
        tr_asig   = a_sig[0,0] + a_sig[1,1] + a_sig[2,2]
        a_eps = np.empty_like(sig)
        for i in range(3):
            for j in range(3):
                a_eps[i,j] = 2.0*mu_field * a_sig[i,j] + (lam_field * tr_asig) * (1.0 if i==j else 0.0)

        # grad wrt u: -div a_eps
        gu += div_eps(a_eps, spacing)

        # grad wrt mu: 2 * eps : a_sig, but only at the face
        # compute local contribution and scatter to that face slice
        gmu_face = 2.0 * np.sum(eps * a_sig, axis=(0,1))  # [Nx,Ny,Nz], nonzero only on that face
        if gmu is None:
            continue
        if gmu.ndim == 4:  # [1,Nx,Ny,Nz]
            gmu[0] += gmu_face
        else:
            gmu += gmu_face
    return gu, gmu

def objective_and_grad_mixed_with_data(
    x, shape_u, shape_mu, lam, spacing, dbc,
    nfaces, use_logmu,
    H_mul, HT_mul, d, w=None
):
    nu = np.prod(shape_u)
    u = x[:nu].reshape(shape_u)
    par = x[nu:].reshape(shape_mu)
    mu = np.exp(par) if use_logmu else par

    # clamp Dirichlet
    u_c = clamp_dirichlet(u, dbc)

    # interior residual and gradients
    eps = sym_grad(u_c, spacing)
    sig = stress_isotropic(eps, mu, lam)
    rp  = div_sigma(sig, spacing)

    J   = 0.5*np.sum(rp**2)
    gu  = L_apply(rp, mu, lam, spacing)
    gmu = grad_mu_interior(u_c, mu, lam, spacing, rp)

    # boundary terms and their gradients
    def div_eps(A, spacing):
        hx, hy, hz = spacing
        out = np.zeros((3,)+A.shape[2:], dtype=A.dtype)
        out[0] = -(np.gradient(A[0,0], hx, axis=0, edge_order=2) +
                   np.gradient(A[0,1], hy, axis=1, edge_order=2) +
                   np.gradient(A[0,2], hz, axis=2, edge_order=2))
        out[1] = -(np.gradient(A[1,0], hx, axis=0, edge_order=2) +
                   np.gradient(A[1,1], hy, axis=1, edge_order=2) +
                   np.gradient(A[1,2], hz, axis=2, edge_order=2))
        out[2] = -(np.gradient(A[2,0], hx, axis=0, edge_order=2) +
                   np.gradient(A[2,1], hy, axis=1, edge_order=2) +
                   np.gradient(A[2,2], hz, axis=2, edge_order=2))
        return out

    for f in nfaces:
        t = traction_on_face(sig, f.axis, f.side)   # [3, face...]
        tres = t - f.traction
        J += 0.5 * f.weight * np.sum(tres**2)

        # scatter adjoint a_sigma with only column 'axis' nonzero at the face
        a_sig = np.zeros_like(sig)
        # build an all-slices tuple
        idx = [slice(None), slice(None), slice(None), slice(None), slice(None)]
        # fix column j = axis in the stress tensor
        idx[1] = f.axis
        # pick the face on the spatial axis
        idx[2 + f.axis] = (-1 if f.side == -1 else 0)

        # now assign in one shot; shapes match: (3, face..., face...)
        a_sig[tuple(idx)] = f.weight * tres

        # backprop through σ(ε(u),μ): a_eps = 2 μ a_sig + λ tr(a_sig) I
        mu_field  = mu.reshape(sig.shape[2:])
        lam_field = np.broadcast_to(lam, sig.shape[2:])
        tr_asig   = a_sig[0,0] + a_sig[1,1] + a_sig[2,2]
        a_eps = np.empty_like(sig)
        for i in range(3):
            for j in range(3):
                a_eps[i,j] = 2.0*mu_field*a_sig[i,j] + (lam_field*tr_asig)*(1.0 if i==j else 0.0)

        # grad wrt u: -div a_eps
        gu += div_eps(a_eps, spacing)

        # grad wrt mu: 2 * ε(u) : a_sig, restricted to the face
        gmu_face = 2.0*np.sum(eps * a_sig, axis=(0,1))  # [Nx,Ny,Nz], zeros except on face
        gmu[0] += gmu_face

    # zero gradients on Dirichlet-constrained dofs
    gu = gu.copy();
    gu[dbc.mask] = 0.0

    # chain rule if using log-μ
    if use_logmu:
        gmu = gmu * mu

    #---- data discrepancy term - ---
    y_pred = H_mul(u_c)  # Hu
    if w is None:
        r = y_pred - d  # residual
        J += 0.5 * np.dot(r, r)
        gu += HT_mul(r)  # ∇_u J_data = H^T r
    else:
        r = (y_pred - d) * w  # weighted residual
        J += 0.5 * np.dot(r, r)
        gu += HT_mul(w * r)  # H^T (W^T W (Hu - d)) with diag W

    g = np.concatenate([gu.ravel(), gmu.ravel()])
    return J, g

df = pd.read_csv("linear_symcube_p10.csv")   # columns: x,y,z,ux,uy,uz

# pull coordinates
xyz_all = df[["x","y","z"]].to_numpy(float)

# build per-sample, per-component lists
sample_xyz_list = []
comps_list = []
d_list = []

for comp_name, comp_id in (("ux",0), ("uy",1), ("uz",2)):
    vals = df[comp_name].to_numpy(float)
    mask = np.isfinite(vals)
    if np.any(mask):
        sample_xyz_list.append(xyz_all[mask])    # shape (m_c, 3)
        comps_list.append(np.full(mask.sum(), comp_id, dtype=int))
        d_list.append(vals[mask])

# concatenate into one observation vector
sample_xyz = np.vstack(sample_xyz_list)          # (m, 3)
comps      = np.concatenate(comps_list)          # (m,)
d          = np.concatenate(d_list)              # (m,)

def build_obs_trilinear(x, y, z, sample_xyz, comps, weights=None):
    sample_xyz = np.asarray(sample_xyz, float)
    m = sample_xyz.shape[0]
    comps = np.asarray(comps, int)
    if comps.size == 1:
        comps = np.full(m, comps.item(), int)
    if weights is None:
        w = np.ones(m)
    else:
        w = np.asarray(weights).reshape(-1)

    Nx, Ny, Nz = len(x), len(y), len(z)
    hx = np.diff(x).mean(); hy = np.diff(y).mean(); hz = np.diff(z).mean()

    def axis_precompute(coord, grid, h, N):
        t = np.clip((coord - grid[0]) / h, 0, N-1-1e-12)
        i0 = np.floor(t).astype(int)
        i1 = np.clip(i0+1, 0, N-1)
        a1 = t - i0
        a0 = 1.0 - a1
        return i0, i1, a0, a1

    i0x,i1x,a0x,a1x = axis_precompute(sample_xyz[:,0], x, hx, Nx)
    i0y,i1y,a0y,a1y = axis_precompute(sample_xyz[:,1], y, hy, Ny)
    i0z,i1z,a0z,a1z = axis_precompute(sample_xyz[:,2], z, hz, Nz)

    w000 = a0x*a0y*a0z; w100 = a1x*a0y*a0z; w010 = a0x*a1y*a0z; w001 = a0x*a0y*a1z
    w110 = a1x*a1y*a0z; w101 = a1x*a0y*a1z; w011 = a0x*a1y*a1z; w111 = a1x*a1y*a1z
    corners = [(i0x,i0y,i0z,w000),(i1x,i0y,i0z,w100),(i0x,i1y,i0z,w010),(i0x,i0y,i1z,w001),
               (i1x,i1y,i0z,w110),(i1x,i0y,i1z,w101),(i0x,i1y,i1z,w011),(i1x,i1y,i1z,w111)]

    def H_mul(u):
        # u: [3,Nx,Ny,Nz] -> y[m]
        y = np.zeros(m, dtype=u.dtype)
        for (ix,iy,iz,wc) in corners:
            y += wc * u[comps, ix, iy, iz]
        return y

    def HT_mul(r):
        # r: [m] -> grad shape [3,Nx,Ny,Nz]
        g = np.zeros((3, Nx, Ny, Nz), dtype=r.dtype)
        for (ix,iy,iz,wc) in corners:
            np.add.at(g, (comps, ix, iy, iz), wc * r)
        return g

    return H_mul, HT_mul, w

# grid
Nx=20; Ny=20; Nz=10;
hx=hy=hz=1.0/(Nx-1)
spacing=(hx,hy,hz)
lam = 650.0

x = np.linspace(0.0, (Nx-1)*hx, Nx)
y = np.linspace(0.0, (Ny-1)*hy, Ny)
z = np.linspace(0.0, (Nz-1)*hz, Nz)

H_mul, HT_mul, w = build_obs_trilinear(x, y, z, sample_xyz, comps, weights=None)


# fields
u0  = np.zeros((3, Nx, Ny, Nz))
mu0 = np.ones((1, Nx, Ny, Nz))*1.0

# Dirichlet mask: x- : ux=0; y- : uy=0; z- : uz=0
mask = np.zeros_like(u0, dtype=bool)
mask[0, 0, :, :] = True   # x-   -> u_x
mask[1, :, 0, :] = True   # y-   -> u_y
mask[2, :, :, 0] = True   # z-   -> u_z
value = np.zeros_like(u0)

dbc = DirichletBC(mask=mask, value=value)

alphaN = 1.0                 # boundary penalty (can scale by face area if you like)

# x+ face: shape [3, Ny, Nz], zero traction
t_xp = np.zeros((3, Ny, Nz))

# y+ face: shape [3, Nx, Nz], zero traction
t_yp = np.zeros((3, Nx, Nz))

# z+ face: shape [3, Nx, Ny], pressure p along +z
p = -10.0                      # set your pressure value
t_zp = np.zeros((3, Nx, Ny)); t_zp[2] = p

face_xp = NeumannFace(axis=0, side=-1, traction=t_xp, weight=alphaN*hy*hz)
face_yp = NeumannFace(axis=1, side=-1, traction=t_yp, weight=alphaN*hx*hz)
face_zp = NeumannFace(axis=2, side=-1, traction=t_zp, weight=alphaN*hx*hy)

nfaces = [face_xp, face_yp, face_zp]

x0 = np.concatenate([u0.ravel(), np.log(mu0).ravel()])  # use log-μ for positivity

obj = lambda x: objective_and_grad_mixed_with_data(
    x, u0.shape, mu0.shape, lam, spacing,
    dbc, nfaces, use_logmu=True,
    H_mul=H_mul, HT_mul=HT_mul, d=d, w=None   # or w if you built weights
)

res = minimize(obj, x0, method="L-BFGS-B", jac=True,
               options=dict(maxiter=1200, gtol=1e-12, ftol=1e-12))

u_opt  = res.x[:u0.size].reshape(u0.shape)
eta_opt = res.x[u0.size:].reshape(mu0.shape)
mu_opt  = np.exp(eta_opt)
print("final loss:", res.fun)



x = np.linspace(0, (Nx-1)*hx, Nx)
y = np.linspace(0, (Ny-1)*hy, Ny)
z = np.linspace(0, (Nz-1)*hz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")   # shape [Nx,Ny,Nz]
grid = pv.StructuredGrid(X, Y, Z)

# flatten grid order must match (x-fastest vs. z-fastest)
u_vec = np.moveaxis(u_opt, 0, -1).reshape(-1, 3)  # shape [Nx*Ny*Nz, 3]
mu_flat = mu_opt.ravel()                          # shape [Nx*Ny*Nz]

grid["displacement"] = u_vec
grid["shear_modulus"] = mu_flat

grid.save("elastic_solution.vtk")