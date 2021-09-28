import taichi as ti
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

ti.init(ti.cpu)

# Display
gui_y = 500
gui_x = 800
display = ti.field(ti.f64, shape=(gui_x, gui_y)) # field for display

# Model parameters
nely = 50
nelx = 80
n_node = (nelx+1) * (nely+1)
ndof = 2 * n_node

# FEM variables
K = ti.field(ti.f64, shape=(ndof, ndof))
F = ti.field(ti.f64, shape=(ndof))
U = ti.field(ti.f64, shape=(ndof))
Ke = ti.field(ti.f64, shape=(8,8))

fixed_dofs = list(range(0, 2 * (nely + 1)))
all_dofs = list(range(0, 2 * (nelx + 1) * (nely + 1)))
free_dofs = np.array(list(set(all_dofs) - set(fixed_dofs)))
n_free_dof = len(free_dofs)

free_dofs_vec = ti.field(ti.i32, shape=n_free_dof)
K_freedof = ti.field(ti.f64, shape=(n_free_dof, n_free_dof))
F_freedof = ti.field(dtype=ti.f64, shape=(n_free_dof))
U_freedof = ti.field(dtype=ti.f64, shape=(n_free_dof))

# BESO parameters
E = 1.
nu = 0.3
rmin = 4
volfrac = 0.5 # volume limit
penalty = 3
xmin = 1e-3
ert = 0.02

# BESO variables
xe = ti.field(ti.f64, shape=(nely, nelx))
dc = ti.field(ti.f64, shape=(nely, nelx))  # derivative of compliance
compliance = ti.field(ti.f64, shape=()) # compliance
dc_old = ti.field(ti.f64, shape=(nely, nelx))  # derivative of compliance


def examples(case=0):
    if  case == 0:
        F[2*(nelx+1)*(nely+1)-nely-1] = -1.
    if case == 1:
        F[2*nelx*(nely+1)-1] = -1.


@ti.kernel
def initialize():
    # 1. initialize rho
    for I in ti.grouped(xe):
        xe[I] = 1

    # 2. set boundary condition
    for i in range(n_free_dof):
        F_freedof[i] = F[free_dofs_vec[i]]


@ti.kernel
def display_sampling():
    s_x = int(gui_x / nelx)
    s_y = int(gui_y / nely)
    for i, j in ti.ndrange(gui_x, gui_y):
        elx = i // s_x
        ely = j // s_y
        display[i, gui_y - j] = 1. - xe[ely, elx] # Note:  transpose rho here


@ti.func
def clamp(x: ti.template(), ely, elx):
    return x[ely, elx] if 0 <= ely < nely and 0 <= elx < nelx else 0.


def filt(x, dc):
    nely, nelx = x.shape
    rminf = math.floor(rmin)
    dcf = np.zeros((nely, nelx))

    for i in range(nelx):
        for j in range(nely):
            sum_ = 0.
            for k in range(max(i - rminf, 0), min(i + rminf + 1, nelx)):
                for l in range(max(j - rminf, 0), min(j + rminf + 1, nely)):
                    fac = rmin - math.sqrt((i - k) ** 2. + (j - l) ** 2.)
                    sum_ += max(0., fac)
                    dcf[j, i] = dcf[j, i] + max(0., fac) * dc[l, k]
            dcf[j, i] = dcf[j, i] / sum_
    return dcf


@ti.kernel
def averaging_dc():
        for ely, elx in ti.ndrange(nely, nelx):
            dc[ely, elx] = (dc[ely, elx] + dc_old[ely, elx]) * 0.5


@ti.kernel
def assemble_k():
    for I in ti.grouped(K):
        K[I] = 0.

    # 1. Assemble Stiffness Matrix
    for ely, elx in ti.ndrange(nely, nelx):
        n1 = (nely + 1) * elx + ely + 1
        n2 = (nely + 1) * (elx + 1) + ely + 1
        edof = ti.Vector([2*n1 -2, 2*n1 -1, 2*n2 -2, 2*n2 -1, 2*n2, 2*n2+1, 2*n1, 2*n1+1])

        for i, j in ti.static(ti.ndrange(8, 8)):
            K[edof[i], edof[j]] += xe[ely, elx] ** penalty * Ke[i, j]

    # 2. Get K_freedof
    for i, j in ti.ndrange(n_free_dof,n_free_dof):
        K_freedof[i, j] = K[free_dofs_vec[i], free_dofs_vec[j]]


@ti.kernel
def backward_map_u():
    # mapping U_freedof backward to U
    for i in range(n_free_dof):
        idx = free_dofs_vec[i]
        U[idx] = U_freedof[i]


def get_ke():
    k = np.array(
        [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
         nu / 6, 1 / 8 - 3 * nu / 8])

    Ke_ = E / (1. - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                         [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                         [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                         [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                         [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                         [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                         [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                         [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

    Ke.from_numpy(Ke_)


@ti.kernel
def get_dc():
    for ely, elx in ti.ndrange(nely, nelx):
        n1 = (nely + 1) * elx + ely + 1
        n2 = (nely + 1) * (elx + 1) + ely + 1
        Ue = ti.Vector([U[2*n1 -2], U[2*n1 -1], U[2*n2-2], U[2*n2-1], U[2*n2], U[2*n2+1], U[2*n1], U[2*n1+1]])

        t = ti.Vector([0.,0.,0.,0.,0.,0.,0.,0.])
        for i in ti.static(range(8)):
            for j in ti.static(range(8)):
                t[i] += Ke[i, j] * Ue[j]
        d = 0.
        for i in ti.static(range(8)):
            d += 0.5 * Ue[i] * t[i] # d = Ue' * Ke * Ue

        compliance[None] += xe[ely, elx] ** penalty * d

        dc[ely, elx] = xe[ely, elx] ** (penalty - 1) * d


def solver():
    KG = K_freedof.to_numpy()
    Fv = F_freedof.to_numpy()
    U_freedof.from_numpy(spsolve(csr_matrix(KG),Fv)) # scipy solver, will be replaced


def beso(crtvol):
    dc_np = dc.to_numpy()
    l1 = 0 # dc_np.min()
    l2 = 1e5 # dc_np.max()
    tarvol = crtvol * nely * nelx
    x = xe.to_numpy()
    while l2 - l1 > 1e-5:
        lmid = (l2 + l1) / 2.
        for ely in range(0, nely):
            for elx in range(0,nelx):
                x[ely, elx] = 1 if dc_np[ely, elx] > lmid else xmin
        if (sum(sum(x)) - tarvol) > 0:
            l1 = lmid
        else:
            l2 = lmid
    return x


if __name__ == '__main__':
    gui = ti.GUI('Taichi TopOpt', res=(gui_x, gui_y))
    video_manager = ti.VideoManager(output_dir='./img', framerate=2, automatic_build=False)

    examples(0)
    free_dofs_vec.from_numpy(free_dofs)
    initialize()
    get_ke()
    change = 1.
    volume = 1.
    history_C = []
    while gui.running:
        x_old = xe.to_numpy()
        iter = 0
        while change > 1e-3:
            iter += 1
            compliance[None] = 0.
            dc_old = dc
            volume = max(volfrac, volume * (1-ert))
            assemble_k()
            solver()
            backward_map_u()
            get_dc()
            dc.from_numpy(filt(x_old, dc.to_numpy()))
            if iter > 1: averaging_dc()
            history_C.append(compliance[None])
            x = beso(volume)

            # check convergence
            if iter > 10:
                change = abs((sum(history_C[iter - 5:iter]) - sum(history_C[iter - 10:iter - 5])) / sum(history_C[iter - 5:iter]))

            x_old = x
            xe.from_numpy(x)
            display_sampling()
            video_manager.write_frame(display)
            print(f"iter: {iter}, volume = {volume}, compliance = {compliance[None]}, change = {change}")
            print(f'\rFrame {iter} is recorded', end=''+'\n')
            gui.set_image(display)
            gui.show()
        video_manager.make_video(gif=True)
        gui.close()