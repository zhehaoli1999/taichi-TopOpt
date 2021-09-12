import taichi as ti
import numpy as np
from utils import *
import time

# ti.init(ti.cpu, kernel_profiler=True)
ti.init(ti.cpu)

gui_y = 500
gui_x = 2 * gui_y
display = ti.field(ti.f32, shape=(gui_x, gui_y)) # field for display

nely = 10
nelx = 2 * nely
n_node = (nelx+1) * (nely+1)
ndof = 2 * n_node

E = 1.
nu = 0.3
volfrac = 0.5 # volume limit
simp_penal = 3

rho = ti.field(ti.f32, shape=(nely, nelx))
K = ti.field(ti.f32, shape=(ndof, ndof))
F = ti.field(ti.f32, shape=(ndof))
U = ti.field(ti.f32, shape=(ndof))
Ke = ti.field(ti.f32, shape=(8,8))

fixed_dofs = list(range(0, 2 * (nely + 1), 2))
fixed_dofs.append(2 * (nelx + 1) * (nely + 1) - 1)
all_dofs = list(range(0, 2 * (nelx + 1) * (nely + 1)))
free_dofs = np.array(list(set(all_dofs) - set(fixed_dofs)))
n_free_dof = len(free_dofs)

free_dofs_vec = ti.field(ti.int32, shape=n_free_dof)
K_freedof = ti.field(ti.f32, shape=(n_free_dof, n_free_dof))
F_freedof = ti.field(dtype=ti.f32, shape=(n_free_dof))
U_freedof = ti.field(dtype=ti.f32, shape=(n_free_dof))

dc = ti.field(ti.f32, shape=(nely, nelx))  # derivative of compliance

# for cg
# r = ti.field(dtype=ti.f32, shape=(n_free_dof))
# p = ti.field(dtype=ti.f32, shape=(n_free_dof))
# Ap = ti.field(dtype=ti.f32, shape=(n_free_dof))

# for mgpcg
n_mg_levels = 4
pre_and_post_smoothing = 2
bottom_smoothing = 50

r = [ti.field(dtype=ti.f32) for _ in range(n_mg_levels)]  # residual
z = [ti.field(dtype=ti.f32) for _ in range(n_mg_levels)]  # M^-1 r

p = ti.field(dtype=ti.f32)  # conjugate gradient
Ap = ti.field(dtype=ti.f32)  # matrix-vector product

alpha = ti.field(ti.f32)
beta = ti.field(ti.f32)

ti.root.pointer(ti.i, [n_free_dof // 4]).dense(ti.i, 4).place(p, Ap)

for l in range(n_mg_levels):
    ti.root.pointer(ti.i, [n_free_dof // (4 * 2**l)]).dense(ti.i, 4).place(r[l], z[l])

ti.root.place(alpha, beta)

A, x, b = ti.static(K_freedof, U_freedof, F_freedof)  # variable alias

@ti.kernel
def display_sampling():
    s_x = int(gui_x / nelx)
    s_y = int(gui_y / nely)
    for i, j in ti.ndrange(gui_x, gui_y):
        elx = i // s_x
        ely = j // s_y
        display[i, gui_y - j] = 1. - rho[ely, elx] # Note:  transpose rho here

# Optimality Criteria
def OC():
    l1 = 0.
    l2 = 1e5
    move = 0.2
    x = rho.to_numpy()
    dc_np = dc.to_numpy()
    while l2 - l1 > 1e-4:
        lmid = (l2 + l1) / 2.
        t = np.sqrt( np.abs(dc_np) / lmid)
        xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1., np.minimum(x+move, x*t))))

        if (sum(sum(xnew)) - volfrac * nely * nelx) > 0:
            l1 = lmid
        else:
            l2 = lmid

    return xnew

@ti.func
def clamp(x: ti.template(), ely, elx):
    return x[ely, elx] if 0 <= ely < nely and 0 <= elx < nelx else 0.

@ti.kernel
def derivative_filter():
    rmin = 1.2
    for ely, elx in ti.ndrange(nely, nelx):
        dc[ely, elx] = rmin * clamp(rho, ely, elx) * dc[ely, elx] +  \
                       (rmin - 1) * (clamp(rho, ely-1, elx) * clamp(dc, ely-1, elx) + \
                                    clamp(rho, ely+1, elx) * clamp(dc, ely+1, elx) + \
                                    clamp(rho, ely, elx-1) * clamp(dc, ely, elx-1) + \
                                    clamp(rho, ely, elx+1) * clamp(dc, ely, elx+1))

        weight = rmin * clamp(rho, ely, elx)  + \
                (rmin - 1) * (clamp(rho, ely-1, elx) + clamp(rho, ely+1, elx) + \
                                clamp(rho, ely, elx-1) + clamp(rho, ely, elx+1))

        dc[ely, elx] = dc[ely, elx] / weight

@ti.kernel
def assemble_K():
    for I in ti.grouped(K):
        K[I] = 0.

    # 1. Assemble Stiffness Matrix
    for ely, elx in ti.ndrange(nely, nelx):
        n1 = (nely + 1) * elx + ely + 1
        n2 = (nely + 1) * (elx + 1) + ely + 1
        edof = ti.Vector([2*n1 -2, 2*n1 -1, 2*n2 -2, 2*n2 -1, 2*n2, 2*n2+1, 2*n1, 2*n1+1])

        for i, j in ti.static(ti.ndrange(8, 8)):
            K[edof[i], edof[j]] += rho[ely, elx]**simp_penal * Ke[i, j]

    # 2. Get K_freedof
    for i, j in ti.ndrange(n_free_dof,n_free_dof):
        K_freedof[i, j] = K[free_dofs_vec[i], free_dofs_vec[j]]

@ti.kernel
def backward_map_U():
    # mapping U_freedof backward to U
    for i in range(n_free_dof):
        idx = free_dofs_vec[i]
        U[idx] = U_freedof[i]

@ti.kernel
def cg_compute_Ap(A: ti.template(), p:ti.template()):
    for I in range(n_free_dof):
        Ap[I] = 0.

    for i in range(n_free_dof):
        for j in range(n_free_dof):
            Ap[i] += A[i, j] * p[j]
    # for i, j in ti.ndrange((n_free_dof, n_free_dof)): # error

@ti.kernel
def reduce(r1: ti.template(), r2: ti.template()) -> ti.f32:
    result = 0.
    for I in range(r1.shape[0]):
        result += r1[I] * r2[I]  # dot
    return result

@ti.kernel
def multigrid_init():
    A, x, b = ti.static(K_freedof, U_freedof, F_freedof)  # variable alias
    for I in range(n_free_dof):
        r[0][I] = b[I]  # r = b - A * x = b
        z[0][I] = 0.0
        Ap[I] = 0.
        p[I] = 0.
        x[I] = 0.

@ti.kernel
def smooth(l: ti.template()):
    A, b = ti.static(K_freedof, F_freedof)  # variable alias
    # Gauss-Seidel
    for i in range(n_free_dof):
        sigma = 0.
        for j in range(n_free_dof):
            if j != i :
                sigma += A[i,j] * z[l][j]
            if A[i, i] != 0.:
                z[l][i] = (b[i] - sigma) / A[i, i]

@ti.kernel
def restrict(l: ti.template()):
    A, b = ti.static(K_freedof, F_freedof)  # variable alias

    # calculate residual
    res = 0.
    for i in range(n_free_dof):
        sum = 0.
        for j in range(n_free_dof):
            sum += A[i,j] * z[l][j]
        res += b[i] - sum

    for i in range(n_free_dof):
        r[l+1][i // 2] += res * 0.5


@ti.kernel
def prolongate(l: ti.template()):
    for I in ti.grouped(z[l]):
        z[l][I] = z[l + 1][I // 2]  # sampling for interpolation


def apply_preconditioner():
    z[0].fill(0)
    for l in range(n_mg_levels - 1):
        for i in range(pre_and_post_smoothing << l):
            smooth(l)
        z[l + 1].fill(0)
        r[l + 1].fill(0)
        restrict(l)

    for i in range(bottom_smoothing):
        smooth(n_mg_levels - 1)

    for l in reversed(range(n_mg_levels - 1)):
        prolongate(l)
        for i in range(pre_and_post_smoothing << l):
            smooth(l)

@ti.kernel
def cg_update_p():
    for I in ti.grouped(p):
        p[I] = z[0][I] + beta[None] * p[I]

@ti.kernel
def cg_update_x():
    x = ti.static(U_freedof)  # variable alias
    for I in ti.grouped(p):
        x[I] += alpha[None] * p[I]

@ti.kernel
def cg_update_r():
    for I in ti.grouped(p):
        r[0][I] -= alpha[None] * Ap[I]

def mgpcg():
    '''
    :Reference https://en.wikipedia.org/wiki/Conjugate_gradient_method
    :Reference https://en.wikipedia.org/wiki/Multigrid_method
    :Reference https://github.com/taichi-dev/taichi/blob/master/examples/algorithm/mgpcg.py

    '''

    multigrid_init()
    initial_rTr = reduce(r[0], r[0]) # Used to check convergence

    apply_preconditioner() # Get z0 = M^-1 r0

    cg_update_p() # p0 = z0
    old_zTr = reduce(r[0], z[0])

    # cg iteration
    for iter in range(n_free_dof + 50):
        print("debug")
        cg_compute_Ap(A, p)
        pAp = reduce(p, Ap)
        alpha[None] = old_zTr / pAp

        cg_update_x() # x = x + alpha * p
        cg_update_r() # r = r - alpha * Ap

        rTr = reduce(r[0], r[0])  # check convergence
        if rTr < initial_rTr * 1.0e-12:
            break

        apply_preconditioner() # update z:  z_{k+1} = M^-1 r_{k+1}
        new_zTr = reduce(r[0], z[0])
        beta[None] = new_zTr / old_zTr

        cg_update_p()
        old_zTr = new_zTr


def get_Ke():
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
    compliance = 0.
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
            d += Ue[i] * t[i] # d = Ue' * Ke * Ue

        compliance += rho[ely, elx]**simp_penal * d

        dc[ely, elx] = -simp_penal * rho[ely, elx]**(simp_penal -1) * d


@ti.kernel
def initialize():
    # 1. initialize rho
    for I in ti.grouped(rho):
        rho[I] = volfrac
    # 2. set boundary condition
    F[1] = -1.
    for i in range(n_free_dof):
        F_freedof[i] = F[free_dofs_vec[i]]

if __name__ == '__main__':
    # window = ti.ui.Window('Taichi TopoOpt', (nelx , nely))
    # while window.running:
    #     canvas = window.get_canvas()
        # canvas.set_background_color()

    gui = ti.GUI('Taichi TopoOpt', res=(gui_x, gui_y))
    free_dofs_vec.from_numpy(free_dofs)
    initialize()
    get_Ke()

    # print(K_freedof)
    # print(U)
    # print(display)
    # print(dc)

    change = 1.
    print(f"total dof = {ndof}")
    while gui.running:
        x_old = rho.to_numpy()
        iter = 0
        while change > 0.01:
            iter += 1

            assemble_K()
            mgpcg()
            backward_map_U()
            get_dc()
            derivative_filter()

            x = OC()
            volume = sum(sum(x)) / (nely * nelx)
            change = np.max(np.abs(x - x_old))

            print(f"iter: {iter}, volume = {volume}, change = {change}")

            x_old = x

            rho.from_numpy(x)
            display_sampling()

            gui.set_image(display)
            gui.show()

            ti.print_kernel_profile_info()





