import taichi as ti
import numpy as np

# ti.init(ti.cpu, kernel_profiler=True)
ti.init(ti.cpu)

gui_y = 500
gui_x = 2 * gui_y
display = ti.field(ti.f32, shape=(gui_x, gui_y)) # field for display

nely = 8
nelx = 2 * nely
n_node = (nelx+1) * (nely+1)
ndof = 2 * n_node

E = 1.
nu = 0.3
volfrac = 0.5 # volume limit
simp_penal = 3

rho = ti.field(ti.f32, shape=(nely, nelx))
# K = ti.field(ti.f32, shape=(ndof, ndof))
F = ti.field(ti.f32, shape=(ndof))
U = ti.field(ti.f32, shape=(ndof))
Ke = ti.field(ti.f32, shape=(8,8))

dc = ti.field(ti.f32, shape=(nely, nelx))  # derivative of compliance

# for mgpcg
n_mg_levels = 2
pre_and_post_smoothing = 2
bottom_smoothing = 50
use_multigrid = False

K = [ti.field(dtype=ti.f32) for _ in range(n_mg_levels)]
for l in range(n_mg_levels):
    # ti.root.pointer(ti.ij, [ndof // (1 * 2**l)]).dense(ti.ij, 1).place(K[l])
    ti.root.dense(ti.ij, ndof // 2**l).place(K[l])

K_freedof = [ti.field(ti.f32) for _ in range(n_mg_levels)]
F_freedof = [ti.field(ti.f32) for _ in range(n_mg_levels)]
freedof_idx = [ti.field(dtype=ti.int32) for _ in range(n_mg_levels)] # free dof indices
r = [ti.field(dtype=ti.f32) for _ in range(n_mg_levels)]  # residual
z = [ti.field(dtype=ti.f32) for _ in range(n_mg_levels)]  # M^-1 r

n_freedof_list = [] # a list of num of free dofs in each level
freedofs_list = []

for l in range(n_mg_levels):
    nelx_l = nelx // 2 ** l
    nely_l = nely // 2 ** l
    ndof_l = ndof // 2 ** l

    # Set fixed nodes
    fixed_dofs = list(range(0, 2 * (nely_l + 1), 2))
    fixed_dofs.append(2 * (nelx_l + 1) * (nely_l + 1) - 1)

    # Get free dofs
    all_dofs = list(range(0, 2*(nelx_l + 1) * (nely_l + 1)))
    free_dofs = np.array(list(set(all_dofs) - set(fixed_dofs)))

    freedofs_list.append(free_dofs)
    n_free_dof_l = len(free_dofs)
    n_freedof_list.append(n_free_dof_l)

    # Place taichi fields
    ti.root.dense(ti.ij, (n_free_dof_l, n_free_dof_l)).place(K_freedof[l])
    ti.root.dense(ti.i, n_free_dof_l).place(freedof_idx[l])
    ti.root.dense(ti.i, n_free_dof_l).place(r[l], z[l], F_freedof[l])

n_freedof_np = np.array(n_freedof_list, dtype=int)
n_freedof = ti.field(ti.int32, shape=(n_mg_levels))
n_freedof.from_numpy(n_freedof_np)  # num of free dofs in each level

# print(f"n_freedof of each level: {n_freedof}")

for l in range(n_mg_levels):
   freedof_idx[l].from_numpy(freedofs_list[l])
   F_freedof[l][0] = -1
   # print(f"free_dof_vec[{l}]: {freedof_idx[l]}")


U_freedof = ti.field(dtype=ti.f32, shape=(n_freedof[0]))
# F_freedof = ti.field(dtype=ti.f32, shape=(n_freedof[0]))

p = ti.field(dtype=ti.f32, shape=(n_freedof[0]))  # conjugate gradient
Ap = ti.field(dtype=ti.f32, shape=(n_freedof[0]))  # matrix-vector product
# ti.root.pointer(ti.i, [ndof // 1]).dense(ti.i, 1).place(p, Ap)
# ti.root.dense(ti.i, n_freedof[0]).place(p)
# ti.root.dense(ti.i, n_freedof[0]).place(Ap)

alpha = ti.field(ti.f32)
beta = ti.field(ti.f32)
ti.root.place(alpha, beta)

A, x, b = ti.static(K_freedof, U_freedof, F_freedof)  # variable alias

@ti.kernel
def initialize():
    # 1. initialize rho
    for I in ti.grouped(rho):
        rho[I] = volfrac
    # 2. set boundary condition
    # F[1] = -1.
    # for i in range(n_freedof[0]):
    #     F_freedof[i] = F[freedof_idx[0][i]]


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

# @ti.kernel
# def assemble_K():
#     for I in ti.grouped(K):
#         K[I] = 0.
#
#     # 1. Assemble Stiffness Matrix
#     for ely, elx in ti.ndrange(nely, nelx):
#         n1 = (nely + 1) * elx + ely + 1
#         n2 = (nely + 1) * (elx + 1) + ely + 1
#         edof = ti.Vector([2*n1 -2, 2*n1 -1, 2*n2 -2, 2*n2 -1, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
#
#         for i, j in ti.static(ti.ndrange(8, 8)):
#             K[edof[i], edof[j]] += rho[ely, elx]**simp_penal * Ke[i, j]
#
#     # 2. Get K_freedof
#     for i, j in ti.ndrange(n_freedof,n_freedof):
#         K_freedof[i, j] = K[freedof_idx[i], freedof_idx[j]]

@ti.kernel
def mg_assemble_K(l : ti.template()):
    # for l in range(n_mg_levels):
    nely_l = nely // 2**l
    nelx_l = nelx // 2**l
    for I in ti.grouped(K[l]):
        K[l][I] = 0.

    # 1. Assemble Stiffness Matrix
    for ely, elx in ti.ndrange(nely_l, nelx_l):
        n1 = (nely_l + 1) * elx + ely + 1
        n2 = (nely_l + 1) * (elx + 1) + ely + 1
        edof = ti.Vector([2*n1 -2, 2*n1 -1, 2*n2 -2, 2*n2 -1, 2*n2, 2*n2+1, 2*n1, 2*n1+1])

        for i, j in ti.static(ti.ndrange(8, 8)):
            K[l][edof[i], edof[j]] += rho[ely * 2**l, elx * 2**l]**simp_penal * Ke[i, j] #FIXME problem with round error

    # 2. Get K_freedof
    n_free_dof_l = n_freedof[l]
    for i, j in ti.ndrange(n_free_dof_l, n_free_dof_l):
        K_freedof[l][i, j] = K[l][freedof_idx[l][i], freedof_idx[l][j]]


@ti.kernel
def backward_map_U():
    # mapping U_freedof backward to U
    for i in range(n_freedof[0]):
        idx = freedof_idx[0][i]
        U[idx] = U_freedof[i]

@ti.kernel
def multigrid_init():
    for I in range(n_freedof[0]):
        r[0][I] = b[0][I]  # r = b - A * x = b
        z[0][I] = 0.0
        Ap[I] = 0.
        p[I] = 0.
        x[I] = 0.

@ti.kernel
def smooth(l: ti.template()):
    # Gauss-Seidel #TODO: red-black GS or Jacobi for parallelization
    for i in range(n_freedof[l]):
        sigma = 0.
        for j in range(n_freedof[l]):
            if j != i :
                sigma += A[l][i, j] * z[l][j]
            if A[l][i, i] != 0.:
                z[l][i] = (r[l][i] - sigma) / A[l][i, i]

@ti.func
def clamp_vec(v: ti.template(), l: ti.template(), n):
    return v[l][n] if 0 <= n < n_freedof[l] else 0.

@ti.func
def down_sampling(r: ti.template(), i):
    i /

@ti.kernel
def restrict(l: ti.template()):
    # calculate residual
    for i in range(n_freedof[l]):
        sum = 0.
        for j in range(n_freedof[l]):
            sum += A[l][i,j] * z[l][j]
        r[l][i] = b[l][i] - sum # get residual

    for i in range(n_freedof[l+1]):
        # r[l+1][i] = (clamp_vec(r, l, i * 2) + clamp_vec(r, l, i*2 + 1)) / 2. # down sampling #TODO
        r[l + 1][i] = clamp_vec(r, l, i * 2) # down sampling


@ti.kernel
def prolongate(l: ti.template()):
    for I in range(n_freedof[l]):
        z[l][I] = clamp_vec(z, l + 1, I // 2)  # sampling for interpolation


def apply_preconditioner():
    z[0].fill(0)
    for l in range(n_mg_levels - 1):
        for i in range(pre_and_post_smoothing << l): # equals to pre_and_post_smoothing // 2**l
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
def cg_compute_Ap(A: ti.template(), p:ti.template()):
    for I in range(ndof):
        Ap[I] = 0.

    for i in range(ndof):
        for j in range(ndof):
            Ap[i] += A[i, j] * p[j]
    # for i, j in ti.ndrange((n_freedof, n_freedof)): # error

@ti.kernel
def reduce(r1: ti.template(), r2: ti.template()) -> ti.f32:
    result = 0.
    for I in range(r1.shape[0]):
        result += r1[I] * r2[I]  # dot
    return result

@ti.kernel
def cg_update_p():
    for I in ti.grouped(p):
        p[I] = z[0][I] + beta[None] * p[I]

@ti.kernel
def cg_update_x():
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

    if use_multigrid:
        apply_preconditioner() # Get z0 = M^-1 r0
    else:
        z[0].copy_from(r[0])

    cg_update_p() # p0 = z0
    old_zTr = reduce(r[0], z[0])

    # cg iteration
    for iter in range(n_freedof[0] + 50):
        cg_compute_Ap(A[0], p)
        pAp = reduce(p, Ap)
        alpha[None] = old_zTr / pAp

        cg_update_x() # x = x + alpha * p
        cg_update_r() # r = r - alpha * Ap

        rTr = reduce(r[0], r[0])  # check convergence
        print(f"mgpcg res: {rTr / initial_rTr}")
        if rTr < initial_rTr * 1.0e-12:
            break

        if use_multigrid:
            apply_preconditioner()  #  update z:  z_{k+1} = M^-1 r_{k+1}
        else:
            z[0].copy_from(r[0])

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



if __name__ == '__main__':
    # window = ti.ui.Window('Taichi TopoOpt', (nelx , nely))
    # while window.running:
    #     canvas = window.get_canvas()
        # canvas.set_background_color()

    gui = ti.GUI('Taichi TopoOpt', res=(gui_x, gui_y))
    initialize()
    get_Ke()

    # print(K_freedof)
    # print(U)
    # print(display)
    # print(dc)

    change = 1.


    for l in range(n_mg_levels):
        mg_assemble_K(l)
    # print(K_freedof[1])

    mgpcg()
    print(U_freedof)
    # print(f"total dof = {ndof}")
    # while gui.running:
    #     x_old = rho.to_numpy()
    #     iter = 0
    #     while change > 0.01:
    #         iter += 1
    #
    #         for l in range(n_mg_levels):
    #             mg_assemble_K(l)
    #         mgpcg()
    #         backward_map_U()
    #         get_dc()
    #         derivative_filter()
    #
    #         x = OC()
    #         volume = sum(sum(x)) / (nely * nelx)
    #         change = np.max(np.abs(x - x_old))
    #
    #         print(f"iter: {iter}, volume = {volume}, change = {change}")
    #
    #         x_old = x
    #
    #         rho.from_numpy(x)
    #         display_sampling()
    #
    #         gui.set_image(display)
    #         gui.show()

            # ti.print_kernel_profile_info()





