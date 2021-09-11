import taichi as ti
import numpy as np
import utils

ti.init(arch=ti.cpu)

gui_x = 1000
gui_y = 500
display = ti.field(ti.f32, shape=(gui_x, gui_y))


nelx = 128
nely = 64
n_node = (nelx+1) * (nely+1)
ndof = 2 * n_node

E = 1.
nu = 0.3
volfrac = 0.3

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
idx_to_dof = np.array([free_dofs[i] for i in range(n_free_dof)])

idx_to_dof_vec = ti.field(ti.int32, shape=n_free_dof)
free_dofs_vec = ti.field(ti.int32, shape=n_free_dof)
K_effective = ti.field(ti.f32, shape=(n_free_dof, n_free_dof))
F_effective = ti.field(ti.f32, shape=(n_free_dof))
U_effective = ti.field(ti.f32, shape=(n_free_dof))

dc = ti.field(ti.f32, shape=(nely, nelx)) # derivative

@ti.kernel
def render():
    s_x = int(gui_x / nelx)
    s_y = int(gui_y / nely)
    for i, j in ti.ndrange(gui_x, gui_y):
        elx = i % s_x
        ely = j % s_y
        display[i, j] = rho[ely, elx] # Note:  transpose rho here


def OC(volfrac):
    l1 = 0.
    l2 = 1e5
    move = 0.2
    x = rho.to_numpy()
    dc_np = dc.to_numpy()
    while l2 - l1 > 1e-4:
        lmid = (l2 - l1) / 2.
        t = np.sqrt( - dc_np / lmid)
        xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1., np.minimum(x+move, x*t))))

        if (sum(sum(xnew)) - volfrac * nely * nelx) > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew

@ti.func
def filter():
    pass

@ti.func
def fem(nelx, nely, rho, penal):
    for ely, elx in ti.ndrange(nely, nelx):
        n1 = (nely + 1) * (elx - 1) + ely
        n2 = (nely + 1) * elx + ely
        edof = ti.Vector([2*n1 -2, 2*n1 -1, 2*n2 -2, 2*n2 -1, 2*n2, 2*n2+1, 2*n1, 2*n1+1])

        for i, j in ti.static(ti.ndrange(8, 8)):
            K[edof[i], edof[j]] += rho[ely, elx]**penal * Ke[i, j]

    for i, j in ti.ndrange(n_free_dof,n_free_dof):
        K_effective[i, j] = K[free_dofs_vec[i],free_dofs_vec[j]]
    for i in range(n_free_dof):
        F_effective[i] = F[free_dofs_vec[i]]

    jacobi_iter(5)


@ti.func
def jacobi_iter(max_iter):
    for I in ti.grouped(U_effective):
        U_effective[I] = 0.

    for iter in range(max_iter):
        for i in range(n_free_dof):
            sigma = 0.
            for j in range(n_free_dof):
                if j != i :
                    sigma += K_effective[i,j] * U_effective[j]
            if K_effective[i, i] != 0.:
                U_effective[i] = (F_effective[i] - sigma) / K_effective[i, i]

    # mapping backward
    for i in range(n_free_dof):
        idx = idx_to_dof_vec[i]
        U[idx] = U_effective[i]

@ti.func
def cg_iter():
    for I in ti.grouped(U_effective):
        U_effective[I] = 0.
    pass #TODO


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
    free_dofs_vec.from_numpy(free_dofs)
    idx_to_dof_vec.from_numpy(idx_to_dof)


@ti.kernel
def topo_opt(penal: int):
    fem(nelx, nely, rho, 3)

    compliance = 0.
    for ely, elx in ti.ndrange(nely, nelx):
        n1 = (nely + 1) * (elx - 1)+ely
        n2 = (nely + 1) * elx + ely
        Ue = ti.Vector([U[2*n1 -2], U[2*n1 -1], U[2*n2-2], U[2*n2-1], U[2*n2], U[2*n2+1], U[2*n1], U[2*n1+1]])

        t = ti.Vector([0.,0.,0.,0.,0.,0.,0.,0.])
        utils.scalar_mat_mul_vec(Ke, Ue, t)

        d = utils.vec_dot(Ue, t)
        compliance += rho[ely, elx]**penal * d

        dc[ely, elx] = -penal * rho[ely, elx]**(penal -1) * d


@ti.kernel
def initialize():
    # 1. initialize rho
    for I in ti.grouped(rho):
        rho[I] = volfrac
    # 2. set boundary condition
    F[1] = -1.

if __name__ == '__main__':
    # window = ti.ui.Window('Taichi TopoOpt', (nelx , nely))
    # while window.running:
    #     canvas = window.get_canvas()
        # canvas.set_background_color()

    gui = ti.GUI('Taichi TopoOpt', res=(gui_x, gui_y))
    initialize()
    get_Ke()
    topo_opt(3)
    x = OC(volfrac)
    rho.from_numpy(x)
    print(x)
    render()

    print(display)

    while gui.running:
        gui.set_image(display)
        gui.show()





