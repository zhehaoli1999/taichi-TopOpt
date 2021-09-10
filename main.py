import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

nelx = 128
nely = 64
n_node = (nelx+1) * (nely+1)
ndof = 2 * n_node

E = 1.
nu = 0.3
volfrac = 0.3

rho = ti.field(ti.f32, shape=(nelx, nely))
K = ti.field(ti.f32, shape=(ndof, ndof))
F = ti.field(ti.f32, shape=(ndof, 1))
U = ti.field(ti.f32, shape=(ndof, 1))
Ke = ti.field(ti.f32, shape=(8,8))

fixed_dofs = list(range(0, 2*(nely+1), 2))
fixed_dofs.append(2*(nelx+1)*(nely+1)-1)
all_dofs = list(range(0, 2*(nelx+1)*(nely+1)))
free_dofs = list(set(all_dofs) - set(fixed_dofs))


@ti.func
def fem(nelx, nely, rho, penal):
    for elx, ely in ti.ndrange(nelx, nely):
        n1 = (nely + 1) * (elx - 1) + ely
        n2 = (nely + 1) * elx + ely
        edof = ti.Vector([2*n1 -1, 2*n1, 2*n2 -1, 2*n2, 2*n2+1, 2*n2+2, 2*n1+1, 2*n1+2])

        for i, j in ti.static(ti.ndrange(8, 8)):
            K[edof[i], edof[j]] += rho[ely, elx]**penal * Ke[i, j]

    # use mgpcg to solve linear system

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
def topo_opt():
    fem(nelx, nely, rho, 3)

@ti.kernel
def initialize():
    # 1. initialize rho
    for I in ti.grouped(rho):
        rho[I] = volfrac
    # 2. set boundary condition
    F[1, 0] = -1.

if __name__ == '__main__':
    initialize()
    get_Ke()
    print(Ke)
    topo_opt()
    print(K)





