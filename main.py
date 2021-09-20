import taichi as ti
import numpy as np
from solver import fem_mgpcg

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
K = ti.field(ti.f32, shape=(ndof, ndof))
F = ti.field(ti.f32, shape=(ndof))
U = ti.field(ti.f32, shape=(ndof))
Ke = ti.field(ti.f32, shape=(8,8))

dc = ti.field(ti.f32, shape=(nely, nelx))  # derivative of compliance

# set fixed dof
fixed_dofs = list(range(0, 2 * (nely + 1), 2))
fixed_dofs.append(2 * (nelx + 1) * (nely + 1) - 1)
fixed_dofs_vec = ti.field(ti.f32, shape=(len(fixed_dofs)))
fixed_dofs_vec.from_numpy(np.array(fixed_dofs))

# set force
F[1] = -1.

@ti.kernel
def init_rho():
    for I in ti.grouped(rho):
        rho[I] = volfrac


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
    gui = ti.GUI('Taichi TopoOpt', res=(gui_x, gui_y))
    init_rho()
    get_Ke()
    assemble_K()

    solver = fem_mgpcg(nelx=nelx, nely=nely, fixed_dof=fixed_dofs_vec, K=K, F=F)

    change = 1.

    print(f"total dof = {ndof}")
    while gui.running:
        x_old = rho.to_numpy()
        iter = 0
        while change > 0.01:
            iter += 1

            solver.solve()

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

            # ti.print_kernel_profile_info()





