import taichi as ti
import numpy as np

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
rmin = 3
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
r = ti.field(dtype=ti.f32, shape=(n_free_dof))
p = ti.field(dtype=ti.f32, shape=(n_free_dof))
Ap = ti.field(dtype=ti.f32, shape=(n_free_dof))

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

@ti.func
def cg_compute_Ap(A: ti.template(), p:ti.template()):
    for I in range(n_free_dof):
        Ap[I] = 0.

    for i in range(n_free_dof):
        for j in range(n_free_dof):
            Ap[i] += A[i, j] * p[j]
    # for i, j in ti.ndrange((n_free_dof, n_free_dof)): # error

@ti.func
def reduce(r: ti.template()):
    result = 0.
    for I in range(r.shape[0]):
        result += r[I] * r[I]  # rsnew = r' * r
    return result

@ti.kernel
def conjungate_gradient():
    A, x, b = ti.static(K_freedof, U_freedof, F_freedof) # variable alias

    # init
    for I in ti.grouped(b):
        x[I] = 0.
        r[I] = b[I]  # r = b - A * x = b
        p[I] = r[I]

    rsold = reduce(r)

    # cg iteration
    for iter in range(n_free_dof + 50):
        cg_compute_Ap(A, p)

        beta = 0.
        for I in range(n_free_dof):
            beta += p[I] * Ap[I] # p' * Ap
        alpha = rsold / beta

        for I in range(n_free_dof):
            x[I] += alpha * p[I]  # x = x + alpha * Ap
            r[I] -= alpha * Ap[I] # r = r - alpha * Ap

        rsnew = reduce(r)

        if ti.sqrt(rsnew) < 1e-10:
            break

        for I in range(n_free_dof):
            p[I] = r[I] + (rsnew / rsold) * p[I] # p = r + (rsnew / rsold) * p

        rsold = rsnew

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
def get_dc() -> ti.f32:
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
    return compliance

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
            conjungate_gradient()
            backward_map_U()
            compliance = get_dc()
            derivative_filter()

            x = OC()
            volume = sum(sum(x)) / (nely * nelx)
            change = np.max(np.abs(x - x_old))

            print(f"iter: {iter}, volume = {volume}, compliance = {compliance}, change = {change}")

            x_old = x

            rho.from_numpy(x)
            display_sampling()

            gui.set_image(display)
            gui.show()

            # ti.print_kernel_profile_info()


