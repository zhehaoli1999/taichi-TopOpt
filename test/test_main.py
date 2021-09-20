import numpy as np

nely = 4
nelx = 2 * nely
n_node = (nelx + 1) * (nely + 1)
ndof = 2 * n_node

E = 1.
nu = 0.3
volfrac = 0.5  # volume limit
simp_penal = 3

rho = np.zeros((ndof, ndof))
K = np.zeros((ndof, ndof))
F = np.zeros(ndof)
U = np.zeros(ndof)
Ke = np.zeros((8, 8))

# for mgpcg
n_mg_levels = 1
pre_and_post_smoothing = 4
bottom_smoothing = 50
use_multigrid = True

A = []
b = []

r = []  # residual
z = []  # M^-1 r

Itp = []  # interpolation operator #FIXME: [0] is useless

# Set fixed nodes
fixed_dofs = list(range(0, 2 * (nely + 1), 2))
fixed_dofs.append(2 * (nelx + 1) * (nely + 1) - 1)

# Get free dofs
# all_dofs = list(range(0, 2 * (nelx + 1) * (nely + 1)))
# free_dofs = np.array(list(set(all_dofs) - set(fixed_dofs)), dtype=int)
# n_freedof = len(free_dofs)

for l in range(n_mg_levels):
    nely_l = nely // 2**l
    nelx_l = nelx // 2**l

    ndof_l = 2 * (nely_l + 1) * (nelx_l + 1)

    A.append(np.zeros((ndof_l, ndof_l)))
    r.append(np.zeros(ndof_l))
    z.append(np.zeros(ndof_l))
    b.append(np.zeros(ndof_l))

    if l >= 1:
        nely1 = nely // 2 ** (l-1)
        nelx1 = nelx // 2 ** (l-1)
        ndof1 = 2 * (nely1 + 1) * (nelx1 + 1)
        Itp.append(np.zeros((ndof1, ndof_l)))
    if l == 0:
        Itp.append(np.zeros(1))

x = np.zeros(ndof)
p = np.zeros(ndof)
Ap = np.zeros(ndof)

def check_is_free(l, node_x_l, node_y_l, add=0):
    node_x = node_x_l * 2**l
    node_y = node_y_l * 2**l

    dof_idx = 2 * (node_x * (nely + 1) + node_y) + add
    if dof_idx in fixed_dofs:
        return False
    else:
        return  True

def init_Itp(l):

    # initialize to be 0
    for i in range(Itp[l].shape[0]):
        for j in range(Itp[l].shape[1]):
            Itp[l][i][j] = 0.

    # Require: l >= 1
    nely_2h = nely // 2**l
    nelx_2h = nelx // 2**l

    nely_h = nely // 2**(l-1)
    nelx_h = nelx // 2 ** (l - 1)

    dim = 2
    for node_y_2h in range(nely_2h + 1):
        for node_x_2h in range(nelx_2h + 1):
            node_y_h = node_y_2h * 2 # Get node coordinate in level h
            node_x_h = node_x_2h * 2

            # get indices of dof
            dof_idx_2h = [dim * (node_x_2h * (nely_2h + 1) + node_y_2h), dim * (node_x_2h * (nely_2h + 1) + node_y_2h) + 1]
            dof_idx_h = [dim * (node_x_h * (nely_h + 1) + node_y_h), dim * (node_x_h * (nely_h + 1) + node_y_h) + 1]

            # bilinear interpolation
            # situation of weight = 1
            for t in range(2):
                # if dof_idx_h[t] in freedof_idx[l-1]: # check if node in level h is fixed
                if check_is_free(l-1, node_x_h, node_y_h, t):
                    Itp[l][dof_idx_h[t]][dof_idx_2h[t]] = 1.

            # situation of weight = 1 / 2.
            for i in [-1, +1]:
                x, y = node_x_h + i, node_y_h
                if 0 <= x < nelx_h:
                    dof_h = [dim * (x * (nely_h + 1) + y), dim * (x * (nely_h + 1) + y) + 1]
                    for t in range(2):
                        # if dof_h[t] in freedof_idx[l - 1]: # check if fixed
                        if check_is_free(l - 1, x, y, t):
                            Itp[l][dof_h[t]][dof_idx_2h[t]] = 1 / 2.

            for j in [-1, +1]:
                x, y = node_x_h, node_y_h + j
                if 0 <= y < nely_h:
                    dof_h = [dim * (x * (nely_h + 1) + y), dim * (x * (nely_h + 1) + y) + 1]
                    for t in range(2):
                        # if dof_h[t] in freedof_idx[l - 1]: # check if fixed
                        if check_is_free(l - 1, x, y, t):
                            Itp[l][dof_h[t]][dof_idx_2h[t]] = 1 / 2.

            # situation of weight = 1 / 4.
            for i in [-1, +1]:
                for j in [-1, +1]:
                    x = node_x_h + i
                    y = node_y_h + j
                    if 0<= x < nelx_h and 0 <= y < nely_h:
                        dof_h = [dim * (x* (nely_h + 1) + y), dim * (x * (nely_h + 1) + y) + 1]
                        for t in range(2):
                            # if dof_h[t] in freedof_idx[l-1]:
                            if check_is_free(l - 1, x, y, t):
                                Itp[l][dof_h[t]][dof_idx_2h[t]] = 1 / 4.




def initialize():
    # 1. initialize rho
    for i in range(ndof):
        for j in range(ndof):
            rho[i][j] = volfrac

    # 2. initialize Itp
    for l in range(1, n_mg_levels):
        init_Itp(l)

    # 3. initialize F and b
    F[1] = -1.
    b[0] = F
    for l in range(1, n_mg_levels):
        b[l] = b[l - 1] @ Itp[l] # TODO


def mg_get_Kl(l):
    # Require l >= 1
    # Use Garlekin coarsening: K[l+1] = R[l+1] @ K[l] @ I[l+1], where R = transpose(I)
    global A

    A[l] = Itp[l].transpose() @ A[l-1] @ Itp[l]

    #     n1 = n_freedof // 2 ** (l - 1)
    #     n2 = n_freedof // 2 ** l
    # for i in range(n2):
    #     for j in range(n2):
    #         A[l][i][j] = 0.
    #         for t in range(n1):
    #             s = 0.
    #             for m in range(n1):
    #                 s += A[l - 1][t][m] * Itp[l][m][j]
    #             s = s * Itp[l][t][i]
    #             A[l][i][j] += s
    # print(A[l])
    # Temp = Itp[l].transpose() @ A[l - 1] @ Itp[l]
    # print(Temp)


def assemble_K():
    global K, A

    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i][j] = 0.

    # 1. Assemble Stiffness Matrix
    for ely in range(nely):
        for elx in range(nelx):
            n1 = (nely + 1) * elx + ely + 1
            n2 = (nely + 1) * (elx + 1) + ely + 1
            edof = [2 * n1 - 2, 2 * n1 - 1, 2 * n2 - 2, 2 * n2 - 1, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1]

            for i in range(8):
                for j in range(8):
                    K[edof[i]][edof[j]] += rho[ely][elx] ** simp_penal * Ke[i][j]

    # 2. Get A[0]
    A[0] = K


def multigrid_init():
    for I in range(ndof):
        r[0][I] = b[0][I]  # r = b - A * x = b
        z[0][I] = 0.0
        Ap[I] = 0.
        p[I] = 0.
        x[I] = 0.

def smooth(l):
    # Gauss-Seidel #TODO: red-black GS or Jacobi for parallelization
    n_node_y_l = nely // 2**l + 1
    n_node_x_l = nelx // 2**l + 1
    for x in range(n_node_x_l):
        for y in range(n_node_y_l):

            # Get two dof
            for t in range(2):
                i = 2 * (x * n_node_y_l + y) + t
                if check_is_free(l, x, y, t): # Check boundary condition
                    sigma = 0.
                    for j in range(A[l].shape[1]):
                        if j != i:
                            sigma += A[l][i][j] * z[l][j]
                    if A[l][i][i] != 0.:
                        z[l][i] = (r[l][i] - sigma) / A[l][i][i]
                else:
                    z[l][i] = 0.


def restrict(l):
    # calculate residual
    n_node_y_l = nely // 2 ** l + 1
    n_node_x_l = nelx // 2 ** l + 1
    for x in range(n_node_x_l):
        for y in range(n_node_y_l):

            # Get two dof
            for t in range(2):
                i = 2 * (x * n_node_y_l + y) + t
                if check_is_free(l, x, y, t):  # Check boundary condition
                    sum = 0.
                    for j in range(A[l].shape[1]):
                        sum += A[l][i][j] * z[l][j]
                    r[l][i] = b[l][i] - sum  # get residual
                else:
                    r[l][i] = 0.

    # down sample residual on fine grid to coarse grid
    # n_freedof_2l = n_freedof // 2 ** (l + 1)
    # for i in range(n_freedof_2l):
    #     for j in range(n_freedof_l):
    #         r[l + 1][i] += r[l][j] * Itp[l + 1][j][i]  # r[l+1] = r[l] @ I[l+1]
    r[l+1] = r[l] @ Itp[l+1]


def prolongate(l):
    # interpolate coarse to fine
    # for i in range(n_freedof_l):
    #     for j in range(n_freedof_2l):
    #         z[l][i] = Itp[l + 1][i][j] * z[l + 1][j]  # z[l] = z[l+1] @ I[l+1]^T

    z[l] = z[l+1] @ Itp[l+1].transpose()

def apply_preconditioner():
    z[0].fill(0)
    for l in range(n_mg_levels - 1):
        for i in range(pre_and_post_smoothing << l):  # equals to pre_and_post_smoothing // 2**l
            smooth(l)
            # print("============")
            # print(z[0])
        z[l + 1].fill(0)
        r[l + 1].fill(0)
        restrict(l)
    # print("============")
    # print(r[1])
    for i in range(bottom_smoothing):
        smooth(n_mg_levels - 1)
    # print("============")
    # print(A[1])
    # print("============")
    # print(z[1])
    # print("============")
    # print(z[0])
    # print("============")
    # print(z[1])

    for l in reversed(range(n_mg_levels - 1)):
        prolongate(l)
        for i in range(pre_and_post_smoothing << l):
            smooth(l)

def cg_compute_Ap(A, p):
    for I in range(ndof):
        Ap[I] = 0.

    for i in range(ndof):
        for j in range(ndof):
            Ap[i] += A[i][j] * p[j]
    # for i, j in ti.ndrange((n_freedof, n_freedof)): # error


def reduce(r1, r2):
    result = 0.
    for I in range(r1.shape[0]):
        result += r1[I] * r2[I]  # dot
    return result


def cg_update_p(beta):
    for i in  range(len(p)):
        p[i] = z[0][i] + beta * p[i]


def cg_update_x(alpha):
    for i in  range(len(p)):
        if i in fixed_dofs:
            x[i] = 0.
        else:
            x[i] += alpha * p[i]


def cg_update_r(alpha):
    for i in  range(len(p)):
        if i in fixed_dofs:
            r[0][i] = 0.
        else:
            r[0][i] -= alpha * Ap[i]


def mgpcg():
    '''
    :Reference https://en.wikipedia.org/wiki/Conjugate_gradient_method
    :Reference https://en.wikipedia.org/wiki/Multigrid_method
    :Reference https://github.com/taichi-dev/taichi/blob/master/examples/algorithm/mgpcg.py

    '''
    multigrid_init()
    initial_rTr = reduce(r[0], r[0])  # Used to check convergence

    if use_multigrid:
        apply_preconditioner()  # Get z0 = M^-1 r0
    else:
        z[0] = r[0]

    cg_update_p(beta=0.)  # p0 = z0
    old_zTr = reduce(r[0], z[0])

    # for iter in range(ndof+50):
    #     xold = x
    # #     # print(z[0])
    #     apply_preconditioner()
    #     print(np.linalg.norm(xold - x))

    # cg iteration
    for iter in range(ndof + 50):
        cg_compute_Ap(A[0], p)
        pAp = reduce(p, Ap)

        alpha = old_zTr / pAp

        cg_update_x(alpha)  # x = x + alpha * p
        cg_update_r(alpha)  # r = r - alpha * Ap

        rTr = reduce(r[0], r[0])  # check convergence
        print(f"mgpcg res: {rTr / initial_rTr}")
        if rTr < initial_rTr * 1.0e-12:
            break

        if use_multigrid:
            apply_preconditioner()  # update z:  z_{k+1} = M^-1 r_{k+1}
        else:
            z[0] = r[0]

        new_zTr = reduce(r[0], z[0])

        beta = new_zTr / old_zTr

        cg_update_p(beta)
        old_zTr = new_zTr


def get_Ke():
    k = np.array(
        [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
         nu / 6, 1 / 8 - 3 * nu / 8])

    global Ke
    Ke = E / (1. - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                         [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                         [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                         [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                         [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                         [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                         [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                         [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])



if __name__ == '__main__':

    initialize()
    get_Ke()

    assemble_K()

    for l in range(1, n_mg_levels):
        mg_get_Kl(l)

    mgpcg()
