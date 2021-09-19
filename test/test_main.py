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
n_mg_levels = 2
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
all_dofs = list(range(0, 2 * (nelx + 1) * (nely + 1)))
free_dofs = np.array(list(set(all_dofs) - set(fixed_dofs)), dtype=int)
n_freedof = len(free_dofs)

for l in range(n_mg_levels):
    n_freedof_l = n_freedof // 2 ** l

    A.append(np.zeros((n_freedof_l, n_freedof_l)))
    r.append(np.zeros(n_freedof_l))
    z.append(np.zeros(n_freedof_l))
    b.append(np.zeros(n_freedof_l))

    if l >= 1:
        Itp.append(np.zeros((n_freedof // 2 ** (l - 1), n_freedof // 2 ** l)))
    if l == 0:
        Itp.append(np.zeros(1))

freedof_idx = free_dofs

x = np.zeros(n_freedof)
p = np.zeros(n_freedof)
Ap = np.zeros(n_freedof)

# alpha = 0.
# beta = 0.

F[1] = -1.
for l in range(n_mg_levels):
    for i in range(n_freedof // 2 ** l):
        b[l][i] = F[freedof_idx[i * 2 ** l]]  # sampling


def initialize():
    # 1. initialize rho
    for i in range(ndof):
        for j in range(ndof):
            rho[i][j] = volfrac

def mg_get_Kl(l):
    # Require l >= 1
    # Use Garlekin coarsening: K[l+1] = R[l+1] @ K[l] @ I[l+1]
    # R = transpose(I)
    n1 = n_freedof // 2 ** (l - 1)
    n2 = n_freedof // 2 ** l

    global A

    A[l] = Itp[l].transpose() @ A[l-1] @ Itp[l]

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
    for i in range(n_freedof):
        for j in range(n_freedof):
            A[0][i][j] = K[freedof_idx[i]][freedof_idx[j]]
            # print(A[0][i][j])


def backward_map_U():
    # mapping x backward to U
    for i in range(n_freedof):
        idx = freedof_idx[i]
        U[idx] = x[i]


def multigrid_init():
    for I in range(n_freedof):
        r[0][I] = b[0][I]  # r = b - A * x = b
        z[0][I] = 0.0
        Ap[I] = 0.
        p[I] = 0.
        x[I] = 0.


def init_Itp(l):
    # Require: l >= 1
    n1 = n_freedof // 2 ** (l - 1)
    n2 = n_freedof // 2 ** l

    nely1 = int(np.sqrt(n1))
    n_node1 = nely1 + 1

    for i in range(Itp[l].shape[0]):
        for j in range(Itp[l].shape[1]):
            Itp[l][i][j] = 0.

    for i in range(n2):
        Itp[l][i * 2][i] = 1.

        a = [i * 2 - 2, i * 2 + 2, i * 2 - n_node1, i * 2 + n_node1]
        for t in range(4):
            if 0 <= a[t] < n1:
                Itp[l][a[t]][i] = 1 / 2.

        # b = [i * 2 - n_node1 - 2, i * 2 - n_node1 + 2, i * 2 + n_node1 - 2, i * 2 + n_node1 + 2]
        # for t in range(4):
        #     if 0 <= b[t] < n1:
        #         Itp[l][b[t]][i] = 1 / 4.


def smooth(l):
    # Gauss-Seidel #TODO: red-black GS or Jacobi for parallelization
    n_freedof_l = n_freedof // 2 ** l
    for i in range(n_freedof_l):
        sigma = 0.
        for j in range(n_freedof_l):
            if j != i:
                sigma += A[l][i][j] * z[l][j]
            if A[l][i][i] != 0.:
                z[l][i] = (r[l][i] - sigma) / A[l][i][i]


def restrict(l):
    # calculate residual
    n_freedof_l = n_freedof // 2 ** l
    for i in range(n_freedof_l):
        sum = 0.
        for j in range(n_freedof_l):
            sum += A[l][i][j] * z[l][j]
        r[l][i] = b[l][i] - sum  # get residual

    # down sample residual on fine grid to coarse grid
    # n_freedof_2l = n_freedof // 2 ** (l + 1)
    # for i in range(n_freedof_2l):
    #     for j in range(n_freedof_l):
    #         r[l + 1][i] += r[l][j] * Itp[l + 1][j][i]  # r[l+1] = r[l] @ I[l+1]
    r[l+1] = r[l] @ Itp[l+1]


def prolongate(l):
    n_freedof_l = n_freedof // 2 ** l
    n_freedof_2l = n_freedof // 2 ** (l + 1)

    # interpolate coarse to fine
    # for i in range(n_freedof_l):
    #     for j in range(n_freedof_2l):
    #         z[l][i] = Itp[l + 1][i][j] * z[l + 1][j]  # z[l] = z[l+1] @ I[l+1]^T

    z[l] = z[l+1] @ Itp[l+1].transpose()

def apply_preconditioner():
    z[0].fill(0)
    # print(z[0])
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
    # print("============")
    # print(z[0])


def cg_compute_Ap(A, p):
    for I in range(n_freedof):
        Ap[I] = 0.

    for i in range(n_freedof):
        for j in range(n_freedof):
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
        x[i] += alpha * p[i]


def cg_update_r(alpha):
    for i in  range(len(p)):
        r[0][i] -= alpha * Ap[i]


def mgpcg():
    '''
    :Reference https://en.wikipedia.org/wiki/Conjugate_gradient_method
    :Reference https://en.wikipedia.org/wiki/Multigrid_method
    :Reference https://github.com/taichi-dev/taichi/blob/master/examples/algorithm/mgpcg.py

    '''
    multigrid_init()
    initial_rTr = reduce(r[0], r[0])  # Used to check convergence

    for iter in range(100):
        apply_preconditioner()
        print(sum(r[0]))

    # if use_multigrid:
    #     apply_preconditioner()  # Get z0 = M^-1 r0
    # else:
    #     z[0] = r[0]

    # print(z[0])

    # cg_update_p(beta=0.)  # p0 = z0
    # old_zTr = reduce(r[0], z[0])

    # cg iteration
    # for iter in range(n_freedof + 50):
    #     cg_compute_Ap(A[0], p)
    #     pAp = reduce(p, Ap)
    #
    #     alpha = old_zTr / pAp
    #
    #     cg_update_x(alpha)  # x = x + alpha * p
    #     cg_update_r(alpha)  # r = r - alpha * Ap
    #
    #     rTr = reduce(r[0], r[0])  # check convergence
    #     print(f"mgpcg res: {rTr / initial_rTr}")
    #     if rTr < initial_rTr * 1.0e-12:
    #         break
    #
    #     if use_multigrid:
    #         apply_preconditioner()  # update z:  z_{k+1} = M^-1 r_{k+1}
    #     else:
    #         z[0] = r[0]
    #
    #     new_zTr = reduce(r[0], z[0])
    #
    #     beta = new_zTr / old_zTr
    #
    #     cg_update_p(beta)
    #     old_zTr = new_zTr


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
        init_Itp(l)

    for l in range(1, n_mg_levels):
        mg_get_Kl(l)

    mgpcg()
