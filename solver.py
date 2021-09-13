import taichi as ti

# @ti.func
# def GS_solver(max_iter): # Guass-Seidel Solver: May not converge
#     for iter in range(max_iter):
#         for i in range(n_free_dof):
#             sigma = 0.
#             for j in range(n_free_dof):
#                 if j != i :
#                     sigma += K_freedof[i,j] * U_freedof[j]
#             if K_freedof[i, i] != 0.:
#                 U_freedof[i] = (F_freedof[i] - sigma) / K_freedof[i, i]


# for cg
r = ti.field(dtype=ti.f32, shape=(n_free_dof))
p = ti.field(dtype=ti.f32, shape=(n_free_dof))
Ap = ti.field(dtype=ti.f32, shape=(n_free_dof))

@ti.func
def cg_compute_Ap(A: ti.template(), p:ti.template()):
    for I in range(n_free_dof):
        Ap[I] = 0.

    for i in range(n_free_dof):
        for j in range(n_free_dof):
            Ap[i] += A[i, j] * p[j]
    # for i, j in ti.ndrange((n_free_dof, n_free_dof)): # error

@ti.func
def reduce(r1: ti.template(), r2: ti.template()):
    result = 0.
    for I in range(r1.shape[0]):
        result += r1[I] * r2[I]  # dot
    return result

@ti.kernel
def conjungate_gradient():
    A, x, b = ti.static(K_freedof, U_freedof, F_freedof) # variable alias

    # init
    for I in ti.grouped(b):
        x[I] = 0.
        r[I] = b[I]  # r = b - A * x = b
        p[I] = r[I]

    rsold = reduce(r, r)

    # cg iteration
    for iter in range(n_free_dof + 50):
        cg_compute_Ap(A, p)

        beta = reduce(p, Ap)
        alpha = rsold / beta

        for I in range(n_free_dof):
            x[I] += alpha * p[I]  # x = x + alpha * Ap
            r[I] -= alpha * Ap[I] # r = r - alpha * Ap

        rsnew = reduce(r, r)

        if ti.sqrt(rsnew) < 1e-10:
            break

        for I in range(n_free_dof):
            p[I] = r[I] + (rsnew / rsold) * p[I] # p = r + (rsnew / rsold) * p

        rsold = rsnew
