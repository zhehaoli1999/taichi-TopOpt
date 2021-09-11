import taichi as ti

@ti.func
def scalar_mat_mul_vec(A: ti.template(), b: ti.template(), result:ti.template()):
    # assert A.shape()[1] == b.n
    for i in ti.static(range(b.n)):
        result[i] = 0.0
    for i, j in ti.static(ti.ndrange(A.shape[0], A.shape[1])):
        result[i] += A[i,j] * b[j]

@ti.func
def vec_dot(r1: ti.template(), r2: ti.template()):
    result = 0.
    for i in ti.static(range(r1.n)):
        result += r1[i] * r2[i]
    return result