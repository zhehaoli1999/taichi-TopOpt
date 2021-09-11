import taichi as ti

@ti.func
def scalar_mat_mul_vec(A: ti.template(), b: ti.template(), result:ti.template()):
    # assert A.shape()[1] == b.n / b.shape[0]
    for i in ti.static(range(A.shape[1])):
        result[i] = 0.0
    for i, j in ti.static(ti.ndrange(A.shape[0], A.shape[1])):
        result[i] += A[i,j] * b[j]


@ti.func
def vec_dot(r1: ti.template(), r2: ti.template()):
    result = 0.
    for i in ti.static(range(r1.n)):
        result += r1[i] * r2[i]
    return result


@ti.func
def scalar_dot(r1: ti.template(), r2: ti.template()):
    result = 0.
    for i in ti.static(range(r1.shape[0])):
        result += r1[i] * r2[i]
    return result

@ti.func
def scalar_minus(r1: ti.template(), r2: ti.template(), result: ti.template()):
    for i in ti.static(range(r1.shape[0])):
        result[i] = r1[i] - r2[i]

@ti.func
def scalar_add(r1: ti.template(), r2: ti.template(), result: ti.template()):
    for i in ti.static(range(r1.shape[0])):
        result[i] = r1[i] + r2[i]

@ti.func
def scalar_mul_const(r1: ti.template(), c: ti.f32, result:ti.template()):
    for i in ti.static(range(r1.shape[0])):
        result[i] = r1[i] * c

@ti.func
def scalar_copy(r1: ti.template(), r2: ti.template()):
    for i in ti.static(range(r1.shape[0])):
        r2[i] = r1[i]