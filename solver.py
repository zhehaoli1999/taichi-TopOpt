import taichi as ti

@ti.func
def cg(A: ti.template(), b:ti.template(), x: ti.template()):
    t  = ti.field(ti.f32, shape=())
    scalar_mat_mul_vec(A, x, t)
    r = b - t
    pass #TODO: 感觉taichi很多东西都不直观：不知道这样子是否可行：比如：两个scalar field能否直接相减？