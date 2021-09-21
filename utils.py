import taichi as ti

@ti.kernel
def print_2Dfield(A: ti.template()):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            print(A[i,j], end=",")
        print("\n")