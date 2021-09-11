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