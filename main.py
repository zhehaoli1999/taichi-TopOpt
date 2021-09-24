import scipy.io as spi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

A = csr_matrix(spi.loadmat('KG.mat')['KG'], dtype=np.double)
A_array = A.toarray()
b = spi.loadmat('Fv.mat')['Fv']
U_matlab = spi.loadmat('Uv.mat')['Uv']
x = spsolve(A,b)
x_np = np.linalg.solve(A.toarray(),b)
er = np.linalg.norm(U_matlab-x)
er2 = np.linalg.norm(U_matlab-x_np)
print(x[0])
print(x_np[0])
print(U_matlab[0])
print(er)
print(er2)

# A2 = csr_matrix(spi.loadmat('beso_KG.mat')['A'])
# b2 = spi.loadmat('beso_Fv.mat')['b']
# x2 = spsolve(A2,b2.T)
#
# c = np.linalg.norm(A2.toarray()-A.toarray())
# d = np.linalg.norm(b2.T-b)
# e = np.linalg.norm(x2-x)
# print(c)
# print(d)
# print(e)