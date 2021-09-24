import scipy.io as spi
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

dc_beso = spi.loadmat('dc_beso.mat')['dc']
dc_mat = spi.loadmat('dc_mat.mat')['dc']
er2 = np.linalg.norm(dc_mat-dc_beso)
print(dc_mat.min())
print(dc_beso.min())

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