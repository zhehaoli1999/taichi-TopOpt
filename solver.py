import taichi as ti
from utils import *

@ti.data_oriented
class fem_mgpcg:
    def __init__(self, nelx, nely, fixed_dofs, K, F, dim=2, n_mg_levels=4, dtype=ti.f32, use_multigrid=True):
        self.use_multigrid = use_multigrid

        self.n_mg_levels = n_mg_levels
        self.pre_and_post_smoothing = 4
        self.bottom_smoothing = 50

        self.dim = dim
        self.real = dtype
        self.fixed_dofs = fixed_dofs  # fixed dofs: ti.Vector

        self.A = [ti.field(dtype=self.real) for _ in range(self.n_mg_levels)]  # stiffness matrix, A[0] = K
        self.r = [ti.field(dtype=self.real) for _ in range(self.n_mg_levels)]  # residual vector
        self.z = [ti.field(dtype=self.real) for _ in range(self.n_mg_levels)]  # M^-1 r
        self.Itp = [ti.field(dtype=ti.f32) for _ in range(self.n_mg_levels - 1)]  # Interpolation matrix I
        self.Itp_mask = [ti.field(dtype=ti.i8) for _ in range(self.n_mg_levels - 1)]
        # self.R = [ti.field(dtype=ti.f32) for _ in range(self.n_mg_levels-1)] # R = I^T

        self.nelx = nelx
        self.nely = nely
        self.ndof = 2 * (nelx + 1) * (nely + 1)

        self.b = ti.field(self.real)  # force
        self.x = ti.field(self.real)  # solution
        self.p = ti.field(self.real)  # used in conjugate gradient
        self.Ap = ti.field(self.real)  # used in conjugate gradient: A @ p
        ti.root.dense(ti.i, self.ndof).place(self.b, self.x, self.p, self.Ap)

        self.alpha = ti.field(self.real)
        self.beta = ti.field(self.real)
        self.sum = ti.field(self.real) # ti.func must annotate type of return value, need to use sum to unify dtype
        ti.root.place(self.alpha, self.beta, self.sum)

        for l in range(n_mg_levels):
            nely_l = self.nely // 2 ** l
            nelx_l = self.nelx // 2 ** l
            ndof_l = 2 * (nely_l + 1) * (nelx_l + 1)  # calculate num of freedom in each level

            ti.root.dense(ti.ij, ndof_l).place(self.A[l]) # TODO: Leverage sparcity
            ti.root.dense(ti.i, ndof_l).place(self.r[l], self.z[l])

            if l < n_mg_levels - 1:
                nely_2l = nely // 2 ** (l + 1)
                nelx_2l = nelx // 2 ** (l + 1)
                ndof_2l = 2 * (nely_2l + 1) * (nelx_2l + 1)

                ti.root.dense(ti.ij, (ndof_l, ndof_2l)).place(self.Itp[l])  # shape of Itp[l] is (ndof_l x ndof_2l)
                ti.root.dense(ti.ij, (ndof_l, ndof_2l)).place(self.Itp_mask[l])
                self.Itp_mask[l].fill(0)

                # initialize
        print("mgpcg solver initializing...")

        for l in range(n_mg_levels - 1):
            self.init_Itp(l)

        self.set_b(F)
        self.set_A0(K)

        for l in range(1, n_mg_levels):
            self.get_kl(l)

        print("mgpcg solver initialized")

    def re_init(self, K):
        self.set_A0(K)
        for l in range(1, self.n_mg_levels):
            self.get_kl(l)

    @ti.func
    def is_in(self, num, array):
        ans = 0
        for i in range(array.shape[0]):
            if num == array[i]:
                ans = 1
                break
        return ans

    @ti.func
    def check_is_free(self, l, node_x_l, node_y_l, add=0):
        '''
        Check if the dof indicated by ${add} in level ${l} with node coordinate (node_x_l, node_y_l) is fixed
        '''
        node_x = node_x_l * 2 ** l
        node_y = node_y_l * 2 ** l

        dof_idx = 2 * (node_x * (self.nely + 1) + node_y) + add
        ans = 0
        if self.is_in(dof_idx, self.fixed_dofs):
        # if dof_idx in self.fixed_dofs:
            ans = 0  # False
        else:
            ans = 1  # True
        return ans

    @ti.kernel
    def init_Itp(self, l: ti.template()):
        '''
        Set Interploation matrix with bilinear interpolation rules.
        '''
        # initialize to be 0
        for I in ti.grouped(self.Itp[l]):
            self.Itp[l][I] = 0.

        nely_2h = self.nely // 2 ** (l + 1)
        nelx_2h = self.nelx // 2 ** (l + 1)

        nely_h = self.nely // 2 ** l
        nelx_h = self.nelx // 2 ** l

        dim = 2
        v = ti.Vector([-1, +1])
        for node_y_2h, node_x_2h  in ti.ndrange(nely_2h + 1, nelx_2h + 1):
            node_y_h = node_y_2h * 2  # Get node coordinate in level h
            node_x_h = node_x_2h * 2

            # get indices of dof
            dof_idx_2h = ti.Vector([dim * (node_x_2h * (nely_2h + 1) + node_y_2h), \
                                    dim * (node_x_2h * (nely_2h + 1) + node_y_2h) + 1])
            dof_idx_h = ti.Vector([dim * (node_x_h * (nely_h + 1) + node_y_h), \
                                   dim * (node_x_h * (nely_h + 1) + node_y_h) + 1])

            # bilinear interpolation
            # situation of weight = 1
            for t in ti.static(range(2)):
                # if dof_idx_h[t] in freedof_idx[l-1]: # check if node in level h is fixed
                if self.check_is_free(l, node_x_h, node_y_h, t):
                    self.Itp[l][dof_idx_h[t], dof_idx_2h[t]] = 1.
                    self.Itp_mask[l][dof_idx_h[t], dof_idx_2h[t]] = 1

            # situation of weight = 1 / 2.
            # for i in ti.Vector([-1, +1]):  #FIXME: a bug to report "TypeError: Can only iterate through Taichi fields/snodes (via template) or dense arrays (via any_arr)"
            for ii in ti.static(range(2)):
                i = v[ii]
                x, y = node_x_h + i, node_y_h
                if 0 <= x < nelx_h:
                    dof_h = ti.Vector([dim * (x * (nely_h + 1) + y), dim * (x * (nely_h + 1) + y) + 1])
                    for t in ti.static(range(2)):
                        # if dof_h[t] in freedof_idx[l - 1]: # check if fixed
                        if self.check_is_free(l, x, y, t):
                            self.Itp[l][dof_h[t], dof_idx_2h[t]] = 1 / 2.
                            self.Itp_mask[l][dof_h[t], dof_idx_2h[t]] = 1

            for jj in ti.static(range(2)):
                j = v[jj]
                x, y = node_x_h, node_y_h + j
                if 0 <= y < nely_h:
                    dof_h = ti.Vector([dim * (x * (nely_h + 1) + y), dim * (x * (nely_h + 1) + y) + 1])
                    for t in ti.static(range(2)):
                        # if dof_h[t] in freedof_idx[l - 1]: # check if fixed
                        if self.check_is_free(l, x, y, t):
                            self.Itp[l][dof_h[t], dof_idx_2h[t]] = 1 / 2.
                            self.Itp_mask[l][dof_h[t], dof_idx_2h[t]] = 1

            # situation of weight = 1 / 4.
            for ii, jj in ti.static(ti.ndrange(2,2)):
                    i = v[ii]
                    j = v[jj]
                    x = node_x_h + i
                    y = node_y_h + j
                    if 0 <= x < nelx_h and 0 <= y < nely_h:
                        dof_h = ti.Vector([dim * (x * (nely_h + 1) + y), dim * (x * (nely_h + 1) + y) + 1])
                        for t in ti.static(range(2)):
                            # if dof_h[t] in freedof_idx[l-1]:
                            if self.check_is_free(l, x, y, t):
                                self.Itp[l][dof_h[t], dof_idx_2h[t]] = 1 / 4.
                                self.Itp_mask[l][dof_h[t], dof_idx_2h[t]] = 1

    @ti.kernel
    def get_kl(self, l: ti.template()):
        '''
        Get A[l] with Galerkin coarsening: K[l+1] = R[l] @ K[l] @ I[l], where R = transpose(I)
        :param l: l should >= 1
        '''
        n1 = self.A[l - 1].shape[0]
        n2 = self.A[l].shape[0]
        # Use Garlekin coarsening: K[l+1] = R[l] @ K[l] @ I[l], where R = transpose(I)
        for i, j in ti.ndrange(n2, n2):
            self.A[l][i, j] = 0.
            for t in range(n1):
                if self.Itp_mask[l - 1][t, i] == 1:
                    s = 0.
                    for m in range(n1):
                        if self.Itp_mask[l - 1][m, j] == 1:
                            s += self.A[l - 1][t, m] * self.Itp[l - 1][m, j]
                    s = s * self.Itp[l - 1][t, i]
                    self.A[l][i, j] += s

    @ti.kernel
    def set_A0(self, K: ti.template()):
        for i, j in ti.ndrange(self.ndof, self.ndof):
            self.A[0][i, j] = K[i, j]

    @ti.kernel
    def set_b(self, F: ti.template()):
        for i in range(self.ndof):
            self.b[i] = F[i]

    @ti.kernel
    def multigrid_init(self):
        for i in range(self.ndof):
            self.r[0][i] = self.b[i]  # r = b - A * x = b
            self.z[0][i] = 0.
            self.Ap[i] = 0.
            self.p[i] = 0.
            self.x[i] = 0.

    @ti.kernel
    def smooth(self, l: ti.template()):
        # Gauss-Seidel #TODO: red-black GS or Jacobi for parallelization
        n_node_y_l = self.nely // 2 ** l + 1
        n_node_x_l = self.nelx // 2 ** l + 1
        for x, y in ti.ndrange(n_node_x_l, n_node_y_l):
            # Get two dof
            for t in range(2):
                i = 2 * (x * n_node_y_l + y) + t
                if self.check_is_free(l, x, y, t):  # Check boundary condition
                    sigma = 0.
                    for j in range(self.A[l].shape[1]):
                        if j != i:
                            sigma += self.A[l][i, j] * self.z[l][j]
                    if self.A[l][i, i] != 0.:
                        self.z[l][i] = (self.r[l][i] - sigma) / self.A[l][i,i]
                else:
                    self.z[l][i] = 0.

    @ti.kernel
    def restrict(self, l: ti.template()):
        # 1. calculate residual
        n_node_y_l = self.nely // 2 ** l + 1
        n_node_x_l = self.nelx // 2 ** l + 1
        for x, y in ti.ndrange(n_node_x_l, n_node_y_l):
            # Get two dof
            for t in range(2):
                i = 2 * (x * n_node_y_l + y) + t
                if self.check_is_free(l, x, y, t):  # Check boundary condition
                    sum = 0.
                    for j in range(self.A[l].shape[1]):
                        sum += self.A[l][i, j] * self.z[l][j]
                    self.r[l][i] = self.r[l][i] - sum  # get residual, Note: should not be r[l][i] = b[l][i] - sum
                else:
                    self.r[l][i] = 0.

        # 2. down sample residual on fine grid to coarse grid
        for i in range(self.r[l + 1].shape[0]):
            self.r[l + 1][i] = 0.
            for j in range(self.r[l].shape[0]):
                self.r[l + 1][i] += self.r[l][j] * self.Itp[l][j, i]  # r[l+1] = r[l] @ I[l]

    @ti.kernel
    def prolongate(self, l: ti.template()):
        for i in range(self.z[l].shape[0]):
            self.z[l][i] = 0.
            for j in range(self.z[l + 1].shape[0]):
                self.z[l][i] += self.Itp[l][i, j] * self.z[l + 1][j]  # z[l] = z[l+1] @ I[l]^T


    def apply_preconditioner(self):
        # V-cycle multigrid
        self.z[0].fill(0)
        for l in range(self.n_mg_levels - 1):
            for i in range(self.pre_and_post_smoothing << l):  # equals to pre_and_post_smoothing // 2**l
                self.smooth(l)
            self.z[l + 1].fill(0)
            self.r[l + 1].fill(0)
            self.restrict(l)

        for i in range(self.bottom_smoothing):
            self.smooth(self.n_mg_levels - 1)

        for l in reversed(range(self.n_mg_levels - 1)):
            self.prolongate(l)
            for i in range(self.pre_and_post_smoothing << l):
                self.smooth(l)

    @ti.kernel
    def cg_compute_Ap(self):
        for I in range(self.ndof):
            self.Ap[I] = 0.

        for i in range(self.ndof):
            for j in range(self.ndof):
                self.Ap[i] += self.A[0][i, j] * self.p[j]

    @ti.kernel
    def reduce(self, r1: ti.template(), r2: ti.template()):
        self.sum[None] = 0.
        result = 0.
        for I in range(r1.shape[0]):
            result += r1[I] * r2[I]  # dot
        self.sum[None] = result

    @ti.kernel
    def cg_update_p(self):
        for i in range(self.p.shape[0]):
            self.p[i] = self.z[0][i] + self.beta[None] * self.p[i]

    @ti.kernel
    def cg_update_x(self):
        for i in range(self.p.shape[0]):
            if self.is_in(i, self.fixed_dofs):
                self.x[i] = 0.
            else:
                self.x[i] += self.alpha[None] * self.p[i]

    @ti.kernel
    def cg_update_r(self):
        for i in range(self.p.shape[0]):
            if self.is_in(i, self.fixed_dofs):
                self.r[0][i] = 0.
            else:
                self.r[0][i] -= self.alpha[None] * self.Ap[i]

    def solve(self, U, max_iters=-1, eps=1e-12, abs_tol=1e-12, rel_tol=1e-12, verbose=True):
        '''
        Solve KU = F using MGPCG (MultiGrid Preconditioned Conjugate Gradient descent)

        :parameter max_iters: Specify the maximal iterations. -1 for no limit.
        :param eps: Specify a non-zero value to prevent ZeroDivisionError.
        :parameter abs_tol: Specify the absolute tolerance of loss.
        :parameter rel_tol: Specify the tolerance of loss relative to initial loss.
        :parameter verbose: Specify whether to print iteration information
        '''
        if verbose:
            print(f"mg level: {self.n_mg_levels}")

        self.multigrid_init()

        self.reduce(self.r[0], self.r[0])
        initial_rTr =  self.sum[None] # Used to check convergence

        tol = max(abs_tol, initial_rTr * rel_tol)

        if self.use_multigrid:
            self.apply_preconditioner()  # Get z0 = M^-1 r0
        else:
            self.z[0] = self.r[0]

        self.cg_update_p()  # p0 = z0

        self.reduce(self.r[0], self.z[0])
        old_zTr = self.sum[None]

        # cg iteration
        iter = 0
        while max_iters == -1 or iter < max_iters:
            self.cg_compute_Ap()
            self.reduce(self.p, self.Ap)
            pAp = self.sum[None]

            self.alpha[None] = old_zTr / (pAp + eps)

            self.cg_update_x()  # x = x + alpha * p
            self.cg_update_r()  # r = r - alpha * Ap

            self.reduce(self.r[0], self.r[0])  # check convergence
            rTr = self.sum[None]

            if verbose:
                print(f"iter{iter}, mgpcg relative res: {rTr / initial_rTr}")

            if rTr < tol:
                break

            if self.use_multigrid:
                self.apply_preconditioner()  # update z:  z_{k+1} = M^-1 r_{k+1}
            else:
                self.z[0] = self.r[0]

            self.reduce(self.r[0], self.z[0])
            new_zTr = self.sum[None]

            self.beta[None] = new_zTr / (old_zTr + eps)

            self.cg_update_p()
            old_zTr = new_zTr

            iter += 1

        for i in range(self.ndof):
            U[i] = self.x[i]
