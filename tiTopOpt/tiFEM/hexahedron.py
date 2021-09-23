from element import *
from intergration import *


class Hexahedron(Element):
    def __init__(self, nodes, E=1., nu=0.3):
        if len(nodes) != 8:
            raise AttributeError("A hexahedron must include 8 nodes.")
        Element.__init__(self, nodes)
        self.E = E
        self.nu = nu
        self.ndof = 3
        self.Ke = np.zeros(shape=(len(nodes) * self.ndof, len(nodes) * self.ndof))

    def calc_D(self):
        a = self.E / ((1. + self.nu) * (1. - 2. * self.nu))
        c1 = 1. - self.nu
        c2 = (1. - 2. * self.nu) * 0.5
        self.D = a * np.array([[c1, self.nu, self.nu, 0., 0., 0.],
                          [self.nu, c1, self.nu, 0., 0., 0.],
                          [self.nu, self.nu, c1, 0., 0., 0.],
                          [0., 0., 0., c2, 0., 0.],
                          [0., 0., 0., 0., c2, 0.],
                          [0., 0., 0., 0., 0., c2]])

    def calc_B(self, *intv_pts):
        s = intv_pts[0]
        t = intv_pts[1]
        u = intv_pts[2]

        x = [nd.x for nd in self.nodes]
        y = [nd.y for nd in self.nodes]
        z = [nd.z for nd in self.nodes]

        N1s, N1t, N1u = -(1.-t) * (1. - u) * 0.125, -(1. - s) * (1. - u) * 0.125, -(1. - s) * (1. - t) * 0.125
        N2s, N2t, N2u = (1. - t) * (1. - u) * 0.125, -(1. + s) * (1. - u) * 0.125, -(1. + s) * (1. - t) * 0.125
        N3s, N3t, N3u = (1. + t) * (1. - u) * 0.125, (1. + s) * (1. - u) * 0.125, -(1. + s) * (1. + t) * 0.125
        N4s, N4t, N4u = -(1. + t) * (1. - u) * 0.125, (1. - s) * (1. - u) * 0.125, -(1. - s) * (1. + t) * 0.125
        N5s, N5t, N5u = -(1. - t) * (1. + u) * 0.125, -(1. - s) * (1. + u) * 0.125, (1. - s) * (1. - t) * 0.125
        N6s, N6t, N6u = (1. - t) * (1. + u) * 0.125, -(1. + s) * (1. + u) * 0.125, (1. + s) * (1. - t) * 0.125
        N7s, N7t, N7u = (1. + t) * (1. + u) * 0.125, (1. + s) * (1. + u) * 0.125, (1. + s) * (1. + t) * 0.125
        N8s, N8t, N8u = -(1. + t) * (1. + u) * 0.125, (1. - s) * (1. + u) * 0.125, (1. - s) * (1. + t) * 0.125

        Ns = [N1s, N2s, N3s, N4s, N5s, N6s, N7s, N8s]
        Nt = [N1t, N2t, N3t, N4t, N5t, N6t, N7t, N8t]
        Nu = [N1u, N2u, N3u, N4u, N5u, N6u, N7u, N8u]

        xs = sum(Ns[i] * x[i] for i in range(8))
        xt = sum(Nt[i] * x[i] for i in range(8))
        xu = sum(Nu[i] * x[i] for i in range(8))

        ys = sum(Ns[i] * y[i] for i in range(8))
        yt = sum(Nt[i] * y[i] for i in range(8))
        yu = sum(Nu[i] * y[i] for i in range(8))

        zs = sum(Ns[i] * z[i] for i in range(8))
        zt = sum(Nt[i] * z[i] for i in range(8))
        zu = sum(Nu[i] * z[i] for i in range(8))

        MJ = np.array([[xs, ys, zs],
                       [xt, yt, zt],
                       [xu, yu, zu]])

        J = np.linalg.det(MJ)
        J_v = np.linalg.inv(MJ)

        Nx = [J_v[0, 0] * Ns[i] + J_v[0, 1] * Nt[i] + J_v[0, 2] * Nu[i] for i in range(8)]
        Ny = [J_v[1, 0] * Ns[i] + J_v[1, 1] * Nt[i] + J_v[1, 2] * Nu[i] for i in range(8)]
        Nz = [J_v[2, 0] * Ns[i] + J_v[2, 1] * Nt[i] + J_v[2, 2] * Nu[i] for i in range(8)]

        B = np.array(
            [[Nx[0], 0, 0, Nx[1], 0, 0, Nx[2], 0, 0, Nx[3], 0, 0, Nx[4], 0, 0, Nx[5], 0, 0, Nx[6], 0, 0, Nx[7], 0, 0],
             [0, Ny[0], 0, 0, Ny[1], 0, 0, Ny[2], 0, 0, Ny[3], 0, 0, Ny[4], 0, 0, Ny[5], 0, 0, Ny[6], 0, 0, Ny[7], 0],
             [0, 0, Nz[0], 0, 0, Nz[1], 0, 0, Nz[2], 0, 0, Nz[3], 0, 0, Nz[4], 0, 0, Nz[5], 0, 0, Nz[6], 0, 0, Nz[7]],
             [Ny[0], Nx[0], 0, Ny[1], Nx[1], 0, Ny[2], Nx[2], 0, Ny[3], Nx[3], 0, Ny[4], Nx[4], 0, Ny[5], Nx[5], 0, Ny[6], Nx[6], 0, Ny[7], Nx[7], 0],
             [0, Nz[0], Ny[0], 0, Nz[1], Ny[1], 0, Nz[2], Ny[2], 0, Nz[3], Ny[3], 0, Nz[4], Ny[4], 0, Nz[5], Ny[5], 0, Nz[6], Ny[6], 0, Nz[7], Ny[7]],
             [Nz[0], 0, Nx[0], Nz[1], 0, Nx[1], Nz[2], 0, Nx[2], Nz[3], 0, Nx[3], Nz[4], 0, Nx[4], Nz[5], 0, Nx[5], Nz[6], 0, Nx[6], Nz[7], 0, Nx[7]]])
        return B, J

    def calc_Ke(self):
        self.calc_D()

        glq = Intergration(2) # 2 sample points
        for i in range(len(glq.Xi)):
            for j in range(len(glq.Xi)):
                for k in range(len(glq.Xi)):
                    B, J = self.calc_B(glq.Xi[i], glq.Xi[j], glq.Xi[k])
                    self.Ke += glq.w[i] * glq.w[j] * glq.w[k] * np.matmul(np.matmul(B.T, self.D), B) * J


if __name__ == '__main__':
    a = Node(0.,0.,0.)
    b = Node(1.,0.,0.)
    c = Node(1.,1.,0.)
    d = Node(0.,1.,0.)
    e = Node(0.,0.,1.)
    f = Node(1.,0.,1.)
    g = Node(1.,1.,1.)
    h = Node(0.,1.,1.)
    ele = Hexahedron([a,b,c,d,e,f,g,h])
    print(ele.nodes)
    print(ele.ndof)
    ele.calc_D()
    print(ele.D)
    ele.calc_Ke()
    print(ele.Ke)