from node import *
from integration import *


# ====== element base ======
class Element:
    def __init__(self, nodes, E=1., nu=0.3):
        for nd in nodes:
            assert issubclass(type(nd), Node), "Must be Node type"

        self.nodes = nodes # the nodes in this element
        self.dim = nodes[0].dim
        self.cont_nds = [] # the indices of the connected nodes
        self.adj_elems = [] # the adjacent elements
        self.ID = None  # index
        self.E = E  # Young's modulus
        self.nu = nu  # Possion's ratio
        self.rho = 1.0
        self.volume = None  # volume

        self.ndof = len(nodes) * self.dim
        self.D = None
        self.B = None
        self.J = None
        self.Ke = None  # elemental stiffness matrix
        self.Me = None  # elemental mass matrix
        self.Xe = 1.  # design variable


# ====== 2D elements ======
class Triangle(Element):
    def __init__(self, nodes, E=1., nu=0.3):
        if len(nodes) != 3:
            raise AttributeError("A triangle must include 3 nodes.")
        Element.__init__(self, nodes)
        self.E = E
        self.nu = nu
        self.ndof = 2
        self.D = []
        self.B = []
        self.Ke = []
        self.calc_volume()
        self.type_abaqus = "SHELL"

    def calc_volume(self):
        v1 = np.array([self.nodes[0].x, self.nodes[0].y])-np.array([self.nodes[1].x, self.nodes[1].y])
        v2 = np.array([self.nodes[0].x, self.nodes[0].y])-np.array([self.nodes[2].x, self.nodes[2].y])
        area = np.cross(v1, v2)/2.
        self.volume = area

    def calc_D(self):
        a = self.E / (1 - self.nu ** 2)
        self.D = a * np.array([[1., self.nu, 0.],
                          [self.nu, 1., 0.],
                          [0., 0., (1 - self.nu) / 2.]])

    def calc_B(self):
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y
        belta1 = y2 - y3
        belta2 = y3 - y1
        belta3 = y1 - y2
        gama1 = x3 - x2
        gama2 = x1 - x3
        gama3 = x2 - x1

        self.B = 1. / (2. * self.volume) * np.array([[belta1, 0, belta2, 0, belta3, 0],
                                            [0., gama1, 0, gama2, 0, gama3],
                                            [gama1, belta1, gama2, belta2, gama3, belta3]])

    def calc_Ke(self):
        self.calc_B()
        self.calc_D()
        self.Ke = self.volume * np.dot(np.dot(self.B.T, self.D), self.B)


class Quadrangle(Element):
    def __init__(self, nodes, E=1., nu=0.3, t=1.):
        if len(nodes) != 4:
            raise AttributeError("A quadangle must include 4 nodes.")
        Element.__init__(self, nodes)
        self.E = E
        self.nu = nu
        self.ndof = 2
        self.t = t
        self.Ke = np.zeros(shape=(len(nodes) * self.ndof, len(nodes) * self.ndof))
        self.type_abaqus = "SHELL"

    def calc_D(self):
        a = self.E / (1 - self.nu ** 2)
        self.D = a * np.array([[1., self.nu, 0.],
                          [self.nu, 1., 0.],
                          [0., 0., (1 - self.nu) / 2.]])

    def calc_B(self, *intv_pts):
        s = intv_pts[0] * 1.0
        t = intv_pts[1] * 1.0
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y
        x4, y4 = self.nodes[3].x, self.nodes[3].y

        a = 0.25 * (y1 * (s - 1) + y2 * (-1 - s) + y3 * (1 + s) + y4 * (1 - s))
        b = 0.25 * (y1 * (t - 1) + y2 * (1 - t) + y3 * (1 + t) + y4 * (-1 - t))
        c = 0.25 * (x1 * (t - 1) + x2 * (1 - t) + x3 * (1 + t) + x4 * (-1 - t))
        d = 0.25 * (x1 * (s - 1) + x2 * (-1 - s) + x3 * (1 + s) + x4 * (1 - s))

        B100 = -0.25 * a * (1 - t) + 0.25 * b * (1 - s)
        B111 = -0.25 * c * (1 - s) + 0.25 * d * (1 - t)
        B120 = B111
        B121 = B100

        B200 = 0.25 * a * (1 - t) + 0.25 * b * (1 + s)
        B211 = -0.25 * c * (1 + s) - 0.25 * d * (1 - t)
        B220 = B211
        B221 = B200

        B300 = 0.25 * a * (1 + t) - 0.25 * b * (1 + s)
        B311 = 0.25 * c * (1 + s) - 0.25 * d * (1 + t)
        B320 = B311
        B321 = B300

        B400 = -0.25 * a * (1 + t) - 0.25 * b * (1 - s)
        B411 = 0.25 * c * (1 - s) + 0.25 * d * (1 + t)
        B420 = B411
        B421 = B400

        B = np.array([[B100, 0, B200, 0, B300, 0, B400, 0],
                      [0, B111, 0, B211, 0, B311, 0, B411],
                      [B120, B121, B220, B221, B320, B321, B420, B421]])

        X = np.array([x1, x2, x3, x4])
        Y = np.array([y1, y2, y3, y4]).reshape(4, 1)
        J = np.array([[0, 1 - t, t - s, s - 1],
                       [t - 1, 0, s + 1, -s - t],
                       [s - t, -s - 1, 0, t + 1],
                       [1 - s, s + t, -t - 1, 0]])
        J = np.dot(np.dot(X, J), Y) / 8.
        return B,J

    def calc_Ke(self):
        self.calc_D()

        glq = integration(2) # 2 sample points
        for i in range(len(glq.Xi)):
            for j in range(len(glq.Xi)):
                B, J = self.calc_B(glq.Xi[i], glq.Xi[j])
                B /= J
                self.Ke += glq.w[i] * glq.w[j] * self.t * np.dot(np.dot(B.T, self.D), B) * J


# ====== 3D elements ======
class Tetrahedron(Element):
    def __init__(self, nodes, E=1., nu=0.3):
        if len(nodes) != 4:
            raise AttributeError("A tetrahedron must include 4 nodes.")
        Element.__init__(self, nodes)
        self.E = E
        self.nu = nu
        self.ndof = 3
        self.calc_volume()
        self.type_abaqus = "SOLID"

    def calc_volume(self):
        V = np.ones((4,4))
        for i,nd in enumerate(self.nodes):
            V[i,1:] = nd.pos
        self.volume = abs(np.linalg.det(V)/6.)

    def calc_D(self):
        a = self.E / ((1. + self.nu) * (1. - 2. * self.nu))
        c1 = 1. - self.nu
        c2 = (1. - self.nu) / 2.
        self.D = a * np.array([[c1, self.nu, self.nu, 0., 0., 0.],
                          [self.nu, c1, self.nu, 0., 0., 0.],
                          [self.nu, self.nu, c1, 0., 0., 0.],
                          [0., 0., 0., c2, 0., 0.],
                          [0., 0., 0., 0., c2, 0.],
                          [0., 0., 0., 0., 0., c2]])

    def calc_B(self):
        A = np.ones((4, 4))
        belta = np.zeros(4)
        gama = np.zeros(4)
        delta = np.zeros(4)
        for i, nd in enumerate(self.nodes):
            A[i, 1:] = nd.pos

        for i in range(4):
            belta[i] = (-1) ** (i + 1) * np.linalg.det(np.delete(np.delete(A, i, 0), 1, 1))
            gama[i] = (-1) ** (i + 2) * np.linalg.det(np.delete(np.delete(A, i, 0), 2, 1))
            delta[i] = (-1) ** (i + 1) * np.linalg.det(np.delete(np.delete(A, i, 0), 3, 1))

        self.B = 1. / (6. * self.volume) * np.array([[belta[0], 0., 0., belta[1], 0., 0., belta[2], 0., 0., belta[3], 0., 0.],
                                               [0., gama[0], 0., 0., gama[1], 0., 0., gama[2], 0., 0., gama[3], 0.],
                                               [0., 0., delta[0], 0., 0., delta[1], 0., 0., delta[2], 0., 0., delta[3]],
                                               [gama[0], belta[0], 0., gama[1], belta[1], 0., gama[2], belta[2], 0,
                                                gama[3],
                                                belta[3], 0.],
                                               [0., delta[0], gama[0], 0., delta[1], gama[1], 0., delta[2], gama[2], 0.,
                                                delta[3], gama[3]],
                                               [delta[0], 0., belta[0], delta[1], 0., belta[1], delta[2], 0., belta[2],
                                                delta[3], 0, belta[3]]])

    def calc_Ke(self):
        self.calc_B()
        self.calc_D()
        self.Ke = self.volume * np.dot(np.dot(self.B.T, self.D), self.B)


class Hexahedron(Element):
    def __init__(self, nodes, E=1., nu=0.3):
        if len(nodes) != 8:
            raise AttributeError("A hexahedron must include 8 nodes.")
        Element.__init__(self, nodes)
        self.E = E
        self.nu = nu
        self.ndof = 3
        self.Ke = np.zeros(shape=(len(nodes) * self.ndof, len(nodes) * self.ndof))
        self.type_abaqus = "SOLID"

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

        glq = integration(2) # 2 sample points
        for i in range(len(glq.Xi)):
            for j in range(len(glq.Xi)):
                for k in range(len(glq.Xi)):
                    B, J = self.calc_B(glq.Xi[i], glq.Xi[j], glq.Xi[k])
                    self.Ke += glq.w[i] * glq.w[j] * glq.w[k] * np.matmul(np.matmul(B.T, self.D), B) * J


if __name__ == '__main__':
    # ===== Triangle =====
    tri_0 = Node(0., 0., 0.)
    tri_1 = Node(1., 0., 0.)
    tri_2 = Node(1., 1., 0.)
    tri_ele = Triangle([tri_0, tri_1, tri_2])
    print(tri_ele.nodes)
    print(tri_ele.ndof)
    tri_ele.calc_D()
    print(tri_ele.D)
    tri_ele.calc_Ke()
    print(tri_ele.Ke)

    # ===== Quadrangle =====
    quad_0 = Node(0., 0., 0.)
    quad_1 = Node(1., 0., 0.)
    quad_2 = Node(1., 1., 0.)
    quad_3 = Node(1., 2., 0.)
    quad_ele = Quadrangle([quad_0,quad_1,quad_2,quad_3])
    print(quad_ele.nodes)
    print(quad_ele.ndof)
    quad_ele.calc_D()
    print(quad_ele.D)
    quad_ele.calc_Ke()
    print(quad_ele.Ke)

    # ===== Tetrahedron =====
    tet_0 = Node(0., 0., 0.)
    tet_1 = Node(1., 0., 0.)
    tet_2 = Node(1., 1., 0.)
    tet_3 = Node(1., 1., 1.)
    tet_ele = Tetrahedron([tet_0,tet_1,tet_2,tet_3])
    print(tet_ele.nodes)
    print(tet_ele.ndof)
    tet_ele.calc_D()
    print(tet_ele.D)
    tet_ele.calc_Ke()
    print(tet_ele.Ke)

    # ===== hexahedron =====
    hex_0 = Node(0.,0.,0.)
    hex_1 = Node(1.,0.,0.)
    hex_2 = Node(1.,1.,0.)
    hex_3 = Node(0.,1.,0.)
    hex_4 = Node(0.,0.,1.)
    hex_5 = Node(1.,0.,1.)
    hex_6 = Node(1.,1.,1.)
    hex_7 = Node(0.,1.,1.)
    hex_ele = Hexahedron([hex_0,hex_1,hex_2,hex_3,hex_4,hex_5,hex_6,hex_7])
    print(hex_ele.nodes)
    print(hex_ele.ndof)
    hex_ele.calc_D()
    print(hex_ele.D)
    hex_ele.calc_Ke()
    print(hex_ele.Ke)