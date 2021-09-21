from element import *


class Tetrahedron(Element):
    def __init__(self, nodes, E=1., nu=0.3):
        if len(nodes) != 4:
            raise AttributeError("A tetrahedron must include 4 nodes.")
        Element.__init__(self, nodes)
        self.E = E
        self.nu = nu
        self.ndof = 3
        self.calc_volume()

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


if __name__ == '__main__':
    a = Node(0,0,0)
    b = Node(1,0,0)
    c = Node(1,1,0)
    d = Node(1,1,1)
    ele = Tetrahedron([a,b,c,d])
    print(ele.nodes)
    print(ele.ndof)
    ele.calc_D()
    print(ele.D)
    ele.calc_Ke()
    print(ele.Ke)