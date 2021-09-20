from element import *


class Tetrahedron(Element):
    def __init__(self, nodes, E=1., nu=0.3):
        if len(nodes) != 4:
            raise AttributeError("A quadangle must include 4 nodes.")
        Element.__init__(self, nodes)
        self.E = E
        self.nu = nu
        self.ndof = 3

        # calc volume
        print(self.nodes.shape)
        v1 = np.array([self.nodes[0, 0], self.nodes[0, 1]])-np.array([self.nodes[1, 0], self.nodes[1, 1]])
        v2 = np.array([self.nodes[0, 0], self.nodes[0, 1]])-np.array([self.nodes[2, 0], self.nodes[2, 1]])
        area = np.cross(v1, v2)/2.
        self.volume = area

    def calc_D(self):
        a = self.E / ((1. + self.nu) * (1. - 2. * self.nu))
        c1 = 1. - self.nu
        c2 = (1. - self.nu) / 2.
        D = a * np.array([[c1, self.nu, self.nu, 0., 0., 0.],
                          [self.nu, c1, self.nu, 0., 0., 0.],
                          [self.nu, self.nu, c1, 0., 0., 0.],
                          [0., 0., 0., c2, 0., 0.],
                          [0., 0., 0., 0., c2, 0.],
                          [0., 0., 0., 0., 0., c2]])
        self.D.from_numpy(D)

def _calc_B_for_tetra3d11(nodes, volume):
    A = np.ones((4, 4))
    belta = np.zeros(4)
    gama = np.zeros(4)
    delta = np.zeros(4)
    for i, nd in enumerate(nodes):
        A[i, 1:] = nd.coord

    for i in range(4):
        belta[i] = (-1) ** (i + 1) * np.linalg.det(np.delete(np.delete(A, i, 0), 1, 1))
        gama[i] = (-1) ** (i + 2) * np.linalg.det(np.delete(np.delete(A, i, 0), 2, 1))
        delta[i] = (-1) ** (i + 1) * np.linalg.det(np.delete(np.delete(A, i, 0), 3, 1))

    B = 1. / (6. * volume) * np.array([[belta[0], 0., 0., belta[1], 0., 0., belta[2], 0., 0., belta[3], 0., 0.],
                                       [0., gama[0], 0., 0., gama[1], 0., 0., gama[2], 0., 0., gama[3], 0.],
                                       [0., 0., delta[0], 0., 0., delta[1], 0., 0., delta[2], 0., 0., delta[3]],
                                       [gama[0], belta[0], 0., gama[1], belta[1], 0., gama[2], belta[2], 0, gama[3],
                                        belta[3], 0.],
                                       [0., delta[0], gama[0], 0., delta[1], gama[1], 0., delta[2], gama[2], 0.,
                                        delta[3], gama[3]],
                                       [delta[0], 0., belta[0], delta[1], 0., belta[1], delta[2], 0., belta[2],
                                        delta[3], 0, belta[3]]])
    return B
