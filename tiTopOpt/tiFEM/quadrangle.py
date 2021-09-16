from element import *


class Quadrangle(Element):
    def __init__(self, nodes, E=1., nu=0.3):
        if len(nodes) != 4:
            raise AttributeError("A quadangle must include 4 nodes.")
        Element.__init__(self, nodes)
        self.E = E
        self.nu = nu

    def init_keys(self):
        self.set_eIk(("sx", "sy", "sxy"))

    def init_unknowns(self):
        for nd in self.nodes:
            nd.init_unknowns("Ux", "Uy")
        self._ndof = 2

    def calc_D(self):
        a = self.E / (1 - self.nu ** 2)
        self._D = a * np.array([[1., self.nu, 0.],
                          [self.nu, 1., 0.],
                          [0., 0., (1 - self.nu) / 2.]])

    def calc_B(self, *intv_pts):
        s = intv_pts[0] * 1.0
        t = intv_pts[1] * 1.0
        x1, y1 = self.nodes[0].x, self.nodes[0].y
        x2, y2 = self.nodes[1].x, self.nodes[1].y
        x3, y3 = self.nodes[2].x, self.nodes[2].y
        x4, y4 = self.nodes[3].x, self.nodes[3].y

        a = 1 / 4 * (y1 * (s - 1) + y2 * (-1 - s) + y3 * (1 + s) + y4 * (1 - s))
        b = 1 / 4 * (y1 * (t - 1) + y2 * (1 - t) + y3 * (1 + t) + y4 * (-1 - t))
        c = 1 / 4 * (x1 * (t - 1) + x2 * (1 - t) + x3 * (1 + t) + x4 * (-1 - t))
        d = 1 / 4 * (x1 * (s - 1) + x2 * (-1 - s) + x3 * (1 + s) + x4 * (1 - s))

        B100 = -1 / 4 * a * (1 - t) + 1 / 4 * b * (1 - s)
        B111 = -1 / 4 * c * (1 - s) + 1 / 4 * d * (1 - t)
        B120 = B111
        B121 = B100

        B200 = 1 / 4 * a * (1 - t) + 1 / 4 * b * (1 + s)
        B211 = -1 / 4 * c * (1 + s) - 1 / 4 * d * (1 - t)
        B220 = B211
        B221 = B200

        B300 = 1 / 4 * a * (1 + t) - 1 / 4 * b * (1 + s)
        B311 = 1 / 4 * c * (1 + s) - 1 / 4 * d * (1 + t)
        B320 = B311
        B321 = B300

        B400 = -1 / 4 * a * (1 + t) - 1 / 4 * b * (1 - s)
        B411 = 1 / 4 * c * (1 - s) + 1 / 4 * d * (1 + t)
        B420 = B411
        B421 = B400

        self._B = np.array([[B100, 0, B200, 0, B300, 0, B400, 0],
                      [0, B111, 0, B211, 0, B311, 0, B411],
                      [B120, B121, B220, B221, B320, B321, B420, B421]])

        X = np.array([x1, x2, x3, x4])
        Y = np.array([y1, y2, y3, y4]).reshape(4, 1)
        _J = np.array([[0, 1 - t, t - s, s - 1],
                       [t - 1, 0, s + 1, -s - t],
                       [s - t, -s - 1, 0, t + 1],
                       [1 - s, s + t, -t - 1, 0]])
        self._J = np.dot(np.dot(X, _J), Y) / 8.

