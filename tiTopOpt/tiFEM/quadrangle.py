from element import *
from intergration import *


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

        glq = Intergration(2) # 2 sample points
        for i in range(len(glq.Xi)):
            for j in range(len(glq.Xi)):
                B, J = self.calc_B(glq.Xi[i], glq.Xi[j])
                B /= J
                self.Ke += glq.w[i] * glq.w[j] * self.t * np.dot(np.dot(B.T, self.D), B) * J


if __name__ == '__main__':
    a = Node(0,0,0)
    b = Node(1,0,0)
    c = Node(1,1,0)
    d = Node(1,2,0)
    ele = Quadrangle([a,b,c,d])
    print(ele.nodes)
    print(ele.ndof)
    ele.calc_D()
    print(ele.D)
    ele.calc_Ke()
    print(ele.Ke)