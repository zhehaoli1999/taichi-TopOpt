from element import *


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

        # calc volume
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


if __name__ == '__main__':
    a = Node(0,0,0)
    b = Node(1,0,0)
    c = Node(1,1,0)
    ele = Triangle([a,b,c])
    print(ele.nodes)
    print(ele.ndof)
    ele.calc_D()
    print(ele.D)
    ele.calc_Ke()
    print(ele.Ke)
