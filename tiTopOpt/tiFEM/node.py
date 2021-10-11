from init import *


@ti.data_oriented
class Node(object):
    def __init__(self, *pos):
        self.dim = len(pos)
        if  self.dim == 3:
            self.pos = ti.Vector([pos[0], pos[1], pos[2]])
        elif self.dim == 2:
            self.pos = ti.Vector([pos[0], pos[1], 0.])
        else:
            raise AttributeError("Node dimension must be 2 or 3")

        self.ID = None  # index
        self.force = ti.Vector([0., 0., 0.]) # force vector
        self.disp = ti.Vector([0., 0., 0.]) # displacement vector
        self.cont_elems = [] # connected elements
        self.adj_nds = [] # adjacent nodes

    def __repr__(self):
        return "Node: %r" % (self.pos,)

    def __eq__(self, other):
        assert issubclass(type(other), Node), "Must be Node type."
        assert other.dim == self.dim, "The dimensions are different."
        if ((self.pos[0] == other.pos[0]) &\
            (self.pos[1] == other.pos[1]) &\
            (self.pos[2] == other.pos[2])):
            return True
        else:
            return False

    def clear_force(self):
        self.force = ti.Vector([0., 0., 0.])

    def clear_disp(self):
        self.disp = ti.Vector([0., 0., 0.])


if __name__ == '__main__':
    ti.init()
    nd = Node(2,3,4)
    print(nd)
    nd.force=ti.Vector([3,4,5])
    print(nd.force)
    nd.disp=ti.Vector([1,0,1])
    print(nd.disp)
    nd.ID=4
    print(nd.ID)
    nd.pos[1]=2
    print(nd)
    nd2 = Node(2,2,4)
    print(nd2)
    print(nd == nd2)