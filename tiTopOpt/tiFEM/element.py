from node import *


@ti.data_oriented
class Element:
    def __init__(self, nodes, E=1., nu=0.3, t=1.):
        self.dim = nodes[0].dim
        self.nodes = ti.field(dtype=ti.f64, shape=(len(nodes), self.dim)) # elemental nodes
        self.nodeID = ti.field(dtype=ti.i32, shape=(len(nodes)))
        self.set_nodes(nodes)
        self.ID = None  # index
        self.E =  E # Young's modulus
        self.nu = nu # Possion's ratio
        self.t = 1. # thickness
        self.volume = None # volume

        self.ndof = len(nodes) * self.dim

        self.D = ti.field(ti.f32, shape=(self.ndof, self.ndof))
        self.B = ti.field(ti.f32, shape=(self.ndof, self.ndof))
        self.J = ti.field(ti.f32, shape=(self.ndof, self.ndof))
        self.Ke = ti.field(ti.f32, shape=(self.ndof, self.ndof)) # elemental stiffness matrix
        self.Me = ti.field(ti.f32, shape=(self.ndof, self.ndof)) # elemental mass matrix
        self.Xe = 1. # design variable

    @ti.pyfunc
    def set_nodes(self, nodes):
        for i in range(len(nodes)):
            assert issubclass(type(nodes[i]), Node), "All nodes must be the Node type."
            assert nodes[i].dim ==  self.dim, "The dimension of each node must be the same."
            for j in range(self.dim):
                self.nodes[i, j] = nodes[i].pos[j]


if __name__ == '__main__':
    ti.init()
    nd0 = Node(0,1)
    nd1 = Node(1,2)
    nd2 = Node(2,3)
    ele = Element((nd0,nd1,nd2))
    print(ele.nodes)
    print(ele.ndof)