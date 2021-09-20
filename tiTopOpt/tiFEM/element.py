from node import *


class Element:
    def __init__(self, nodes, E=1., nu=0.3):
        for nd in nodes:
            assert issubclass(type(nd), Node), "Must be Node type"

        self.nodes = nodes
        self.dim = nodes[0].dim
        self.nodeID = []
        self.ID = None  # index
        self.E = E  # Young's modulus
        self.nu = nu  # Possion's ratio
        self.volume = None  # volume

        self.ndof = len(nodes) * self.dim
        self.D = None
        self.B = None
        self.J = None
        self.Ke = None  # elemental stiffness matrix
        self.Me = None  # elemental mass matrix
        self.Xe = 1.  # design variable


if __name__ == '__main__':
    nd0 = Node(0,1)
    nd1 = Node(1,2)
    nd2 = Node(2,3)
    ele = Element((nd0,nd1,nd2))
    print(ele.nodes)
    print(ele.ndof)