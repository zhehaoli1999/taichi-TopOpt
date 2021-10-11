from element import *


class Model:
    def __init__(self, nodes, elements, loads, supports):
        for nd in nodes:
            assert issubclass(type(nd), Node), "Must be Node type"

        self.nodes = nodes
        self.elements = elements
        self.dim = nodes[0].dim
        self.E = elements[0].E
        self.nu = elements[0].nu
        self.rho = elements[0].rho
        self.init_index()
        self.cal_connected_nodes()
        self.loads = {}
        self.supports = {}
        self.apply_loads_BCs()

    # Initilize node ID and element ID
    def init_index(self):
        for i in range(len(self.nodes)):
            self.nodes[i].ID = i

        for i in range(len(self.elements)):
            self.elements[i].ID = i
            for nd in self.elements[i].nodes:
                self.elements[i].cont_nds.append(nd.ID)

    def apply_loads_BCs(self):
        for nd in self.nodes:
            if nd.force != []:
                self.loads[nd.ID] = nd.force
            if nd.disp != []:
                self.supports[nd.ID] = nd.disp


    # Find the connected nodes of each element
    def cal_connected_nodes(self):
        for elem in self.elements:
            for nd in elem.nodes:
                nd.cont_elems.append(elem.ID)

    # Find the adjacent nodes of each node
    def cal_adjacent_nodes(self):
        for nd in self.nodes:
            for id in nd.cont_elems:
                for adj_nd in self.elements[id].nodes:
                    if  adj_nd.ID != nd.ID:
                        nd.adj_nds.append(adj_nd.ID)

    # Find the adjacent elements of each element
    def cal_adjacent_elements(self):
        for elem in self.elements:
            for nd in self.nodes:
                for adj_elem in nd.cont_elems:
                    if (adj_elem != elem.ID) & (adj_elem not in elem.adj_elems):
                        elem.adj_elems.append(adj_elem)


if __name__ == '__main__':
    nd0 = Node(0., 0.)
    nd1 = Node(0., 1.)
    nd2 = Node(1., 0.)
    nd3 = Node(1., 1.)
    nd4 = Node(2., 1.)
    nodes = [nd0, nd1, nd2, nd3, nd4]

    nd4.set_force([-1,0])
    nd2.set_disp([0, 0])

    ele0 = Triangle([nd0, nd1, nd2])
    ele1 = Triangle([nd1, nd2, nd3])
    ele2 = Triangle([nd2, nd3, nd4])
    elems = [ele0,ele1, ele2]

    loads = []
    supports = []

    model = Model(nodes,elems, loads, supports)
    model.cal_adjacent_nodes()
    model.cal_adjacent_elements()
    print(model.elements[0].cont_nds)
    print(model.nodes[3].ID)
    print(model.elements[0].nodes[0].adj_nds)
    print(model.elements[1].adj_elems)
    print(model.loads)
    print(model.supports)