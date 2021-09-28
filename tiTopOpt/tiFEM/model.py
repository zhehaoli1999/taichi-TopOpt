from element import *


class Model:
    def __init__(self, nodes, elements):
        for nd in nodes:
            assert issubclass(type(nd), Node), "Must be Node type"

        self.nodes = nodes
        self.elements = elements
        self.dim = nodes[0].dim

    def init_index(self):
        for i in range(len(self.nodes)):
            self.nodes[i].ID = i

        for i in range(len(self.elements)):
            self.elements[i].ID = i


    # def cal_adjacent_nodes(self):
    #     for


    def cal_adjacent_elements(self):
        for elem in self.elements:
            for nd in elem.nodes:
                self.nodes[nd.ID].adj_elems.append(elem)
                nd.adj_elems.append(elem)

        for nd in self.nodes:
            for adj_elem in nd.adj_elems:
                elem.adj_elems.append(adj_elem)


if __name__ == '__main__':
    nd0 = Node(0., 0.)
    nd1 = Node(0., 1.)
    nd2 = Node(1., 0.)
    nd3 = Node(1., 1.)

    nodes = [nd0,nd1,nd2,nd3]
    ele0 = Triangle([nd0,nd1,nd2])
    ele1 = Triangle([nd1,nd2,nd3])
    elems = [ele0,ele1]

    model = Model(nodes,elems)
    model.init_index()
    print(model.elements[0].nodeID)

