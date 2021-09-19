import numpy as np
import taichi as ti


@ti.data_oriented
class Node(object):
    def __init__(self, *pos):
        if len(pos) != 2 and len(pos) != 3:
            raise AttributeError("Node dimension must be 2 or 3")
        self.dim = len(pos)  # dimension
        self.ID = None  # index
        self.pos = ti.Vector(pos, dt=ti.f64)  # position
        self.force = ti.field(ti.f64, shape=self.dim)  # force vector
        self.disp = ti.field(ti.f64, shape=self.dim)  # displacement vector
        self.dof = ti.Vector(np.zeros(self.dim),dt=ti.f64)  # degree-of-freedom

    def __repr__(self):
        return str(self.dim) + "D Node: " + str(self.pos)

    def __getitem__(self, index):
        return self.pos[index]

    def __setitem__(self, index, val):
        l = self.pos
        l[index] = val
        self.pos = l

    def __eq__(self, other):
        assert issubclass(type(other), Node), "Must be Node type"
        a = self.pos.to_numpy()
        b = other.pos.to_numpy()
        if np.isclose(a, b).all():
            return True
        else:
            return False

    def set_force(self, *force):
        if len(force) != self.dim:
            raise AttributeError("The dimension of the force vector must match with the nodal dimension.")
        self.force = ti.Vector(force, dt=ti.f64)

    def set_disp(self, *disp):
        if len(disp) != self.dim:
            raise AttributeError("The dimension of the displacement vector must match with the nodal dimension.")
        self.disp = ti.Vector(disp, dt=ti.f64)

    @ti.pyfunc
    def set_ID(self, var):
        self.ID = var


if __name__ == '__main__':
    ti.init()
    nd = Node(2,3,4)
    print(nd)
    nd.set_force(3,4,5)
    print(nd.force)
    nd.set_disp(0,1,0)
    print(nd.disp)
    nd.set_ID(4)
    print(nd.ID)
    print(nd.dof)
    nd[2]=0
    print(nd)
    nd2 = Node(2,3,0)
    print(nd == nd2)