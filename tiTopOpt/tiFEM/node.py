import numpy as np


class Node(object):
    def __init__(self, *pos):
        self.init_pos(*pos) # position
        self.set_dof() # degree-of-freedom
        self._ID = None # index
        self._force = [] # force vector
        self._disp = [] # displacement vector

    def __repr__(self):
        return "Node:%r" % (self.pos,)

    def __getitem__(self, index):
        return self.pos[index]

    def __setitem__(self, index, val):
        l = self.pos
        l[index] = val
        self.pos = l

    def __eq__(self, other):
        assert issubclass(type(other), Node), "Must be Node type"
        a = np.array(self.pos)
        b = np.array(other.pos)
        if np.isclose(a, b).all():
            return True
        else:
            return False

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = val

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = val

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, val):
        self._z = val

    @property
    def dofX(self):
        return self._dofX

    @dofX.setter
    def dofX(self, bool):
        self._dofX = bool

    @property
    def dofY(self):
        return self._dofY

    @dofY.setter
    def dofY(self, bool):
        self._dofY = bool

    @property
    def dofZ(self):
        return self._dofZ

    @dofZ.setter
    def dofZ(self, bool):
        self._dofZ = bool

    @property
    def ID(self):
        return self._ID

    @ID.setter
    def ID(self, val):
        self._ID = val

    @property
    def force(self):
        return self._force

    @property
    def disp(self):
        return self._disp

    def init_pos(self, *pos):
        self._x = 0.
        self._y = 0.
        self._z = 0.

        if len(pos) == 1:
            self.dim = len(pos[0])
            if self.dim == 2:
                self._x = pos[0][0] * 1.
                self._y = pos[0][1] * 1.
                self.pos = (self.x, self.y)
            elif self.dim == 3:
                self._x = pos[0][0] * 1.
                self._y = pos[0][1] * 1.
                self._z = pos[0][2] * 1.
                self.pos = (self.x, self.y, self.z)
            else:
                raise AttributeError("Node dimension must be 2 or 3")

        elif len(pos) == 2:
            self.pos = tuple(pos)
            self.dim = 2
            self._x = pos[0] * 1.
            self._y = pos[1] * 1.
            self.pos = (self.x, self.y)
        elif len(pos) == 3:
            self.pos = tuple(pos)
            self.dim = 3
            self._x = pos[0] * 1.
            self._y = pos[1] * 1.
            self._z = pos[2] * 1.
            self.pos = (self.x, self.y, self.z)
        else:
            raise AttributeError("Node dimension must be 2 or 3")

    def set_dof(self):
        self._dofX = False
        self._dofY = False
        self._dofZ = False

    def set_force(self, *forces):
        for force in forces:
            self._force += force

    def clear_force(self):
        self._force = []

    def get_force(self):
        return self._force

    def set_disp(self, *disps):
        for disp in disps:
            self._disp += disp

    def clear_disp(self):
        self._force = []

        def get_disp(self):
            return self._disp
