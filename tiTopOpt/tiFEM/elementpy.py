from nodepy import *


class Element(object):
    def __init__(self, nodes):
        for nd in nodes:
            assert issubclass(type(nd), Node), "Must be Node type"

        self._nodes = nodes  # elemental nodes
        self._dim = 0  # dimension
        self._type = self.__class__.__name__  # elemental type
        self._ID = None  # index
        self._eIk = None
        self._ndof = None  # nodal degree-of-freedom
        self._volume = None  # elemental volume
        self.init_keys()

        self._E = None  # Young's modulus
        self._nu = None  # Possion's ratio
        self._t = 1.  # thickness
        self._D = None
        self._B = None
        self._J = None
        self._Ke = None  # elemental stiffness matrix
        self._Me = None  # elemental mass matrix
        self.Xe = 1.  # design variable

        self._stress = dict.fromkeys(self.eIk, 0.)
        self.init_nodes(nodes)  # initialize nodes
        self.dens = 2
        self.t = 1  # thickness

    def __repr__(self):
        return "%s Element: %r" % (self.elem_type, self.nodes,)

    def __getitem__(self, key):
        return self._nodes[key]

    @property
    def volume(self):
        return self._volume

    @property
    def B(self):
        return self._B

    @property
    def D(self):
        return self._D

    @property
    def J(self):
        return self._J

    @property
    def dim(self):
        return self.nodes[0].dim

    @property
    def eIk(self):
        return self._eIk

    @property
    def Ke(self):
        return self._Ke

    @Ke.setter
    def Ke(self, val):
        self._Ke = val

    @property
    def Me(self):
        return self._Me

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, val):
        self._E = val

    @property
    def nu(self):
        return self._nu

    @nu.setter
    def nu(self, val):
        self._nu = val

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, val):
        self._t = val

    @property
    def Xe(self):
        return self._Xe

    @Xe.setter
    def Xe(self, val):
        self._Xe = val

    @property
    def ndof(self):
        return self._ndof

    @property
    def nodes(self):
        return self._nodes

    @property
    def elem_type(self):
        return self._type

    @property
    def ID(self):
        return self._ID

    @property
    def non(self):
        return len(self._nodes)

    @property
    def stress(self):
        return self._stress

    def init_nodes(self, nodes):
        pass

    def init_unknowns(self):
        pass

    def init_keys(self):
        pass

    def func(self, x):
        pass

    def func_jac(self, x):
        pass

    def calc_Ke(self):
        pass

    def set_eIk(self, val):
        self._eIk = val

    def get_eIk(self):
        return self._eIk

    def set_ndof(self, val):
        self._ndof = val

    def get_ndof(self):
        return self._ndof

    def get_element_type(self):
        return self.elem_type

    def get_nodes(self):
        return self._nodes

    def calc_D(self):
        pass

    def calc_B(self, *intv_pts):
        pass

    def calc_Ke(self):
        self.calc_B()
        self.calc_D()
        self._Ke = self.volume * np.dot(np.dot(self.B.T, self.D), self.B)

    def evaluate(self):
        u = np.array([[nd.disp[key] for nd in self.nodes for key in nd.nAk[:self.ndof]]])
        self._undealed_stress = np.dot(np.dot(self.D, self.B), u.T)
        self.distribute_stress()

    def distribute_stress(self):
        n = len(self.eIk)
        for i, val in enumerate(self.eIk):
            self._stress[val] += self._undealed_stress[i::n]
