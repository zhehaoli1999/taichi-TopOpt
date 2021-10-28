# Element class

A class for supporting various finite element types.

### Members

| Members | Description                | Type          | Shape             |
| ------- | -------------------------- | ------------- | ----------------- |
| dim     | Elemental dimension        | int32         |                   |
| nodes   | Elemental nodes            | ti.field(f64) | (len(nodes), dim) |
| ID      | Elemental ID (Default: -1) | int32         |                   |
| E       | Young's modulus            | f32           |                   |
| nu      | Possion's ratio            | f32           |                   |



# Triangle Class (Element)

A class for construct a 2D triangular element.

### Members

| Members     | Description                 | Type          | Shape  |
| ----------- | --------------------------- | ------------- | ------ |
| dim         | Elemental dimension         | int32         |        |
| nodes       | Elemental nodes             | ti.field(f64) | (3, 2) |
| nd_len      | Number of the nodes         | int32         |        |
| ndof        | Number of degree-of-freedom | int32         |        |
| ID          | Elemental ID (Default: -1)  | int32         |        |
| E           | Young's modulus             | f32           |        |
| nu          | Possion's ratio             | f32           |        |
| volume      | Elemental volume (area)     | f64           |        |
| type_abaqus | Elemental type in Abaqus    | string        |        |
| Ke          | Elemental stiffness matrix  | ti.field(f64) | (6, 6) |

### Functions

| Functions   | Description                          | Return |
| ----------- | ------------------------------------ | ------ |
| calc_area() | Calculate triangular areas           | f64    |
| calc_Ke()   | Calculate elemental stiffness matrix | None   |

### Example

```Python
# ===== Triangle =====
tri_0 = Node(0., 0., 0.)
tri_1 = Node(1., 0., 0.)
tri_2 = Node(1., 1., 0.)
tri_ele = Triangle([tri_0, tri_1, tri_2])
tri_ele.calc_Ke()
```



# Quadrangle Class (Element)

A class for construct a 2D quadrangle element.

### Members

| Members     | Description                                                  | Type          | Shape  |
| ----------- | ------------------------------------------------------------ | ------------- | ------ |
| dim         | Elemental dimension                                          | int32         |        |
| nodes       | Elemental nodes                                              | ti.field(f64) | (4, 2) |
| nd_len      | Number of the nodes                                          | int32         |        |
| ndof        | Number of degree-of-freedom                                  | int32         |        |
| ID          | Elemental ID (Default: -1)                                   | int32         |        |
| E           | Young's modulus                                              | f32           |        |
| nu          | Possion's ratio                                              | f32           |        |
| type_abaqus | Elemental type in Abaqus                                     | string        |        |
| Ke          | Elemental stiffness matrix                                   | ti.field(f64) | (8, 8) |
| intv        | [Legendre-Gauss Quadrature](http://mathworld.wolfram.com/Legendre-GaussQuadrature.html#:~:text=Legendre-Gauss quadrature is a numerical integration method also,the Legendre polynomials%2C which occur symmetrically about 0.) | integration   |        |

### Functions

| Functions | Description                          | Return |
| --------- | ------------------------------------ | ------ |
| calc_Ke() | Calculate elemental stiffness matrix | None   |

### Examples

```Python
# ===== Quadrangle =====
quad_0 = Node(0., 0., 0.)
quad_1 = Node(1., 0., 0.)
quad_2 = Node(1., 1., 0.)
quad_3 = Node(1., 2., 0.)
quad_ele = Quadrangle([quad_0,quad_1,quad_2,quad_3])
quad_ele.calc_Ke()
```



# Tetrahedron Class (Element)

A class for construct a 3D tetrahedral element.

### Members

| Members     | Description                 | Type          | Shape    |
| ----------- | --------------------------- | ------------- | -------- |
| dim         | Elemental dimension         | int32         |          |
| nodes       | Elemental nodes             | ti.field(f64) | (4, 3)   |
| nd_len      | Number of the nodes         | int32         |          |
| ndof        | Number of degree-of-freedom | int32         |          |
| ID          | Elemental ID (Default: -1)  | int32         |          |
| E           | Young's modulus             | f32           |          |
| nu          | Possion's ratio             | f32           |          |
| volume      | Elemental volume (area)     | f64           |          |
| type_abaqus | Elemental type in Abaqus    | string        |          |
| Ke          | Elemental stiffness matrix  | ti.field(f64) | (12, 12) |

### Functions

| Functions     | Description                          | Return |
| ------------- | ------------------------------------ | ------ |
| calc_volume() | Calculate tetrahedral volume         | f64    |
| calc_Ke()     | Calculate elemental stiffness matrix | None   |

### Examples

```Python
# ===== Tetrahedron =====
tet_0 = Node(0., 0., 0.)
tet_1 = Node(1., 0., 0.)
tet_2 = Node(1., 1., 0.)
tet_3 = Node(1., 1., 1.)
tet_ele = Tetrahedron([tet_0,tet_1,tet_2,tet_3])
tet_ele.calc_Ke()
```



# Hexahedron Class (Element)

A class for construct a 3D hexahedral element.

### Members

| Members     | Description                                                  | Type          | Shape    |
| ----------- | ------------------------------------------------------------ | ------------- | -------- |
| dim         | Elemental dimension                                          | int32         |          |
| nodes       | Elemental nodes                                              | ti.field(f64) | (8, 3)   |
| nd_len      | Number of the nodes                                          | int32         |          |
| ndof        | Number of degree-of-freedom                                  | int32         |          |
| ID          | Elemental ID (Default: -1)                                   | int32         |          |
| E           | Young's modulus                                              | f32           |          |
| nu          | Possion's ratio                                              | f32           |          |
| type_abaqus | Elemental type in Abaqus                                     | string        |          |
| Ke          | Elemental stiffness matrix                                   | ti.field(f64) | (24, 24) |
| intv        | [Legendre-Gauss Quadrature](http://mathworld.wolfram.com/Legendre-GaussQuadrature.html#:~:text=Legendre-Gauss quadrature is a numerical integration method also,the Legendre polynomials%2C which occur symmetrically about 0.) | integration   |          |

### Functions

| Functions | Description                          | Return |
| --------- | ------------------------------------ | ------ |
| calc_Ke() | Calculate elemental stiffness matrix | None   |

### Examples

```Python
# ===== Hexahedron =====
hex_0 = Node(0.,0.,0.)
hex_1 = Node(1.,0.,0.)
hex_2 = Node(1.,1.,0.)
hex_3 = Node(0.,1.,0.)
hex_4 = Node(0.,0.,1.)
hex_5 = Node(1.,0.,1.)
hex_6 = Node(1.,1.,1.)
hex_7 = Node(0.,1.,1.)
hex_ele = Hexahedron([hex_0,hex_1,hex_2,hex_3,hex_4,hex_5,hex_6,hex_7])
hex_ele.calc_Ke()
```

