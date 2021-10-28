# Node class

A class for implementing elemental nodes.

### Members

| Members | Description               | Type      | Shape  |
| ------- | ------------------------- | --------- | ------ |
| pos     | Nodal position            | ti.Vector | 2 or 3 |
| dim     | Nodal dimension           | int32     |        |
| ID      | Nodal ID (Default: -1)    | int32     |        |
| force   | Nodal force vector        | ti.Vector | 2 or 3 |
| disp    | Nodal displacement vector | ti.Vector | 2 or 3 |

### Functions

| Functions     | Description                      | Return |
| ------------- | -------------------------------- | ------ |
| clear_force() | Clear nodal force vector.        | None   |
| clear_disp()  | Clear nodal displacement vector. | None   |

### Example

```python
nd = Node(2.,3.,4.)
print(nd) # Node: [2. 3. 4.]
nd.force=ti.Vector([3.,4.,5.])
print(nd.force) # [3. 4. 5.]
nd.disp=ti.Vector([1.,0.,1.])
print(nd.disp) # [0. 1. 0.]
nd.ID = 4
print(nd.ID) # 4
nd.pos[1]=2
print(nd) # Node: [2. 2. 4.]
nd2 = Node(2.,2.,4.)
nd3 = Node(2.,3.,4.)
print(nd == nd2) # True
print(nd == nd3) # False
```

