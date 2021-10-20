# Node class

A class for implementing elemental nodes.

| Members    | Description                            | Type    |
| ---------- | -------------------------------------- | ------- |
| pos        | Nodal position                         | List    |
| dim        | Nodal dimension                        | Int32   |
| x          | X coordinate                           | float32 |
| y          | Y coordinate                           | float32 |
| z          | Z coordinate (if `dim = 2`, `z = 0.0`) | float32 |
| ID         | Nodal ID                               | Int32   |
| force      | Nodal force vector                     | List    |
| disp       | Nodal displacement vector              | List    |
| cont_elems | The indices of the connected elements  | List    |
| adj_nds    | The indices of the adjacent nodes      | List    |

| Functions         | Description                                                  | Return |
| ----------------- | ------------------------------------------------------------ | ------ |
| init_pos(*pos)    | Initilize the position.                                      | None   |
| set_force(*force) | Set nodal force vector. All input force vectors will be summed into one vector. | None   |
| clear_force()     | Clear nodal force vector.                                    | None   |
| set_disp(*disps)  | Set nodal displacement vector. All input displacement vectors will be summed into one vector. | None   |
| clear_disp()      | Clear nodal displacement vector.                             | None   |

### Example

```python
nd = Node([2,3,4])
print(nd) # Node:(2.0, 3.0, 4.0)
nd.set_force([3,4,5])
print(nd.force) # [3, 4, 5]
nd.set_disp([0.,1.,0.])
print(nd.disp) # [0.0, 1.0, 0.0]
nd.ID = 4
print(nd.ID) # 4
nd.y = 2.
print(nd) # Node:(2.0, 3.0, 4.0)
```

