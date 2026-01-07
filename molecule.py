import numpy as np
from torch import Tensor

# atoms through the second period
ATOM_DICT = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10
}

class Molecule:
    def __init__(self, n_atoms: int, properties: list[str]) -> None:
        self.n_atoms: int = n_atoms
        self.idx: int = int(properties[1])
        self.properties: np.ndarray = np.array([float(property) for property in properties[2:]])
        self.z_list: np.ndarray = np.zeros(self.n_atoms, dtype='int64')
        self.coords: Tensor = Tensor∏
    def read_coords(self, xyz_list: list[str]) -> None:
        for (i, line) in enumerate(xyz_list):
            parts: list[str] = line.split()
            self.z_list[i] = ATOM_DICT[parts[0]]
            print(ATOM_DICT[parts[0]])