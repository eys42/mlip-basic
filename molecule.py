import numpy as np
import torch
import random
random.seed(0)

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

"""
Properties schema:
0: A (rotational constant, GHz), 1: B (rotational constant, GHz), 2: C (rotational constant, GHz)
3: mu (dipole moment, Debye), 4: alpha (isotropic polarizability, a_0^3)
5: epsilon_HOMO (energy of HOMO, Hartree), 6: epsilon_LUMO (energy of LUMO, Hartree), 7: epsilon_gap (HOMO-LUMO gap, Hartree)
8: <R^2> (electronic spatial extent, a_0^2), 9: zpve (zero-point vibrational energy, Hartree)
10: U_0 (internal energy at 0 K, Hartree), 11: U (internal energy at 298.15 K, Hartree)
12: H (enthalpy at 298.15 K, Hartree), 13: G (Gibbs free energy at 298.15 K, Hartree), 14: C_v (heat capacity at 298.15 K, cal/(mol K))
"""
class Molecule:
    def __init__(self, n_atoms: int, properties: list[str] | None = None) -> None:
        self.n_atoms: int = n_atoms
        if properties is not None:
            self.idx: int = int(properties[1])
            self.properties: torch.Tensor = torch.tensor([float(property) for property in properties[2:]], dtype=torch.float64)
        self.z_list: torch.Tensor = torch.zeros(n_atoms, dtype=torch.int64)
        self.coords: torch.Tensor = torch.zeros((3, n_atoms), dtype=torch.float64)
        self.dist_matrix: torch.Tensor = torch.zeros((n_atoms, n_atoms), dtype=torch.float64)
    
    def set_attributes(self, idx: int, properties: torch.Tensor, z_list: torch.Tensor, coords: torch.Tensor) -> None:
        """
        Set the molecule's index, properties, z_list, and coords.
        
        :param idx: Index in QM9 dataset
        :type idx: int
        :param properties: Length 15 tensor of molecular properties from QM9 dataset
        :type properties: torch.Tensor
        :param z_list: Length n_atoms tensor of atomic numbers
        :type z_list: torch.Tensor
        :param coords: 3 x n_atoms tensor of atomic coordinates
        :type coords: torch.Tensor
        """
        self.idx: int = idx
        self.properties: torch.Tensor = properties
        self.z_list: torch.Tensor = z_list
        self.coords: torch.Tensor = coords
    
    def coords_from_XYZ(self, xyz_list: list[str]) -> None:
        """
        Reads XYZ specification lines and populates z_list and coords tensor.

        :param xyz_list: List of XYZ lines
        :type xyz_list: list[str]
        """
        for (i, line) in enumerate(xyz_list):
            parts: list[str] = line.split()
            self.z_list[i] = ATOM_DICT[parts[0]]
            try:
                self.coords[:,i] = torch.tensor([float(parts[1]), float(parts[2]), float(parts[3])], dtype=torch.float64)
            except Exception as e:
                # Handle nonstandard usage of '*^'
                self.coords[:,i] = torch.tensor([float(parts[1].replace('*^', 'e')),
                                           float(parts[2].replace('*^', 'e')),
                                           float(parts[3].replace('*^', 'e'))], dtype=torch.float64)
                
    def generate_distance_matrix(self, print_matrix: bool = True) -> None:
        """
        Generates distance matrix for the molecule.
        """
        for i in range(self.n_atoms):
            for j in range(i, self.n_atoms):
                dist = torch.linalg.vector_norm(self.coords[:,i] - self.coords[:,j])
                self.dist_matrix[i,j] = dist
                self.dist_matrix[j,i] = dist
        if print_matrix:
            print(self.dist_matrix)
    
    def apply_random_rotation(self) -> 'Molecule':
        """
        Apply a random 3D rotation to the molecule's coordinates and return a new Molecule instance with rotated coordinates.

        :return: New Molecule instance with rotated coordinates
        :rtype: Molecule
        """
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, 2 * np.pi)
        psi = random.uniform(0, 2 * np.pi)
        costheta, sintheta = np.cos(theta), np.sin(theta)
        cosphi, sinphi = np.cos(phi), np.sin(phi)
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        # composition of rotation matrices about the x, y, and z axes
        rotation_matrix = torch.tensor([[cosphi*cospsi, -cosphi*sinpsi, sinphi],
                                        [cospsi*sintheta*sinphi+costheta*sinpsi, costheta*cospsi-sintheta*sinphi*sinpsi, -cosphi*sintheta],
                                        [-costheta*cospsi*sinphi+sintheta*sinpsi, cospsi*sintheta+costheta*sinphi*sinpsi, costheta*cosphi]],
                                        dtype=torch.float64)
        new_coords = rotation_matrix @ self.coords
        new_molecule: Molecule = Molecule(self.n_atoms)
        new_molecule.set_attributes(self.idx, self.properties, self.z_list.detach().clone(), new_coords)
        return new_molecule
    
    def apply_random_translation(self) -> 'Molecule':
        """
        Apply a random 3D translation within the unit Angstrom sphere to the molecule's coordinates
        and return a new Molecule instance with translated coordinates.

        :return: New Molecule instance with translated coordinates
        :rtype: Molecule
        """
        translation_vector = torch.tensor([random.uniform(-1, 1),
                                           random.uniform(-1, 1),
                                           random.uniform(-1, 1)], dtype=torch.float64)
        translation_vector = translation_vector / torch.linalg.vector_norm(translation_vector)
        translation_vector = translation_vector * random.uniform(0, 1)
        new_coords = self.coords + translation_vector.unsqueeze(1)
        new_molecule: Molecule = Molecule(self.n_atoms)
        new_molecule.set_attributes(self.idx, self.properties, self.z_list.detach().clone(), new_coords)
        return new_molecule
        