import numpy as np
from torch import Tensor, tensor, get_default_dtype, zeros, linalg, cat
np.random.seed(0)

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
    # atomic Z values H-F
    atom_dict = {
        'H': 1,
        'He': 2,
        'Li': 3,
        'Be': 4,
        'B': 5,
        'C': 6,
        'N': 7,
        'O': 8,
        'F': 9,
    }

    # exact masses in Da of most common isotopes
    exact_mass_list = [1.007825, 4.002603, 7.016005, 9.012183, 11.009305, 12.0, 14.003074, 15.994915, 18.998403]

    def __init__(self, n_atoms: int, properties: list[str] | None = None) -> None:
        self.n_atoms: int = n_atoms
        if properties is not None:
            self.idx: int = int(properties[1])
            self.properties: Tensor = tensor([float(property) for property in properties[2:]], dtype=get_default_dtype())
        self.z_list: Tensor = zeros(n_atoms, dtype=get_default_dtype())
        self.coords: Tensor = zeros((3, n_atoms), dtype=get_default_dtype())
    
    def set_attributes(self, idx: int, properties: Tensor, z_list: Tensor, coords: Tensor) -> None:
        """
        Set the molecule's index, properties, z_list, and coords.
        
        :param idx: Index in QM9 dataset
        :type idx: int
        :param properties: Length 15 tensor of molecular properties from QM9 dataset
        :type properties: Tensor
        :param z_list: Length n_atoms tensor of atomic numbers
        :type z_list: Tensor
        :param coords: 3 x n_atoms tensor of atomic coordinates
        :type coords: Tensor
        """
        self.idx: int = idx
        self.properties: Tensor = properties
        self.z_list: Tensor = z_list
        self.coords: Tensor = coords
    
    def coords_from_XYZ(self, xyz_list: list[str]) -> None:
        """
        Reads XYZ specification lines and populates z_list and coords tensor.

        :param xyz_list: List of XYZ lines
        :type xyz_list: list[str]
        """
        for (i, line) in enumerate(xyz_list):
            parts: list[str] = line.split()
            self.z_list[i] = self.atom_dict[parts[0]]
            try:
                self.coords[:,i] = tensor([float(parts[1]), float(parts[2]), float(parts[3])], dtype=get_default_dtype())
            except Exception as e:
                # Handle nonstandard usage of '*^'
                self.coords[:,i] = tensor([float(parts[1].replace('*^', 'e')),
                                           float(parts[2].replace('*^', 'e')),
                                           float(parts[3].replace('*^', 'e'))], dtype=get_default_dtype())
    
    def apply_random_rotation(self) -> 'Molecule':
        """
        Apply a random 3D rotation to the molecule's coordinates and return a new Molecule instance with rotated coordinates.

        :return: New Molecule instance with rotated coordinates
        :rtype: Molecule
        """
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        psi = np.random.uniform(0, 2 * np.pi)
        costheta, sintheta = np.cos(theta), np.sin(theta)
        cosphi, sinphi = np.cos(phi), np.sin(phi)
        cospsi, sinpsi = np.cos(psi), np.sin(psi)
        # composition of rotation matrices about the x, y, and z axes
        rotation_matrix = tensor([[cosphi*cospsi, -cosphi*sinpsi, sinphi],
                                        [cospsi*sintheta*sinphi+costheta*sinpsi, costheta*cospsi-sintheta*sinphi*sinpsi, -cosphi*sintheta],
                                        [-costheta*cospsi*sinphi+sintheta*sinpsi, cospsi*sintheta+costheta*sinphi*sinpsi, costheta*cosphi]],
                                        dtype=get_default_dtype())
        new_coords = rotation_matrix @ self.coords
        new_molecule: Molecule = Molecule(self.n_atoms)
        new_molecule.set_attributes(self.idx, self.properties, self.z_list.detach().clone(), new_coords)
        return new_molecule
                
    def compute_center_of_mass(self) -> Tensor:
        """
        Computes the center of mass of the molecule using exact atomic masses of the most abundant isotopes.

        :return: Center of mass coordinates as a 3-element tensor
        :rtype: Tensor
        """
        total_mass: float = 0.0
        com: Tensor = zeros(3, dtype=get_default_dtype())
        for i in range(self.n_atoms):
            mass = self.exact_mass_list[int(self.z_list[i].item()) - 1]
            total_mass += mass
            com += self.coords[:, i] * mass
        com /= total_mass
        self.com: Tensor = com
        return com
    
    def translate_to_center_of_mass(self) -> None:
        """
        Translates the molecule's coordinates so that the center of mass is at the origin.
        """
        com: Tensor = self.compute_center_of_mass().unsqueeze(1)
        self.coords -= com
        self.compute_center_of_mass()
                
    def generate_distance_matrix(self, print_matrix: bool = True) -> None:
        """
        Generates distance matrix for the molecule.

        :param print_matrix: Whether to print the distance matrix
        :type print_matrix: bool
        """
        self.dist_matrix: Tensor = zeros((self.n_atoms, self.n_atoms), dtype=get_default_dtype())
        for i in range(self.n_atoms):
            for j in range(i, self.n_atoms):
                dist = linalg.vector_norm(self.coords[:,i] - self.coords[:,j])
                self.dist_matrix[i,j] = dist
                self.dist_matrix[j,i] = dist
        if print_matrix:
            print(self.dist_matrix)
    
    def apply_random_translation(self) -> 'Molecule':
        """
        Apply a random 3D translation within the unit Angstrom sphere to the molecule's coordinates
        and return a new Molecule instance with translated coordinates.

        :return: New Molecule instance with translated coordinates
        :rtype: Molecule
        """
        translation_vector = tensor([np.random.uniform(-1, 1),
                                           np.random.uniform(-1, 1),
                                           np.random.uniform(-1, 1)], dtype=get_default_dtype())
        translation_vector = translation_vector / linalg.vector_norm(translation_vector)
        translation_vector = translation_vector * np.random.uniform(0, 1)
        new_coords = self.coords + translation_vector.unsqueeze(1)
        new_molecule: Molecule = Molecule(self.n_atoms)
        new_molecule.set_attributes(self.idx, self.properties, self.z_list.detach().clone(), new_coords)
        return new_molecule
    
    def one_hot_encode_Z(self, Z_max: int = 9) -> Tensor:
        """
        One-hot encodes the atomic numbers (Z values) of the molecule.

        :param Z_max: Maximum atomic number to consider for one-hot encoding
        :type Z_max: int
        :return: One-hot encoded tensor of shape (n_atoms, Z_max)
        :rtype: Tensor
        """
        Z_tensor = zeros((self.n_atoms, Z_max), dtype=get_default_dtype())
        for i in range(self.n_atoms):
            Z: int | float = self.z_list[i].item()
            Z_tensor[i, int(Z) - 1] = 1.0
        return Z_tensor

    def generate_combined_input_tensor(self, Z_max: int = 9) -> Tensor:
        """
        Generates a combined input tensor of shape (n_atoms, Z_max + 3).
        The first Z_max columns are the one-hot encoded atomic numbers, and the last 3 columns are the x, y, z coordinates.

        :param Z_max: Maximum atomic number to consider for one-hot encoding
        :type Z_max: int
        """
        self.combined_input_tensor = cat((self.one_hot_encode_Z(Z_max), self.coords.T), dim=1)
        return self.combined_input_tensor