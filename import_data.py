from os import listdir, path, getcwd
from time import time
from molecule import Molecule
from torch import load, save

class QM9DataImport:
    @staticmethod
    def import_data_from_XYZ(folder: str, generate_combined_input_tensor: bool = False, Z_max: int = 9) -> list[Molecule]:
        """
        Imports the entire QM9 dataset from XYZ files in the given folder.

        :param folder: Folder containing the QM9 XYZ files
        :type folder: str    
        :param generate_combined_input_tensor: Whether to generate combined input tensor for each molecule
        :type generate_combined_input_tensor: bool
        :param Z_max: Maximum atomic number for one-hot encoding
        :type Z_max: int
        :return: List of Molecule objects of the QM9 dataset
        :rtype: list[Molecule]
        """
        filenames: list[str] = []
        t1: float = time()
        for filename in listdir(path.join(getcwd(), folder)):
            filenames.append(filename)
        dataset: list[Molecule] = []
        n_problematic = 0
        for filename in filenames:
            try:
                molecule = QM9DataImport.molecule_from_XYZ(filename, folder)
                if generate_combined_input_tensor:
                    molecule.generate_combined_input_tensor(Z_max=Z_max)
                dataset.append(molecule)
            except Exception:
                print(filename)
                print(Exception)
                n_problematic += 1
        t2: float = time()
        print(f'Time to import dataset: {t2 - t1:.1f} seconds')
        print(f'Number of problematic files: {n_problematic}')
        print(f'Number of molecules imported: {len(dataset)}')
        print(f'Total number of files: {len(filenames)}')
        return dataset
    
    @staticmethod
    def molecule_from_XYZ(filename: str, folder: str) -> Molecule:
        """
        Imports a Molecule from an XYZ file.
        
        :param filename: Filename of molecule XYZ file
        :type filename: str
        :param folder: Folder containing the XYZ files
        :type folder: str
        :return: Molecule object
        :rtype: Molecule
        """
        with open(path.join(getcwd(), folder, filename)) as file:
            n_atoms: int = int(file.readline())
            lines: list[str] = file.readlines()
            properties: list[str] = lines[0].split()
            molecule: Molecule = Molecule(n_atoms, properties)
            molecule.coords_from_XYZ(lines[1:n_atoms+1])
            molecule.translate_to_center_of_mass()
            return molecule
    
    @staticmethod
    def save_dataset_to_pt(dataset: list[Molecule]) -> None:
        """
        Saves the entire QM9 dataset to a .pt file.
        
        :param dataset: List of Molecule objects
        :type dataset: list[Molecule]
        """
        t1: float = time()
        dataset_as_list: list[dict] = []
        for molecule in dataset:
            dataset_as_list.append({
                'n_atoms': molecule.n_atoms,
                'idx': molecule.idx,
                'properties': molecule.properties,
                'z_list': molecule.z_list,
                'coords': molecule.coords
            })
        save(dataset_as_list, path.join(getcwd(), 'QM9_dataset.pt'))
        t2: float = time()
        print(f'Time to save dataset as .pt file: {t2 - t1:.1f} seconds')
        print(f'Number of molecules saved: {len(dataset)}')

    @staticmethod
    def load_dataset_from_pt(filepath: str, generate_combined_input_tensor: bool = False, Z_max: int = 9) -> list[Molecule]:
        """
        Loads the entire QM9 dataset from a .pt file.
        
        :param filepath: Path to the .pt file
        :type filepath: str
        :param generate_combined_input_tensor: Whether to generate combined input tensor for each molecule
        :type generate_combined_input_tensor: bool
        :param Z_max: Maximum atomic number for one-hot encoding
        :type Z_max: int
        :return: List of Molecule objects
        :rtype: list[Molecule]
        """
        t1: float = time()
        dataset_as_list = load(filepath)
        dataset: list[Molecule] = []
        for molecule_dict in dataset_as_list:
            molecule = Molecule(molecule_dict['n_atoms'])
            molecule.set_attributes(molecule_dict['idx'], molecule_dict['properties'], molecule_dict['z_list'], molecule_dict['coords'])
            dataset.append(molecule)
            if generate_combined_input_tensor:
                molecule.generate_combined_input_tensor(Z_max=Z_max)
        t2: float = time()
        print(f'Time to load dataset from .pt file: {t2 - t1:.1f} seconds')
        print(f'Number of molecules loaded: {len(dataset)}')
        return dataset
    
if __name__ == '__main__':
    test_molecule = QM9DataImport.molecule_from_XYZ('dsgdb9nsd_000001.xyz', 'QM9data')
    test_molecule.generate_distance_matrix(print_matrix=True)
    print(test_molecule.compute_center_of_mass())
    test_molecule.translate_to_center_of_mass()
    print(test_molecule.compute_center_of_mass())
    test_molecule.generate_distance_matrix(print_matrix=True)