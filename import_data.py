import os
import time
from molecule import Molecule

FOLDER = 'QM9data'

def import_data() -> list[Molecule]:
    filenames: list[str] = []
    t1: float = time.time()
    for filename in os.listdir(os.path.join(os.getcwd(), FOLDER)):
        filenames.append(filename)
    QM9_dataset: list[Molecule] = []
    n_problematic = 0
    for filename in filenames:
        try:
            QM9_dataset.append(import_molecule(filename))
        except Exception as e:
            print(filename)
            print(e)
            n_problematic += 1
    t2: float = time.time()
    print(f'Time to import dataset: {t2 - t1:.1f} seconds')
    print(f'Number of problematic files: {n_problematic}')
    print(f'Number of molecules imported: {len(QM9_dataset)}')
    print(f'Total number of files: {len(filenames)}')
    return QM9_dataset

def import_molecule(filename: str) -> Molecule:
    """
    Imports a Molecule from an XYZ file.
    
    :param filename: Filename of molecule XYZ file
    :type filename: str
    :return: Molecule object
    :rtype: Molecule
    """
    with open(os.path.join(os.getcwd(), FOLDER, filename)) as file:
        n_atoms: int = int(file.readline())
        lines: list[str] = file.readlines()
        properties: list[str] = lines[0].split()
        molecule: Molecule = Molecule(n_atoms, properties)
        molecule.coords_from_XYZ(lines[1:n_atoms+1])
        return molecule

if __name__ == '__main__':
    QM9_dataset: list[Molecule] = import_data()