import pandas as pd
import numpy as np
import os
import time
from molecule import Molecule

FOLDER = 'QM9data'
# data schema:
# 0: n_atoms 1: 
def import_data():
    files_list = []
    t1 = time.time()
    for filename in os.listdir(os.path.join(os.getcwd(),FOLDER)):
        files_list.append(filename)
    dataset = np.zeros((len(files_list), 16), dtype='float64')
    t2 = time.time()
    print(t2 - t1)
    with open(os.path.join(os.getcwd(),FOLDER,files_list[-1])) as file:
        n_atoms = int(file.readline())
        lines = file.readlines()
        properties = lines[0].split()
        molecule = Molecule(n_atoms, properties)
        molecule.read_coords(lines[1:n_atoms+1])

if __name__ == '__main__':
    import_data()