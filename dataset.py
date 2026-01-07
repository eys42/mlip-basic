from torch import utils, Tensor, get_default_dtype, nested, jagged, stack
from molecule import Molecule

class MLIPDataset(utils.data.Dataset):
    def __init__(self, molecules: list[Molecule]) -> None:
        self.molecules: list[Molecule] = molecules
        #self.U0_list: Tensor = tensor([molecule.properties[10] for molecule in molecules], dtype=get_default_dtype())

    def __len__(self) -> int:
        return len(self.molecules)

    # TODO: complete this method - maybe apply a random transformation to the coords??
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        molecule: Molecule = self.molecules[idx]
        # TODO: fix this behavior - start out with a transposed tensor
        return (molecule.combined_input_tensor.T, molecule.properties[10])
    
    def __getitems__(self, idxs: list[int]) -> list[tuple[Tensor, Tensor]]:
        samples: list[tuple] = []
        for idx in idxs:
            samples.append(self.__getitem__(idx))
        return samples
    
def collate_nested(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    inputs, outputs = zip(*batch)
    return nested.as_nested_tensor(list(inputs), layout=jagged).contiguous(), stack(list(outputs))