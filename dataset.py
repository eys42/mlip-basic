import torch
from molecule import Molecule

class MLIPDataset(torch.utils.data.Dataset):
    def __init__(self, molecules: list[Molecule]) -> None:
        self.molecules: list[Molecule] = molecules
        #self.U0_list: torch.Tensor = torch.tensor([molecule.properties[10] for molecule in molecules], dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.molecules)

    # TODO: complete this method - maybe apply a random transformation to the coords??
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        molecule: Molecule = self.molecules[idx]
        return (molecule.combined_input_tensor.T, molecule.properties[10])
    
    def __getitems__(self, idxs: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        samples: list[tuple] = []
        for idx in idxs:
            samples.append(self.__getitem__(idx))
        return samples
    
def collate_nested(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, outputs = zip(*batch)
    return torch.nested.as_nested_tensor(list(inputs), layout=torch.jagged).contiguous(), torch.stack(list(outputs))
