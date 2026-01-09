from torch import utils, Tensor, get_default_dtype, nested, jagged, stack, get_default_device
from molecule import Molecule

class MLIPDataset(utils.data.Dataset):
    def __init__(self, molecules: list[Molecule]) -> None:
        self.molecules: list[Molecule] = molecules
        self.input_tensor_list: list[Tensor] = [molecule.combined_input_tensor for molecule in self.molecules]
        self.output_tensor_list: list[Tensor] = [molecule.properties[10] for molecule in self.molecules]

    def __len__(self) -> int:
        return len(self.molecules)

    # TODO: complete this method - maybe apply a random transformation to the coords??
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return (self.input_tensor_list[idx], self.output_tensor_list[idx])
    
    def __getitems__(self, idxs: list[int]) -> list[tuple[Tensor, Tensor]]:
        samples: list[tuple] = []
        for idx in idxs:
            samples.append(self.__getitem__(idx))
        return samples
    
def collate_nested(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
    inputs, outputs = zip(*batch)
    return (nested.as_nested_tensor(list(inputs), layout=jagged).contiguous().to(get_default_dtype()),
    stack(list(outputs)).to(get_default_dtype()))