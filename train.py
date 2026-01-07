from sklearn.model_selection import train_test_split
import torch
from model import Model
from import_data import QM9DataImport
from molecule import Molecule
torch.manual_seed(0)

def train_model(model: Model, dataset: list[Molecule], batch_size: int = 32, epochs: int = 100, lr: float = 0.001) -> None:
    """
    Docstring for train_model
    
    :param model: Description
    :type model: Model
    :param dataset: Description
    :type dataset: list[Molecule]
    :param epochs: Description
    :type epochs: int
    :param lr: Description
    :type lr: float
    """
    # make train, test, val splits (train = 80%, test = 10%, val = 10%)
    train_molecules, test_molecules = train_test_split(dataset, test_size=0.2, random_state=0, shuffle=False)
    test_molecules, val_molecules = train_test_split(test_molecules, test_size=0.5, random_state=0, shuffle=False)
    train_dataset = torch.utils.data.DataLoader(MLIPDataset(train_molecules), batch_size=batch_size, shuffle=True, collate_fn=collate_nested)
    test_dataset = torch.utils.data.DataLoader(MLIPDataset(test_molecules), batch_size=batch_size, shuffle=False, collate_fn=collate_nested)
    val_dataset = torch.utils.data.DataLoader(MLIPDataset(val_molecules), batch_size=batch_size, shuffle=False, collate_fn=collate_nested)

    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()

class MLIPDataset(torch.utils.data.Dataset):
    def __init__(self, molecules: list[Molecule]) -> None:
        self.molecules: list[Molecule] = molecules
        self.U0_list: torch.Tensor = torch.tensor([molecule.properties[10] for molecule in molecules], dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.molecules)

    # TODO: complete this method - maybe apply a random transformation to the coords??
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        molecule: Molecule = self.molecules[idx]
        return (molecule.combined_input_tensor, molecule.properties[10])
    
    def __getitems__(self, idxs: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        samples: list[tuple] = []
        for idx in idxs:
            samples.append(self.__getitem__(idx))
        return samples
    
def collate_nested(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, outputs = zip(*batch)
    return torch.nested.as_nested_tensor(list(inputs)), torch.stack(list(outputs))

