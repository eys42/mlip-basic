from sklearn.model_selection import train_test_split
import torch
from model import Model
from dataset import MLIPDataset, collate_nested
from molecule import Molecule
from tqdm import tqdm

torch.manual_seed(0)

def train_model(model: Model, dataset: list[Molecule], batch_size: int = 32, epochs: int = 100, lr: float = 0.001, use_mps: bool = False) -> None:
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
    :param use_mps: Description
    :type use_mps: bool
    """
    if use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    # make train, test, val splits (train = 80%, test = 10%, val = 10%)
    train_molecules, test_molecules = train_test_split(dataset, test_size=0.2, random_state=0, shuffle=False)
    test_molecules, val_molecules = train_test_split(test_molecules, test_size=0.5, random_state=0, shuffle=False)
    train_dataset = torch.utils.data.DataLoader(MLIPDataset(train_molecules), batch_size=batch_size, shuffle=True, collate_fn=collate_nested)
    test_dataset = torch.utils.data.DataLoader(MLIPDataset(test_molecules), batch_size=batch_size, shuffle=False, collate_fn=collate_nested)
    val_dataset = torch.utils.data.DataLoader(MLIPDataset(val_molecules), batch_size=batch_size, shuffle=False, collate_fn=collate_nested)
    
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss()
    for n in tqdm(range(1, epochs + 1)):
        train_loss: float = train_epoch(model, train_dataset, optimizer, loss_fn)
        val_loss: float = evaluate_model(model, val_dataset, loss_fn)
        if n % 10 == 0 or n == 1 or n == epochs:
            torch.save(model.state_dict(), 'model_checkpoint.pt')
            print(f'Epoch {n:02d}: Training loss={train_loss:.4f} (MSE, Ha^2), Validation loss={val_loss:.4f} (MSE, Ha^2)')
    model.load_state_dict(torch.load('model_checkpoint.pt'))
    test_loss: float = evaluate_model(model, test_dataset, loss_fn)
    print(f'Test loss={test_loss:.4f} (MSE, Ha^2)')

def train_epoch(model: Model, dataset: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module) -> float:
    """
    Docstring for train_epoch
    
    :param model: Description
    :type model: Model
    :param dataset: Description
    :type dataset: torch.utils.data.DataLoader
    :param optimizer: Description
    :type optimizer: torch.optim.Optimizer
    :param loss_fn: Description
    :type loss_fn: torch.nn.Module
    :return: Description
    :rtype: float
    """
    total_loss: float = 0.0
    for x_batch, y_batch in dataset:
        y_hat = model(x_batch)
        loss = loss_fn(y_hat.view(-1), y_batch.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
    return total_loss / len(dataset)

@torch.no_grad()
def evaluate_model(model: Model, dataset: torch.utils.data.DataLoader, loss_fn: torch.nn.Module) -> float:
    """
    Docstring for evaluate_model
    
    :param model: Description
    :type model: Model
    :param dataset: Description
    :type dataset: torch.utils.data.DataLoader
    :param loss_fn: Description
    :type loss_fn: torch.nn.Module
    :return: Description
    :rtype: float
    """
    total_loss: float = 0.0
    for x_batch, y_batch in dataset:
        y_hat = model(x_batch)
        loss = loss_fn(y_hat.view(-1), y_batch.view(-1))
        total_loss += loss.item() * y_batch.size(0)
    return total_loss / len(dataset)