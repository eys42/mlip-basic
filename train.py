from sklearn.model_selection import train_test_split
from torch import nn, utils, optim, save, load, no_grad, manual_seed, get_default_device, randn
from model import Model
from dataset import MLIPDataset, collate_nested
from molecule import Molecule
from tqdm import tqdm
import wandb
manual_seed(0)

def train_model(model: Model, dataset: list[Molecule], batch_size: int = 32, epochs: int = 100, lr: float = 0.001, chkfile: str = 'model_checkpoint.pt') -> None:
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
    # make train, test, val splits (train = 80%, test = 10%, val = 10%)
    train_molecules, test_molecules = train_test_split(dataset, test_size=0.2, random_state=0, shuffle=False)
    test_molecules, val_molecules = train_test_split(test_molecules, test_size=0.5, random_state=0, shuffle=False)
    train_dataset = utils.data.DataLoader(MLIPDataset(train_molecules), batch_size=batch_size, shuffle=True, collate_fn=collate_nested)
    test_dataset = utils.data.DataLoader(MLIPDataset(test_molecules), batch_size=batch_size, shuffle=False, collate_fn=collate_nested)
    val_dataset = utils.data.DataLoader(MLIPDataset(val_molecules), batch_size=batch_size, shuffle=False, collate_fn=collate_nested)
    
    # dummy input to allocate space on GPU
    if get_default_device().type == 'mps' or get_default_device().type == 'cuda':
        model = model.to(get_default_device())
        with no_grad():
            dummy_input = randn(1, 5, 12).to(get_default_device()) 
            _ = model(dummy_input)
    
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=lr)
    loss_fn: nn.MSELoss = nn.MSELoss()
    for n in tqdm(range(1, epochs + 1)):
        train_loss: float = train_epoch(model, train_dataset, optimizer, loss_fn)
        val_loss: float = evaluate_model(model, val_dataset, loss_fn)
        wandb.log({'epoch': n, 'train_loss': train_loss, 'val_loss': val_loss})
        if n % 10 == 0 or n == 1 or n == epochs:
            save(model.state_dict(), chkfile)
            print(f'Epoch {n:02d}: Training loss={train_loss:.4f} (MSE, Ha^2), Validation loss={val_loss:.4f} (MSE, Ha^2)')
    model.load_state_dict(load(chkfile))
    test_loss: float = evaluate_model(model, test_dataset, loss_fn)
    print(f'Test loss={test_loss:.4f} (MSE, Ha^2)')

def train_epoch(model: Model, dataset: utils.data.DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module) -> float:
    """
    Docstring for train_epoch
    
    :param model: Description
    :type model: Model
    :param dataset: Description
    :type dataset: utils.data.DataLoader
    :param optimizer: Description
    :type optimizer: optim.Optimizer
    :param loss_fn: Description
    :type loss_fn: nn.Module
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

@no_grad()
def evaluate_model(model: Model, dataset: utils.data.DataLoader, loss_fn: nn.Module) -> float:
    """
    Docstring for evaluate_model
    
    :param model: Description
    :type model: Model
    :param dataset: Description
    :type dataset: utils.data.DataLoader
    :param loss_fn: Description
    :type loss_fn: nn.Module
    :return: Description
    :rtype: float
    """
    total_loss: float = 0.0
    for x_batch, y_batch in dataset:
        y_hat = model(x_batch)
        loss = loss_fn(y_hat.view(-1), y_batch.view(-1))
        total_loss += loss.item() * y_batch.size(0)
    return total_loss / len(dataset)