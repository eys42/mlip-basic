from sklearn.model_selection import train_test_split
from torch import nn, utils, optim, save, load, no_grad, manual_seed, Generator, device
from model import Model
from dataset import MLIPDataset, collate_nested
from molecule import Molecule
from tqdm import tqdm
import wandb
manual_seed(0)

def train_model(model: Model, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.ReduceLROnPlateau, dataset: list[Molecule], batch_size: int = 32, epochs: int = 100, chkfile: str = 'model_checkpoint.pt', torch_device: device | None = None) -> None:
    """
    Load data and train model.
    
    :param model: Model to be trained
    :type model: Model
    :param optimizer: Optimizer for training
    :type optimizer: optim.Optimizer
    :param scheduler: Learning rate scheduler for training
    :type scheduler: optim.lr_scheduler.ReduceLROnPlateau
    :param dataset: list of Molecule objects representing the QM9 dataset
    :type dataset: list[Molecule]
    :param batch_size: batch size for training
    :type batch_size: int
    :param epochs: number of training epochs
    :type epochs: int
    :param chkfile: path to checkpoint file
    :type chkfile: str
    :param torch_device: pytorch device to use for training
    :type torch_device: device | None
    """
    # make train, test, val splits (train = 80%, test = 10%, val = 10%)
    train_molecules, test_molecules = train_test_split(dataset, test_size=0.2, random_state=0, shuffle=False)
    test_molecules, val_molecules = train_test_split(test_molecules, test_size=0.5, random_state=0, shuffle=False)
    train_dataset = MLIPDataset(train_molecules, move_to_device=(torch_device is not None))
    test_dataset = MLIPDataset(test_molecules, move_to_device=(torch_device is not None))
    val_dataset = MLIPDataset(val_molecules, move_to_device=(torch_device is not None))
    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_nested,
                                          generator=Generator(device='cpu'), pin_memory=True, num_workers=8, persistent_workers=True)
    test_dataloader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_nested,
                                         generator=Generator(device='cpu'), pin_memory=True, num_workers=8, persistent_workers=True)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_nested,
                                        generator=Generator(device='cpu'), pin_memory=True, num_workers=8, persistent_workers=True)
    
    loss_fn: nn.MSELoss = nn.MSELoss()
    if torch_device is not None:
        model = model.to(torch_device)
    running_lr = optimizer.param_groups[0]['lr']
    print(f'Starting training for {epochs} epochs with initial learning rate {running_lr}')
    for n in tqdm(range(1, epochs + 1)):
        train_loss: float = train_epoch(model, train_dataloader, optimizer, loss_fn, torch_device=torch_device)
        val_loss: float = evaluate_model(model, val_dataloader, loss_fn, torch_device=torch_device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != running_lr:
            print(f'Learning rate changed from {running_lr} to {current_lr} at epoch {n}')
            running_lr = current_lr
        if n % 10 == 0 or n == 1 or n == epochs:
            save(model.state_dict(), chkfile)
            test_loss: float = evaluate_model(model, test_dataloader, loss_fn, torch_device=torch_device)
            print(f'Epoch {n:03d}: Training loss={train_loss:.4f} (MSE, Ha^2), Validation loss={val_loss:.4f} (MSE, Ha^2), Test loss={test_loss:.4f} (MSE, Ha^2)')
            wandb.log({'epoch': n, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})
        else:
            print(f'Epoch {n:03d}: Training loss={train_loss:.4f} (MSE, Ha^2), Validation loss={val_loss:.4f} (MSE, Ha^2)')
            wandb.log({'epoch': n, 'train_loss': train_loss, 'val_loss': val_loss})
    model.load_state_dict(load(chkfile))
    test_loss: float = evaluate_model(model, test_dataloader, loss_fn, torch_device=torch_device)
    print(f'Test loss={test_loss:.4f} (MSE, Ha^2)')

def train_epoch(model: Model, dataset: utils.data.DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, torch_device: device | None = None) -> float:
    """
    Train model for one epoch.

    :param model: Model to be trained
    :type model: Model
    :param dataset: Training dataset DataLoader
    :type dataset: utils.data.DataLoader
    :param optimizer: Optimizer for training
    :type optimizer: optim.Optimizer
    :param loss_fn: Loss function
    :type loss_fn: nn.Module
    :param torch_device: pytorch device to use for training
    :type torch_device: device | None
    :return: Training loss for the epoch
    :rtype: float
    """
    total_loss: float = 0.0
    for x_batch, y_batch in dataset:
        if torch_device is not None:
            x_batch = x_batch.to(torch_device)
            y_batch = y_batch.to(torch_device)
        y_hat = model(x_batch)
        loss = loss_fn(y_hat.view(-1), y_batch.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_batch.size(0)
    return total_loss / len(dataset)

@no_grad()
def evaluate_model(model: Model, dataset: utils.data.DataLoader, loss_fn: nn.Module, torch_device: device | None = None) -> float:
    """
    Evaluate the model without computing gradients.

    :param model: Model to be evaluated
    :type model: Model
    :param dataset: Evaluation dataset DataLoader
    :type dataset: utils.data.DataLoader
    :param loss_fn: Loss function
    :type loss_fn: nn.Module
    :param torch_device: pytorch device to use for evaluation
    :type torch_device: device | None
    :return: Evaluation loss
    :rtype: float
    """
    total_loss: float = 0.0
    for x_batch, y_batch in dataset:
        if torch_device is not None:
            x_batch = x_batch.to(torch_device)
            y_batch = y_batch.to(torch_device)
        y_hat = model(x_batch)
        loss = loss_fn(y_hat.view(-1), y_batch.view(-1))
        total_loss += loss.item() * y_batch.size(0)
    return total_loss / len(dataset)