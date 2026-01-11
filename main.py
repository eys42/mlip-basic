from molecule import Molecule
from torch import set_default_dtype, set_default_device, get_default_device, float32, load, cuda, device, save, optim
from model import Model
from train import train_model
from import_data import QM9DataImport
from os import path, getcwd
from uuid import uuid4
from dotenv import load_dotenv
import wandb
import atexit
import sys

def cleanup_wrapper(model: Model, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.ReduceLROnPlateau, chkfile: str):
    """
    Returns a cleanup function to save model checkpoint and log W&B artifact on exit.
    
    :param model: Current model
    :type model: Model
    :param optimizer: Current optimizer
    :type optimizer: optim.Optimizer
    :param scheduler: Current learning rate scheduler
    :type scheduler: optim.lr_scheduler.ReduceLROnPlateau
    :param chkfile: Path to checkpoint file
    :type chkfile: str
    """
    def cleanup() -> None:
        save(model.state_dict(), chkfile)
        save(optimizer.state_dict(), 'optimizer_checkpoint.pt')
        save(scheduler.state_dict(), 'scheduler_checkpoint.pt')
        artifact = wandb.Artifact('mlip-basic-qm9', type='model')
        artifact.add_file('model_checkpoint.pt')
        artifact.add_file('optimizer_checkpoint.pt')
        artifact.add_file('scheduler_checkpoint.pt')
        wandb.log_artifact(artifact)
        wandb.finish()
        print(f'Model checkpoint saved to {chkfile} and W&B artifact logged on exit.')
    return cleanup

if __name__ == '__main__':
    # get args on command line
    args = sys.argv[1:]
    args_dict = {arg.split('=')[0]: arg.split('=')[1] for arg in args}
    config = {
        'batch_size': int(args_dict.get('--batch-size', 32)),
        'epochs': int(args_dict.get('--epochs', 50)),
        'learning_rate': float(args_dict.get('--learning-rate', 0.001)),
        'nhead': int(args_dict.get('--nhead', 4)),
        'd_model': int(args_dict.get('--d-model', 64)),
        'dim_feedforward': int(args_dict.get('--dim-feedforward', 128)),
        'num_layers': int(args_dict.get('--num-layers', 4)),
        'Z_MAX': int(args_dict.get('--Z-MAX', 9))
    }
    
    # configure default datatype and device
    load_dotenv()
    set_default_dtype(float32)
    use_cuda: bool = True
    chkfile: str = 'model_checkpoint.pt'

    # initialize wandb
    load_wandb_artifact: bool = True
    if '--use-wandb-artifact' in args_dict and args_dict['--use-wandb-artifact'].lower() == 'false':
        load_wandb_artifact = False
    wandbname = f'mlip-basic-qm9_{str(uuid4())}'
    wandb.init(
        project='mlip-basic-qm9',
        name=wandbname,
        config=config
    )

    # load dataset
    QM9_dataset: list[Molecule]
    if path.exists(path.join(getcwd(), 'QM9_dataset.pt')):
        QM9_dataset = QM9DataImport.load_dataset_from_pt(
            path.join(getcwd(), 'QM9_dataset.pt'),
            generate_combined_input_tensor=True, Z_max=wandb.config.Z_MAX)
    else:
        QM9_dataset = QM9DataImport.load_dataset_from_XYZ(
            'QM9data',
            generate_combined_input_tensor=True, Z_max=wandb.config.Z_MAX)
        QM9DataImport.save_dataset_to_pt(QM9_dataset)
    
    # initialize model
    model: Model = Model(in_features=wandb.config.Z_MAX + 3, nhead=wandb.config.nhead, d_model=wandb.config.d_model, dim_feedforward=wandb.config.dim_feedforward, num_layers=wandb.config.num_layers)
    artifact_version = 'latest'
    artifact_dir = getcwd()
    if path.exists(chkfile) and not '--wandb-artifact-version' in args_dict:
        model.load_state_dict(load(chkfile))
        print(f'Loaded model checkpoint from {chkfile}')
    elif load_wandb_artifact:
        api = wandb.Api()
        if '--wandb-artifact-version' in args_dict:
            artifact_version = args_dict['--wandb-artifact-version']
        artifact = wandb.use_artifact(f'mlip-basic-qm9:{artifact_version}', type='model')
        artifact_dir = artifact.download()
        model.load_state_dict(load(path.join(artifact_dir, 'model_checkpoint.pt'), map_location=get_default_device()))
        print(f'Loaded model checkpoint from W&B artifact mlip-basic-qm9:{artifact_version}')
    # set default device for training
    torch_device = get_default_device()
    if use_cuda and cuda.is_available():
        torch_device = device('cuda')
        print(f'Using CUDA ({cuda.get_device_name(0)}) for training.')
    else:
        set_default_device(torch_device)
        print(f'Using {torch_device.type} for training.')

    if torch_device is not None:
        model = model.to(torch_device)
    # initialize optimizer and scheduler
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scheduler: optim.lr_scheduler.ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=12, threshold=0.0001, cooldown=5)
    if path.exists(path.join(artifact_dir, 'optimizer_checkpoint.pt')) and not '--reset-optimizer' in args_dict:
        optimizer.load_state_dict(load(path.join(artifact_dir, 'optimizer_checkpoint.pt'), map_location=torch_device))
        print(f'Loaded optimizer checkpoint from {path.join(artifact_dir, "optimizer_checkpoint.pt")}')
    elif '--reset-optimizer' in args_dict:
        print('Optimizer checkpoint not loaded from artifact.')
    if path.exists(path.join(artifact_dir, 'scheduler_checkpoint.pt')) and not '--reset-scheduler' in args_dict:
        scheduler.load_state_dict(load(path.join(artifact_dir, 'scheduler_checkpoint.pt'), map_location=torch_device))
        print(f'Loaded scheduler checkpoint from {path.join(artifact_dir, "scheduler_checkpoint.pt")}')
    elif '--reset-scheduler' in args_dict:
        print('Scheduler checkpoint not loaded from artifact.')
    # start wandb logging and register cleanup function
    wandb.watch(model, log_freq=100)
    atexit.register(cleanup_wrapper(model, optimizer, scheduler, chkfile))

    print('Beginning training:')
    train_model(model, optimizer, scheduler, QM9_dataset, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs, chkfile=chkfile, torch_device=torch_device)