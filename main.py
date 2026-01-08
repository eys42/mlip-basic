from molecule import Molecule
from torch import set_default_dtype, set_default_device, get_default_device, float64, float32, load, backends, cuda, device, save
from model import Model
from train import train_model
from import_data import QM9DataImport
from os import path, getcwd
from uuid import uuid4
from dotenv import load_dotenv
import wandb
import atexit
import sys


def cleanup_wrapper(model: Model, chkfile: str):
    """
    Generates a cleanup function to save model checkpoint and log W&B artifact on exit.
    
    :param model: Current model
    :type model: Model
    :param chkfile: Path to checkpoint file
    :type chkfile: str
    """
    def cleanup() -> None:
        save(model.state_dict(), chkfile)
        artifact = wandb.Artifact('mlip-basic-qm9', type='model')
        artifact.add_file('model_checkpoint.pt')
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
        'num_layers': int(args_dict.get('--num-layers', 4)),
        'Z_MAX': int(args_dict.get('--Z-MAX', 9))
    }
    
    # configure default datatype and device
    load_dotenv()
    set_default_dtype(float64)
    use_cuda: bool = True
    chkfile: str = 'model_checkpoint.pt'

    # initialize wandb
    load_wandb_artifact: bool = True
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
        QM9_dataset = QM9DataImport.import_data_from_XYZ(
            'QM9data',
            generate_combined_input_tensor=True, Z_max=wandb.config.Z_MAX)
        QM9DataImport.save_dataset_to_pt(QM9_dataset)
    
    # initialize model
    model: Model = Model(in_features=wandb.config.Z_MAX + 3, nhead=wandb.config.nhead, d_model=wandb.config.d_model, num_layers=wandb.config.num_layers)
    if path.exists(chkfile) and not '--wandb-artifact-version' in args_dict:
        model.load_state_dict(load(chkfile))
        print(f'Loaded model checkpoint from {chkfile}')
    elif load_wandb_artifact:
        api = wandb.Api()
        artifact_version = 'latest'
        if '--wandb-artifact-version' in args_dict:
            artifact_version = args_dict['--wandb-artifact-version']
        artifact = wandb.use_artifact(f'mlip-basic-qm9:{artifact_version}', type='model')
        artifact_dir = artifact.download()
        model.load_state_dict(load(path.join(artifact_dir, 'model_checkpoint.pt')))
        print(f'Loaded model checkpoint from W&B artifact mlip-basic-qm9:{artifact_version}')
    # set default device for training
    torch_device = get_default_device()
    if use_cuda and cuda.is_available():
        torch_device = device('cuda')
        print(f'Using CUDA ({cuda.get_device_name(0)}) for training.')
    else:
        set_default_device(torch_device)
        print(f'Using {torch_device.type} for training.')
    
    # start wandb logging and register cleanup function
    wandb.watch(model, log_freq=100)
    atexit.register(cleanup_wrapper(model, chkfile))

    print('Beginning training:')
    train_model(model, QM9_dataset, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs, lr=wandb.config.learning_rate, chkfile=chkfile, torch_device=torch_device)