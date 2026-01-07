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


def cleanup_wrapper(model: Model, chkfile: str):
    def cleanup() -> None:
        wandb.finish()
        save(model.state_dict(), chkfile)
        print(f'W&B exited and model checkpoint saved to {chkfile} on exit.')
    return cleanup

if __name__ == '__main__':
    # configure default datatype and device
    load_dotenv()
    set_default_dtype(float64)
    use_mps: bool = False
    use_cuda: bool = True
    chkfile = 'model_checkpoint.pt'

    wandb.init(
        project='mlip-basic-qm9',
        name=f'mlip-basic-qm9_{str(uuid4())}',
        config={
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'nhead': 4,
            'd_model': 64,
            'num_layers': 4,
            'Z_MAX': 9
        }
    )

    # load dataset
    QM9_dataset: list[Molecule]
    if path.exists(path.join(getcwd(), 'QM9_dataset.pt')):
        QM9_dataset = QM9DataImport.load_dataset_from_pt(
            path.join(getcwd(), 'QM9_dataset.pt'),
            generate_combined_input_tensor=True)
    else:
        QM9_dataset = QM9DataImport.import_data_from_XYZ(
            'QM9data',
            generate_combined_input_tensor=True)
        QM9DataImport.save_dataset_to_pt(QM9_dataset)
    
    # initialize model
    model: Model = Model(in_features=wandb.config.Z_MAX + 3, nhead=wandb.config.nhead, d_model=wandb.config.d_model, num_layers=wandb.config.num_layers)
    if path.exists(chkfile):
        model.load_state_dict(load(chkfile))
        print(f'Loaded model checkpoint from {chkfile}')

    # set default device for training
    torch_device = get_default_device()
    if use_mps and backends.mps.is_available():
        torch_device = device('mps')
        set_default_device(torch_device)
        model = model.to(torch_device)
        print('Using MPS device (Apple Silicon Metal API) for training.')
    elif use_cuda and cuda.is_available():
        torch_device = device('cuda')
        set_default_device(torch_device)
        model = model.to(torch_device)
        print(f'Using CUDA {cuda.get_device_name(0)} for training.')
    else:
        set_default_device(torch_device)
        print(f'Using {torch_device.type} for training.')
    
    wandb.watch(model, log_freq=100)
    atexit.register(cleanup_wrapper(model, chkfile))

    print('Beginning training:')
    train_model(model, QM9_dataset, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs, lr=wandb.config.learning_rate, chkfile=chkfile)