from torch import nn, utils, load, manual_seed, Generator, get_default_device
from time import time
from os import path, getcwd
from dotenv import load_dotenv
from model import Model
from dataset import MLIPDataset, collate_nested
from import_data import QM9DataImport
from molecule import Molecule
from train import evaluate_model
import wandb
import numpy as np
import sys
import atexit

def cleanup_wrapper(time_arr, loss_arr, j):
    def cleanup() -> None:
        print(f'Average test loss over {j+1} evaluations: {np.mean(loss_arr[:j+1]):.4f} (MSE, Ha^2) ± {np.std(loss_arr[:j+1]):.4f}')
        print(f'Average evaluation time over {j+1} evaluations: {np.mean(time_arr[:j+1]):.1f} seconds ± {np.std(time_arr[:j+1]):.1f} seconds')
    return cleanup

if __name__ == '__main__':
    manual_seed(0)
    np.random.seed(0)
    load_dotenv()
    args = sys.argv[1:]
    args_dict = {arg.split('=')[0]: arg.split('=')[1] for arg in args}
    config = {
        'nhead': 4,
        'd_model': 64,
        'num_layers': 4,
        'batch_size': int(args_dict.get('--batch-size', 64)),
        'Z_MAX': 9,
        'n_evals': int(args_dict.get('--n-evals', 10)),
        'artifact_version': args_dict.get('--wandb-artifact-version', 'latest')
    }
    QM9_dataset: list[Molecule]
    if path.exists(path.join(getcwd(), 'QM9_dataset.pt')):
        QM9_dataset = QM9DataImport.load_dataset_from_pt(
            path.join(getcwd(), 'QM9_dataset.pt'),
            generate_combined_input_tensor=True, Z_max=config['Z_MAX'])
    else:
        QM9_dataset = QM9DataImport.load_dataset_from_XYZ(
            'QM9data',
            generate_combined_input_tensor=True, Z_max=config['Z_MAX'])
        QM9DataImport.save_dataset_to_pt(QM9_dataset)
    # generator function to split dataset into n_parts
    def split_fn(dataset, n_parts):
        n = len(dataset) // n_parts
        for i in range(0, len(dataset), n):
            yield dataset[i:i + n]
    dataset_split = split_fn(QM9_dataset, config['n_evals'])
    loss_fn: nn.MSELoss = nn.MSELoss()
    model: Model = Model(in_features=config['Z_MAX'] + 3, nhead=config['nhead'], d_model=config['d_model'], num_layers=config['num_layers'])
    api = wandb.Api()
    artifact = api.artifact(f'mlip-basic-qm9/mlip-basic-qm9:{config['artifact_version']}', type='model')
    artifact_dir = artifact.download()
    print('artifact_dir: ', artifact_dir)
    model.load_state_dict(load(path.join(artifact_dir, 'model_checkpoint.pt'), map_location=get_default_device()))
    print(f'Loaded model checkpoint from W&B artifact mlip-basic-qm9:{config["artifact_version"]}')
    
    # perform n_evals evaluations and print statistics
    n_evals: int = config['n_evals']
    time_arr: np.ndarray = np.zeros(n_evals)
    loss_arr: np.ndarray = np.zeros(n_evals)
    j = 0
    for i, test_molecules in enumerate(dataset_split):
        if i < n_evals:
            t1 = time()
            test_dataset = utils.data.DataLoader(MLIPDataset(test_molecules), batch_size=config['batch_size'], shuffle=False, collate_fn=collate_nested,
                                                generator=Generator(device='cpu'), pin_memory=True, num_workers=4)
            test_loss: float = evaluate_model(model, test_dataset, loss_fn)
            t2 = time()
            time_arr[i] = t2 - t1
            loss_arr[i] = test_loss
            print(f'Evaluation {i+1}/{n_evals}: Test loss={test_loss:.4f} (MSE, Ha^2), Time={t2 - t1:.1f} seconds')
            j = i
    print(f'Average test loss over {n_evals} evaluations: {np.mean(loss_arr):.4f} (MSE, Ha^2) ± {np.std(loss_arr):.4f}')
    print(f'Average evaluation time over {n_evals} evaluations: {np.mean(time_arr):.1f} seconds ± {np.std(time_arr):.1f} seconds')
    atexit.register(cleanup_wrapper(time_arr, loss_arr, j))