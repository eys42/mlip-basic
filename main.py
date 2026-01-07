from molecule import Molecule
from torch import set_default_dtype, float64, load, backends, device
from model import Model
from train import train_model
from import_data import QM9DataImport
from os import path, getcwd

if __name__ == '__main__':
    # configure default datatype and device
    set_default_dtype(float64)
    use_mps: bool = False

    # load dataset
    QM9_dataset: list[Molecule] = QM9DataImport.load_dataset_from_pt(
        path.join(getcwd(), 'QM9_dataset.pt'),
        generate_combined_input_tensor=True)
    Z_MAX: int = 9  # max atomic number in QM9 dataset (F)
    model: Model = Model(in_features=Z_MAX + 3, nhead=4, d_model=64, num_layers=4)
    if path.exists('model_checkpoint.pt'):
        model.load_state_dict(load('model_checkpoint.pt'))
        print('Loaded model checkpoint from model_checkpoint.pt')

    if use_mps and backends.mps.is_available():
        device = device("mps")
    else:
        device = device("cpu")
    model.to(device)
    print('Beginning training:')
    train_model(model, QM9_dataset, batch_size=32, epochs=50, lr=0.001)