from molecule import Molecule
import torch
from model import Model
from train import train_model
from import_data import QM9DataImport
import os

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    QM9_dataset: list[Molecule] = QM9DataImport.load_dataset_from_pt(
        os.path.join(os.getcwd(), 'QM9_dataset.pt'),
        generate_combined_input_tensor=True)
    Z_MAX: int = 9  # max atomic number in QM9 dataset (F)
    model: Model = Model(in_features=Z_MAX + 3, nhead=4, d_model=64, num_layers=4)
    if os.path.exists('model_checkpoint.pt'):
        model.load_state_dict(torch.load('model_checkpoint.pt'))
        print('Loaded model checkpoint from model_checkpoint.pt')
    print('Beginning training:')
    train_model(model, QM9_dataset, batch_size=32, epochs=50, lr=0.001)