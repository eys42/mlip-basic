# mlip-basic

A machine-learned interatomic potential trained to predict internal (electronic) energy from molecular geometries of small organic molecules.

## Training data

This model is trained on the [QM9 dataset](https://www.nature.com/articles/sdata201422) of ~134k small, neutral, diamagnetic organic molecules (HCONF) containing at most 9 heavy atoms (CONF) for which optimized geometries and thermochemical properties (among other data) were calculated at the B3LYP/6-31G(2df,p) level of theory.

The data was preprocessed by translating all molecular geometries to the center of mass, and performing an 80/10/10 train/test/validation split. During each training epoch, as well as during validation and testing, a random 3D rotation was applied to each training example in order to encode approximate SO(3) invariance.

## Model architecture

This model predicts $U_0$, the density functional theory (DFT)-computed internal (electronic) energy at 0 K, from an input optimized molecular geometry.

A transformer architecture with scaled dot-product attention was used. A forward pass through the transformer predicts a per-atom energy $\epsilon_i$ for each atom $i$. $\epsilon_i$ is subsequently summed over $i$ to give the predicted energy $U_0$, in Hartree. A mean squared error (MSE) loss function was used to evaluate the model's performance on each forward pass.

## Author

Ethan Song 

Email: [eys42@cornell.edu](mailto:eys42@cornell.edu)