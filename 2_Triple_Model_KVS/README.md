ICCS 2023: Convolutional Recurrent Autoencoder for Molecular-Continuum Coupling

Piet Jarmatz, Sebastian Lerdo, Philipp Neumann

## 2_Triple_Model_KVS

This subdirectory contains the triple model approach as used for the KVS based datasets.

### data_preprocessing.py

This script allows to perform the required data preprocessing including data
cleaning and data visualization. Data cleaning is necessary since the initial
MaMiCo output files use faulty delimiters. Visualization of the raw data aims
to validate correctness/plausibility so as to reduce the likelihood of faulty
simulation data.

### model.py

This script contains all the custom PyTorch models used for the triple model
approach. In particular:

- `DoubleConv`

- `AE_u_i`

![alt text][ConvAE_triple]

- `RNN`

![alt text][RNN]

- `Hybrid_MD_RNN_AE_u_i`

![alt text][ConvRecAE_triple]

### dataset.py

This script contains all the custom PyTorch Dataset implementations used for
the triple model approach.

### utils.py

This script contains necessary auxillary functions from loading from file to
getting the model specific DataLoader. In particular:

- `get_AE_loaders`

- `get_RNN_loaders`

- `get_Hybrid_loaders`

### plotting.py

This script contains all plotting functionalities used for the triple model
approach. In particular:

- `plot_flow_profile`

- `plot_flow_profile_std`

### trial_1.py

This script focuses on training the convolutional autoencoder as used in the
triple model approach. Afterwards, the latentspaces are extracted from the
trained model.

### trial_2.py

This script focuses on training the recurrent neural networks on the basis of the
extracted latent spaces. Afterwards, the hybrid model is validated.

[ConvAE_triple]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/ConvAe_triple.drawio.svg "Convolutional autoencoder as employed in the triple model approach"

[RNN]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/LatentspaceRNN.drawio.svg "Recurrent neural network as employed in the triple model approach"

[ConvRecAE_triple]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/ConvRecAE_triple.drawio.svg "Convolutional recurrent autoencoder as employed in the triple model approach"


