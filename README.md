ICCS 2023: Convolutional Recurrent Autoencoder for Molecular-Continuum Coupling

Piet Jarmatz, Sebastian Lerdo, Philipp Neumann

# Abstract


Molecular-continuum coupled flow simulations are used in many
applications to build a bridge across spatial or temporal scales.
Hence, they allow to investigate effects beyond flow scenarios
modeled by any single-scale method alone, such as a discrete particle
system or a partial differential equation solver. On the particle
side of the coupling, often molecular dynamics (MD) is used to obtain
trajectories based on pairwise molecule interaction potentials. However,
since MD is computationally expensive and macroscopic flow quantities
sampled from MD systems often highly fluctuate due to thermal noise,
the applicability of molecular-continuum methods is limited. If
machine learning (ML) methods can learn and predict MD based flow
data, then this can be used as a noise filter or even to replace MD
computations – both generates potential for tremendous speed-up of
molecular-continuum simulations, aiming to enable new applications
emerging on the horizon.

In this paper, we develop an advanced hybrid ML model for MD data in
the context of coupled molecular-continuum flow simulations. Our model
is based on recent know-how from computer vision and speech recognition,
applied to a computational fluid dynamics context: A convolutional
autoencoder (ConvAE) deals with the spatial extent of the flow data, while a
recurrent neural network (RNN) is used to capture its temporal correlation.
We use the open source coupling tool MaMiCo to generate MD datasets for ML
training and implement the hybrid model as a PyTorch-based filtering
module for MaMiCo. 

The ML models are trained with real MD data from
different flow scenarios including a Couette flow validation setup
and a three-dimensional vortex street test case. Our results show that
the convolutional recurrent hybrid model is able to learn and predict
smooth flow quantities, even for very noisy MD input data. We furthermore
demonstrate that also the more complex vortex street continuum flow
data can accurately be reproduced by the ML module, without access
to any corresponding continuum flow information.

# Overview

## 0_Data_Generation

## 1_Single_Model_Couette

![alt text][ConvRecAE_single]

## 2_Triple_Model_KVS

![alt text][ConvRecAE_triple]

## X_Graveyard


[ConvRecAE_single]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/ConvRecAE_single.drawio.svg "Convolutional  recurrent autoencoder as employed in the single model approach"


[ConvRecAE_triple]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/ConvRecAE.drawio.svg "Convolutional recurrent autoencoder as employed in the triple model approach"


[ConvAE]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/ConvAe.drawio.svg "Convolutional autoencoder as employed in the triple model approach"


[RNN]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/LatentspaceRNN.drawio.svg "Recurrent neural network as employed in the triple model approach"


[molcont]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/ConvRecAE.drawio.svg "Convolutional recurrent autoencoder as employed in the triple model approach"
