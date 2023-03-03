ICCS 2023: Convolutional Recurrent Autoencoder for Molecular-Continuum Coupling

Piet Jarmatz, Sebastian Lerdo, Philipp Neumann

## 0_Data_Generation

This subdirectory contains a file structure of configurations in order to
generate the MD datasets. In this paper, we use the macro-micro coupling tool (MaMiCo)
to generate datasets based on Couette and Karman-Vortex-Street scenarios for molecular-continuum coupled flow simulations.

A few key simulation parameters are given in the following table.

**TO DO PIET: table of noteworthy MaMiCo parameters**

| Parameter         | Description           | Value   |
| ------------------|:---------------------:| -------:|
| col 3 is          | right-aligned         | $1600   |


### 0_Couette

![alt text][Couette]

### 1_KVS

![alt text][KVS]


[Couette]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/Schematic_Couette_Setup.drawio.svg "Schematic of the Couette setup"

[KVS]: https://github.com/HSU-HPC/MaMiCo_hybrid_ml/blob/master/3_Figures/Schematic_KVS_Setup.drawio.svg "Schematic of the KVS setup"

