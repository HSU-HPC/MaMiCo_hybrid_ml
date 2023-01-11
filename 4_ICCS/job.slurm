#!/bin/bash
#SBATCH --job-name=data_preprocessing # specifies a user-defined job name

#SBATCH --nodes=1             # number of compute nodes to be used
#SBATCH --ntasks=1           # number of MPI processes
#SBATCH --partition=small #
#SBATCH --cpus-per-task=1    # number of cores per process
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/data_preprocessing_%j.log
# commands to be executed

cd /beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/

python3 data_preprocessing.py
