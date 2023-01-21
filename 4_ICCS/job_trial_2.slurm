#!/bin/bash
#SBATCH --job-name=trial_2_RNN # specifies a user-defined job name

#SBATCH --nodes=1             # number of compute nodes to be used
#SBATCH --ntasks=1           # number of MPI processes
#SBATCH --partition=small_gpu #
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1    # number of cores per process
#SBATCH --time=03:00:00
#SBATCH --output=/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/trial_2_RNN_%j.log
# commands to be executed

cd /beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/4_ICCS/

python trial_2.py
