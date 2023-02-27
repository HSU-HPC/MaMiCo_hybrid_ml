#!/bin/bash
#SBATCH --job-name=analysis_4_%j # specifies a user-defined job name

#SBATCH --nodes=1             # number of compute nodes to be used
#SBATCH --ntasks=1           # number of MPI processes
#SBATCH --partition=small_gpu #
#SBATCH --gpus=2
#SBATCH --cpus-per-task=1    # number of cores per process
#SBATCH --time=12:00:00
#SBATCH --output=/beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/3_Constituent_Hybrid_approach/analysis_4_%j.log
# commands to be executed

cd /beegfs/project/MaMiCo/mamico-ml/ICCS/MD_U-Net/3_Constituent_Hybrid_approach/
python analysis_4.py
