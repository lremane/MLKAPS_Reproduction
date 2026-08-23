#!/bin/bash
#SBATCH --account=paj2615
#SBATCH --partition=dc-cpu-devel
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH --job-name=mlkaps_pdgeqrf
#SBATCH --output=mlkaps_pdgeqrf_%j.out
#SBATCH --error=mlkaps_pdgeqrf_%j.err

# Load modules
module purge
module load Stages/2026 GCC OpenMPI
module load ScaLAPACK/2.2.2-fb
module load Python/3.13.5

# Activate the MLKAPS virtual environment
source /p/scratch/weatherai/remane1/MLKAPS/venv2026/bin/activate

# Run MLKAPS
mlkaps examples/Scalapack-PDGEQRF/config.json