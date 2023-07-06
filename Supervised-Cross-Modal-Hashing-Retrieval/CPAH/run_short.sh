#!/bin/bash
#SBATCH -o /home/users/m/mikriukov/projects/cpah/out_gpu_short.log
#SBATCH -J cpah
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_short
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:tesla:1

echo "Loading venv..."
source /home/users/m/mikriukov/venvs/DADH/bin/activate

echo "Loading cuda..."
module load nvidia/cuda/10.1

echo "Running 128..."
python3 main.py train --flag $1 --proc short --bit 128
python3 main.py test --flag $1 --proc short --bit 128

echo "Running 64..."
python3 main.py train --flag $1 --proc short --bit 64
python3 main.py test --flag $1 --proc short --bit 64

echo "Running 32..."
python3 main.py train --flag $1 --proc short --bit 32
python3 main.py test --flag $1 --proc short --bit 32

echo "Running 16..."
python3 main.py train --flag $1 --proc short --bit 16
python3 main.py test --flag $1 --proc short --bit 16

echo "Running 8..."
python3 main.py train --flag $1 --proc short --bit 8
python3 main.py test --flag $1 --proc short --bit 8