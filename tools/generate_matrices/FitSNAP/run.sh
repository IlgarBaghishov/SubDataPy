#!/bin/sh
#
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -t 01:00:00
#SBATCH -J qSNAP
#SBATCH -o ll_out
#SBATCH -A che190010

mpirun -np 32 python -u qSNAP.py