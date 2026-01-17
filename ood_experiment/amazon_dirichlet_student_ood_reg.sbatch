#!/bin/bash -l
#SBATCH --job-name=amazon_dirichlet_student_ood_reg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --account=plgittossl-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --mem=92G
#SBATCH --output=amazon_dirichlet_student_ood_reg-%j.out
#SBATCH --error=amazon_dirichlet_student_ood_reg-%j.err

source .env/bin/activate
python amazon_dirichlet_student_ood.py --mode learnable
