#!/bin/bash
#SBATCH --job-name=train_doc_gnn
#SBATCH --output=./results/train_doc_gat_%j.out
#SBATCH --error=./results/train_doc_gat_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --exclude=babel-12-9,babel-7-9,babel-13-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zmeng33@emory.edu
# babel-12-9 (L40S) has memory ECC error, babel-7-9 python not respond, babel-13-1 is much slower

# # Load the CUDA module
module load cuda-11.8
# 
# # Activate the Conda environment
# source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path if Conda is installed elsewhere
# conda activate cs584              # Replace with your environment name

source ~/.bashrc
conda activate cs584

# Train GCN model
# python arxiv_main.py --model gcn --epochs 100 --hidden_dim 256
# accelerate launch --config_file accelerate_config/single_gpu.yaml arxiv_main.py --model gcn --epochs 100 --hidden_dim 256 
# CUDA_VISIBLE_DEVICES=0,1 python arxiv_main.py --model gcn --epochs 100 --hidden_dim 256 
# python arxiv_main.py --model gcn --epochs 100 --hidden_dim 256 --batch_size 128 --run_name gcn


# # Train GAT model with specific parameters
# python arxiv_main.py --model gat --epochs 200 --hidden_dim 512 --heads 8 --run_name gat

# # Train GIN model with custom configuration
python arxiv_main.py --model sage --epochs 200 --hidden_dim 384 --num_layers 4 --run_name sage --batch_size 32


