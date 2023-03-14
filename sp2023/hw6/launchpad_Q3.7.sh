#!/bin/bash

#SBATCH -A cs601_gpu
#SBATCH --partition=mig_class
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name="HW6 CS 601.471/671 homework"
#SBATCH --reservation=MIG
#SBATCH --qos=qos_mig_class


module load anaconda

# init virtual environment if needed
# conda create -n toy_classification_env python=3.7

conda activate toy_classification_env # open the Python environment

# pip install -r requirements.txt # install Python dependencies

# runs your code
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 5  2>&1 > bs64_lr1e-4_ep5.log # --small_subset True
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 7  2>&1 > bs64_lr1e-4_ep7.log # --small_subset True
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-4 --num_epochs 9  2>&1 > bs64_lr1e-4_ep9.log # --small_subset True
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 5e-4 --num_epochs 5  2>&1 > bs64_lr5e-4_ep5.log # --small_subset True
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 5e-4 --num_epochs 7  2>&1 > bs64_lr5e-4_ep7.log # --small_subset True
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 5e-4 --num_epochs 9  2>&1 > bs64_lr5e-4_ep9.log # --small_subset True
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-3 --num_epochs 5  2>&1 > bs64_lr1e-3_ep5.log # --small_subset True
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-3 --num_epochs 7  2>&1 > bs64_lr1e-3_ep7.log # --small_subset True
srun python classification.py  --experiment "overfit"  --device cuda --model "distilbert-base-uncased" --batch_size "64" --lr 1e-3 --num_epochs 9  2>&1 > bs64_lr1e-3_ep9.log # --small_subset True
