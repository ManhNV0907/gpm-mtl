#!/bin/bash  -ex   

#SBATCH --job-name=standard-train

#SBATCH --output=/lustre/scratch/client/vinai/users/hoangpv7/implement/MOO-SAM/multi-mnist/logs/%A.out 

#SBATCH --error=/lustre/scratch/client/vinai/users/hoangpv7/implement/MOO-SAM/multi-mnist/logs/%A.err

#SBATCH --partition=applied

#SBATCH --gpus=3

#SBATCH --nodes=1

#SBATCH --mem-per-gpu=100G

#SBATCH --cpus-per-gpu=40

#SBATCH --mail-type=all

#SBATCH --mail-user=v.HoangPV7@vinai.io

source /sw/software/miniconda3/bin/activate

conda activate /lustre/scratch/client/vinai/users/hoangpv7/ml

cd /lustre/scratch/client/vinai/users/hoangpv7/implement/MOO-SAM/multi-mnist


CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 1 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 1 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 1 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 2 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 2 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 2 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 5 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 5 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=0 python standard-train.py --dset multi_mnist \
                --rho 5 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 1 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 1 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 1 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 2 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 2 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 2 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 5 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 5 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=1 python standard-train.py --dset multi_fashion \
                --rho 5 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 1 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 1 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 1 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 2 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 2 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 2 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 5 \
                --adaptive True \
                --seed 0 \
                --n_epochs 200 &

CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 5 \
                --adaptive True \
                --seed 1 \
                --n_epochs 200 &


CUDA_VISIBLE_DEVICES=2 python standard-train.py --dset multi_fashion_and_mnist \
                --rho 5 \
                --adaptive True \
                --seed 2 \
                --n_epochs 200 &
wait
