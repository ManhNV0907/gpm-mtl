# baseline
# for seed in 0 1 2 3 4; do
#     python standard-train.py --dset multi_mnist --seed $seed --rho 0 &
#     python standard-train.py --dset multi_fashion_and_mnist --seed $seed --rho 0 &
#     python standard-train.py --dset multi_fashion --seed $seed --rho 0
# done
# for seed in 4 0
# do
#     for rho in 0 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1 2
#     do
#         CUDA_VISIBLE_DEVICES=0  python standard-train-single-task-lib.py --dset multi_mnist \
#                                                 --rho $rho \
#                                                 --adaptive False \
#                                                 --seed $seed \
#                                                 --n_epochs 200 &

#         CUDA_VISIBLE_DEVICES=1  python standard-train-single-task-lib.py --dset multi_mnist \
#                                                 --rho $rho \
#                                                 --adaptive True \
#                                                 --seed $seed \
#                                                 --n_epochs 200 
#     done
# done


# for seed in 0 2 3 4
    
# do 
#     for rho in 2 
#     do
#         CUDA_VISIBLE_DEVICES=0  python standard-train-single-task.py --dset multi_mnist \
#                                                 --rho 2 \
#                                                 --adaptive True \
#                                                 --seed $seed \
#                                                 --n_epochs 200  &
#         CUDA_VISIBLE_DEVICES=1  python standard-train-single-task-lib.py --dset multi_mnist \
#                                                 --rho 2 \
#                                                 --adaptive True \
#                                                 --seed $seed \
#                                                 --n_epochs 200 
#     done
# done


