for dset in "multi_fashion" "multi_fashion_and_mnist" "multi_mnist"; do
    for seed in 0 1 2; do


        python train.py --dset $dset \
            --rho 0 \
            --adaptive True \
            --method pcgrad \
            --seed $seed \
            --n_epochs 200
        python train.py --dset $dset \
            --rho 0 \
            --adaptive True \
            --method a-pcgrad \
            --seed $seed \
            --n_epochs 200


    done
done

