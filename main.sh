wandb login

python3 train.py \
    --data-dir ./cityscapes_small \
    --batch-size 2 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 2 \
    --seed 42 \
    --experiment-id "unet-training-test" \