# Model Batch Generation 

```shell

CUDA_VISIBLE_DEVICES=3 python image_benign_train_random_params.py --data_type audio --dataset speechcommands --attack_name blend --model_name audiocnn --pratio 0.1 --num_workers 4 --epochs 100

```