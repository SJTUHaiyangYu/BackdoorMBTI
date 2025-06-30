cd ../resources/sbat
python train.py --data_type image --dataset gtsrb  --attack_name sbat --model_name resnet18 --pratio 0.1 --num_workers 4 --epochs 50