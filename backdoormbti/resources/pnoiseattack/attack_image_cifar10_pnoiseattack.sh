#!/bin/bash

DATA_TYPE="image"
DATASET="cifar10"
ATTACK_NAME="pnoiseattack"
MODEL="resnet18"
PRATIO=0.2
NUM_WORKERS=4

python ../scripts/atk_train.py --data_type $DATA_TYPE --dataset $DATASET --attack_name $ATTACK_NAME --model $MODEL --pratio $PRATIO --num_workers $NUM_WORKERS

# 实验运行设备:Linux dell-PowerEdge-R750 6.5.0-28-generic #29~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Apr  4 14:39:20 UTC 2 x86_64 x86_64 x86_64 GNU/Linux

# 显存消耗情况
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA RTX A5000               Off |   00000000:CA:00.0 Off |                  Off |
# | 30%   48C    P2             85W /  230W |    9657MiB /  24564MiB |     14%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+
                                                                                         
# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |    0   N/A  N/A      1626      G   /usr/lib/xorg/Xorg                              4MiB |
# |    0   N/A  N/A    733089      C   python                                        510MiB |
# +-----------------------------------------------------------------------------------------+

# 数据集：cifar10

# 实验结果
# Epoch 299: 100%|██████████████████████████████████████████████████████████████| 782/782 [00:23<00:00, 33.44it/s, v_num=1, train_loss_step=0.0015, train_loss_epoch=0.00313]
# INFO:attack:training finished
# INFO:attack:loading test data
# Files already downloaded and verified
# INFO:attack:test data loaded
# INFO:attack:start testing
# LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
# Testing DataLoader 1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 141/141 [00:01<00:00, 126.10it/s]
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃        Test metric        ┃       DataLoader 0        ┃       DataLoader 1        ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
# │         test_asr          │                           │            1.0            │
# │      test_clean_acc       │    0.8700000047683716     │                           │
# │          test_ra          │                           │            0.0            │
# └───────────────────────────┴───────────────────────────┴───────────────────────────┘
# INFO:attack:test finished
# INFO:attack:results.json save in: /home/hst/BackdoorMMT/scripts/../logs/image-pnoiseattack-cifar10-resnet18-pratio-0.2--normal/results.json