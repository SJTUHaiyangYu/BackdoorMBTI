pwd
cd  ../resources/ssba
dataset=$1
flag=$2
image_resolution=1
python dataset_convert_into_images.py --dataset ${dataset} #prepare clean data
if [ "$dataset" == "cifar10" -o "$dataset" == "gtsrb" ];then
  image_resolution=32
elif [ "$dataset" == "tiny"  -o "$dataset" == "celeba" ]; then
image_resolution=64
fi
python train.py \
  --data_dir ../../data/${dataset}_seperate_images/train \
  --output_dir ./data/${dataset}/output \
  --EXP_NAME customized_experiment_name \
  --random_seed 0 \
  --fingerprint_length 100 \
  --image_resolution ${image_resolution} \
  --num_epochs 20 \
  --batch_size 64 \
  --use_residual 0 \
  --use_modulated 0 \
  --fc_layers 0 \
  --fused_conv 0
python embed_fingerprints.py \
  --encoder_path ./data/${dataset}/output/checkpoints/customized_experiment_name_encoder.pth \
  --data_dir ../../data/${dataset}_seperate_images/train \
  --output_dir ./poison_data/${dataset}/train \
  --image_resolution ${image_resolution} \
  --identical_fingerprints \
  --check \
  --decoder_path ./data/${dataset}/output/checkpoints/customized_experiment_name_decoder.pth \
  --batch_size 32 \
  --seed 0 \
  --encode_method bch \
  --secret abcd \
  --use_residual 0 \
  --use_modulated 0 \
  --fused_conv 0 \
  --fc_layers 0 \
  --cuda 0
mkdir -p "./package/${dataset}"
python utils/pack_images.py \
  --path ./poison_data/${dataset}/train  \
  --save_file_path ./package/${dataset}/train.npy
python embed_fingerprints.py \
    --encoder_path ./data/${dataset}/output/checkpoints/customized_experiment_name_encoder.pth \
    --data_dir ../../data/${dataset}_seperate_images/test \
    --output_dir ./poison_data/${dataset}/test \
    --image_resolution ${image_resolution} \
    --identical_fingerprints \
    --check \
    --decoder_path ./data/${dataset}/output/checkpoints/customized_experiment_name_decoder.pth \
    --batch_size 32 \
    --seed 0 \
    --encode_method bch \
    --secret abcd \
    --use_residual 0 \
    --use_modulated 0 \
    --fused_conv 0 \
    --fc_layers 0 \
    --cuda 0
# pack images into npy file
python utils/pack_images.py \
  --path ./poison_data/${dataset}/test \
  --save_file_path ./package/${dataset}/test.npy
