DATASET=qnli
SEED="1234"
GPU=2
EPOCHS=10
RATIO=1.0
BATCH_SIZE=16
BACKBONE=roberta_large

# Step1
for seed in $SEED
do
  CUDA_VISIBLE_DEVICES=$GPU python train.py --train_type 0611_base_noisy --noisy_label_criteria random --noisy_label_ratio 0.2 --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
done
