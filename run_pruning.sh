DATASET=sst2
SEED="1234"
#SUB="0.09 0.13 0.17 0.25 0.33 0.5 0.66 0.83"
SUB="0.09"
#BASE="rand, easy, hard, ambig, forget, el2n, ent, dens"
BASE=rand
GPU=2
EPOCHS=10
BATCH_SIZE=16
BACKBONE=roberta_large

for seed in $SEED
do
for sub_ratio in $SUB 
do
  CUDA_VISIBLE_DEVICES=$GPU python train.py --data_ratio $sub_ratio --selected $BASE --train_type 0108_base --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 1 --selected_idx 0108_sst2_large_coreset.npy --train_type 0109_base_coreset --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE

  #CUDA_VISIBLE_DEVICES=$GPU python train.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 0101_wino_large_info_dens_dpp.npy --train_type 0101_base_info_dens_dpp --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 0101_wino_large_info_reduce_dens_dpp.npy --train_type 0101_base_info_reduce_dens_dpp --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 0102_wino_large_new_info_dens_dpp.npy --train_type 0104_base_new_info_dens_dpp --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 0102_wino_large_new_info_reduce_dens_dpp.npy --train_type 0104_base_new_info_reduce_dens_dpp --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
   
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 1 --selected_idx 0105_wino_large_real_info_dens_dpp.npy --train_type 0105_base_info_real_dens_dpp --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  #CUDA_VISIBLE_DEVICES=$GPU python train.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 0105_wino_large_false_info_dens_dpp.npy --train_type 0105_base_info_false_dens_dpp --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
done
done
