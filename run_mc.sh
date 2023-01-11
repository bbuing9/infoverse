DATASET=wino
SEED="3456"
SUB="0.83 0.66 0.5 0.33 0.25 0.17 0.13 0.09"
GPU=0
EPOCHS=7
RATIO=1.0
BATCH_SIZE=8
BACKBONE=roberta_mc_large

# Step1
for seed in $SEED
do
for sub in $SUB 
do
  CUDA_VISIBLE_DEVICES=$GPU python train_tf.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 230111_wino_dpp_coreset.npy --train_type 230111_base_info_coreset --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  CUDA_VISIBLE_DEVICES=$GPU python train_tf.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 230111_wino_large_false_info_dens_dpp_training_dynamics.npy --train_type 230111_base_info_dens_dpp_train_dynamics --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  CUDA_VISIBLE_DEVICES=$GPU python train_tf.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 230111_wino_large_false_info_dens_dpp_training_ens_uncertainty.npy --train_type 230111_base_info_dens_dpp_ens --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  CUDA_VISIBLE_DEVICES=$GPU python train_tf.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 230111_wino_large_false_info_dens_dpp_training_mc_ens_uncertainty.npy --train_type 230111_base_info_dens_dpp_mc_ens --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  CUDA_VISIBLE_DEVICES=$GPU python train_tf.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 230111_wino_large_false_info_dens_dpp_static.npy --train_type 230111_base_info_dens_dpp_static --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
  CUDA_VISIBLE_DEVICES=$GPU python train_tf.py --model_lr 1e-5 --num_sub $sub --grad_accumulation 8 --selected_idx 230111_wino_large_false_info_dens_dpp_pre_train.npy --train_type 230111_base_info_dens_dpp_pre --data_ratio $RATIO --batch_size $BATCH_SIZE --epochs $EPOCHS --dataset $DATASET --seed $seed --backbone $BACKBONE
done
done
