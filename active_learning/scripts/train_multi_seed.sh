### change these variables if needed
DATA_DIR=data
TASK_NAME=dbpedia
MODEL_TYPE=bert
MODEL_NAME=/nlp/users/yekyung.kim/git/LINDA/LINDA/pretrained_models/bert-base-uncased #bert-base-uncased
OUTPUT=models/$SEED/$TASK_NAME/base
SEEDS='123 124 125 126 127'
### end

for SEED in $SEEDS
do
  echo -e "\n\nSTARTING SEED $SEED \n\n"
  python -m src.train \
      --model_type $MODEL_TYPE \
      --model_name_or_path $MODEL_NAME \
      --task_name $TASK_NAME \
      --do_train \
      --do_test \
      --per_gpu_train_batch_size 32 \
      --per_gpu_eval_batch_size 32 \
      --data_dir $DATA_DIR/$TASK_NAME \
      --max_seq_length 128 \
      --learning_rate 2e-5 \
      --num_train_epochs 5 \
      --output_dir $OUTPUT \
      --seed $SEED \
      --base_model $MODEL_NAME

done

