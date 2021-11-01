cache_dir=/Users/alex/.cache/torch/transformers/named-models
project_dir=/Users/alex/Projects/usc-research/edit-pathways/modeling/

model_type=roberta
if [[ $model_type == 'gpt2' ]]
then
  pretrained_model="$cache_dir/gpt2-medium"
  frozen_layers="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
else
  pretrained_model="$cache_dir/roberta-base"
  frozen_layers="0 1 2 3 4 5 6 7 8 9"
fi
##

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

python $SCRIPT_DIR/../model_runner.py \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --experiment sentence \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --local \
        --do_eval \
        --train_data_file_s3 "$project_dir/data/sentence-data-small.csv" \
        --notes "Sentence Discriminator" \
        --freeze_transformer \
        --do_regression \
        --use_poisson_regression \
        --loss_weighting .25 .25 .25 .25 \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --num_gpus 0 \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \