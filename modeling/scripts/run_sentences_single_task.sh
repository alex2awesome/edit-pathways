DEFAULT_REPO='git+https://bbgithub.dev.bloomberg.com/aspangher/edit-project.git'
DEFAULT_BRANCH='master'
DEFAULT_PACKAGE=$DEFAULT_REPO@$DEFAULT_BRANCH

DEFAULT_JOB_SIZE='Custom'
#DEFAULT_FRAMEWORK='pytorch-1.6-python-3.7'
DEFAULT_FRAMEWORK='python-3.7-rhel-cuda-10.2'
DEFAULT_GIT_IDENTIY='spectro-oauth-aspangher'
DEFAULT_HADOOP_IDENTITY='aspangher-cluster-test'
DEFAULT_BCS_IDENTITY='aspangher-cluster-test'
ENV=bb
## gpus
num_nodes=1
num_gpus_per_node=1
if [[ $num_nodes -gt 1 ]]
then
  APPROACH='distributed-pytorch'
  worker_args="--node-num-gpus $num_gpus_per_node --num-workers $num_nodes --node-num-cores 4 --node-memory 80Gi"
else
  APPROACH='single'
  worker_args="--node-num-gpus $num_gpus_per_node --node-num-cores 4 --node-memory 80Gi"
fi

model_type=roberta
if [[ $model_type == 'gpt2' ]]
then
  pretrained_model='gpt2-medium-expanded-embeddings'
  frozen_layers="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22"
else
  pretrained_model='roberta-base'
  frozen_layers="0 1 2 3 4"
fi
##

katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module modeling.runner \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --namespace s-ai-classification \
        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/edit-pathways/tensorboard \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --experiment sentence \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --do_eval \
        --num_dataloader_cpus 0 \
        --train_data_file_s3 training-data/training_data_short_15__sampled_50000.csv \
        --notes "Sentence Operations, Sentence Level, Docs < 15,> 5, downsampled 50,000" \
        --freeze_encoder_layers $frozen_layers \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --use_positional \
        --use_doc_emb \
        --doc_embed_arithmetic \
        --loss_weighting .25 .25 .25 .25 \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \
        --do_operations \


katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module modeling.runner \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --namespace s-ai-classification \
        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/edit-pathways/tensorboard \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --experiment sentence \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --do_eval \
        --num_dataloader_cpus 0 \
        --train_data_file_s3 training-data/training_data_short_15__sampled_50000.csv \
        --notes "Addition, Classification, Sentence Level, Docs < 15,> 5, downsampled 50,000" \
        --freeze_encoder_layers $frozen_layers \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --use_positional \
        --use_doc_emb \
        --doc_embed_arithmetic \
        --loss_weighting .25 .25 .25 .25 \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \
        --do_addition \


katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module modeling.runner \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --namespace s-ai-classification \
        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/edit-pathways/tensorboard \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --experiment sentence \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --do_eval \
        --num_dataloader_cpus 0 \
        --train_data_file_s3 training-data/training_data_short_15__sampled_50000.csv \
        --notes "Addition, Normal Regression, Sentence Level, Docs < 15,> 5, downsampled 50,000" \
        --freeze_encoder_layers $frozen_layers \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --use_positional \
        --use_doc_emb \
        --doc_embed_arithmetic \
        --loss_weighting .25 .25 .25 .25 \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \
        --do_addition \
        --do_regression \
        --use_contextual_layers

katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module modeling.runner \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --namespace s-ai-classification \
        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/edit-pathways/tensorboard \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --experiment sentence \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --do_eval \
        --num_dataloader_cpus 0 \
        --train_data_file_s3 training-data/training_data_short_15__sampled_50000.csv \
        --notes "Addition, Poisson Regression, Sentence Level, Docs < 15,> 5, downsampled 50,000" \
        --freeze_encoder_layers $frozen_layers \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --use_positional \
        --use_doc_emb \
        --doc_embed_arithmetic \
        --loss_weighting .25 .25 .25 .25 \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \
        --do_addition \
        --do_regression \
        --use_poisson_regression \
        --use_contextual_layers \


katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module modeling.runner \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --namespace s-ai-classification \
        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/edit-pathways/tensorboard \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --experiment sentence \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --do_eval \
        --num_dataloader_cpus 0 \
        --train_data_file_s3 training-data/training_data_short_15__sampled_50000.csv \
        --notes "Refactoring Classification, Sentence Level, Docs < 15,> 5, downsampled 50,000" \
        --freeze_encoder_layers $frozen_layers \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --use_positional \
        --use_doc_emb \
        --doc_embed_arithmetic \
        --loss_weighting .25 .25 .25 .25 \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \
        --do_refactor \
        --use_contextual_layers \

katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module modeling.runner \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --namespace s-ai-classification \
        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/edit-pathways/tensorboard \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --experiment sentence \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --do_eval \
        --num_dataloader_cpus 0 \
        --train_data_file_s3 training-data/training_data_short_15__sampled_50000.csv \
        --notes "Refactoring, Normal Regression, Sentence Level, Docs < 15,> 5, downsampled 50,000" \
        --freeze_encoder_layers $frozen_layers \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --use_positional \
        --use_doc_emb \
        --doc_embed_arithmetic \
        --loss_weighting .25 .25 .25 .25 \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \
        --do_refactor \
        --do_regression \
        --use_contextual_layers \


katie compute run \
        $APPROACH \
        --compute-framework $DEFAULT_FRAMEWORK \
        --node-size $DEFAULT_JOB_SIZE \
        $worker_args \
        --python-module modeling.runner \
        --identities hadoop=$DEFAULT_HADOOP_IDENTITY bcs=$DEFAULT_BCS_IDENTITY git=$DEFAULT_GIT_IDENTIY \
        --pip-packages $DEFAULT_PACKAGE \
        --namespace s-ai-classification \
        --tensorboard-log-dir hdfs:///projects/ai_classification/aspangher/edit-pathways/tensorboard \
        --env NCCL_ASYNC_ERROR_HANDLING=1 NCCL_LL_THRESHOLD=0 NCCL_DEBUG=INFO env=$ENV \
        -- \
        --model_type $model_type \
        --pretrained_files_s3 $pretrained_model \
        --experiment sentence \
        --batch_size 1 \
        --num_train_epochs 3 \
        --do_train \
        --do_eval \
        --num_dataloader_cpus 0 \
        --train_data_file_s3 training-data/training_data_short_15__sampled_50000.csv \
        --notes "Refactoring, Poisson Regression, Sentence Level, Docs < 15,> 5, downsampled 50,000" \
        --freeze_encoder_layers $frozen_layers \
        --dropout .1 \
        --accumulate_grad_batches 1 \
        --learning_rate 1e-4 \
        --warmup_steps 0 \
        --use_positional \
        --use_doc_emb \
        --doc_embed_arithmetic \
        --loss_weighting .25 .25 .25 .25 \
        --num_contextual_layers 2 \
        --num_sent_attn_heads 2 \
        --do_refactor \
        --do_regression \
        --use_poisson_regression \
        --use_contextual_layers \

