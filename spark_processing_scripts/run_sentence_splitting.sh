# cluster ds-ob-dev02
HADOOP_IDENTITY="aspangher-cluster-test"
BCS_IDENTITY="aspangher-bcs-test"
GIT_IDENTITY="spectro-oauth-aspangher"
NAMESPACE="aspangher"

# 4 for BB authored news, 6 for web-scraped content
#for PVF_LEVEL in 4 6; do
katie spark run \
    --namespace ${NAMESPACE} \
    --identities ${HADOOP_IDENTITY} ${BCS_IDENTITY} ${GIT_IDENTITY} \
    --job-name sentence-parse \
    --python-mode \
    --spark-framework spark-2.4-python-3.7 \
    --python-package-uris git+https://bbgithub.dev.bloomberg.com/aspangher/edit-project \
    --py-uri /job/.local/bin/run_pyspark.py \
    --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.5 \
    --cluster-size Custom \
    --worker-count 50 \
    --worker-cores 5 \
    --worker-memory 20G \
    --driver-memory 20G \
    --driver-cores 5 \
    --sync-launch tail \
    -- \
      --db_name guardian \
      --num_files 20000 \
      --continuous \
      --split_sentences \
      --output_format pkl \
