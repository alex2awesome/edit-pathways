import spark_processing_scripts.pipeline_steps as sps
import pyspark.sql.functions as F


SENTENCE_SIM_THRESH = .44
APPROX_JOIN_CUTOFF = .5


def get_pipelines(sentence=False, env='bb'):
    if sentence:
        return (sps.get_split_sentence_pipeline(),)
    else:
        top_sentence_pipeline_x, top_sentence_pipeline_y = sps.get_sentence_pipelines()
        return (
            sps.get_sparknlp_pipeline(env=env),
            sps.get_explode_pipeline(),
            sps.get_similarity_pipeline(),
            top_sentence_pipeline_x,
            top_sentence_pipeline_y
        )


def run_spark_sentences(df, spark, sentence_pipeline):
    sdf = spark.createDataFrame(df)
    sdf = sdf.repartition('entry_id', 'version').cache()

    # Process the input data to split sentences, tokenize and get BERT embeddings
    sentence_processed_df = sentence_pipeline.fit(sdf).transform(sdf)
    return sentence_processed_df.select('entry_id', 'version', 'sent_idx', 'sentence')


def run_spark(df, spark, sparknlp_pipeline, explode_pipeline, similarity_pipeline, top_sentence_pipeline_x, top_sentence_pipeline_y):
    sdf = spark.createDataFrame(df)
    sdf = sdf.repartition('entry_id', 'version').cache()

    # Process the input data to split sentences, tokenize and get BERT embeddings
    spark_processed_df = sparknlp_pipeline.fit(sdf).transform(sdf)
    spark_processed_df = spark_processed_df.cache()

    # Explode the sentences
    exploded_sdf = explode_pipeline.transform(spark_processed_df).na.drop(subset='word_embedding')
    exploded_sdf = exploded_sdf.cache()

    # Hash the BERT embeddings for the Approximate Join
    similarity_model = similarity_pipeline.fit(exploded_sdf)
    sim_sdf = similarity_model.transform(exploded_sdf)
    sim_sdf = sim_sdf.cache()

    # Approximate Join
    word_pair_matched_sdf = (
        similarity_model
        .stages[1]
        .approxSimilarityJoin(sim_sdf, sim_sdf, APPROX_JOIN_CUTOFF, distCol="distance")
        .where(
            (F.col("datasetA.entry_id") == F.col("datasetB.entry_id")) &
            (F.col("datasetA.version") + 1 == F.col("datasetB.version"))
        )
        .select(
             F.col("datasetA.entry_id").alias("entry_id"),
             F.col("datasetA.version").alias("version_x"),
             F.col("datasetB.version").alias("version_y"),
             F.col("datasetA.sent_idx").alias("sent_idx_x"),
             F.col("datasetB.sent_idx").alias("sent_idx_y"),
             F.col("datasetA.word_idx").alias("word_idx_x"),
             F.col("datasetB.word_idx").alias("word_idx_y"),
             F.col("datasetA.num_words").alias("num_words"),
             F.col("datasetA.token").alias("token_x"),
             F.col("datasetB.token").alias("token_y"),
             F.col("distance")
        )
    )
    key_cols = ['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'sent_idx_y']
    word_pair_matched_sdf = word_pair_matched_sdf.repartition(*key_cols).cache()


    ## Get bipartite graph from both directions
    sent_pairs_x_sdf = (
        top_sentence_pipeline_x
            .transform(word_pair_matched_sdf)
            .withColumnRenamed('avg_sentence_distance', 'avg_sentence_distance_x')
            .cache()
    )
    sent_pairs_y_sdf = (
        top_sentence_pipeline_y
            .transform(word_pair_matched_sdf)
            .withColumnRenamed('avg_sentence_distance', 'avg_sentence_distance_y')
            .cache()
    )

    ## Join and drop duplicates
    final_sdf = (
         sent_pairs_x_sdf
             .join(sent_pairs_y_sdf, on=key_cols, how='outer')
             .dropDuplicates(key_cols)
    )

    return final_sdf