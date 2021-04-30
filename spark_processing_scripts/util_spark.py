import spark_processing_scripts.pipeline_steps as sps
import pyspark.sql.functions as F


SENTENCE_SIM_THRESH = .44
APPROX_JOIN_CUTOFF = .5


def run_spark(df, spark):
    sdf = spark.createDataFrame(df)
    sdf = sdf.repartition('entry_id', 'version').cache()

    # Process the input data to split sentences, tokenize and get BERT embeddings
    sparknlp_pipeline = sps.get_sparknlp_pipeline()
    spark_processed_df = sparknlp_pipeline.fit(sdf).transform(sdf)
    spark_processed_df = spark_processed_df.cache()

    # Explode the sentences
    exploded_sdf = sps.get_explode_pipeline().transform(spark_processed_df)
    exploded_sdf = exploded_sdf.cache()

    # Hash the BERT embeddings for the Approximate Join
    similarity_model = sps.get_similarity_pipeline().fit(exploded_sdf)
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

    top_sentence_pipeline_x, top_sentence_pipeline_y = sps.get_sentence_pipelines()

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



def get_word_matching_sql(side):
    """Generate the SQL necessary to transform each side. Side \in {'x', 'y'}"""

    word_pair_min_distance_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                word_idx_%(side)s,
                MIN(num_words) as num_words_total_list,
                MIN(distance) as min_word_distance
        FROM __THIS__ 
        GROUP BY entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                word_idx_%(side)s
      """ % ({'side': side})

    sentence_pair_min_distance_sql = """
        SELECT entry_id,
               version_x,
               version_y,
               sent_idx_x,
               sent_idx_y,
               (sum_min_word_distance + %(approx_join_cutoff)f * ( num_words_total - num_matched_words )) / num_words_total AS avg_sentence_distance
        FROM (
           SELECT entry_id,
                  version_x,
                  version_y,
                  sent_idx_x,
                  sent_idx_y,
                  SUM(min_word_distance) AS sum_min_word_distance,
                  COUNT(1) AS num_matched_words,
                  MIN(num_words_total_list) AS num_words_total
           FROM __THIS__
                GROUP BY entry_id,
                   version_x,
                   version_y,
                   sent_idx_x,
                   sent_idx_y
          )
      """ % ({'approx_join_cutoff': APPROX_JOIN_CUTOFF})

    sentence_min_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                avg_sentence_distance
           FROM (
                    SELECT *, ROW_NUMBER() OVER (
                         PARTITION BY entry_id, 
                                      version_x, 
                                      version_y, 
                                      sent_idx_%(side)s
                         ORDER BY avg_sentence_distance ASC
                ) AS rn FROM __THIS__
        )
         where rn = 1
    """ % ({'side': side})

    threshold_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_%(join_side)s,
                CASE 
                    WHEN (avg_sentence_distance < %(sentence_sim)f ) THEN sent_idx_%(other_side)s
                    ELSE NULL
                END AS sent_idx_%(other_side)s,
                CASE 
                    WHEN (avg_sentence_distance < %(sentence_sim)f ) THEN avg_sentence_distance
                    ELSE NULL
                END AS avg_sentence_distance
            FROM __THIS__
    """ % ({
        'join_side': side,
        'other_side': list({'x', 'y'} - set(side))[0],
        'sentence_sim': SENTENCE_SIM_THRESH
    })

    return word_pair_min_distance_sql, sentence_pair_min_distance_sql, sentence_min_sql, threshold_sql