from pyspark.sql import SparkSession
import spark_processing_scripts.pipeline_steps as sps
import spark_processing_scripts.util_general as sug
import spark_processing_scripts.util_spark as sus
import pyspark.sql.functions as F
import os

def run_spark(df, spark):
    sdf = spark.createDataFrame(df)
    sdf = sdf.repartition('entry_id', 'version').cache()

    # Process the input data to split sentences, tokenize and get BERT embeddings
    spark_processed_df = sps.get_sparknlp_pipeline().fit(sdf).transform(sdf)
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
        .approxSimilarityJoin(sim_sdf, sim_sdf, sus.APPROX_JOIN_CUTOFF, distCol="distance")
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
    word_pair_matched_sdf = word_pair_matched_sdf.repartition(key_cols).cache()

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

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--db_name', type=str, help='Name of the source DB to use.')
    parser.add_argument('--start', type=int, help='Start entry_id index.')
    parser.add_argument('--num_files', type=int, help='Number of entry_ids to use.')
    args = parser.parse_args()

    print('downloading data %s...' % args.db_name)
    sug.download_data(args.db_name)

    # spark_processing_scripts
    spark = (
        SparkSession.builder
          .config("spark.executor.instances", "30")
          .config("spark.driver.memory", "20g")
          .config("spark.executor.memory", "20g")
          .config("spark.sql.shuffle.partitions", "2000")
          .config("spark.executor.cores", "5")
          .config("spark.kryoserializer.buffer.max", "2000M")
          .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark_processing_scripts-nlp_2.11:2.7.5")
          .getOrCreate()
    )

    # read dataframe
    df = sug.read_spark_df(args.num_files, args.start, args.db_name)

    # process via spark_processing_scripts
    print('running spark...')
    output_sdf = run_spark(df, spark)

    # to disk
    output_fname = 'db_%s__start_%s__end_%s' % (args.db_name, args.start, args.start + args.num_files)
    outfile_s3_path = os.path.join('s3://aspangher', 'edit-pathways', 'spark_processing_scripts-output', output_fname)
    output_sdf.write.mode("overwrite").parquet(outfile_s3_path)


if __name__ == "__main__":
    main()