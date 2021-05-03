from pyspark.sql import SparkSession
import spark_processing_scripts.util_spark as sus
import spark_processing_scripts.util_general as sug
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row, SparkSession, SQLContext

def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--db_name', type=str, help='Name of the source DB to use.')
    parser.add_argument('--start', type=int, default=0, help='Start entry_id index.')
    parser.add_argument('--num_files', type=int, default=500, help='Number of entry_ids to use.')
    parser.add_argument('--input_format', type=str, default='csv', help="Input format for the previously-stored runs.")
    parser.add_argument('--output_format', type=str, default='csv', help="Output format.")
    parser.add_argument('--continuous', action='store_true', help='Whether to keep iterating or not...')
    parser.add_argument('--split_sentences', action='store_true', help="Whether to just perform sentence-splitting.")

    args = parser.parse_args()

    spark = (
        SparkSession.builder
            .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.5")
            .config("spark.executor.instances", "40")
            .config("spark.driver.memory", "20g")
            .config("spark.executor.memory", "20g")
            .config("spark.sql.shuffle.partitions", "2000")
            .config("spark.executor.cores", "5")
            .config("spark.kryoserializer.buffer.max", "2000M")
            .getOrCreate()
    )

    sqlContext = SQLContext(spark)

    print('downloading source data %s...' % args.db_name)
    full_db = sug.download_pq_to_df(args.db_name)

    df = full_db
    pipelines = sus.get_pipelines(sentence=args.split_sentences)
    while len(df) > 0:
        print('downloading prefetched data...')
        prefetched_df = sug.download_prefetched_data(args.db_name, split_sentences=args.split_sentences)

        # read dataframe
        df = sug.get_rows_to_process_df(
            args.num_files, args.start, prefetched_df['entry_id'].drop_duplicates(), full_db
        )

        # process via spark_processing_scripts
        print('running spark...')
        if args.split_sentences:
            output_sdf = sus.run_spark_sentences(df, spark, *pipelines)
        else:
            output_sdf = sus.run_spark(df, spark, *pipelines)

        sug.upload_files_to_s3(
            output_sdf, args.output_format,
            args.db_name, args.start, args.start + args.num_files,
            args.split_sentences
        )
        #
        if args.continuous:
            sqlContext.clearCache()
        ##
        if not args.continuous:
            break

if __name__ == "__main__":
    main()