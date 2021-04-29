from pyspark.sql import SparkSession
import spark_processing_scripts.util_spark as sus
import spark_processing_scripts.util_general as sug


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--db_name', type=str, help='Name of the source DB to use.')
    parser.add_argument('--start', type=int, default=0, help='Start entry_id index.')
    parser.add_argument('--num_files', type=int, default=500, help='Number of entry_ids to use.')
    parser.add_argument('--input_format', type=str, default='csv', help="Input format for the previously-stored runs.")
    parser.add_argument('--output_format', type=str, default='csv', help="Output format.")

    args = parser.parse_args()

    print('downloading source data %s...' % args.db_name)
    sug.download_data(args.db_name)

    print('downloading prefetched data...')
    prefetched_df = sug.download_prefetched_data(args.db_name)

    # spark
    spark = (
        SparkSession.builder
            .config("spark.executor.instances", "30")
            .config("spark.driver.memory", "20g")
            .config("spark.executor.memory", "20g")
            .config("spark.sql.shuffle.partitions", "2000")
            .config("spark.executor.cores", "5")
            .config("spark.kryoserializer.buffer.max", "2000M")
            .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.5")
            .getOrCreate()
    )

    # read dataframe
    df = sug.get_files_to_process_df(
        args.num_files, args.start, prefetched_df['entry_id'].drop_duplicates(), args.db_name
    )

    # process via spark_processing_scripts
    print('running spark...')
    output_sdf = sus.run_spark(df, spark)

    sug.upload_files_to_s3(output_sdf, args.output_format, args.db_name, args.start, args.start + args.num_files)



if __name__ == "__main__":
    main()