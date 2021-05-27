import spark_processing_scripts.util_spark as sus
import spark_processing_scripts.util_general as sug
from pyspark.sql import SparkSession, SQLContext
import pandas as pd

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
    parser.add_argument('--env', type=str, default='bb', help="Whether we're running on Bloomberg or somewhere else.")

    args = parser.parse_args()

    # see what data we already have
    print('downloading prefetched data...')
    if not args.split_sentences:
        if args.env == 'bb':
            prefetched_entry_ids = sug.download_prefetched_data(args.db_name, split_sentences=args.split_sentences)
        else:
            prefetched_entry_ids = sug.read_prefetched_data(args.db_name, split_sentences=args.split_sentences)
    else:
        prefetched_entry_ids = None

    print('downloading source data %s...' % args.db_name)
    if args.env == 'bb':
        to_fetch_df = sug.download_pq_to_df(args.db_name, prefetched_entry_ids)
    else:
        to_fetch_df = sug.get_rows_to_process_sql(args.db_name, prefetched_entry_ids=prefetched_entry_ids)
    if to_fetch_df is None:
        print('Done!!!')
        return

    if args.env == 'bb':
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

    else:
        import findspark
        findspark.init()
        spark = (
            SparkSession.builder
                .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.7.5")
                # .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp-gpu_2.12:3.0.0")
                .config("spark.executor.instances", "40")
                .config("spark.driver.memory", "20g")
                .config("spark.executor.memory", "20g")
                .config("spark.sql.shuffle.partitions", "2000")
                .config("spark.executor.cores", "5")
                .config("spark.kryoserializer.buffer.max", "2000M")
                .config('spark.driver.maxResultSize', '5g')
                .getOrCreate()
        )

    sqlContext = SQLContext(spark)

    pipelines = sus.get_pipelines(sentence=args.split_sentences, env=args.env)
    num_tries = 3
    file_count = -1

    # loop spark job
    while (len(to_fetch_df) > 0) and (to_fetch_df is not None):
        # keep an internal counter so we don't have to keep hitting S3 to count output files
        file_count += 1

        # read dataframe
        df, to_fetch_df = sug.get_rows_to_process_df(
            args.num_files, args.start, prefetched_entry_ids, to_fetch_df
        )
        print('FETCHING IDs: %s' % ', '.join(list(map(str, df['entry_id'].drop_duplicates().tolist()))))
        print('LEN(DF): %s' % str(len(df)))
        print('START: %s' % args.start)

        # process via spark_processing_scripts
        print('running spark...')
        if args.split_sentences:
            output_sdf = sus.run_spark_sentences(df, spark, *pipelines)
        else:
            output_sdf = sus.run_spark(df, spark, *pipelines)

        output_df = output_sdf.toPandas()
        if len(output_df) == 0:
            if num_tries > 0:
                print('ZERO-LEN DF, retrying...')
                num_tries -= 1
                continue
            else:
                if len(to_fetch_df['entry_id'].drop_duplicates()) < 5:
                    if args.env == 'bb':
                        to_fetch_df = sug.download_pq_to_df(args.db_name, prefetched_entry_ids)
                    else:
                        to_fetch_df = sug.get_rows_to_process_sql(
                            args.db_name, prefetched_entry_ids=prefetched_entry_ids
                        )
                    continue
                else:
                    print('ZERO-LEN DF, TOO MANY RETRIES, breaking....')
                    break

        print('VALID DATA, UPLOADING...')
        ## cache prefetched_df, instead of pulling it each time.
        if prefetched_entry_ids is not None:
            prefetched_entry_ids = pd.concat([
                prefetched_entry_ids,
                output_df['entry_id'].drop_duplicates()
            ])
        else:
            prefetched_entry_ids = output_df['entry_id'].drop_duplicates()

        ### upload data
        if args.env == 'bb':
            file_count = sug.upload_files_to_s3(
                output_df, args.output_format,
                args.db_name, args.start, args.start + args.num_files,
                args.split_sentences,
                file_count=file_count
            )
        else:
            file_count = sug.dump_files_locally(
                output_df,
                args.output_format,
                args.db_name,
                args.start, args.start + args.num_files,
                args.split_sentences,
                file_count=file_count
            )

        if len(to_fetch_df['entry_id'].drop_duplicates()) < 5:
            if args.env == 'bb':
                to_fetch_df = sug.download_pq_to_df(args.db_name, prefetched_entry_ids)
            else:
                to_fetch_df = sug.get_rows_to_process_sql(args.db_name, prefetched_entry_ids=prefetched_entry_ids)

        # clean up
        if args.continuous:
            sqlContext.clearCache()
        ##
        if not args.continuous:
            break

if __name__ == "__main__":
    main()