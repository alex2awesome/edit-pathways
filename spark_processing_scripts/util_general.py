import sys
sys.path.append('../')
import util.util_data_access as uda
import gzip
import os
import shutil, re
import pandas as pd
import sqlite3
import s3fs
from tqdm.auto import tqdm

conn_mapper_dict = {
    'nyt': 'newssniffer-nytimes',
    'wp': 'newssniffer-washpo',
    'ap': 'ap',
    'guardian': 'newssniffer-guardian',
    'bbc-1': 'bbc',
    'bbc-2': 'newssniffer-bbc',
    'reuters': 'reuters',
    'cnn': 'cnn',
    'cbc': 'cbc',
    'fox': 'fox',
    'independent': 'newssniffer-independent',
    'dailymail': 'dailymail',
    'therebel': 'therebel',
    'torontostar': 'torontostar',
    'torontosun': 'torontosun',
    'calgaryherald': 'calgaryherald',
    'globemail': 'globemail',
    'canadaland': 'canadaland',
    'whitehouse': 'whitehouse',
    'lapresse': 'lapresse',
    'nationalpost': 'nationalpost',
    'telegraph': 'telegraph',
}

s3_db_dir = 's3://aspangher/edit-pathways/dbs'
s3_csv_dir = 's3://aspangher/edit-pathways/csvs'
s3_pq_dir = 's3://aspangher/edit-pathways/pqs'
s3_output_dir_main = 's3://aspangher/edit-pathways/spark_processing_scripts-output'
s3_output_dir_sentences = 's3://aspangher/edit-pathways/spark_processing_scripts-output_sentences'
pq_pat= r'df_%(news_source)s__start_\d+__end_\d+__num_\d+/'
csv_pat= r'df_%(news_source)s__start_\d+__end_\d+__num_\d+.csv.gz'
get_pq_files = lambda s3_path, news_source: list(filter(lambda y:
                                                        re.search(pq_pat % {'news_source': news_source}, y),
                                                        get_fs().ls(os.path.join(s3_path, news_source))
                                                        ))

get_csv_files = lambda s3_path, news_source: list(filter(lambda y:
                                                    re.search(csv_pat % {'news_source': news_source}, y),
                                                    get_fs().ls(os.path.join(s3_path, news_source))
                                      ))
fn_template_csv = '%(news_source)s/df_%(news_source)s__start_%(start)s__end_%(end)s__num_%(num_files)s.csv.gz'
fn_template_pkl = '%(news_source)s/df_%(news_source)s__start_%(start)s__end_%(end)s__num_%(num_files)s.pkl'
fn_template_pq = '%(news_source)s/df_%(news_source)s__start_%(start)s__end_%(end)s__num_%(num_files)s'

_fs = None
def get_fs():
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'http://s3.dev.obdc.bcs.bloomberg.com'})
    return _fs


def _download_prefetched_data_pq(news_source, split_sentences):
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    import pyarrow.parquet as pq
    fs = get_fs()
    files = get_pq_files(news_source)
    df_list = []
    for f_path in files:
        df_list.append(
            pq.ParquetDataset('s3://' + f_path, filesystem=fs)
                .read_pandas()
                .to_pandas()
        )
    if len(df_list) > 0:
        return pd.concat(df_list)
    else:
        return


def _download_prefetched_data_csv(news_source, split_sentences, show_progress):
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    fs = get_fs()
    files = get_csv_files(s3_path, news_source)
    df_list = []
    f_iter = files if not show_progress else tqdm(files)
    for f_path in f_iter:
        with fs.open('s3://' + f_path) as f:
            df = pd.read_csv(f, index_col=0)
        df_list.append(df)
    if len(df_list) > 0:
        return pd.concat(df_list)
    else:
        return


def download_prefetched_data(news_source, split_sentences=False, format='csv', show_progress=False):
    if format == 'csv':
        return _download_prefetched_data_csv(news_source, split_sentences, show_progress)
    else:
        return _download_prefetched_data_pq(news_source, split_sentences)


def download_pq_to_df(conn_name):
    fname = conn_mapper_dict[conn_name]
    fpath = os.path.join(s3_pq_dir, fname + '.pq')
    with get_fs().open(fpath) as f:
        return pd.read_parquet(f)


def download_csv_to_df(conn_name):
    fname = conn_mapper_dict[conn_name]
    fpath = os.path.join(s3_csv_dir, fname + '.csv.gz')
    with get_fs().open(fpath) as f:
        return pd.read_csv(f, index_col=0, compression='gzip')


def download_sqlite_db(conn_name):
    import tempfile
    fname = conn_mapper_dict[conn_name]
    remote_db_path = os.path.join(s3_db_dir, fname+'.db')
    fs = get_fs()

    with tempfile.NamedTemporaryFile() as fp:
        fs.download(remote_db_path, fp.name)
        return fp.name


def get_rows_to_process_df(num_entries, start_idx, prefetched_df, full_df):
    if prefetched_df is not None:
        prefetched_entry_ids = prefetched_df['entry_id'].drop_duplicates().values
    else:
        prefetched_entry_ids = []

    return (
        full_df
            .loc[lambda df: df['num_versions'] > 1]
            .loc[lambda df: df['num_versions'] < 40]
            .loc[lambda df: df['entry_id'].isin(
                df['entry_id']
                    .drop_duplicates()
                    .loc[lambda s: ~s.isin(prefetched_entry_ids)]
                    .sort_values()
                    .iloc[start_idx : start_idx + num_entries]
            )]
            .assign(summary=lambda df: df['summary'].str.replace('</p><p>', ' '))
    )


def get_rows_to_process_sql(num_entries, start_idx, prefetched_entry_ids, db_fp):
    with sqlite3.connect(db_fp) as conn:
        df = pd.read_sql('''
             SELECT * from entryversion 
             WHERE entry_id IN (
                SELECT distinct entry_id 
                    FROM entryversion
                    WHERE entry_id NOT IN (%(prefetched_ids)s)
                    AND num_versions > 1 
                    AND num_versions < 40
                    ORDER BY entry_id 
                    LIMIT %(num_entries)d OFFSET %(start_idx)d
                )
         ''' % {
            'num_entries': num_entries,
            'start_idx': start_idx,
            'prefetched_ids': ', '.join(list(map(str, prefetched_entry_ids))),
        }, con=conn)
        df = df.assign(summary=lambda df: df['summary'].str.replace('</p><p>', ' '))
        return df


def _upload_files_to_s3_pq(output_sdf, news_source, start, num_records_per_file, split_sentences):
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    num_prefetched_files = len(get_pq_files(s3_path, news_source))
    output_fname = fn_template_pq % {
        'news_source': news_source,
        'start': (start + num_prefetched_files) * num_records_per_file,
        'end': (start + num_prefetched_files + 1) * num_records_per_file,
        'num_files': num_prefetched_files + 1
    }
    outfile_s3_path = os.path.join(s3_path, output_fname)
    output_sdf.write.mode("overwrite").parquet(outfile_s3_path)


def _upload_files_to_s3_pkl(output_sdf, news_source, start, num_records_per_file, split_sentences):
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    num_prefetched_files = len(get_csv_files(s3_path, news_source))
    output_fname = fn_template_pkl % {
        'news_source': news_source,
        'start': (start + num_prefetched_files) * num_records_per_file,
        'end': (start + num_prefetched_files + 1) * num_records_per_file,
        'num_files': num_prefetched_files + 1
    }
    ##
    outfile_s3_path = os.path.join(s3_path, output_fname)
    print('OUTFILE: %s' % outfile_s3_path)
    output_df = output_sdf.toPandas()
    with get_fs().open(outfile_s3_path, 'wb') as f:
        output_df.to_pickle(output_df)


def _upload_files_to_s3_csv(output_sdf, news_source, start, num_records_per_file, split_sentences):
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    num_prefetched_files = len(get_csv_files(s3_path, news_source))
    output_fname = fn_template_csv % {
        'news_source': news_source,
        'start': (start + num_prefetched_files) * num_records_per_file,
        'end': (start + num_prefetched_files + 1) * num_records_per_file,
        'num_files': num_prefetched_files + 1
    }
    ##
    outfile_s3_path = os.path.join(s3_path, output_fname)
    print('OUTFILE: %s' % outfile_s3_path)
    output_df = output_sdf.toPandas()
    bytes_to_write = output_df.to_csv(None, compression='gzip').encode()
    with get_fs().open(outfile_s3_path, 'wb') as f:
        f.write(bytes_to_write)


def upload_files_to_s3(output_sdf, output_format, news_source, start, end, split_sentences):
    if output_format == 'pq':
        _upload_files_to_s3_pq(output_sdf, news_source, start, end, split_sentences)
    elif output_format == 'csv':
        _upload_files_to_s3_csv(output_sdf, news_source, start, end, split_sentences)
    elif output_format == 'pkl':
        _upload_files_to_s3_pkl(output_sdf, news_source, start, end, split_sentences)




















############################################


def dump_db_to_s3():
    """Doesn't work..."""
    conn = sqlite3.connect('newssniffer-nytimes.db')
    dump_lines = '\n'.join(list(conn.iterdump())).encode()
    with get_fs().open('s3://aspangher/edit-pathways/db-dumps/newssniffer-nytimes-sqlite-dump.txt', 'wb') as f:
        f.write(dump_lines)

def dump_csv_to_s3():
    with sqlite3.connect('newssniffer-nytimes.db') as conn:
        df = pd.read_sql('''
             SELECT * from entryversion
         ''', con=conn)
    df.to_csv('newssniffer-nytimes.csv.gz', compression='gzip')
    util_data_access.upload_file('newssniffer-nytimes.csv.gz', 'edit-pathways/csvs/newssniffer-nytimes.csv.gz')

def load_sqlite_to_spark():
    from pyspark.sql import SQLContext
    sqlContext = SQLContext(spark)
    sdf = (
        sqlContext.read.format('jdbc')
            #       .options(url='jdbc:sqlite:s3://aspangher/edit-pathways/dbs/newssniffer-nytimes.db', dbtable='entryversion', driver='org.sqlite.JDBC')
            .options(url='jdbc:sqlite:%s' % db_local_path, dbtable='entryversion', driver='org.sqlite.JDBC')
            .load()
    )