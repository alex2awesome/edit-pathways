import sys
sys.path.append('../')
import util.util_data_access as uda
import gzip
import os
import shutil, re
import pandas as pd
import sqlite3
import s3fs

conn_mapper_dict = {
    'nyt': 'newssniffer-nytimes.db',
    'wp': 'newssniffer-washpo.db',
    'ap': 'ap.db',
    'guardian': 'newssniffer-guardian.db',
    'bbc-1': 'bbc.db',
    'bbc-2': 'newssniffer-bbc.db',
    'reuters': 'reuters.db',
    'cnn': 'cnn.db',
    'cbc': 'cbc.db',
    'fox': 'fox.db',
    'independent': 'newssniffer-independent.db',
    'dailymail': 'dailymail.db',
    'therebel': 'therebel.db',
    'torontostar': 'torontostar.db',
    'torontosun': 'torontosun.db',
    'calgaryherald': 'calgaryherald.db',
    'globemail': 'globemail.db',
    'canadaland': 'canadaland.db',
    'whitehouse': 'whitehouse.db',
    'lapresse': 'lapresse.db',
    'nationalpost': 'nationalpost.db',
    'telegraph': 'telegraph.db',
}

s3_output_dir = 's3://aspangher/edit-pathways/spark_processing_scripts-output/'
pq_pat= r'df_%(news_source)s__start_\d+__end_\d+__num_\d+/'
csv_pat= r'df_%(news_source)s__start_\d+__end_\d+__num_\d+.csv.gz'
get_pq_files = lambda x: list(filter(lambda y: re.search(pq_pat % {'news_source': x}, y), get_fs().ls(s3_output_dir) ))
get_csv_files = lambda x:list(filter(lambda y: re.search(csv_pat % {'news_source': x}, y), get_fs().ls(s3_output_dir) ))
fn_template_csv = 'db_%(news_source)s__start_%(start)s__end_%(end)s__num_%(num_files)s.csv.gz'
fn_template_pq = 'db_%(news_source)s__start_%(start)s__end_%(end)s__num_%(num_files)s'

_fs = None
def get_fs():
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'http://s3.dev.obdc.bcs.bloomberg.com'})
    return _fs


def _download_prefetched_data_pq(news_source):
    import pyarrow.parquet as pq
    fs = get_fs()
    files = get_pq_files(news_source)
    df_list= []
    for f_path in files:
        df_list.append(
            pq.ParquetDataset('s3://' + f_path, filesystem=fs)
                .read_pandas()
                .to_pandas()
        )
    return pd.concat(df_list)


def _download_prefetched_data_csv(news_source):
    fs = get_fs()
    files = get_csv_files(news_source)
    df_list = []
    for f_path in files:
        with fs.open('s3://' + f_path) as f:
            df = pd.read_csv(f, index_col=0)
        df_list.append(df)
    return pd.concat(df_list)


def download_prefetched_data(news_source, format='csv'):
    if format == 'csv':
        return _download_prefetched_data_csv(news_source)
    else:
        return _download_prefetched_data_pq(news_source)


def download_data(conn_name):
    fname = conn_mapper_dict[conn_name]
    if not os.path.exists(fname):
        zipped_fname = '%s.gz' % fname
        remote_fname = os.path.join('edit-pathways', 'dbs', zipped_fname)
        uda.download_file(zipped_fname, remote_fname)

        with gzip.open(zipped_fname, 'rb') as f_in:
            with open(fname, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def upload_file(fname, remote_path):
    uda.upload_file()


def get_files_to_process_df(num_entries, start_idx, prefetched_entry_ids, db_name):
    with sqlite3.connect(conn_mapper_dict[db_name]) as conn:
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


def _upload_files_to_s3_pq(output_sdf, news_source, start, end):
    num_files = get_pq_files(news_source)
    output_fname = fn_template_pq % {
        'newws_source': news_source,
        'start': start,
        'end': end,
        'num_files': num_files
    }
    outfile_s3_path = os.path.join(s3_output_dir, output_fname)
    output_sdf.write.mode("overwrite").parquet(outfile_s3_path)


def _upload_files_to_s3_csv(output_sdf, news_source, start, end):
    fs = get_fs()
    num_files = get_csv_files(news_source)
    output_fname = fn_template_csv % {
        'newws_source': news_source,
        'start': start,
        'end': end,
        'num_files': num_files
    }
    ##
    outfile_s3_path = os.path.join(s3_output_dir, output_fname)
    output_df = output_sdf.toPandas()
    bytes_to_write = output_df.to_csv(None, compression='gzip').encode()
    with fs.open(outfile_s3_path, 'wb') as f:
        f.write(bytes_to_write)


def upload_files_to_s3(output_sdf, output_format, news_source, start, end):
    if output_format == 'pq':
        _upload_files_to_s3_pq(output_sdf, news_source, start, end)
    else:
        _upload_files_to_s3_csv(output_sdf, news_source, start, end)