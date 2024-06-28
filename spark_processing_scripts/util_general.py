import sys
sys.path.append('../')
import util.util_data_access as uda
import gzip
import os, glob
import shutil, re
import pandas as pd
import sqlite3
import s3fs
from tqdm.auto import tqdm
import unidecode


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
pkl_pat= r'df_%(news_source)s__start_\d+__end_\d+__num_\d+.pkl'
get_pq_files = lambda s3_path, news_source: list(filter(lambda y:
                                                        re.search(pq_pat % {'news_source': news_source}, y),
                                                        get_fs().ls(os.path.join(s3_path, news_source))
                                                        ))

get_files = lambda s3_path, news_source, file_pat: list(filter(lambda y:
                                                    re.search(file_pat % {'news_source': news_source}, y),
                                                    get_fs().ls(os.path.join(s3_path, news_source))
                                      ))
fn_template_csv = '%(news_source)s/df_%(news_source)s__start_%(start)s__end_%(end)s__num_%(num_files)s.csv.gz'
fn_template_pkl = '%(news_source)s/df_%(news_source)s__start_%(start)s__end_%(end)s__num_%(num_files)s.pkl'
fn_template_pq = '%(news_source)s/df_%(news_source)s__start_%(start)s__end_%(end)s__num_%(num_files)s'

pluslab_output_dir = 'data_output'
get_pluslab_output_dir = lambda db_name: os.path.join(pluslab_output_dir, db_name)

_fs = None
def get_fs():
    global _fs
    if _fs is None:
        _fs = s3fs.S3FileSystem(
            default_fill_cache=False,
            client_kwargs={'endpoint_url': 'http://s3.dev.obdc.bcs.bloomberg.com'}
        )
    return _fs


def _download_prefetched_data_pq(news_source):
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


def _download_prefetched_data_csv(f_iter, fs, entry_id_list):
    for f_path in f_iter:
        with fs.open('s3://' + f_path) as f:
            df = pd.read_csv(f, index_col=0)
            if 'entry_id' in df:
                entry_ids = df['entry_id'].drop_duplicates()
            else:
                entry_ids = pd.Series()
        entry_id_list.append(entry_ids)
    return entry_id_list


def _download_prefetched_data_pkl(f_iter, fs, entry_id_list):
    for f_path in f_iter:
        with fs.open('s3://' + f_path) as f:
            df = pd.read_pickle(f, compression='gzip')
            if 'entry_id' in df:
                entry_ids = df['entry_id'].drop_duplicates()
            else:
                entry_ids = pd.Series()
        entry_id_list.append(entry_ids)
    return entry_id_list


def download_prefetched_data(news_source, split_sentences=False, format='csv', show_progress=False):
    # prep
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    fs = get_fs()
    # check if news source is even in the s3_path
    if len(list(filter(lambda x: x.endswith(news_source), fs.ls(s3_path)))) == 0:
        return
    #
    pat = csv_pat if (format == 'csv') and (not split_sentences) else pkl_pat
    files = get_files(s3_path, news_source, pat)
    entry_id_list = []
    f_iter = files if not show_progress else tqdm(files)

    # fetch data
    if split_sentences:
        entry_id_list = _download_prefetched_data_pkl(f_iter, fs, entry_id_list)
    else:
        if format == 'csv':
            entry_id_list = _download_prefetched_data_csv(f_iter, fs, entry_id_list)
        else:
            entry_id_list = _download_prefetched_data_pq(f_iter, fs, entry_id_list)

    # return
    if len(entry_id_list) > 0:
        return pd.concat(entry_id_list).drop_duplicates()
    else:
        return



def read_prefetched_data(news_source, split_sentences=False, format='csv', show_progress=False):
    """
    For running on Nanyun's pluslab

    :param news_source:
    :param split_sentences:
    :param format:
    :param show_progress:
    :return:
    """
    if format == 'csv':
        data_dir = get_pluslab_output_dir(news_source)
        prefetched_data = []
        for file in glob.glob(os.path.join(data_dir, '*')):
            df = pd.read_csv(file)
            prefetched_data.append(df['entry_id'].drop_duplicates())
        if len(prefetched_data) > 0:
            return pd.concat(prefetched_data)
        else:
            return None


def download_pq_to_df(conn_name, prefetched_entry_ids, prefetched_file_idx=0, show_progress=False):
    prefetched_entry_id_list = prefetched_entry_ids.values if (prefetched_entry_ids is not None) else []
    fname = conn_mapper_dict[conn_name]
    file_list = get_fs().ls(s3_pq_dir)
    file_pattern = re.compile(r'%s-\d+.pq' % fname)
    full_file_list = list(enumerate(filter(lambda x: re.search(file_pattern, x), file_list)))
    file_list = full_file_list[prefetched_file_idx:]
    # visualize
    if show_progress:
        file_list = tqdm(file_list)
    # iterate through any files there might be
    full_dfs = []
    for f_idx, fname in file_list:
        with get_fs().open(fname) as f:
            full_df = pd.read_parquet(f)
        full_df = full_df.loc[lambda df: ~df['entry_id'].isin(prefetched_entry_id_list)]
        if len(full_df['entry_id'].drop_duplicates()) > 5:
            last_one = f_idx == (len(file_list) - 1)
            return f_idx + 1, last_one, full_df
        else:
            print('FOUND UNFECTCHED IDS: %s' % str(full_df['entry_id'].drop_duplicates().values.tolist()))
            full_dfs.append(full_df)
    # if we don't find a data file with unfetched entry_ids > 5
    last_one = prefetched_file_idx >= (len(full_file_list) - 1)
    full_df = pd.concat(full_dfs) if len(full_dfs) > 0 else []
    return prefetched_file_idx + 1, last_one, full_df


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


def get_rows_to_process_df(num_entries, start_idx, full_df, prefetched_entry_ids=None):
    if prefetched_entry_ids is not None:
        prefetched_entry_ids = prefetched_entry_ids.drop_duplicates().values
    else:
        prefetched_entry_ids = []

    if len(full_df) == 0:
        return [], []

    output_df = (
        full_df
            .loc[lambda df: df['num_versions'] > 1]
            .loc[lambda df: df['num_versions'] < 40]
    )
    print('len(output_df): %s' % len(output_df['entry_id'].drop_duplicates()))
    to_get_df = (
        output_df
            .loc[lambda df: df['entry_id'].isin(
                df['entry_id']
                    .drop_duplicates()
                    .loc[lambda s: ~s.isin(prefetched_entry_ids)]
                    .sort_values()
                    .iloc[start_idx : start_idx + num_entries]
            )]
            .assign(summary=lambda df: df['summary'].str.replace('</p><p>', ' '))
            .assign(summary=lambda df: df['summary'].apply(unidecode.unidecode))
    )
    print('len(to_get_df): %s' % len(to_get_df['entry_id'].drop_duplicates()))
    left_to_process_df = output_df.loc[lambda df: ~df['entry_id'].isin(to_get_df['entry_id'])]
    print('len(left_to_process_df): %s' % len(left_to_process_df['entry_id'].drop_duplicates()))
    return to_get_df, left_to_process_df


def get_rows_to_process_sql(db_name, num_entries=None, start_idx=None, prefetched_entry_ids=[]):
    db_fp = conn_mapper_dict[db_name] + '.db'
    if prefetched_entry_ids is None:
        prefetched_entry_ids = []

    sql = '''
             SELECT * from entryversion 
             WHERE entry_id IN (
                SELECT distinct entry_id 
                    FROM entryversion
                    WHERE entry_id NOT IN (%(prefetched_ids)s)
                    AND num_versions > 1 
                    AND num_versions < 40
                    ORDER BY entry_id 
                )
         ''' % {'prefetched_ids': ', '.join(list(map(str, prefetched_entry_ids)))}

    if num_entries is not None and start_idx is not None:
        sql += 'LIMIT %(num_entries)d OFFSET %(start_idx)d' % {
            'num_entries': num_entries,
            'start_idx': start_idx,
        }

    with sqlite3.connect(db_fp) as conn:
        df = pd.read_sql(sql, con=conn)
        if len(df['entry_id'].drop_duplicates()) < 10:
            return
        df = df.assign(summary=lambda df: df['summary'].str.replace('</p><p>', ' '))
        return df


def _upload_files_to_s3_pq(output_sdf, news_source, start, num_records_per_file, split_sentences, file_count=0):
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    if file_count == 0:
        file_count = len(get_pq_files(s3_path, news_source))
    output_fname = fn_template_pq % {
        'news_source': news_source,
        'start': (start + file_count) * num_records_per_file,
        'end': (start + file_count + 1) * num_records_per_file,
        'num_files': file_count + 1
    }
    outfile_s3_path = os.path.join(s3_path, output_fname)
    output_sdf.write.mode("overwrite").parquet(outfile_s3_path)
    return file_count

def _upload_files_to_s3_pkl(output_df, news_source, start, num_records_per_file, split_sentences, file_count=0):
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    if file_count == 0:
        file_count = len(get_files(s3_path, news_source, csv_pat))
    output_fname = fn_template_pkl % {
        'news_source': news_source,
        'start': (start + file_count) * num_records_per_file,
        'end': (start + file_count + 1) * num_records_per_file,
        'num_files': file_count + 1
    }
    ##
    outfile_s3_path = os.path.join(s3_path, output_fname)
    print('OUTFILE: %s' % outfile_s3_path)
    with get_fs().open(outfile_s3_path, 'wb') as f:
        output_df.to_pickle(f, compression='gzip')
    return file_count

def _upload_files_to_s3_csv(output_df, news_source, start, num_records_per_file, split_sentences, file_count=0):
    s3_path = s3_output_dir_main if not split_sentences else s3_output_dir_sentences
    if file_count == 0:
        file_count = len(get_files(s3_path, news_source, csv_pat))
    output_fname = fn_template_csv % {
        'news_source': news_source,
        'start': (start + file_count) * num_records_per_file,
        'end': (start + file_count + 1) * num_records_per_file,
        'num_files': file_count + 1
    }
    ##
    outfile_s3_path = os.path.join(s3_path, output_fname)
    print('OUTFILE: %s' % outfile_s3_path)
    bytes_to_write = output_df.to_csv(None, compression='gzip').encode()
    with get_fs().open(outfile_s3_path, 'wb') as f:
        f.write(bytes_to_write)
    return file_count


def upload_files_to_s3(output_df, output_format, news_source, start, end, split_sentences, file_count):
    if output_format == 'pq':
        return _upload_files_to_s3_pq(output_df, news_source, start, end, split_sentences, file_count)
    elif output_format == 'csv':
        return _upload_files_to_s3_csv(output_df, news_source, start, end, split_sentences, file_count)
    elif output_format == 'pkl':
        return _upload_files_to_s3_pkl(output_df, news_source, start, end, split_sentences, file_count)


def dump_files_locally(output_df, output_format, news_source, start, end, split_sentences, file_count):
    if output_format == 'csv':
        out_dir = get_pluslab_output_dir(news_source)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if file_count == 0:
            file_count = len(os.listdir(out_dir))

        output_fname = fn_template_csv % {
            'news_source': news_source,
            'start': (start + file_count) * end,
            'end': (start + file_count + 1) * end,
            'num_files': file_count + 1
        }
        out_path = os.path.join(pluslab_output_dir, output_fname)
        output_df.to_csv(out_path, compression='gzip')
        return file_count





















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