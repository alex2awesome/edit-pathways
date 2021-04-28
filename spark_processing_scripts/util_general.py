import sys
sys.path.append('../')
import util.util_data_access as uda
import gzip
import os
import shutil
import pandas as pd
import sqlite3

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

def download_data(conn_name):
    fname = conn_mapper_dict[conn_name]
    if not os.path.exists(fname):
        zipped_fname = '%s.gz' % fname
        remote_fname = os.path.join('edit-pathways', 'dbs', zipped_fname)
        uda.download_file(zipped_fname, remote_fname)

        with gzip.open(zipped_fname, 'rb') as f_in:
            with open(fname, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def read_spark_df(num_entries, start_idx, db_name):
    with sqlite3.connect(conn_mapper_dict[db_name]) as conn:
        df = pd.read_sql('''
             SELECT * from entryversion 
             WHERE entry_id IN (
                SELECT distinct entry_id 
                    FROM entryversion ORDER BY entry_id LIMIT %(num_entries)d OFFSET %(start_idx)d
                )
         ''' % {'num_entries': num_entries, 'start_idx': start_idx}, con=conn)
        df = df.assign(summary=lambda df: df['summary'].str.replace('</p><p>', ' '))
        return df