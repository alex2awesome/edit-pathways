import os, argparse
import sqlite3
import pandas as pd
from util import util_newssniffer_parsing as unp
import warnings
warnings.filterwarnings("ignore")

data_path = '../data/diffengine-diffs/db'
if not os.path.exists(data_path):
    data_path = '../data'

output_path = "output"

conn_mapper_dict = {
    'nyt': os.path.join(data_path, 'newssniffer-nytimes.db'),
    'wp': os.path.join(data_path, 'newssniffer-washpo.db'),
    'ap': os.path.join(data_path, 'ap.db'),
    'guardian': os.path.join(data_path, 'newssniffer-guardian.db'),
    'bbc-1': os.path.join(data_path, 'bbc.db'),
    'bbc-2': os.path.join(data_path, 'newssniffer-bbc.db'),
    'reuters': os.path.join(data_path, 'reuters.db'),
    'cnn': os.path.join(data_path, 'cnn.db'),
    'cbc': os.path.join(data_path, 'cbc.db'),
    'fox': os.path.join(data_path, 'fox.db'),
    'independent': os.path.join(data_path, 'newssniffer-independent.db'),
    'dailymail': os.path.join(data_path, 'dailymail.db'),
    'therebel': os.path.join(data_path, 'therebel.db'),
    'torontostar': os.path.join(data_path, 'torontostar.db'),
    'torontosun': os.path.join(data_path, 'torontosun.db'),
    'calgaryherald': os.path.join(data_path, 'calgaryherald.db'),
    'globemail': os.path.join(data_path, 'globemail.db'),
    'canadaland': os.path.join(data_path, 'canadaland.db'),
    'whitehouse': os.path.join(data_path, 'whitehouse.db'),
    'lapresse': os.path.join(data_path, 'lapresse.db'),
    'nationalpost': os.path.join(data_path, 'nationalpost.db'),
    'telegraph': os.path.join(data_path, 'telegraph.db'),
}

s_client = None
def get_storage_client():
    global s_client
    pass

_ds_client = None
def get_ds_client(refresh=False):
    global _ds_client
    if refresh:
        _ds_client = None
    if _ds_client is None:
        from google.cloud import datastore
        try:
            # running on google cloud
            _ds_client = datastore.Client()
        except:
            # running locally
            import os
            if os.uname().sysname == 'Darwin':
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/alex/.google-cloud/usc-research-data-access.json'
            else:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/alexa/google-cloud/usc-research-c087445cf499.json'
            _ds_client = datastore.Client()
    return _ds_client


def put_entity(ent_type, key_name, to_exclude, data):
    from google.api_core.exceptions import ServiceUnavailable
    num_tries = 5
    ds_client = get_ds_client()
    for idx in range(num_tries):
        try:
            key = ds_client.key(ent_type, key_name)
            e = datastore.Entity(key=key, exclude_from_indexes=to_exclude)
            e.update(data)
            ds_client.put(e)
            break
        except ServiceUnavailable:
            ds_client = get_ds_client(refresh=True)
            print('service unavailable, retrying %s...' % idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_db_name", type=str, default=None)
    parser.add_argument("--n_file", type=int, default=None)
    parser.add_argument("--output_different_db", action="store_true")
    parser.add_argument("--num_version_cutoff", type=int, default=40)
    parser.add_argument('--add_to_datastore', action='store_true')
    parser.add_argument('--add_to_sqlite', action='store_true')
    parser.add_argument('--n_splits', type=int, default=1)
    parser.add_argument('--split_num', type=int, default=0)
    args = parser.parse_args()

    if args.add_to_datastore == True:
        from google.cloud import datastore

    # download dataset
    print('downloading dataset...')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    db_file_path = conn_mapper_dict[args.source_db_name]
    if not os.path.exists(db_file_path):
        from google.cloud import storage
        from google.cloud.storage import Blob
        client = storage.Client()
        bucket = client.get_bucket("usc-data")
        db_name = os.path.basename(db_file_path)
        blob = Blob(os.path.join("edit-pathways", db_name), bucket)
        with open(db_file_path, "wb") as file_obj:
            client.download_blob_to_file(blob, file_obj)

    # get connection
    conn = sqlite3.connect(db_file_path)

    # see if table exists
    sql_tables = pd.read_sql("""
        SELECT 
            name
        FROM 
            sqlite_master 
        WHERE 
            type ='table' AND 
            name NOT LIKE 'sqlite_%';
    """, con=conn)

    ## don't duplicate work
    print('fetching duplicates...')
    to_add = ""
    if ('sentence_stats' in sql_tables.name.values):
        to_add = " AND entry_id not in (SELECT DISTINCT a_id from sentence_stats)"
    if args.add_to_datastore:
        ds_client = get_ds_client()
        q = ds_client.query(kind='edit-paths-sentence-stats').add_filter('source', '=', args.source_db_name)
        r = list(q.fetch())
        a_ids = list(set(map(lambda x: str(x['a_id']), r)))
        to_add = " AND entry_id not in (%s)" % ', '.join(a_ids)

    ## don't select empty documents
    to_add += " AND summary != ''"

    # read data
    print('reading data...')
    select_sql = "SELECT * from entryversion"
    # put a limit on it
    if args.n_file is not None:
        limit_line = """
        WHERE entry_id IN (
            SELECT DISTINCT entry_id from entryversion 
            WHERE num_versions < %s
            and num_versions > 1
            %s
            ORDER BY RANDOM() 
            LIMIT %s
        )""" % (args.num_version_cutoff, to_add, args.n_file)
    else:
        limit_line = " WHERE num_versions < %s AND num_versions > 1 %s" % (args.num_version_cutoff, to_add)
        if args.n_splits != 1:
            import copy
            version_lim = copy.copy(limit_line)
            n_items = pd.read_sql("""select count(DISTINCT entry_id) from entryversion %s""" % version_lim, con=conn).iloc[0][0]
            n_per_split = int(n_items / args.n_splits)
            limit_line += " AND entry_id in (SELECT DISTINCT entry_id from entryversion %s ORDER BY entry_id LIMIT %s OFFSET %s)" % (
                version_lim, n_per_split, args.split_num * n_per_split
            )
    #
    select_sql += '\n' + limit_line
    sample_diffs = pd.read_sql(select_sql, con=conn)

    # last-minute cleanup
    sample_diffs = (sample_diffs
        .loc[lambda df: df['summary'].str.strip() != ''] # make sure even the stripped text isn't null.
        .loc[lambda df: df['entry_id'].isin(df['entry_id'].value_counts().loc[lambda s: s > 1].index)] # make sure only one version.
    )


    ##
    # get sentence stats df
    data_processor_iter = unp.get_sentence_diff_stats(
        sample_diffs,
        get_sentence_vars=True,
        output_type='iter'
    )

    if args.output_different_db == True:
        dir_path, f_name = os.path.dirname(db_file_path), os.path.basename(db_file_path)
        conn = sqlite3.connect('%s/outputs-%s' % (dir_path, f_name))

    for sentence_stats_df, words_stats_df in data_processor_iter:
        ### handle error case
        if (sentence_stats_df is None) and ('error' in words_stats_df.get('status')):
            key_name ='%s-%s-%s-%s' % (args.source_db_name, words_stats_df['a_id'], words_stats_df['version_old'], words_stats_df['version_new'])
            put_entity('edit-paths-sentence-stats', key_name, [], words_stats_df)
            continue

        # sentence stats
        output_sentence_stats_df = (sentence_stats_df
             .assign(version_old=lambda df: df['version_nums'].str.get(0))
             .assign(version_new=lambda df: df['version_nums'].str.get(1))
             .drop(['version_nums', 'vars_old', 'vars_new'], axis=1)
        )
        if args.add_to_sqlite == True:
            output_sentence_stats_df.to_sql('sentence_stats', con=conn, index=False, if_exists='append')

        if args.add_to_datastore:
            ds_client = get_ds_client()
            output_sentence_stats_df.loc[:, 'source'] = args.source_db_name
            #
            for output_dict in output_sentence_stats_df.to_dict(orient='records'):
                key_name = '%s-%s-%s-%s' % (args.source_db_name, output_dict['a_id'], output_dict['version_old'], output_dict['version_new'])
                to_exclude = ['num_added_sents', 'len_new_doc', 'num_removed_sents', 'len_old_doc', 'num_changed_sents']
                put_entity('edit-paths-sentence-stats', key_name, to_exclude, output_dict)

        # sentences
        for vers, a_id, v_old, v_new in sentence_stats_df[['version_nums', 'a_id', 'vars_old', 'vars_new']].itertuples(index=False):
            comb_sent_df = pd.concat([
                    (pd.DataFrame(v_old)
                     .assign(version_old=vers[0])
                     .assign(a_id=a_id)
                     .assign(s_idx=lambda df: df.reset_index()['index'])
                     .rename(columns={'text': 'sent_old', 'tag': 'tag_old'})
                    ), (
                    pd.DataFrame(v_new)
                     .assign(version_new=vers[1])
                     .rename(columns={'text': 'sent_new', 'tag': 'tag_new'})
                    )
                ], axis=1)
            ##
            output_comb_sent_df = comb_sent_df[['a_id', 's_idx', 'sent_old', 'sent_new', 'tag_old', 'tag_new', 'version_old',  'version_new']]
            if args.add_to_sqlite == True:
                output_comb_sent_df.to_sql('sentence_diffs', con=conn, if_exists='append', index=False)
            if args.add_to_datastore:
                output_comb_sent_df.loc[:, 'source'] = args.source_db_name
                #
                for output_dict in output_comb_sent_df.to_dict(orient='records'):
                    key_name = '%s-%s-%s-%s-%s' % (args.source_db_name, output_dict['a_id'], output_dict['version_old'], output_dict['version_new'], output_dict['s_idx'])
                    to_exclude = ['sent_old', 'sent_new', 'tag_old', 'tag_new']
                    put_entity('edit-paths-sentence-diffs', key_name, to_exclude, output_dict)


        if words_stats_df is not None:
            # word stats
            output_word_stats_df = (words_stats_df
                 .assign(version_old=lambda df: df['version_nums'].str.get(0))
                 .assign(version_new=lambda df: df['version_nums'].str.get(1))
                 .drop(['version_nums', 's_old', 's_new'], axis=1)
            )
            if args.add_to_sqlite == True:
                output_word_stats_df.to_sql('word_stats', con=conn, if_exists='append')
            if args.add_to_datastore:
                output_word_stats_df.loc[:, 'source'] = args.source_db_name
                #
                for output_dict in output_word_stats_df.to_dict(orient='records'):
                    key_name = '%s-%s-%s-%s-%s' % (args.source_db_name, output_dict['a_id'], output_dict['version_old'], output_dict['version_new'], output_dict['s_idx'])
                    to_exclude = ['num_removed_words', 'num_added_words', 'len_old_sent', 'len_new_sent']
                    put_entity('edit-paths-word-stats', key_name, to_exclude, output_dict)


            # words
            for vers, a_id, s_idx, v_old, v_new in (
                words_stats_df
                [['version_nums', 'a_id', 's_idx', 's_old', 's_new']]
                .itertuples(index=False)
            ):
                comb_word_df = pd.concat([
                        (pd.DataFrame(v_old)
                         .assign(version_old=vers[0])
                         .assign(a_id=a_id)
                         .assign(s_idx=s_idx)
                         .assign(word_idx=lambda df: df.reset_index()['index'])
                         .rename(columns={'text': 'word_old', 'tag': 'tag_old'})
                        ),
                        (pd.DataFrame(v_new)
                         .assign(version_new=vers[1])
                         .rename(columns={'text': 'word_new', 'tag': 'tag_new'})
                        )
                    ], axis=1)

                output_comb_word_df = comb_word_df[['a_id', 's_idx', 'word_idx', 'word_old', 'word_new', 'tag_old', 'tag_new', 'version_old',  'version_new']]
                if args.add_to_sqlite == True:
                    output_comb_word_df.to_sql('word_diffs', con=conn, if_exists='append', index=False)
                if args.add_to_datastore:
                    output_comb_word_df.loc[:, 'source'] = args.source_db_name
                    for output_dict in output_comb_word_df.to_dict(orient='records'):
                        key_name = '%s-%s-%s-%s-%s-%s' % (args.source_db_name, output_dict['a_id'], output_dict['version_old'], output_dict['version_new'], output_dict['s_idx'], output_dict['word_idx'])
                        to_exclude = ['word_old', 'word_new', 'tag_old', 'tag_new']
                        put_entity('edit-paths-word-diffs', key_name, to_exclude, output_dict)





