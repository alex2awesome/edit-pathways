import pandas as pd
import util_refactorings as ur

def get_split_and_matched_dfs(conn, sents_max=30, sents_min=3):
    low_count_versions = pd.read_sql('''
    with c1 as 
        (SELECT entry_id, 
            CAST(version as INT) as version, 
            COUNT(1) as c from split_sentences 
            GROUP BY entry_id, version)
    SELECT entry_id, version from c1
    WHERE c < %s and c > %s
    '''% (sents_max, sents_min), con=conn)

    # get join keys
    low_count_entry_ids = ', '.join(list(map(str, low_count_versions['entry_id'].unique())))
    joint_keys = low_count_versions.pipe(lambda df: df['entry_id'].astype(str) + '-' + df['version'].astype(str))
    joint_keys = "'%s'" % "', '".join(joint_keys.tolist())

    # matched sentences
    matched_sentences = pd.read_sql('''
        WITH c1 as ( 
        SELECT *, 
        entry_id || '-' || version_x as key_x,
        entry_id || '-' || version_y as key_y 
        FROM matched_sentences 
        )
        SELECT *
        FROM c1
        WHERE key_x in (%s) AND key_y  in (%s)
        ''' % (joint_keys, joint_keys)
    , con=conn)

    # get split sentences
    split_sentences = pd.read_sql('''
        with c1 AS (
            SELECT *, entry_id || '-' || CAST(version AS INT) as key FROM split_sentences
        )
        SELECT entry_id, CAST(version AS INT) as version, sent_idx, sentence 
        FROM c1
        WHERE key IN (%s)
    ''' % joint_keys, con=conn)
    return matched_sentences, split_sentences

def label_sentences_add(doc):
    doc = doc.copy()
    doc = doc.loc[lambda df: ~df[['sent_idx_x', 'sent_idx_y']].isnull().all(axis=1)]
    sent_idxs = doc['sent_idx_y'].dropna().sort_values().tolist()
    additions = doc.loc[lambda df: df['sent_idx_x'].isnull()]['sent_idx_y'].tolist()

    add_aboves = []
    add_belows = []
    idx_in_add_l = 0
    while idx_in_add_l < len(additions):
        a = additions[idx_in_add_l]
        idx_in_sent_l = sent_idxs.index(a)
        cluster_size = 1
        if idx_in_sent_l < len(sent_idxs) - cluster_size:
            add_above = sent_idxs[idx_in_sent_l + cluster_size]
            exists_sent_below = True
            while add_above in additions:
                cluster_size += 1
                if (idx_in_sent_l + cluster_size) < len(sent_idxs):
                    add_above = sent_idxs[idx_in_sent_l + cluster_size]
                    exists_sent_below = True
                else:
                    exists_sent_below = False
                    break
            if exists_sent_below:
                add_aboves.append({
                    'add_above': add_above,
                    'cluster_size': cluster_size
                })
        if idx_in_sent_l > 0:
            add_below = sent_idxs[idx_in_sent_l - 1]
            add_belows.append({
                'add_below': add_below,
                'cluster_size': cluster_size
            })
        idx_in_add_l += cluster_size

    return add_aboves, add_belows


## label each sentence in the old version as:
# 1. Deleted in the new version
# 2. Sentence added above/sentence added below
# 3. Sentence edited
# 4. Sentence refactored

# 5. Sentence split (?)
# 6. Sentence merge (?)

def get_sentence_and_doc_labels(doc, doc_sentences):
    # 1. Detect deletions:
    deleted_labeled_sentences = pd.concat([
        (doc_sentences
            .loc[lambda df: ~df['sent_idx'].astype(int).isin(doc['sent_idx_x'].dropna().astype(int))]
            .assign(deleted_label=True)
            .rename(columns={'version': 'version_x', 'sent_idx': 'sent_idx_x'})
        [['entry_id', 'version_x', 'sent_idx_x', 'deleted_label']]
            )
        ,
        (doc
            .loc[lambda df: df['sent_idx_y'].isnull()]
            .assign(deleted_label=True)
        [['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'deleted_label']]
            )
    ]).drop_duplicates()

    # 2. Sentence additions above/below
    add_aboves, add_belows = label_sentences_add(doc)
    if len(add_aboves) > 0:
        add_above_labeled_sentences = (pd.DataFrame(add_aboves)
            #  .assign(add_above_label=lambda df: df['cluster_size'].apply(lambda x: 'add above ' + str(x)))
            .rename(columns={'cluster_size': 'add_above_label'})
            .merge(doc, how='inner', right_on='sent_idx_y', left_on='add_above')
        [['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'add_above_label']]
            )
    else:
        add_above_labeled_sentences = pd.DataFrame()

    if len(add_belows) > 0:
        add_below_labeled_sentences = (pd.DataFrame(add_belows)
            #  .assign(add_below_label=lambda df: df['cluster_size'].apply(lambda x: 'add below ' + str(x)))
            .rename(columns={'cluster_size': 'add_below_label'})
            .merge(doc, how='inner', right_on='sent_idx_y', left_on='add_below')
        [['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'add_below_label']]
            )
    else:
        add_below_labeled_sentences = pd.DataFrame()
    #         doc['add_below_label'] = 0

    # 3. Sentence edits:
    edited_sentences = (doc
        .loc[lambda df: df['sent_idx_y'].notnull() & df['sent_idx_x'].notnull() & (df['avg_sentence_distance_x'] > .01)]
        .assign(edited_label=True)
    [['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'edited_label']]
        )
    unchanged_sentences = (doc
        .loc[
        lambda df: df['sent_idx_y'].notnull() & df['sent_idx_x'].notnull() & (df['avg_sentence_distance_x'] <= .01)]
        .assign(unchanged_label=True)
    [['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'unchanged_label']]
        )

    # 4. Sentence Refactors
    refactors = ur.find_refactors_for_doc(doc)
    refactored_sentences = (doc
        .loc[lambda df: df.apply(lambda x: (x['sent_idx_x'], x['sent_idx_y']) in refactors, axis=1)]
        .assign(refactored_label=lambda df:
    df
                .pipe(lambda df: df['sent_idx_y'] - df['sent_idx_x'])
                #          .apply(lambda x: 'move %(direction)s %(num_steps)s' % ({
                #              'direction': 'up' if x < 0 else 'down',
                #              'num_steps': abs(int(x))
                #              }))
                )
    [['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'refactored_label']]
        )

    ## Concat to make Sentence-Level DF
    sentence_label_df = (pd.concat([
        deleted_labeled_sentences,
        add_above_labeled_sentences,
        add_below_labeled_sentences,
        edited_sentences,
        unchanged_sentences,
        refactored_sentences,
    ])
         .assign(version_y=lambda df: df['version_y'].fillna(method='bfill'))
         .fillna(False)
         .astype(int)
     )
    if 'add_below_label' not in sentence_label_df:
        sentence_label_df['add_below_label'] = 0
    if 'add_above_label' not in sentence_label_df:
        sentence_label_df['add_above_label'] = 0

    sentence_label_df = (
        sentence_label_df
             .groupby(['entry_id', 'version_x', 'sent_idx_x'])
             .agg({
                'deleted_label': lambda s: max(s),
                'add_above_label': lambda s: max(s),
                'add_below_label': lambda s: max(s),
                'edited_label': lambda s: max(s),
                'unchanged_label': lambda s: max(s),
                'refactored_label': lambda s: min(s)
             })
             .reset_index()
    )

    sentence_label_df = doc_sentences.merge(
        sentence_label_df,
        how='left',
        left_on=['entry_id', 'version', 'sent_idx'],
        right_on=['entry_id', 'version_x', 'sent_idx_x']
    ).drop(['version_x', 'sent_idx_x'], axis=1).fillna(0)

    ## Make Doc-Label DF
    doc_label_df = (sentence_label_df
        .assign(refactored_label=lambda df: (df['refactored_label'] != 0).astype(int))
        .groupby(['entry_id', 'version'])
        .aggregate({
        'deleted_label': 'sum',
        'add_above_label': 'sum',
        'edited_label': 'sum',
        'refactored_label': 'sum',
        'sentence': lambda s: '<SENT>'.join(s)
    })
        .rename(columns={
        'deleted_label': 'num_deleted',
        'add_above_label': 'num_added',
        'edited_label': 'num_edited',
        'refactored_label': 'num_refactored',
        'sentence': 'sentences'
    })
    )

    return sentence_label_df, doc_label_df