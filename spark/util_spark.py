import sqlite3

import pandas as pd

from spark.runner_script import db_name, spark

SENTENCE_SIM_THRESH = .44
APPROX_JOIN_CUTOFF = .5


def get_word_matching_sql(side):
    """Generate the SQL necessary to transform each side. Side \in {'x', 'y'}"""

    word_pair_min_distance_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                word_idx_%(side)s,
                MIN(num_words) as num_words_total_list,
                MIN(distance) as min_word_distance
        FROM __THIS__ 
        GROUP BY entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                word_idx_%(side)s
      """ % ({'side': side})

    sentence_pair_min_distance_sql = """
        SELECT entry_id,
               version_x,
               version_y,
               sent_idx_x,
               sent_idx_y,
               (sum_min_word_distance + %(approx_join_cutoff)f * ( num_words_total - num_matched_words )) / num_words_total AS avg_sentence_distance
        FROM (
           SELECT entry_id,
                  version_x,
                  version_y,
                  sent_idx_x,
                  sent_idx_y,
                  SUM(min_word_distance) AS sum_min_word_distance,
                  COUNT(1) AS num_matched_words,
                  MIN(num_words_total_list) AS num_words_total
           FROM __THIS__
                GROUP BY entry_id,
                   version_x,
                   version_y,
                   sent_idx_x,
                   sent_idx_y
          )
      """ % ({'approx_join_cutoff': APPROX_JOIN_CUTOFF})

    sentence_min_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_x,
                sent_idx_y,
                avg_sentence_distance
           FROM (
                    SELECT *, ROW_NUMBER() OVER (
                         PARTITION BY entry_id, 
                                      version_x, 
                                      version_y, 
                                      sent_idx_%(side)s
                         ORDER BY avg_sentence_distance ASC
                ) AS rn FROM __THIS__
        )
         where rn = 1
    """ % ({'side': side})

    threshold_sql = """
         SELECT entry_id,
                version_x,
                version_y,
                sent_idx_%(join_side)s,
                CASE 
                    WHEN (avg_sentence_distance < %(sentence_sim)f ) THEN sent_idx_%(other_side)s
                    ELSE NULL
                END AS sent_idx_%(other_side)s,
                CASE 
                    WHEN (avg_sentence_distance < %(sentence_sim)f  THEN avg_sentence_distance
                    ELSE NULL
                END AS avg_sentence_distance
            FROM __THIS__
    """ % ({
        'join_side': side,
        'other_side': list({'x', 'y'} - set(side))[0],
        'sentence_sim': SENTENCE_SIM_THRESH
    })

    return word_pair_min_distance_sql, sentence_pair_min_distance_sql, sentence_min_sql, threshold_sql


def read_spark_df(num_entries, start_idx):
    with sqlite3.connect(db_name) as conn:
        df = pd.read_sql('''
             SELECT * from entryversion 
             WHERE entry_id IN (
                SELECT distinct entry_id 
                    FROM entryversion ORDER BY entry_id LIMIT %(num_entries)d OFFSET %(start_idx)d
                )
         ''' % {'num_entries': num_entries, 'start_idx': start_idx}, con=conn)
        df = df.assign(summary=lambda df: df['summary'].str.replace('</p><p>', ' '))
        sdf = spark.createDataFrame(df)
        return sdf