import pandas as pd
import numpy as np
import util.util_refactorings as ur


# zero refactors in this one, but one split edge
input_df = pd.DataFrame([
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 10.0, 'sent_idx_y': 11.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 0.0, 'sent_idx_y': 0.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 9.0, 'sent_idx_y': 10.0, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': 0.16038170542974908, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 7.0, 'sent_idx_y': 7.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 2.0, 'sent_idx_y': 2.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 1.0, 'sent_idx_y': 1.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 8.0, 'sent_idx_y': 8.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 3.0, 'sent_idx_y': 3.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 6.0, 'sent_idx_y': 6.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 4.0, 'sent_idx_y': 4.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 9.0, 'sent_idx_y': 9.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 5.0, 'sent_idx_y': 5.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'},
    {'entry_id': 732754, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 12.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '732754-0', 'key_y': '732754-1'}
])

## one crossing ((12 , 6))
input_df = pd.DataFrame([
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 0.0, 'sent_idx_y': 0.0, 'avg_sentence_distance_x': 0.2375068713555168, 'avg_sentence_distance_y': 0.23750687135551676, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 1.0, 'sent_idx_y': 2.0, 'avg_sentence_distance_x': 0.2650446902309821, 'avg_sentence_distance_y': 0.2650446902309821, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 2.0, 'sent_idx_y': 3.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 3.0, 'sent_idx_y': 4.0, 'avg_sentence_distance_x': 0.2688027257616015, 'avg_sentence_distance_y': 0.2688027257616015, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 4.0, 'sent_idx_y': 5.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 5.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 6.0, 'sent_idx_y': 7.0, 'avg_sentence_distance_x': 0.3907195092752525, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 7.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 8.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 10.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 11.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 12.0, 'sent_idx_y': 6.0, 'avg_sentence_distance_x': 0.10944891052701496, 'avg_sentence_distance_y': 0.10944891052701496, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 13.0, 'sent_idx_y': 7.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 14.0, 'sent_idx_y': 8.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 15.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 16.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 17.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 18.0, 'sent_idx_y': 11.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': np.nan, 'sent_idx_y': 1.0, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': np.nan, 'sent_idx_y': 9.0, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'},
    {'entry_id': 743901, 'version_x': 0, 'version_y': 1, 'sent_idx_x': np.nan, 'sent_idx_y': 10.0, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '743901-0', 'key_y': '743901-1'}
])


if __name__ == "__main__":
    refactoring = ur.find_refactors_for_doc(
        one_doc=input_df#[['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'sent_idx_y']]
        )

    print(refactoring)