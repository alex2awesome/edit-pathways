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



input_df = pd.DataFrame([
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 2.0, 'sent_idx_y': 2.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 9.0, 'sent_idx_y': 6.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 7.0, 'sent_idx_y': 4.0, 'avg_sentence_distance_x': 0.3233337765424261, 'avg_sentence_distance_y': 0.3233337765424261, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 0.0, 'sent_idx_y': 0.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 10.0, 'sent_idx_y': 10.0, 'avg_sentence_distance_x': 0.2244993526073515, 'avg_sentence_distance_y': 0.2244993526073515, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 5.0, 'sent_idx_y': 9.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 8.0, 'sent_idx_y': 5.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 4.0, 'sent_idx_y': 8.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 6.0, 'sent_idx_y': 3.0, 'avg_sentence_distance_x': 0.2363780557250096, 'avg_sentence_distance_y': 0.2363780557250096, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 1.0, 'sent_idx_y': 1.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '842565-0', 'key_y': '842565-1'},
    {'entry_id': 842565, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 3.0, 'sent_idx_y': 7.0, 'avg_sentence_distance_x': 0.14502703111922652, 'avg_sentence_distance_y': 0.14502703111922652, 'key_x': '842565-0', 'key_y': '842565-1'}
])


input_df = pd.DataFrame([
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 10.0, 'sent_idx_y': 5.0, 'avg_sentence_distance_x': 0.15794722522537913, 'avg_sentence_distance_y': 0.15794722522537913, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 12.0, 'sent_idx_y': 7.0, 'avg_sentence_distance_x': 0.2934969179335545, 'avg_sentence_distance_y': 0.2934969179335545, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 11.0, 'sent_idx_y': 6.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 18.0, 'sent_idx_y': 17.0, 'avg_sentence_distance_x': 4.866755764368342e-07, 'avg_sentence_distance_y': 4.866755764368342e-07, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 2.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 17.0, 'sent_idx_y': 16.0, 'avg_sentence_distance_x': 0.0, 'avg_sentence_distance_y': 0.0, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 7.0, 'sent_idx_y': 3.0, 'avg_sentence_distance_x': 0.32805670207838045, 'avg_sentence_distance_y': 0.32805670207838045, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 9.0, 'sent_idx_y': 11.0, 'avg_sentence_distance_x': 0.17931568775166842, 'avg_sentence_distance_y': 0.17931568775166842, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 0.0, 'sent_idx_y': 0.0, 'avg_sentence_distance_x': 0.3700484073083055, 'avg_sentence_distance_y': 0.3700484073083055, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 13.0, 'sent_idx_y': 2.0, 'avg_sentence_distance_x': 0.3553945547529145, 'avg_sentence_distance_y': 0.3553945547529145, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 6.0, 'sent_idx_y': 10.0, 'avg_sentence_distance_x': 0.3851866451866496, 'avg_sentence_distance_y': np.nan, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 5.0, 'sent_idx_y': 10.0, 'avg_sentence_distance_x': 0.3651086829851781, 'avg_sentence_distance_y': 0.3649198550063676, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': np.nan, 'sent_idx_y': 12.0, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 8.0, 'sent_idx_y': 4.0, 'avg_sentence_distance_x': 4.788105030505076e-09, 'avg_sentence_distance_y': 4.788105030505076e-09, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 14.0, 'sent_idx_y': 13.0, 'avg_sentence_distance_x': 0.4136146725083047, 'avg_sentence_distance_y': 0.4136146725083047, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': np.nan, 'sent_idx_y': 1.0, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 4.0, 'sent_idx_y': 9.0, 'avg_sentence_distance_x': 0.3007591346553736, 'avg_sentence_distance_y': 0.3018912521774385, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 16.0, 'sent_idx_y': 15.0, 'avg_sentence_distance_x': 5.174512374800524e-07, 'avg_sentence_distance_y': 5.174512374800524e-07, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 3.0, 'sent_idx_y': 8.0, 'avg_sentence_distance_x': 0.3228540156830339, 'avg_sentence_distance_y': 0.3228540156830339, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 1.0, 'sent_idx_y': np.nan, 'avg_sentence_distance_x': np.nan, 'avg_sentence_distance_y': np.nan, 'key_x': '500211-0', 'key_y': '500211-1'},
    {'entry_id': 500211, 'version_x': 0, 'version_y': 1, 'sent_idx_x': 15.0, 'sent_idx_y': 14.0, 'avg_sentence_distance_x': 0.17788290170777815, 'avg_sentence_distance_y': 0.17788290170777815, 'key_x': '500211-0', 'key_y': '500211-1'}
])

if __name__ == "__main__":
    refactoring = ur.find_refactors_for_doc(
        one_doc=input_df#[['entry_id', 'version_x', 'version_y', 'sent_idx_x', 'sent_idx_y']]
        )

    print(refactoring)