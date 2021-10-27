import numpy as np
from more_itertools import unique_everseen
import pandas as pd
from collections import defaultdict
import copy


def read(pos, tree):
    count = []
    while (pos > 0):
        count += tree[pos]
        pos -= (pos & -pos)
    return count


def update(pos, MAX, edge, tree):
    while (pos <= MAX):
        tree[pos].append(edge)
        pos += (pos & -pos)


def _find_crossings(n, m, e):
    """
    Method to find crossings on a bipartite graph

    n: num nodes on left
    m: num nodes on right
    e: edges -> tuples of (node_idx_left, node_idx_right)
    """
    tree = defaultdict(list)
    k = len(e)
    e = sorted(e)
    res = {}
    for i in range(k):
        r_m = read(m, tree)
        r_e = read(e[i][1], tree)
        c = set(r_m) - set(r_e)
        res[e[i]] = c
        update(e[i][1], m, e[i], tree)
    ##
    return res, tree


def remove_one_crossing(dict_r):
    # 1. select the edge with the most crossings
    max_crossing_val = len(max(dict_r.values(), key=lambda x: len(x)))
    chosen_keys = filter(lambda x: len(x[1]) == max_crossing_val, dict_r.items())
    chosen_keys = list(dict(chosen_keys).keys())

    # 1a. if only one crossing, continue
    if len(chosen_keys) == 1:
        key_to_remove = chosen_keys[0]

    # 1b. if there's multiple keys with the same number of crossings, take the ones moves the farthest.
    else:
        max_move = max(map(lambda x: abs(x[1] - x[0]), chosen_keys))
        chosen_keys = list(filter(lambda x: abs(x[1] - x[0]) == max_move, chosen_keys))
        if len(chosen_keys) == 1:
            key_to_remove = chosen_keys[0]

        # 1c. If there are multiple keys that move the same distance, take the ones that move up.
        # 1d. If there are multiple of these keys, just take the first
        else:
            chosen_keys = list(filter(lambda x: x[1] - x[0] < 0, chosen_keys))
            key_to_remove = chosen_keys[0]

    dict_r.pop(key_to_remove)
    for key, crossings in dict_r.items():
        if key_to_remove in crossings:
            dict_r[key].remove(key_to_remove)

    return key_to_remove, dict_r


def identify_refactor_edges(crossings_dict):
    r_copy = copy.deepcopy(crossings_dict)
    removed_crossings = []
    while any(r_copy.values()):
        removed_crossing, r_copy = remove_one_crossing(r_copy)
        removed_crossings.append(removed_crossing)
    return removed_crossings


def find_refactors_for_doc(one_doc=None, sents_old=None, sents_new=None):
    """
    Method to find refactorings (i.e. whether pairs of sentences cross each other in a bipartite graph)

    params:
    * one_doc: a dataframe containing columns: ['sent_idx_x', 'sent_idx_y']
    OR
    * sents_old: list (or pd.Series) of sentences from the old version
    * sents_new: list (or pd.Series) of sentences from the new version

    returns:
    * num_crossings: the number of sentences that have been refactored,
        i.e. the number of edges in a bipartite graph of sentences that cross each other.
    """
    # drop additions/deletions (these don't affect refactorings)
    if sents_old is None:
        sents_old = one_doc['sent_idx_x']
    if sents_new is None:
        sents_new = one_doc['sent_idx_y']
    sents_old = list(filter(pd.notnull, sents_old))
    sents_new = list(filter(pd.notnull, sents_new))
    sents_old = list(map(int, sents_old))
    sents_new = list(map(int, sents_new))

    # make it not zero-indexed, for bitwise addition
    correct_zero = lambda x: x + 1
    # map missing indices (the result of dropping additions/deletions) to a compressed set.
    sents_old_map = {v: correct_zero(k) for k, v in enumerate(unique_everseen(sents_old))}
    sents_new_map = {v: correct_zero(k) for k, v in enumerate(unique_everseen(sents_new))}

    # prepare input to function
    n = len(sents_old_map)
    m = len(sents_new_map)
    sents_old = list(map(sents_old_map.get, sents_old))
    sents_new = list(map(sents_new_map.get, sents_new))
    e = list(zip(sents_old, sents_new))

    # calculate and return
    crossings, tree = _find_crossings(n, m, e)
    refactors = identify_refactor_edges(crossings)
    if len(refactors) > 0:
        sents_old_map_r = {v: k for k, v in sents_old_map.items()}
        sents_new_map_r = {v: k for k, v in sents_new_map.items()}
        refactors = list(map(lambda x: (sents_old_map_r[x[0]], sents_new_map_r[x[1]]), refactors))
    return refactors