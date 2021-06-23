import numpy as np
from more_itertools import unique_everseen
import pandas as pd


def read(pos, tree):
    count = 0
    while (pos > 0):
        count += tree[pos]
        pos -= (pos & -pos)
    return count


def update(pos, MAX, tree):
    while (pos <= MAX):
        tree[pos] += 1
        pos += (pos & -pos)


def _find_crossings(n, m, e):
    """
    Method to find crossings on a bipartite graph

    n: num nodes on left
    m: num nodes on right
    e: edges -> tuples of (node_idx_left, node_idx_right)
    """
    tree = np.zeros(1000)
    k = len(e)
    e = sorted(e)
    res = 0
    for i in range(k):
        res += (read(m, tree) - read(e[i][1], tree));
        update(e[i][1], m, tree);
    ##
    return res


def find_crossings(sents_old, sents_new):
    """
    Method to find refactorings (i.e. whether pairs of sentences cross each other in a bipartite graph)

    params:
    * sents_old: list (or pd.Series) of sentences from the old version
    * sents_new: list (or pd.Series) of sentences from the new version

    returns:
    * num_crossings: the number of sentences that have been refactored,
        i.e. the number of edges in a bipartite graph of sentences that cross each other.
    """
    # drop additions/deletions (these don't affect refactorings)
    sents_old = list(filter(pd.notnull, sents_old))
    sents_new = list(filter(pd.notnull, sents_new))
    sents_old = list(map(int, sents_old))
    sents_new = list(map(int, sents_new))

    # make it not zero-indexed, for bitwise addition
    correct_zero = lambda x: x
    if min(sents_old) == 0:
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
    return _find_crossings(n, m, e)