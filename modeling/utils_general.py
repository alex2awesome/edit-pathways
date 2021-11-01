import os
import torch
from torch.nn.functional import pad
from modeling.utils_config import TransformersConfig
from torch.autograd import Variable
import numpy as np

def vec_or_nones(vec, output_len):
    return vec if vec is not None else [None] * output_len


def reformat_model_path(x):
    fp_marker = './'
    if os.environ.get('env') == 'bb' and (not x.startswith(fp_marker)):
        return os.path.join(fp_marker, x)
    else:
        return x


def transpose_dict(dicts):
    """Take a dictionary in record-format and translate it into a columnar dict.

    [{'a': 1, 'b':2}, {'a':2, 'b':3}] -> {'a': [1,2], 'b': [2, 3]}
    """
    columns = {}
    for key in dicts[0].keys():
        columns[key] = list(map(lambda d: d[key], dicts))
    return columns


def format_layer_freezes(layers):
    """
    Format the input argument --freeze_encoder_layers
    When run locally, it's a list of strings: ['1', '2',...]
    When run remotely, because of reformatting, it's a list of joined strings: ['1 2 3...']

    """
    if not isinstance(layers, list):
        try:
            return int(layers)
        except:
            return

    if len(layers) == 0:
        return

    if len(layers) == 1:
        if isinstance(layers[0], str) and layers[0].isdigit():
            return int(layers[0])

        layers = layers[0].split()

    return list(map(int, layers))

def format_loss_weighting(vec):
    """
    Format the input argument --loss_weighting
    When run locally, it's a list of strings: ['.01', '.24',...]
    When run remotely, because of reformatting, it's a list of joined strings: ['1 2 3...']
    """
    if not isinstance(vec, list):
        try:
            return 1
        except:
            return

    if len(vec) == 0:
        return

    if len(vec) == 1:
        if isinstance(vec[0], str) and vec[0].isdigit():
            return 1

        vec = vec[0].split()

    vec = Variable(torch.tensor(list(map(float, vec))), requires_grad=False)
    # vec = np.array(list(map(float, vec)))
    vec = vec / vec.sum()
    return vec


def _get_len(x):
    # x is already a length
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, list):
        return len(x)
    else:
        return x.shape.numel()


def get_config(config=None, kwargs={}):
    if config is None:
        config = kwargs.get('config', None)
    if config is None and len(kwargs) > 0:
        config = TransformersConfig.from_dict(kwargs)
    return config

def freeze_all_params(subgraph):
    for p in subgraph.parameters():
        p.requires_grad = False


def reshape_and_pad_sequence(hidden, sequence_lens, device=None):
    """
    Take in a flattened sequence of sentences and reshape it into a padded cube.

    Params:
        * hidden: input token sequence of shape (# tokens in doc X embedding dim )
        * sequence_lens: list of sentence lengths to split the sequence into.

    Returns:
        * output_hidden: matrix of shape (# sentences, max(# tokens in sentence), embedding_dim)
            unless
    """
    if device is None:
        device = get_device()

    # if multiple documents are passed in, we assume every document is the same length.
    # this is true if we're testing candidate sentences.
    max_seq_len = max(sequence_lens)
    cum_seq_lens = torch.cumsum(sequence_lens, dim=0)
    start_idxs = torch.cat((torch.zeros(1, device=device), cum_seq_lens[:-1])).to(torch.int16)
    stop_idxs = cum_seq_lens
    # one document is passed in
    if len(hidden.shape) == 2:
        num_sents, embedding_dim = hidden.shape
        max_seq_len = max(sequence_lens)
        output_hidden = torch.zeros((len(sequence_lens), max_seq_len, embedding_dim), device=device)
        for idx, (s, e) in enumerate(zip(start_idxs, stop_idxs)):
            sentence_emb = hidden[s:e]
            padded_emb = pad(sentence_emb, (0, 0, 0, max_seq_len - (e - s)))
            output_hidden[idx] = padded_emb

    # multiple documents passed in.
    elif len(hidden.shape) == 3:
        num_docs, num_sents, embedding_dim = hidden.shape
        output_hidden = torch.zeros((num_docs, len(sequence_lens), max_seq_len, embedding_dim), device=device)
        for doc_idx in range(num_docs):
            for sent_idx, (s, e) in enumerate(zip(start_idxs, stop_idxs)):
                curr_sent_len = (e - s)
                sentence_emb = hidden[doc_idx][s:e]
                padded_emb = pad(sentence_emb, (0, 0, 0, max_seq_len - curr_sent_len))
                output_hidden[doc_idx][sent_idx] = padded_emb

    return output_hidden


class SuperBlank():
    def __init__(self, *args, **kwargs):
        super().__init__()