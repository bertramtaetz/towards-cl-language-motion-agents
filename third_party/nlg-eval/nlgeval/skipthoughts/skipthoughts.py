"""Skip-thought vectors.

This repository historically used **Theano** to run the pretrained SkipThoughts
encoders. Theano does not work on modern Python versions (e.g. Python 3.12).

To keep SkipThoughtsCosineSimilarity functional, this module implements the
exact same encoder forward-pass in **NumPy**.

The pretrained parameter files (uni_skip.npz / bi_skip.npz) and lookup tables
are unchanged and still downloaded by `nlg-eval --setup`.
"""

import os
import logging
from collections import OrderedDict, defaultdict

import nltk
import numpy as np
import six
from nltk.tokenize import word_tokenize
from scipy.linalg import norm
from six.moves import cPickle as pkl

from nlgeval.utils import get_data_dir


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable sigmoid
    x = x.astype(np.float32, copy=False)
    # Avoid relying on ufunc keyword arguments for broad NumPy compatibility.
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _gru_last_state(embedding: np.ndarray, x_mask: np.ndarray, params: dict, prefix: str, dim: int) -> np.ndarray:
    """Compute last hidden state of the GRU used in the original skip-thoughts code.

    Args:
        embedding: float32 array of shape (T, B, D_in)
        x_mask: float32 array of shape (T, B) with 1 for valid steps, 0 for padding
        params: dict containing numpy arrays for the GRU
        prefix: parameter prefix (e.g. 'encoder' or 'encoder_r')
        dim: hidden size

    Returns:
        float32 array of shape (B, dim)
    """
    # Parameter names match the original code.
    W = params[f'{prefix}_W'].astype(np.float32, copy=False)
    b = params[f'{prefix}_b'].astype(np.float32, copy=False)
    U = params[f'{prefix}_U'].astype(np.float32, copy=False)
    Wx = params[f'{prefix}_Wx'].astype(np.float32, copy=False)
    bx = params[f'{prefix}_bx'].astype(np.float32, copy=False)
    Ux = params[f'{prefix}_Ux'].astype(np.float32, copy=False)

    embedding = embedding.astype(np.float32, copy=False)
    x_mask = x_mask.astype(np.float32, copy=False)

    T, B, _ = embedding.shape

    # Precompute input projections.
    xWb = np.tensordot(embedding, W, axes=(2, 0)).astype(np.float32) + b  # (T,B,2*dim)
    xWxbx = np.tensordot(embedding, Wx, axes=(2, 0)).astype(np.float32) + bx  # (T,B,dim)

    h = np.zeros((B, dim), dtype=np.float32)

    for t in range(T):
        preact = (h @ U).astype(np.float32) + xWb[t]
        r = _sigmoid(preact[:, :dim])
        u = _sigmoid(preact[:, dim:])

        preactx = ((h @ Ux).astype(np.float32) * r) + xWxbx[t]
        h_tilde = np.tanh(preactx).astype(np.float32)

        h_new = u * h + (1.0 - u) * h_tilde
        m = x_mask[t]
        h = (m[:, None] * h_new + (1.0 - m)[:, None] * h).astype(np.float32)

    return h


def load_model():
    """Load the skip-thoughts model and return a dict compatible with the original API."""
    data_dir = get_data_dir()

    path_to_umodel = os.path.join(data_dir, 'uni_skip.npz')
    path_to_bmodel = os.path.join(data_dir, 'bi_skip.npz')

    with open(f'{path_to_umodel}.pkl', 'rb') as f:
        uoptions = pkl.load(f)
    with open(f'{path_to_bmodel}.pkl', 'rb') as f:
        boptions = pkl.load(f)

    # Load parameters directly from the .npz files.
    uparams = {k: v for k, v in np.load(path_to_umodel).items()}
    bparams = {k: v for k, v in np.load(path_to_bmodel).items()}

    # Extractor functions (drop-in replacements for theano.function)
    def f_w2v(embedding, x_mask):
        return _gru_last_state(embedding, x_mask, uparams, prefix='encoder', dim=uoptions['dim'])

    def f_w2v2(embedding, x_mask):
        # Bi encoder: forward + reverse, then concatenate.
        fwd = _gru_last_state(embedding, x_mask, bparams, prefix='encoder', dim=boptions['dim'])
        rev = _gru_last_state(embedding[::-1], x_mask[::-1], bparams, prefix='encoder_r', dim=boptions['dim'])
        return np.concatenate([fwd, rev], axis=1).astype(np.float32)

    utable, btable = load_tables(data_dir)

    return {
        'uoptions': uoptions,
        'boptions': boptions,
        'utable': utable,
        'btable': btable,
        'f_w2v': f_w2v,
        'f_w2v2': f_w2v2,
    }


def load_tables(path_to_tables=None):
    """Load the word->embedding tables (utable/btable)."""
    if path_to_tables is None:
        path_to_tables = get_data_dir()

    words = []
    utable = np.load(os.path.join(path_to_tables, 'utable.npy'), allow_pickle=True, encoding='bytes')
    btable = np.load(os.path.join(path_to_tables, 'btable.npy'), allow_pickle=True, encoding='bytes')
    f = open(os.path.join(path_to_tables, 'dictionary.txt'), 'rb')
    for line in f:
        words.append(line.decode('utf-8').strip())
    f.close()
    utable = OrderedDict(zip(words, utable))
    btable = OrderedDict(zip(words, btable))
    return utable, btable


class Encoder(object):
    """
    Sentence encoder.
    """

    def __init__(self, model):
      self._model = model

    def encode(self, X, use_norm=True, verbose=True, batch_size=128, use_eos=False):
      """
      Encode sentences in the list X. Each entry will return a vector
      """
      return encode(self._model, X, use_norm, verbose, batch_size, use_eos)


def encode(model, X, use_norm=True, verbose=True, batch_size=128, use_eos=False):
    """
    Encode sentences in the list X. Each entry will return a vector
    """
    # first, do preprocessing
    X = preprocess(X)

    # word dictionary and init
    d = defaultdict(lambda: 0)
    for w in model['utable'].keys():
        d[w] = 1
    ufeatures = np.zeros((len(X), model['uoptions']['dim']), dtype='float32')
    bfeatures = np.zeros((len(X), 2 * model['boptions']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i,s in enumerate(captions):
        ds[len(s)].append(i)

    # Get features. This encodes by length, in order to avoid wasting computation
    for k in ds.keys():
        if verbose:
            print(k)
        numbatches = int(len(ds[k]) / batch_size + 1)
        for minibatch in range(numbatches):
            caps = ds[k][minibatch::numbatches]

            if use_eos:
                uembedding = np.zeros((k + 1, len(caps), model['uoptions']['dim_word']), dtype='float32')
                bembedding = np.zeros((k + 1, len(caps), model['boptions']['dim_word']), dtype='float32')
            else:
                uembedding = np.zeros((k, len(caps), model['uoptions']['dim_word']), dtype='float32')
                bembedding = np.zeros((k, len(caps), model['boptions']['dim_word']), dtype='float32')
            for ind, c in enumerate(caps):
                caption = captions[c]
                for j in range(len(caption)):
                    if d[caption[j]] > 0:
                        uembedding[j,ind] = model['utable'][caption[j]]
                        bembedding[j,ind] = model['btable'][caption[j]]
                    else:
                        uembedding[j,ind] = model['utable']['UNK']
                        bembedding[j,ind] = model['btable']['UNK']
                if use_eos:
                    uembedding[-1,ind] = model['utable']['<eos>']
                    bembedding[-1,ind] = model['btable']['<eos>']
            if use_eos:
                uff = model['f_w2v'](uembedding, np.ones((len(caption) + 1, len(caps)), dtype='float32'))
                bff = model['f_w2v2'](bembedding, np.ones((len(caption) + 1, len(caps)), dtype='float32'))
            else:
                uff = model['f_w2v'](uembedding, np.ones((len(caption), len(caps)), dtype='float32'))
                bff = model['f_w2v2'](bembedding, np.ones((len(caption), len(caps)), dtype='float32'))
            if use_norm:
                for j in range(len(uff)):
                    uff[j] /= norm(uff[j])
                    bff[j] /= norm(bff[j])
            for ind, c in enumerate(caps):
                ufeatures[c] = uff[ind]
                bfeatures[c] = bff[ind]
    
    features = np.c_[ufeatures, bfeatures]
    return features


def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X


def nn(model, text, vectors, query, k=5):
    """
    Return the nearest neighbour sentences to query
    text: list of sentences
    vectors: the corresponding representations for text
    query: a string to search
    """
    qf = encode(model, [query])
    qf /= norm(qf)
    scores = np.dot(qf, vectors.T).flatten()
    sorted_args = np.argsort(scores)[::-1]
    sentences = [text[a] for a in sorted_args[:k]]
    print('QUERY: ' + query)
    print('NEAREST: ')
    for i, s in enumerate(sentences):
        print(s, sorted_args[i])


def word_features(table):
    """
    Extract word features into a normalized matrix
    """
    features = np.zeros((len(table), 620), dtype='float32')
    keys = table.keys()
    for i in range(len(table)):
        f = table[keys[i]]
        features[i] = f / norm(f)
    return features


def nn_words(table, wordvecs, query, k=10):
    """
    Get the nearest neighbour words
    """
    keys = table.keys()
    qf = table[query]
    scores = np.dot(qf, wordvecs.T).flatten()
    sorted_args = np.argsort(scores)[::-1]
    words = [keys[a] for a in sorted_args[:k]]
    print('QUERY: ' + query)
    print('NEAREST: ')
    for i, w in enumerate(words):
        print(w)


def _p(pp, name):
    """
    make prefix-appended name
    """
    return '%s_%s' % (pp, name)


def load_params(path, params):
    """Backward compatible parameter loader (unused in the NumPy port)."""
    pp = np.load(path)
    for kk, vv in six.iteritems(params):
        if kk not in pp:
            logging.warning('%s is not in the archive', kk)
            continue
        params[kk] = pp[kk]
    return params


# The remaining code below (parameter initialization / Theano graph building)
# is intentionally removed in the NumPy port. The encoder weights are loaded
# directly from the provided .npz files.


