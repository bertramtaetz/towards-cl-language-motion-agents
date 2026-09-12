# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
import os

# gensim depends on `scipy.linalg.triu` for some utilities. Some SciPy builds in
# this VM do not expose that symbol, so we provide a small compatibility shim.
try:  # pragma: no cover
    import numpy as np
    import scipy.linalg as _scipy_linalg

    if not hasattr(_scipy_linalg, 'triu'):
        _scipy_linalg.triu = np.triu
    if not hasattr(_scipy_linalg, 'tril'):
        _scipy_linalg.tril = np.tril
except Exception:
    pass

try:
    from gensim.models import KeyedVectors
except ImportError:
    from gensim.models import Word2Vec as KeyedVectors

import six
from nlgeval.word2vec.glove2word2vec import glove2word2vec


def txt2bin(filename):
    """Convert a word2vec text file to gensim's native KeyedVectors format.

    Older gensim (<=3) required tweaking `sample_int` on vocab entries.
    In gensim 4, vocabulary internals changed; saving/loading works directly.
    """
    m = KeyedVectors.load_word2vec_format(filename)

    # gensim 3 compatibility: ensure at least one vocab entry has sample_int
    if hasattr(m, 'vocab') and m.vocab:
        m.vocab[next(six.iterkeys(m.vocab))].sample_int = 1

    out = filename.replace('txt', 'bin')
    m.save(out)
    KeyedVectors.load(out, mmap='r')


def generate(path):
    glove_vector_file = os.path.join(path, 'glove.6B.300d.txt')
    output_model_file = os.path.join(path, 'glove.6B.300d.model.txt')

    txt2bin(glove2word2vec(glove_vector_file, output_model_file))


if __name__ == "__main__":
    generate()
