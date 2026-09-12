# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
import os
import numpy as np

from nlgeval.utils import get_data_dir

# gensim imports expect `scipy.linalg.triu` / `tril` in some environments.
# Some SciPy builds here don't expose those symbols, so provide a shim.
try:  # pragma: no cover
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


class Embedding(object):
    def __init__(self):
        path = get_data_dir()
        self.m = KeyedVectors.load(os.path.join(path, 'glove.6B.300d.model.bin'), mmap='r')

        # gensim 4 uses `vectors` / `key_to_index`; gensim 3 used `syn0` / `vocab`.
        self._vectors = getattr(self.m, 'vectors', None)
        if self._vectors is None:
            self._vectors = self.m.syn0
        self.unk = self._vectors.mean(axis=0)

    @property
    def w2v(self):
        return np.concatenate((self._vectors, self.unk[None, :]), axis=0)

    def __getitem__(self, key):
        # Return the index in the vectors matrix (last index reserved for UNK).
        key_to_index = getattr(self.m, 'key_to_index', None)
        if key_to_index is not None:
            idx = key_to_index.get(key)
            return idx if idx is not None else len(self.m.index_to_key)

        # gensim 3 fallback
        try:
            return self.m.vocab[key].index
        except KeyError:
            return len(self.m.syn0)

    def vec(self, key):
        key_to_index = getattr(self.m, 'key_to_index', None)
        if key_to_index is not None:
            idx = key_to_index.get(key)
            return self._vectors[idx] if idx is not None else self.unk

        # gensim 3 fallback
        try:
            return self._vectors[self.m.vocab[key].index]
        except KeyError:
            return self.unk


def eval_emb_metrics(hypothesis, references, emb=None, metrics_to_omit=None):
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.tokenize import word_tokenize
    import numpy as np
    if emb is None:
        emb = Embedding()

    if metrics_to_omit is None:
        metrics_to_omit = set()
    else:
        if 'EmbeddingAverageCosineSimilairty' in metrics_to_omit:
            metrics_to_omit.remove('EmbeddingAverageCosineSimilairty')
            metrics_to_omit.add('EmbeddingAverageCosineSimilarity')

    emb_hyps = []
    avg_emb_hyps = []
    extreme_emb_hyps = []
    for hyp in hypothesis:
        embs = [emb.vec(word) for word in word_tokenize(hyp)]

        avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
        assert not np.any(np.isnan(avg_emb))

        maxemb = np.max(embs, axis=0)
        minemb = np.min(embs, axis=0)
        extreme_emb = list(map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb))

        emb_hyps.append(embs)
        avg_emb_hyps.append(avg_emb)
        extreme_emb_hyps.append(extreme_emb)

    emb_refs = []
    avg_emb_refs = []
    extreme_emb_refs = []
    for refsource in references:
        emb_refsource = []
        avg_emb_refsource = []
        extreme_emb_refsource = []
        for ref in refsource:
            embs = [emb.vec(word) for word in word_tokenize(ref)]

            avg_emb = np.sum(embs, axis=0) / np.linalg.norm(np.sum(embs, axis=0))
            assert not np.any(np.isnan(avg_emb))

            maxemb = np.max(embs, axis=0)
            minemb = np.min(embs, axis=0)
            extreme_emb = list(map(lambda x, y: x if ((x>y or x<-y) and y>0) or ((x<y or x>-y) and y<0) else y, maxemb, minemb))

            emb_refsource.append(embs)
            avg_emb_refsource.append(avg_emb)
            extreme_emb_refsource.append(extreme_emb)
        emb_refs.append(emb_refsource)
        avg_emb_refs.append(avg_emb_refsource)
        extreme_emb_refs.append(extreme_emb_refsource)

    rval = []
    if 'EmbeddingAverageCosineSimilarity' not in metrics_to_omit:
        cos_similarity = list(map(lambda refv: cosine_similarity(refv, avg_emb_hyps).diagonal(), avg_emb_refs))
        cos_similarity = np.max(cos_similarity, axis=0).mean()
        rval.append("EmbeddingAverageCosineSimilarity: %0.6f" % (cos_similarity))
        # For backwards compatibility with an old typo before Nov 20, 2019.
        rval.append("EmbeddingAverageCosineSimilairty: %0.6f" % (cos_similarity))

    if 'VectorExtremaCosineSimilarity' not in metrics_to_omit:
        cos_similarity = list(map(lambda refv: cosine_similarity(refv, extreme_emb_hyps).diagonal(), extreme_emb_refs))
        cos_similarity = np.max(cos_similarity, axis=0).mean()
        rval.append("VectorExtremaCosineSimilarity: %0.6f" % (cos_similarity))

    if 'GreedyMatchingScore' not in metrics_to_omit:
        scores = []
        for emb_refsource in emb_refs:
            score_source = []
            for emb_ref, emb_hyp in zip(emb_refsource, emb_hyps):
                simi_matrix = cosine_similarity(emb_ref, emb_hyp)
                dir1 = simi_matrix.max(axis=0).mean()
                dir2 = simi_matrix.max(axis=1).mean()
                score_source.append((dir1 + dir2) / 2)
            scores.append(score_source)
        scores = np.max(scores, axis=0).mean()
        rval.append("GreedyMatchingScore: %0.6f" % (scores))

    rval = "\n".join(rval)
    return rval


if __name__ == '__main__':
    emb = Embedding()
