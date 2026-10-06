"""Minimal stand-in for gensim.downloader.load('word2vec-google-news-300') used by the
city scripts: serves the GoogleNews vectors for the tokens those scripts look up, from a
cache extracted from the original binary (city_resource/cache/w2v_city_tokens.npz)."""
import os, numpy as np
_C = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'w2v_city_tokens.npz')
class _KV:
    def __init__(self, path):
        d = np.load(path); self._d = dict(zip(d['words'].tolist(), d['vecs']))
    def __contains__(self, k): return k in self._d
    def __getitem__(self, k): return self._d[k]
def load(name):
    assert name == 'word2vec-google-news-300', name
    return _KV(_C)
