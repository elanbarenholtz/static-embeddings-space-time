"""Shared static-embedding utilities for the review revisions.

Tokenization and averaging follow gt_scale/static_vecs.py: words are runs of letters/digits
(with internal apostrophes or hyphens); GloVe is looked up lowercased; Word2Vec and fastText
are looked up case-sensitively with a lowercase fallback. A text is the unweighted mean of the
raw (unnormalized) vectors of its in-vocabulary words; texts with no in-vocabulary word get a
zero vector and are reported. Word2Vec phrase entries (e.g. New_York) are used only for entity
names, not sentences (phrases=False by default here).
"""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import re, gzip, os, json, numpy as np

TOK = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
DATA = _os_sb.path.join(_SBROOT, 'gt_scale/data')
FT = _os_sb.path.join(_SBROOT, 'review_v2/data/wiki-news-300d-1M.vec')
CACHE = _os_sb.path.join(_SBROOT, 'review_v2/cache')
MODELS = ['GloVe', 'Word2Vec', 'fastText']


def words(text):
    return TOK.findall(str(text))


def _need(texts):
    s = set()
    for t in texts:
        ws = words(t)
        s |= set(ws) | {w.lower() for w in ws}
    return s


def _load_text(path, need, header):
    E = {}
    with open(path, encoding='utf8', errors='ignore') as f:
        if header:
            f.readline()
        for line in f:
            w, rest = line.split(' ', 1)
            if w in need:
                E[w] = np.array(rest.split(), dtype=np.float32)
    return E


def _load_w2v(path, need):
    E = {}
    with gzip.open(path, 'rb') as f:
        n, d = map(int, f.readline().split())
        rec = 4 * d
        for _ in range(n):
            wb = bytearray()
            while True:
                c = f.read(1)
                if c == b' ':
                    break
                if c != b'\n':
                    wb += c
            v = f.read(rec)
            w = wb.decode('utf8', errors='ignore')
            if w in need:
                E[w] = np.frombuffer(v, dtype=np.float32).copy()
    return E


def build_cache(name, texts):
    """Extract the vectors needed for `texts` from each embedding file into cache/<name>_<model>.npz."""
    os.makedirs(CACHE, exist_ok=True)
    need = _need(texts)
    for m in MODELS:
        out = f'{CACHE}/{name}_{m}.npz'
        if os.path.exists(out):
            continue
        if m == 'GloVe':
            E = _load_text(f'{DATA}/glove.6B.300d.txt', need, header=False)
        elif m == 'fastText':
            E = _load_text(FT, need, header=True)
        else:
            E = _load_w2v(f'{DATA}/w2v.gz', need)
        ws = sorted(E)
        np.savez(out, words=np.array(ws), vecs=np.stack([E[w] for w in ws]) if ws else np.zeros((0, 300)))
        print('cached', m, len(ws), 'of', len(need), flush=True)


def load(name, model):
    d = np.load(f'{CACHE}/{name}_{model}.npz')
    return dict(zip(d['words'].tolist(), d['vecs']))


def embed(texts, E, model):
    X = np.zeros((len(texts), 300), np.float32); n_in = np.zeros(len(texts), int)
    for i, t in enumerate(texts):
        ws = words(t)
        if model == 'GloVe':
            got = [E[w.lower()] for w in ws if w.lower() in E]
        else:
            got = [E[w] if w in E else E[w.lower()] for w in ws if w in E or w.lower() in E]
        n_in[i] = len(got)
        if got:
            X[i] = np.mean(got, 0)
    return X, n_in
