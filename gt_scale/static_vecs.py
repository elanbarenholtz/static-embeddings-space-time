"""Build averaged static vectors (GloVe 6B 300d, Word2Vec GoogleNews) for G&T entity names.

Two coverage rules are computed: 'all' (every word in vocab) and 'any' (at least one).
Saves static_<model>_<dataset>.npz with vecs, n_in_vocab, n_words.
"""
import re, gzip, sys, numpy as np, pandas as pd

TOK = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")

def words(name):
    return TOK.findall(str(name))

def load_glove(path, need):
    E = {}
    with open(path, encoding='utf8') as f:
        for line in f:
            w, rest = line.split(' ', 1)
            if w in need:
                E[w] = np.array(rest.split(), dtype=np.float32)
    return E

def load_w2v(path, need):
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

def build(model, ds, E, lower):
    df = pd.read_csv(f'data/{ds}.csv')
    vecs = np.zeros((len(df), 300), np.float32)
    nin = np.zeros(len(df), int); nw = np.zeros(len(df), int)
    for i, name in enumerate(df['name']):
        if model == 'Word2Vec':
            phrase = '_'.join(words(name))
            if phrase in E and len(words(name)) > 1:
                vecs[i] = E[phrase]; nin[i] = nw[i] = len(words(name)); continue
        ws = [w.lower() if lower else w for w in words(name)]
        got = [E[w] if w in E else (E.get(w.lower()) if not lower else None) for w in ws]
        got = [g for g in got if g is not None]
        nw[i] = len(ws); nin[i] = len(got)
        if got:
            vecs[i] = np.mean(got, 0)
    np.savez(f'static_{model}_{ds}.npz', vecs=vecs, n_in_vocab=nin, n_words=nw)
    print(model, ds, 'all-coverage', np.mean((nin == nw) & (nw > 0)).round(3),
          'any-coverage', np.mean(nin > 0).round(3), flush=True)

if __name__ == '__main__':
    which = sys.argv[1]
    dss = ['world_place', 'historical_figure']
    allw = set()
    for ds in dss:
        for n in pd.read_csv(f'data/{ds}.csv')['name']:
            ws = words(n)
            allw |= set(ws) | {w.lower() for w in ws} | {'_'.join(ws)}
    if which == 'glove':
        E = load_glove('data/glove.6B.300d.txt', allw)
        for ds in dss: build('GloVe', ds, E, lower=True)
    else:
        E = load_w2v('data/w2v.gz', allw)
        for ds in dss: build('Word2Vec', ds, E, lower=False)
