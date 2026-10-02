"""Table (appendix) of the static baseline on the released entities: GloVe, Word2Vec and fastText,
each scored on the test items it covers (at least one in-vocabulary word), on the authors' split,
with the probe of probe.py (RidgeCV, alphas 1e-2..1e5). fastText vectors (wiki-news-300d-1M.vec)
are built here with the tokenization of review_v2/emb.py (look up as written, then lower case)."""
import os, re, json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
ALPHAS = np.logspace(-2, 5, 29)
TOK = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
FT = '../review_v2/data/wiki-news-300d-1M.vec'
out = {}
for DS in ['world_place', 'historical_figure']:
    df = pd.read_csv(f'data/{DS}.csv'); test = df['is_test'].astype(str).str.lower().eq('true').values
    T = ['latitude', 'longitude'] if DS == 'world_place' else ['death_year']
    Y = df[T].values.astype(float); fin = np.isfinite(Y).all(1)
    names = df['name'].astype(str).tolist()
    if not os.path.exists(f'static_fastText_{DS}.npz'):
        need = set()
        for n in names: ws = TOK.findall(n); need |= set(ws) | {w.lower() for w in ws}
        E = {}
        with open(FT, encoding='utf8', errors='ignore') as f:
            f.readline()
            for line in f:
                w, rest = line.split(' ', 1)
                if w in need: E[w] = np.array(rest.split(), np.float32)
        V = np.zeros((len(names), 300), np.float32); nin = np.zeros(len(names), int)
        for i, n in enumerate(names):
            got = [E[w] if w in E else E[w.lower()] for w in TOK.findall(n) if w in E or w.lower() in E]
            nin[i] = len(got)
            if got: V[i] = np.mean(got, 0)
        np.savez(f'static_fastText_{DS}.npz', vecs=V, n_in_vocab=nin)
    for m in ['GloVe', 'Word2Vec', 'fastText']:
        d = np.load(f'static_{m}_{DS}.npz'); X = d['vecs']; cov = d['n_in_vocab'] > 0
        tr, te = cov & fin & ~test, cov & fin & test
        P = RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr]).predict(X[te]).reshape(int(te.sum()), -1)
        r = [round(float(r2_score(Y[te][:, k], P[:, k])), 3) for k in range(len(T))]
        out[f'{DS}|{m}'] = {'r2': r, 'r2_mean': round(float(np.mean(r)), 3), 'coverage': round(float(cov[fin].mean()), 3), 'n_test': int(te.sum())}
        print(DS, m, out[f'{DS}|{m}'], flush=True)
json.dump(out, open('static_table.json', 'w'), indent=1)
