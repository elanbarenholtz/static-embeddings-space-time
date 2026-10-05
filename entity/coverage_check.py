"""Coverage-selection check: do items with a Wikipedia2Vec entity vector differ in difficulty from those without?
Non-entity representations are trained on ALL training items (not only covered ones) and scored on test items
split by entity coverage."""
import re, sys, json, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
ALPHAS = np.logspace(-2, 5, 29); G = '../gt_scale'
e = np.load('w2v_entities.npz'); EV = dict(zip(e['keys'], e['vecs']))
def _ld(p):
    try: return json.load(open(p))
    except Exception: return {}
T = _ld('figure_titles.json'); T.update({k: v for k, v in json.load(open('figure_titles_sparql.json')).items() if v})

class LowMemRidge:
    """Ridge with LOO-GCV alpha selection via eigendecomposition of X'X (centered), for large n x p."""
    def fit(self, X, Y):
        n, p = X.shape; self.mx = X.mean(0, dtype=np.float64).astype(np.float32); self.my = Y.mean(0)
        Xc = X - self.mx; Yc = Y - self.my
        C = np.zeros((p, p)); 
        for i in range(0, n, 4000): b = Xc[i:i+4000].astype(np.float64); C += b.T @ b
        s2, V = np.linalg.eigh(C); self.V = V; self.s2 = np.clip(s2, 0, None)
        XV = np.empty((n, p), np.float32)
        for i in range(0, n, 4000): XV[i:i+4000] = (Xc[i:i+4000].astype(np.float64) @ V).astype(np.float32)
        XtY = np.zeros((p, Y.shape[1]))
        for i in range(0, n, 4000): XtY += Xc[i:i+4000].astype(np.float64).T @ Yc[i:i+4000]
        VtXtY = V.T @ XtY; best = None
        for a in ALPHAS:
            d = 1.0 / (self.s2 + a)
            H = np.zeros(n)
            for i in range(0, n, 4000): H[i:i+4000] = ((XV[i:i+4000].astype(np.float64) ** 2) * d).sum(1)
            H += 1.0 / n
            err = 0.0
            for i in range(0, n, 4000):
                pred = XV[i:i+4000].astype(np.float64) @ (d[:, None] * VtXtY) + self.my
                err += (((Yc[i:i+4000] + self.my - pred) / (1 - H[i:i+4000])[:, None]) ** 2).sum()
            if best is None or err < best[0]: best = (err, a)
        self.alpha_ = best[1]; self.coef = V @ ((1.0 / (self.s2 + best[1]))[:, None] * VtXtY); return self
    def predict(self, X): return (X - self.mx).astype(np.float64) @ self.coef + self.my

out = {}
LM = False
if sys.argv[1] == '--lm': LM = True; sys.argv.pop(1)
DSS = [sys.argv[1]]; ONLY = sys.argv[2:]
for DS in DSS:
    df = pd.read_csv(f'{G}/data/{DS}.csv'); test = df['is_test'].astype(str).str.lower().eq('true').values
    tg = ['latitude', 'longitude'] if DS == 'world_place' else ['death_year']
    Y = df[tg].values.astype(float)
    keys = df['name'].tolist() if DS == 'world_place' else [T.get(q) or n for q, n in zip(df['wiki_id'], df['name'])]
    cov = np.array([isinstance(k, str) and k.replace(' ', '_') in EV for k in keys])
    gl = np.load(f'{G}/static_GloVe_{DS}.npz')
    ok = np.isfinite(Y).all(1) & (gl['n_in_vocab'] > 0)
    tr = ok & ~test
    sel = {'world_place': {'pythia-2.8b': 24, 'llama-2-7b': 24}, 'historical_figure': {'pythia-2.8b': 28, 'llama-2-7b': 24}}[DS]
    reps = {'GloVe (avg)': gl['vecs'], 'fastText (avg)': np.load(f'{G}/static_fastText_{DS}.npz')['vecs']}
    for M, L in sel.items():
        meta = json.load(open(f'{G}/meta_{DS}_{M}.json')); A = np.load(f'{G}/acts_{DS}_{M}.npy', mmap_mode='r')
        reps[f'{M} L{L}'] = A[meta['layers'].index(L)]
    res = {'n_test_covered': int((ok & test & cov).sum()), 'n_test_uncovered': int((ok & test & ~cov).sum()),
           'std_covered': [float(np.std(Y[ok & test & cov][:, k])) for k in range(len(tg))],
           'std_uncovered': [float(np.std(Y[ok & test & ~cov][:, k])) for k in range(len(tg))]}
    for name, X in reps.items():
        if ONLY and not any(name.startswith(o) for o in ONLY): continue
        X = np.asarray(X, np.float32); m = (LowMemRidge() if name.startswith('llama') or LM else RidgeCV(alphas=ALPHAS)).fit(X[tr], Y[tr]); r = {}
        for tag, mask in [('covered', ok & test & cov), ('uncovered', ok & test & ~cov), ('all', ok & test)]:
            P = m.predict(X[mask]).reshape(int(mask.sum()), -1)
            r[tag] = {'r2': [round(r2_score(Y[mask][:, k], P[:, k]), 3) for k in range(len(tg))],
                      'median_abs': round(float(np.median(np.abs(Y[mask] - P), axis=0).mean()), 2)}
        res[name] = r; print(DS, name, r, flush=True); open('coverage_check.jsonl','a').write(json.dumps({'ds':DS,'rep':name,**r})+'\n')
    print(DS, {k: v for k, v in res.items() if k.startswith(('n_', 'std'))})
    out[DS] = res

