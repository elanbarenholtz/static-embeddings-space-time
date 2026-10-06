"""Semantic subspace ablation on the released places (GloVe name vectors, authors' train/test split).
For each word category (semantic_categories.json): PCA the category's GloVe vectors, keep components covering 90% of
variance (cap 20), project that subspace out of every place's averaged name vector, refit a ridge probe on the training
places and score R^2 on the test places. Compared with 200 matched random word sets (same count, frequency decile,
proper-name status and vector norm; propername_top50k.npy flags words whose capitalised form is more frequent in
GoogleNews), each reduced to the same number of dimensions. Reported: drop in test R^2, z and one-sided empirical p against the matched null,
and the drop beyond that predicted from the share of place-vector variance removed (linear fit over the null sets).
Usage: python semantic_ablation_released.py ; writes semantic_ablation_released.json"""
import json, re, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
P = '..'; V = 50000; N_NULL = 200; rng = np.random.default_rng(0)
cats = json.load(open('semantic_categories.json'))
df = pd.read_csv(f'{P}/gt_scale/data/world_place.csv'); names = [str(n).lower() for n in df['name']]
TOK = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*"); toks = [TOK.findall(n) for n in names]
need = {t for ts in toks for t in ts} | {w for v in cats.values() for w in v}; G = {}; words, W = [], []
with open(f'{P}/gt_scale/data/glove.6B.300d.txt', encoding='utf8') as f:
    for i, line in enumerate(f):
        w, r = line.rstrip().split(' ', 1)
        if i < V: words.append(w); W.append(np.array(r.split(), np.float64))
        if w in need: G[w] = np.array(r.split(), np.float64)
W = np.stack(W); idx = {w: i for i, w in enumerate(words)}
ok = np.array([any(t in G for t in ts) for ts in toks])
X = np.stack([np.mean([G[t] for t in ts if t in G], 0) for ts, o in zip(toks, ok) if o])
test = df['is_test'].values.astype(bool)[ok]; Y = {'Latitude': df['latitude'].values[ok].astype(float), 'Longitude': df['longitude'].values[ok].astype(float)}
tr, te = ~test, test; print('places', len(X), 'train', tr.sum(), 'test', te.sum(), flush=True)
mu = X[tr].mean(0); Xtr = X[tr] - mu; Xte = X[te] - mu; XtX = Xtr.T @ Xtr
alpha = {}; XtY = {}
for t, y in Y.items():
    alpha[t] = float(RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xtr, y[tr]).alpha_); XtY[t] = Xtr.T @ (Y[t][tr] - Y[t][tr].mean())
def probe(Q=None):
    out = {}
    if Q is None: Pm = np.eye(300)
    else: Pm = np.eye(300) - Q @ Q.T
    A = Pm @ XtX @ Pm; Xtea = Xte @ Pm
    for t, y in Y.items():
        w = np.linalg.solve(A + alpha[t] * np.eye(300), Pm @ XtY[t]); pred = Xtea @ w + y[tr].mean()
        out[t] = float(1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[te].mean()) ** 2))
    return out
base = probe(); print('baseline', base, flush=True)
proper = np.load('propername_top50k.npy'); norm = np.linalg.norm(W, axis=1); decile = np.minimum(np.arange(V) * 10 // V, 9)
alpha_ok = np.array([w.isalpha() for w in words])
pools = {(d, p): np.where((decile == d) & (proper == p) & alpha_ok)[0] for d in range(10) for p in (False, True)}
def subspace(ix, k=None, var=0.90, cap=20):
    Vm = W[ix]; U, s, Vt = np.linalg.svd(Vm - Vm.mean(0), full_matrices=False)
    kk = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), var) + 1); kk = min(kk, cap, Vt.shape[0])
    if k is not None: kk = min(k, Vt.shape[0])
    return np.linalg.qr(Vt[:kk].T)[0]
Xc = X - X.mean(0); tot = np.sum(Xc ** 2)
def var_removed(Q): return float(np.sum((Xc @ Q) ** 2) / tot)
out = {'n_places': int(len(X)), 'baseline': base}
for cat, ws in cats.items():
    ix = np.array([idx[w] for w in ws if w in idx]); Q = subspace(ix); k = Q.shape[1]
    b = probe(Q); res = {'n_words': int(len(ix)), 'k': int(k), 'drop': {t: base[t] - b[t] for t in Y}, 'var_removed': var_removed(Q), 'proper_share': float(proper[ix].mean())}
    nd, nv = [], []
    for _ in range(N_NULL):
        pick = []
        for i in ix:
            pool = pools[(decile[i], proper[i])]; cand = rng.choice(pool, min(20, len(pool)), replace=False)
            cand = [c for c in cand if c not in pick and c not in ix] or list(cand); pick.append(cand[int(np.argmin(np.abs(norm[cand] - norm[i])))])
        Qn = subspace(np.array(pick), k=k); nv.append(var_removed(Qn)); bn = probe(Qn); nd.append({t: base[t] - bn[t] for t in Y})
    nv = np.array(nv); res['null_var_removed'] = [float(nv.mean()), float(nv.std())]
    for t in Y:
        dn = np.array([x[t] for x in nd]); A = np.column_stack([np.ones(N_NULL), nv]); coef, *_ = np.linalg.lstsq(A, dn, rcond=None)
        rs = float(np.std(dn - A @ coef)); pred = float(coef[0] + coef[1] * res['var_removed'])
        res[t] = {'null_mean': float(dn.mean()), 'null_sd': float(dn.std()), 'z_matched': float((res['drop'][t] - dn.mean()) / dn.std()), 'p_empirical': float((np.sum(dn >= res['drop'][t]) + 1) / (N_NULL + 1)),
                  'drop_predicted_from_var': pred, 'z_beyond_var': float((res['drop'][t] - pred) / rs)}
    out[cat] = res; print(cat, json.dumps(res), flush=True)
json.dump(out, open('semantic_ablation_released.json', 'w'), indent=1); print('ALLDONE')
