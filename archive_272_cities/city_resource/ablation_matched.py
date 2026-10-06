"""Semantic subspace ablation with matched random word-set controls (review round 2).
For each category (table1_check/categories.json), 200 random word sets are drawn from the 50,000
most frequent GloVe words, matched to the category on: number of in-vocabulary words; frequency
(each category word is replaced by a word from the same GloVe frequency-rank decile); proper-name
status (whether the capitalized form is more frequent than the lowercase form in the GoogleNews
vocabulary); and vector norm (within the decile, the candidate with the closest norm among 20
drawn at random). Each random set's subspace is built exactly as the category's (PCA, 90% of
variance, capped at 20) and truncated or padded to the category's dimensionality k. For every
subspace we record the drop in ten-split R^2 and the share of the city embeddings' variance it
removes. Reported: z of the category drop against the matched-word-set null, and the category
drop minus the drop predicted, from the null sets, by the share of city variance removed
(linear fit), in null-residual s.d. units."""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, os, sys, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
P = _SBROOT
sys.path.insert(0, f'{P}/review_v2')
ALPHAS = np.logspace(-2, 3, 20); N_NULL = 200; rng = np.random.default_rng(0)
rows = json.load(open('run_new/cities272.json'))['cities']
cats = json.load(open(f'{P}/table1_check/categories.json'))
V = 50000; words, W = [], []; need = {t for r in rows for t in r[0].split()} | {w for v in cats.values() for w in v}; G = {}
for i, line in enumerate(open(f'{P}/gt_scale/data/glove.6B.300d.txt', encoding='utf8')):
    w, r = line.rstrip().split(' ', 1)
    if i < V: words.append(w); W.append(np.array(r.split(), np.float64))
    if w in need: G[w] = np.array(r.split(), np.float64)
W = np.stack(W); idx = {w: i for i, w in enumerate(words)}
X = np.array([np.mean([G[t] for t in r[0].split() if t in G], 0) for r in rows])
T = {'Latitude': np.array([r[1] for r in rows], float), 'Longitude': np.array([r[2] for r in rows], float),
     'Temperature': np.array([r[3] for r in rows], float)}
splits = [train_test_split(np.arange(len(rows)), test_size=0.2, random_state=s) for s in range(10)]
def probe(F, y):
    return float(np.mean([r2_score(y[te], RidgeCV(alphas=ALPHAS).fit(F[tr], y[tr]).predict(F[te])) for tr, te in splits]))
base = {t: probe(X, y) for t, y in T.items()}
# proper-name status from GoogleNews vocabulary order (cached)
cache = 'cache/propername.npy'
if os.path.exists(cache):
    proper = np.load(cache)
else:
    import gzip
    need = set(words) | {w.capitalize() for w in words}; rank = {}
    with gzip.open(f'{P}/gt_scale/data/w2v.gz', 'rb') as f:
        n, d = map(int, f.readline().split())
        for i in range(n):
            wb = bytearray()
            while True:
                c = f.read(1)
                if c == b' ': break
                if c != b'\n': wb += c
            f.read(4 * d); w = wb.decode('utf8', errors='ignore')
            if w in need and w not in rank: rank[w] = i
    proper = np.array([rank.get(w.capitalize(), 10**9) < rank.get(w, 10**9) for w in words])
    np.save(cache, proper)
norm = np.linalg.norm(W, axis=1); decile = np.minimum(np.arange(V) * 10 // V, 9)
pools = {(d, p): np.where((decile == d) & (proper == p) & np.array([w.isalpha() for w in words]))[0] for d in range(10) for p in (False, True)}
def subspace(ix, k=None, var=0.90, cap=20):
    Vm = W[ix]; U, s, Vt = np.linalg.svd(Vm - Vm.mean(0), full_matrices=False)
    kk = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), var) + 1); kk = min(kk, cap, Vt.shape[0])
    if k is not None: kk = min(k, Vt.shape[0])
    return np.linalg.qr(Vt[:kk].T)[0]
def city_var_removed(Q):
    Xc = X - X.mean(0); return float(np.sum((Xc @ Q) ** 2) / np.sum(Xc ** 2))
def drops(Q):
    Xa = X - (X @ Q) @ Q.T; return {t: base[t] - probe(Xa, y) for t, y in T.items()}
out = {'baseline': base}
for cat, ws in cats.items():
    ix = np.array([idx[w] for w in ws if w in idx]); Q = subspace(ix); k = Q.shape[1]
    res = {'n_words': int(len(ix)), 'k': int(k), 'drop': drops(Q), 'city_var_removed': city_var_removed(Q),
           'proper_share': float(proper[ix].mean())}
    nd, nv = [], []
    for b in range(N_NULL):
        pick = []
        for i in ix:
            pool = pools[(decile[i], proper[i])]; cand = rng.choice(pool, min(20, len(pool)), replace=False)
            cand = [c for c in cand if c not in pick and c not in ix] or list(cand)
            pick.append(cand[int(np.argmin(np.abs(norm[cand] - norm[i])))])
        Qn = subspace(np.array(pick), k=k); nv.append(city_var_removed(Qn)); nd.append(drops(Qn))
    nv = np.array(nv); res['null_city_var_removed'] = [float(nv.mean()), float(nv.std())]
    for t in T:
        dn = np.array([x[t] for x in nd]); res[t] = {'null_mean': float(dn.mean()), 'null_sd': float(dn.std()),
                                                       'z_matched': float((res['drop'][t] - dn.mean()) / dn.std()),
                                                       'p_matched_empirical': float((np.sum(dn >= res['drop'][t]) + 1) / (N_NULL + 1))}
        A = np.column_stack([np.ones(N_NULL), nv]); coef, *_ = np.linalg.lstsq(A, dn, rcond=None)
        resid_sd = float(np.std(dn - A @ coef)); pred = float(coef[0] + coef[1] * res['city_var_removed'])
        res[t]['drop_predicted_from_var'] = pred; res[t]['z_beyond_var'] = float((res['drop'][t] - pred) / resid_sd)
    out[cat] = res
    print(cat, json.dumps(res), flush=True)
    json.dump(out, open('out_new/ablation_matched.json', 'w'), indent=1)
print('ALLDONE')
