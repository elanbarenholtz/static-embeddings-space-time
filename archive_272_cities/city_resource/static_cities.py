"""Static-embedding analyses on the 272-city set that are not covered by the original
table1_check scripts: eta^2, R^2-vs-eta^2, continent-centroid decomposition, within-continent
ordering, leave-one-continent-out, other-target group dependence, semantic word correlations,
composite scores, shared-company words, combined-subspace ablation. Also recomputes Table 1
and the one-hot baselines so they can be checked against the original scripts.

Usage: PYTHONPATH=shim python static_cities.py <cities.json> <tag>"""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, os, sys, re, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
P = _SBROOT
DATA, TAG = sys.argv[1], sys.argv[2]
OUT = f'out_{TAG}'; os.makedirs(OUT, exist_ok=True)
ALPHAS = np.logspace(-2, 3, 20)
raw = json.load(open(DATA))['cities']
rows = [[c[0].lower()] + list(c[1:]) for c in raw]
n = len(rows)
# ---------- embeddings (as in table1_check/table1_272.py)
V = 50000
vocab, Wv = [], []
need = {t for r in rows for t in r[0].split()}
G = {}
with open(f'{P}/gt_scale/data/glove.6B.300d.txt', encoding='utf8') as f:
    for i, line in enumerate(f):
        w, rest = line.rstrip().split(' ', 1)
        if i < V: vocab.append(w); Wv.append(np.array(rest.split(), np.float32))
        if w in need: G[w] = np.array(rest.split(), np.float64)
        if i >= V and len(G) == len(need): break
Wv = np.stack(Wv); widx = {w: i for i, w in enumerate(vocab)}
import gensim.downloader as api
W2 = api.load('word2vec-google-news-300')
def gv(nm): return np.mean([G[w] for w in nm.split() if w in G], 0)
def wv(nm):
    ph = '_'.join(w.capitalize() for w in nm.split())
    if ph in W2: return np.asarray(W2[ph], np.float64)
    f = [w.capitalize() if w.capitalize() in W2 else w for w in nm.split() if w.capitalize() in W2 or w in W2]
    return np.mean([W2[w] for w in f], 0)
X = {'GloVe': np.array([gv(r[0]) for r in rows]), 'Word2Vec': np.array([wv(r[0]) for r in rows])}
T = {'Longitude': np.array([r[2] for r in rows], float), 'Latitude': np.array([r[1] for r in rows], float),
     'Temperature': np.array([r[3] for r in rows], float),
     'GDP per capita': np.log10(np.array([r[7] for r in rows], float)),
     'Population': np.log10(np.array([r[8] for r in rows], float)),
     'Elevation': np.array([r[6] for r in rows], float), 'Year Founded': np.array([r[5] for r in rows], float)}
T = {k: v for k, v in T.items() if np.std(v) > 0}          # founding year dropped from the new data
cont = np.array([r[4] for r in rows]); conts = sorted(set(cont)); cid = np.array([conts.index(c) for c in cont])
ctry = np.array([str(r[9]) if len(r) > 9 else str(r[7]) for r in rows])     # country (new) or GDP block (old)
splits = [train_test_split(np.arange(n), test_size=0.2, random_state=s) for s in range(10)]
fit = lambda A, y: RidgeCV(alphas=ALPHAS).fit(A, y)

def within_r(p, y, g):
    d = pd.DataFrame({'p': p, 'y': y, 'g': g}); d = d[d.groupby('g').y.transform('size') >= 2]
    return float(np.corrcoef(d.p - d.groupby('g').p.transform('mean'), d.y - d.groupby('g').y.transform('mean'))[0, 1])
def ten(F, y, wr=False):
    r2, w = [], []
    for tr, te in splits:
        p = fit(F[tr], y[tr]).predict(F[te]); r2.append(r2_score(y[te], p))
        if wr: w.append(within_r(p, y[te], cid[te]))
    return (float(np.mean(r2)), float(np.std(r2))) + ((float(np.mean(w)),) if wr else ())
def eta2(y, g):
    m = y.mean(); return float(sum(((y[g == k].mean() - m) ** 2) * (g == k).sum() for k in np.unique(g)) / ((y - m) ** 2).sum())

res = {'n': n, 'n_countries': int(len(set(ctry)))}
# ---------- Table 1 + within-continent r
res['table1'] = {m: {t: ten(F, y, wr=True) for t, y in T.items()} for m, F in X.items()}
for t in T: print(f"{t:15s} GloVe {res['table1']['GloVe'][t]}  W2V {res['table1']['Word2Vec'][t]}", flush=True)
# ---------- one-hots
C1 = np.eye(len(conts))[cid]; cu = sorted(set(ctry)); K1 = np.eye(len(cu))[[cu.index(c) for c in ctry]]
res['onehot'] = {'continent': {t: ten(C1, T[t])[0] for t in T}, 'country': {t: ten(K1, T[t])[0] for t in T}}
print('onehot', res['onehot'], flush=True)
# ---------- eta^2 and R^2 vs eta^2
res['eta2'] = {t: eta2(y, cid) for t, y in T.items()}
res['eta2_country'] = {t: eta2(y, np.array([cu.index(c) for c in ctry])) for t, y in T.items()}
tl = list(T); e = np.array([res['eta2'][t] for t in tl])
for m in X:
    r2 = np.array([res['table1'][m][t][0] for t in tl])
    res.setdefault('r2_vs_eta2', {})[m] = {'pearson': float(stats.pearsonr(e, r2)[0]), 'spearman': float(stats.spearmanr(e, r2)[0])}
    noT = [i for i, t in enumerate(tl) if t != 'Temperature']
    b = np.polyfit(e[noT], r2[noT], 1); res['r2_vs_eta2'][m]['temp_excess_fit_without_temp'] = float(r2[tl.index('Temperature')] - np.polyval(b, e[tl.index('Temperature')]))
    b2 = np.polyfit(e, r2, 1); res['r2_vs_eta2'][m]['temp_excess_fit_all'] = float(r2[tl.index('Temperature')] - np.polyval(b2, e[tl.index('Temperature')]))
    res['r2_vs_eta2'][m]['pearson_without_temp'] = float(stats.pearsonr(e[noT], r2[noT])[0])
both_e = np.concatenate([e, e]); both_r = np.concatenate([[res['table1'][m][t][0] for t in tl] for m in X])
res['r2_vs_eta2']['pooled_pearson'] = float(stats.pearsonr(both_e, both_r)[0])
mean_r = np.mean([[res['table1'][m][t][0] for t in tl] for m in X], 0)
res['r2_vs_eta2']['meanmodel_pearson'] = float(stats.pearsonr(e, mean_r)[0])
noT = [i for i, t in enumerate(tl) if t != 'Temperature']
res['r2_vs_eta2']['pooled_pearson_without_temp'] = float(stats.pearsonr(np.concatenate([e[noT]] * 2), np.concatenate([[res['table1'][m][t][0] for t in np.array(tl)[noT]] for m in X]))[0])
res['r2_vs_eta2']['meanmodel_pearson_without_temp'] = float(stats.pearsonr(e[noT], mean_r[noT])[0])
print('eta2', res['eta2'], res['r2_vs_eta2'], flush=True)
# ---------- continent-centroid decomposition
def decomp(F, y):
    cen, resd = [], []
    for tr, te in splits:
        mu = {k: F[tr][cid[tr] == k].mean(0) for k in np.unique(cid[tr])}
        Fc = np.array([mu[k] for k in cid]); Fr = F - Fc
        cen.append(r2_score(y[te], fit(Fc[tr], y[tr]).predict(Fc[te])))
        resd.append(r2_score(y[te], fit(Fr[tr], y[tr]).predict(Fr[te])))
    return float(np.mean(cen)), float(np.mean(resd))
res['decomp'] = {m: {t: decomp(F, y) for t, y in T.items()} for m, F in X.items()}
print('decomp', res['decomp'], flush=True)
# ---------- leave-one-continent-out (pooled)
def loco(F, y):
    p = np.zeros(n)
    for k in np.unique(cid):
        te = cid == k; p[te] = fit(F[~te], y[~te]).predict(F[te])
    return float(r2_score(y, p))
res['loco'] = {m: {t: loco(F, T[t]) for t in ['Latitude', 'Longitude', 'Temperature']} for m, F in X.items()}
print('loco', res['loco'], flush=True)
# ---------- semantic word correlations with temperature (filter from data_driven_semantic.py)
src = open(f'{P}/data_driven_semantic.py').read()
seg = src[src.index('# FILTER'):src.index('valid_mask')]
ns = {'np': np, 'cities': rows}; exec(seg.split('def is_valid_word')[0], ns)
excl = set()
for k, v in ns.items():
    if k in ('country_words', 'geo_extra', 'city_name_words'): excl |= {x.lower() for x in v}
city_tokens = {t for r in rows for t in r[0].split()}
def valid_word(w, i):
    return i < 20000 and len(w) >= 4 and w.isalpha() and w not in excl and w not in city_tokens
ok = np.array([valid_word(w, i) for i, w in enumerate(vocab)])
Gn = X['GloVe'] / np.linalg.norm(X['GloVe'], axis=1, keepdims=True)
Wn = Wv / np.linalg.norm(Wv, axis=1, keepdims=True)
S = (Wn[ok] @ Gn.T)
def colcorr(S, y):
    Sc = S - S.mean(1, keepdims=True); yc = y - y.mean()
    return (Sc @ yc) / (np.linalg.norm(Sc, axis=1) * np.linalg.norm(yc))
wk = np.array(vocab)[ok]
res['semantic'] = {'n_words': int(ok.sum())}
for t in ['Temperature', 'Latitude']:
    r = colcorr(S, T[t]); o = np.argsort(r)
    res['semantic'][t] = {'top_pos': [(wk[i], float(r[i])) for i in o[::-1][:20]], 'top_neg': [(wk[i], float(r[i])) for i in o[:20]]}
    res['semantic'][t]['anchors'] = {w: float(r[list(wk).index(w)]) for w in ['malaria', 'jungle', 'dengue', 'skiing', 'orchestra', 'skating', 'humid', 'snow'] if w in set(wk)}
print('semantic', json.dumps(res['semantic'])[:1500], flush=True)
cw = Gn @ Wn[widx['cold']] - Gn @ Wn[widx['warm']]
res['composite_cold_warm'] = {'latitude': float(stats.pearsonr(cw, T['Latitude'])[0]), 'abs_latitude': float(stats.pearsonr(cw, np.abs(T['Latitude']))[0]),
                              'temperature': float(stats.pearsonr(cw, T['Temperature'])[0])}
print('composite', res['composite_cold_warm'], flush=True)
# ---------- shared company (linking_words.py logic, on all 272 cities)
okL = np.array([len(w) >= 4 and w.isalpha() and w not in city_tokens for w in vocab])
allmean = Gn.mean(0); order = np.argsort(T['Temperature']); warm, cold = order[-25:], order[:25]
def links(group, anchors, k=12):
    g = Gn[group].mean(0) - allmean
    s_city = Wn @ (g / np.linalg.norm(g)); s_anch = np.mean([Wn @ Wn[widx[a]] for a in anchors], 0)
    rc = np.argsort(np.argsort(-s_city)); ra = np.argsort(np.argsort(-s_anch))
    sc = np.maximum(rc, ra).astype(float); sc[~okL] = 1e9
    for a in anchors: sc[widx[a]] = 1e9
    top = np.argsort(sc)[:k]
    pr = {w: (int(rc[widx[w]]) + 1, int(ra[widx[w]]) + 1) for w in ['tropical', 'humid', 'monsoon', 'snow', 'winter', 'symphony'] if w in widx}
    return [(vocab[i], int(rc[i]) + 1, int(ra[i]) + 1) for i in top], pr
res['links'] = {'warm_cities': [rows[i][0] for i in warm[::-1]], 'cold_cities': [rows[i][0] for i in cold]}
res['links']['warm'], res['links']['warm_probe'] = links(warm, ['malaria', 'jungle'])
res['links']['cold_skiing'], res['links']['cold_probe'] = links(cold, ['skiing'])
res['links']['cold_orchestra'], _ = links(cold, ['orchestra'])
print('links', res['links'], flush=True)
# ---------- combined-subspace ablation (all six vocabularies)
cats = json.load(open(f'{P}/table1_check/categories.json'))
allw = sorted({w for v in cats.values() for w in v if w in widx})
def subspace(words, var=0.90, cap=None):
    Vm = np.array([Wv[widx[w]] for w in words], np.float64); U, s, Vt = np.linalg.svd(Vm - Vm.mean(0), full_matrices=False)
    k = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), var) + 1); k = min(k, cap or k, Vt.shape[0])
    return np.linalg.qr(Vt[:k].T)[0], k
rng = np.random.default_rng(0)
base = {t: ten(X['GloVe'], T[t])[0] for t in ['Latitude', 'Longitude', 'Temperature']}
res['ablation_combined'] = {'n_words': len(allw)}
for label, (Q, k) in {'union_pca90': subspace(allw), 'union_pca90_cap60': subspace(allw, cap=60)}.items():
    Xa = X['GloVe'] - (X['GloVe'] @ Q) @ Q.T
    rnd = {t: [] for t in base}
    for _ in range(100):
        Qr = np.linalg.qr(rng.standard_normal((300, k)))[0]; Xr = X['GloVe'] - (X['GloVe'] @ Qr) @ Qr.T
        for t in base: rnd[t].append(base[t] - ten(Xr, T[t])[0])
    res['ablation_combined'][label] = {'k': k, **{t: {'r2_ablated': ten(Xa, T[t])[0], 'drop': base[t] - ten(Xa, T[t])[0],
                                                     'rand_mean': float(np.mean(rnd[t])), 'rand_sd': float(np.std(rnd[t])),
                                                     'z': float((base[t] - ten(Xa, T[t])[0] - np.mean(rnd[t])) / np.std(rnd[t]))} for t in base}}
print('ablation combined', res['ablation_combined'], flush=True)
json.dump(res, open(f'{OUT}/static_cities.json', 'w'), indent=1)
print('done')
