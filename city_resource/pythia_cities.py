"""Pythia analyses on the 272-city set (Appendix: tab:llm, fig:llmlayers, tab:residual,
fig:residual, error correlations, neighbor dependence), following
mechanism/llm_residualisation.ipynb. Activations are those already extracted
(mechanism/acts_pythia-2.8b.npz, Downloads/acts_cities_pythia-1.4b.npz); only the
targets change.

Usage: PYTHONPATH=shim python pythia_cities.py <cities.json> <tag>"""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, os, sys, numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
P = _SBROOT
DATA, TAG = sys.argv[1], sys.argv[2]
OUT = f'out_{TAG}'; os.makedirs(OUT, exist_ok=True)
ALPHAS = np.logspace(-2, 3, 20); RESID_ALPHAS = np.logspace(0, 5, 12)
rows = [[c[0].lower()] + list(c[1:]) for c in json.load(open(DATA))['cities']]
A28 = np.load(f'{P}/mechanism/acts_pythia-2.8b.npz', allow_pickle=True)
A14 = np.load(_os_sb.path.join(_SBROOT, 'mechanism', 'acts_cities_pythia-1.4b.npz'), allow_pickle=True)
assert [r[0] for r in rows] == list(A28['names']) == list(A14['names'])

# static vectors, as in the notebook
need = {t for r in rows for t in r[0].split()}
G = {}
for line in open(f'{P}/gt_scale/data/glove.6B.300d.txt', encoding='utf8'):
    w, rest = line.split(' ', 1)
    if w in need: G[w] = np.array(rest.split(), np.float64)
import gensim.downloader as api
W = api.load('word2vec-google-news-300')
def gv(nm):
    v = [G[t] for t in nm.split() if t in G]; return np.mean(v, 0) if v else None
def wv(nm):
    ph = '_'.join(t.capitalize() for t in nm.split())
    if ph in W: return np.asarray(W[ph], np.float64)
    f = [t.capitalize() if t.capitalize() in W else t.lower() for t in nm.split() if t.capitalize() in W or t.lower() in W]
    return np.mean([W[t] for t in f], 0).astype(np.float64) if f else None
keep = [i for i, r in enumerate(rows) if gv(r[0]) is not None and wv(r[0]) is not None]
assert len(keep) == len(rows), len(keep)
STATIC = {'GloVe': np.array([gv(r[0]) for r in rows]), 'Word2Vec': np.array([wv(r[0]) for r in rows])}
n = len(rows)
targets = {'Latitude': np.array([r[1] for r in rows], float), 'Longitude': np.array([r[2] for r in rows], float),
           'Temperature': np.array([r[3] for r in rows], float)}
cont = [r[4] for r in rows]; conts = sorted(set(cont)); cid = np.array([conts.index(c) for c in cont])
folds = [train_test_split(np.arange(n), test_size=.2, random_state=s) for s in range(10)]
fit = lambda X, y: RidgeCV(alphas=ALPHAS).fit(X, y)

def within_r(p, y, g):
    d = pd.DataFrame({'p': p, 'y': y, 'g': g}); d = d[d.groupby('g').y.transform('size') >= 2]
    return float(np.corrcoef(d.p - d.groupby('g').p.transform('mean'), d.y - d.groupby('g').y.transform('mean'))[0, 1])

def probe(F, y, wr=False):
    r2, w = [], []
    for tr, te in folds:
        p = fit(F[tr], y[tr]).predict(F[te]); r2.append(r2_score(y[te], p))
        if wr: w.append(within_r(p, y[te], cid[te]))
    out = {'r2': float(np.mean(r2)), 'sd': float(np.std(r2))}
    if wr: out['within_r'] = float(np.mean(w))
    return out

def probe_pca(F, y, k):
    r2 = []
    for tr, te in folds:
        pc = PCA(n_components=k).fit(F[tr]); r2.append(r2_score(y[te], fit(pc.transform(F[tr]), y[tr]).predict(pc.transform(F[te]))))
    return float(np.mean(r2))

res = {'n': n}
# ---- static and continent references
res['static'] = {s: {t: probe(X, y, wr=True) for t, y in targets.items()} for s, X in STATIC.items()}
C1 = np.eye(len(conts))[cid]
res['continent_onehot'] = {t: probe(C1, y)['r2'] for t, y in targets.items()}
# ---- layer sweeps
res['layers'] = {}
for tag, npz in [('pythia-1.4b', A14), ('pythia-2.8b', A28)]:
    acts = npz['acts']; L = acts.shape[0]; res['layers'][tag] = {}
    for t, y in targets.items():
        res['layers'][tag][t] = [probe(acts[l].astype(np.float64), y, wr=True) for l in range(L)]
        b = int(np.argmax([x['r2'] for x in res['layers'][tag][t]]))
        print(tag, t, 'best L', b, res['layers'][tag][t][b], 'L0', res['layers'][tag][t][0]['r2'], flush=True)
best = {m: {t: int(np.argmax([x['r2'] for x in res['layers'][m][t]])) for t in targets} for m in res['layers']}
res['best_layer'] = best
# ---- matched dimensionality (200 PCs fitted on training rows)
res['pca200'] = {'pythia-2.8b': {t: probe_pca(A28['acts'][best['pythia-2.8b'][t]].astype(np.float64), y, 200) for t, y in targets.items()},
                 'GloVe': {t: probe_pca(STATIC['GloVe'], y, 200) for t, y in targets.items()}}
print('pca200', res['pca200'], flush=True)

# ---- residualization
rng = np.random.default_rng(0); PERM = rng.permutation(n)
def split_parts(A, S, tr):
    Pr = RidgeCV(alphas=RESID_ALPHAS).fit(S[tr], A[tr]).predict(S); return Pr, A - Pr
def var_removed(A, R):
    Ac = A - A.mean(0); return float(1 - np.sum((R - R.mean(0)) ** 2) / np.sum(Ac ** 2))
def pca_matched(A, frac):
    Ac = A - A.mean(0); U, s, Vt = np.linalg.svd(Ac, full_matrices=False)
    j = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), frac) + 1); Q = Vt[:max(1, j)].T
    return A - (A @ Q) @ Q.T, j
def random_matched(A, frac, seed):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((A.shape[1], A.shape[1])))
    Ac = A - A.mean(0); pv = ((Ac @ Q) ** 2).sum(0); k = int(np.searchsorted(np.cumsum(pv) / pv.sum(), frac) + 1)
    return A - (A @ Q[:, :k]) @ Q[:, :k].T, k
def resid_layer(A, S, y_items):
    Sp = S[PERM]
    parts = [(tr, te) + split_parts(A, S, tr) + split_parts(A, Sp, tr) for tr, te in folds]
    vr = float(np.mean([var_removed(A, p[3]) for p in parts]))
    Apca, _ = pca_matched(A, vr); Arnd, _ = random_matched(A, vr, 7)
    out = {'var_removed': vr, 'var_removed_shuffled': float(np.mean([var_removed(A, p[5]) for p in parts]))}
    for t, y in y_items:
        o = {}
        for key, getF in [('full', lambda p: A), ('proj', lambda p: p[2]), ('resid', lambda p: p[3]),
                          ('resid_shuffled', lambda p: p[5]), ('null_pca', lambda p: Apca), ('null_random', lambda p: Arnd)]:
            o[key] = float(np.mean([r2_score(y[p[1]], fit(getF(p)[p[0]], y[p[0]]).predict(getF(p)[p[1]])) for p in parts]))
        out[t] = o
    return out
res['resid'] = {'pythia-2.8b': {}}
acts = A28['acts']
for s, S in STATIC.items():
    res['resid']['pythia-2.8b'][s] = []
    for l in range(acts.shape[0]):
        r = resid_layer(acts[l].astype(np.float64), S, list(targets.items()))
        res['resid']['pythia-2.8b'][s].append(r)
        print('resid 2.8b', s, l, round(r['var_removed'], 3), {t: {k: round(v, 3) for k, v in r[t].items()} for t in targets}, flush=True)
res['resid']['pythia-1.4b'] = {}
for s, S in STATIC.items():
    res['resid']['pythia-1.4b'][s] = {t: resid_layer(A14['acts'][best['pythia-1.4b'][t]].astype(np.float64), S, [(t, y)])
                                      for t, y in targets.items()}
print('resid 1.4b', res['resid']['pythia-1.4b'], flush=True)

# ---- error correlations and neighbor dependence (2.8B, best layer)
def _corr(e1, e2, yv, partial=True):
    if partial:
        D = np.column_stack([np.ones(len(yv)), yv])
        e1 = e1 - D @ np.linalg.lstsq(D, e1, rcond=None)[0]; e2 = e2 - D @ np.linalg.lstsq(D, e2, rcond=None)[0]
    return float(np.corrcoef(e1, e2)[0, 1]) if np.std(e1) > 1e-12 and np.std(e2) > 1e-12 else np.nan
def fold_err(F1, F2, y, within=False):
    acc = []
    for tr, te in folds:
        e1 = fit(F1[tr], y[tr]).predict(F1[te]) - y[te]; e2 = fit(F2[tr], y[tr]).predict(F2[te]) - y[te]
        if within:
            for k in range(len(conts)):
                m = cid[te] == k
                if m.sum() >= 5: acc.append((m.sum(), _corr(e1[m], e2[m], y[te][m])))
        else:
            acc.append((len(te), _corr(e1, e2, y[te])))
    acc = [(a, b) for a, b in acc if np.isfinite(b)]; w = np.array([a for a, _ in acc], float)
    return float(np.sum(w * np.array([b for _, b in acc])) / w.sum())
res['err_corr'] = {}; res['neighbors'] = {}
for t, y in targets.items():
    A = acts[best['pythia-2.8b'][t]].astype(np.float64)
    res['err_corr'][t] = {s: {'r': fold_err(A, S, y), 'within': fold_err(A, S, y, True),
                              'random_floor': fold_err(A, rng.normal(size=S.shape), y)} for s, S in STATIC.items()}
    res['err_corr'][t]['GloVe_vs_W2V'] = fold_err(STATIC['GloVe'], STATIC['Word2Vec'], y)
    res['neighbors'][t] = {}
    for nm, F in [('pythia-2.8b', A), ('GloVe', STATIC['GloVe'])]:
        Fn = F / np.linalg.norm(F, axis=1, keepdims=True); Sim = Fn @ Fn.T; np.fill_diagonal(Sim, -np.inf)
        e, m5, gc = [], [], []
        for tr, te in folds:
            p = fit(F[tr], y[tr]).predict(F[te])
            for j, c in enumerate(te):
                s_ = np.sort(Sim[c, tr])[::-1]; e.append(abs(p[j] - y[c])); m5.append(s_[:5].mean()); gc.append(cid[c])
        e, m5, gc = map(np.array, (e, m5, gc))
        D = np.column_stack([np.ones(len(e))] + [(gc == k).astype(float) for k in range(1, len(conts))])
        re_ = e - D @ np.linalg.lstsq(D, e, rcond=None)[0]; rs_ = m5 - D @ np.linalg.lstsq(D, m5, rcond=None)[0]
        r = float(np.corrcoef(re_, rs_)[0, 1]); tt = r * np.sqrt((len(e) - len(conts) - 1) / (1 - r ** 2))
        res['neighbors'][t][nm] = {'r': r, 'p': float(2 * stats.t.sf(abs(tt), len(e) - len(conts) - 1))}
print('err', res['err_corr'], '\nnb', res['neighbors'], flush=True)
json.dump(res, open(f'{OUT}/pythia_cities.json', 'w'), indent=1)

# ---- figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
col = {'Latitude': '#2171b5', 'Longitude': '#e6550d', 'Temperature': '#31a354'}
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, t in zip(axs, targets):
    m = np.array([x['r2'] for x in res['layers']['pythia-2.8b'][t]]); sd = np.array([x['sd'] for x in res['layers']['pythia-2.8b'][t]])
    ax.plot(m, color=col[t], lw=1.8); ax.fill_between(range(len(m)), m - sd, m + sd, color=col[t], alpha=0.15)
    ax.axhline(res['static']['Word2Vec'][t]['r2'], ls='--', color='#555', lw=1, label='Word2Vec')
    ax.set_title(t); ax.set_xlabel('Layer'); ax.set_ylim(-0.1, 1)
axs[0].set_ylabel('Probe $R^2$ (ten random 80/20 splits)'); axs[0].legend(frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/llm_layers_pythia-2.8b.png', dpi=300)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, t in zip(axs, targets):
    R = res['resid']['pythia-2.8b']['GloVe']
    ax.plot([r[t]['full'] for r in R], '-o', ms=2.5, color='#2563eb', label='full activations')
    ax.plot([r[t]['proj'] for r in R], '-', color='#16a34a', label='predictable from GloVe')
    ax.plot([r[t]['resid'] for r in R], '-s', ms=2.5, color='#dc2626', label='residual')
    ax.plot([r[t]['null_random'] for r in R], '--', color='#94a3b8', label='matched-variance null')
    ax.axhline(res['static']['GloVe'][t]['r2'], ls=':', color='#333', lw=1, label='GloVe')
    ax.axhline(0, color='#ccc', lw=0.8); ax.set_title(t); ax.set_xlabel('Layer'); ax.set_ylim(-0.3, 1)
axs[0].set_ylabel('Probe $R^2$'); axs[0].legend(frameon=False, fontsize=7.5, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/residualise_pythia-2.8b.png', dpi=300)
print('done')
