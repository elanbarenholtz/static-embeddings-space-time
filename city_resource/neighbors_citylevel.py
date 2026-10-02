"""Neighbor-availability analysis at the city level (review round 2): each city's absolute probe
error and mean top-5 cosine similarity to its training neighbours are averaged over the outer
splits in which it was a test city; continent is partialled out and the correlation is computed
across cities, so each city contributes one observation."""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, os, numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
P = _SBROOT
ALPHAS = np.logspace(-2, 3, 20)
rows = json.load(open('run_new/cities272_country.json'))['cities']; n = len(rows)
T = {'Latitude': np.array([r[1] for r in rows], float), 'Longitude': np.array([r[2] for r in rows], float),
     'Temperature': np.array([r[3] for r in rows], float)}
conts = sorted({r[4] for r in rows}); cid = np.array([conts.index(r[4]) for r in rows])
need = {w for r in rows for w in r[0].split()}; G = {}
for line in open(f'{P}/gt_scale/data/glove.6B.300d.txt', encoding='utf8'):
    w, rest = line.split(' ', 1)
    if w in need: G[w] = np.array(rest.split(), np.float64)
    if len(G) == len(need): break
Xg = np.array([np.mean([G[w] for w in r[0].split() if w in G], 0) for r in rows])
A = np.load(f'{P}/mechanism/acts_pythia-2.8b.npz', allow_pickle=True)['acts']
best = {t: v['modal_layer'] for t, v in json.load(open('out_new/nested_cities.json'))['pythia-2.8b'].items()}
folds = [train_test_split(np.arange(n), test_size=0.2, random_state=s) for s in range(10)]
out = {}
for t, y in T.items():
    out[t] = {}
    for nm, F in [('pythia-2.8b', A[best[t]].astype(np.float64)), ('GloVe', Xg)]:
        Fn = F / np.linalg.norm(F, axis=1, keepdims=True); S = Fn @ Fn.T; np.fill_diagonal(S, -np.inf)
        err = [[] for _ in range(n)]; sim = [[] for _ in range(n)]
        for tr, te in folds:
            p = RidgeCV(alphas=ALPHAS).fit(F[tr], y[tr]).predict(F[te])
            for j, c in enumerate(te):
                err[c].append(abs(p[j] - y[c])); sim[c].append(np.sort(S[c, tr])[::-1][:5].mean())
        keep = [c for c in range(n) if err[c]]
        e = np.array([np.mean(err[c]) for c in keep]); s = np.array([np.mean(sim[c]) for c in keep]); g = cid[keep]
        D = np.column_stack([np.ones(len(e))] + [(g == k).astype(float) for k in range(1, len(conts))])
        re_ = e - D @ np.linalg.lstsq(D, e, rcond=None)[0]; rs_ = s - D @ np.linalg.lstsq(D, s, rcond=None)[0]
        r = float(np.corrcoef(re_, rs_)[0, 1]); df = len(e) - len(conts) - 1
        tt = r * np.sqrt(df / (1 - r ** 2)); out[t][nm] = {'r': r, 'p': float(2 * stats.t.sf(abs(tt), df)), 'n_cities': len(e)}
    print(t, out[t], flush=True)
json.dump(out, open('out_new/neighbors_citylevel.json', 'w'), indent=1)
