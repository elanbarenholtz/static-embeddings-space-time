"""Training-only layer selection for Pythia on the 272 cities (review round 2).
In each of the ten outer 80/20 splits, every layer is scored by 5-fold CV on the outer training
rows only; the layer with the best inner-CV R^2 is refit on the outer training rows and scored
once on the outer test rows. Reports the mean outer R^2 and within-continent r of this procedure,
the layers chosen, and the modal layer (used for the single-layer analyses)."""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, os, numpy as np, pandas as pd
from collections import Counter
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
ALPHAS = np.logspace(-2, 3, 20)
rows = json.load(open('run_new/cities272_country.json'))['cities']
T = {'Latitude': np.array([r[1] for r in rows], float), 'Longitude': np.array([r[2] for r in rows], float),
     'Temperature': np.array([r[3] for r in rows], float)}
cont = np.array([r[4] for r in rows]); n = len(rows)
def within_r(p, y, g):
    d = pd.DataFrame({'p': p, 'y': y, 'g': g}); d = d[d.groupby('g').y.transform('size') >= 2]
    return float(np.corrcoef(d.p - d.groupby('g').p.transform('mean'), d.y - d.groupby('g').y.transform('mean'))[0, 1])
P = _SBROOT
acts = {'pythia-2.8b': np.load(f'{P}/mechanism/acts_pythia-2.8b.npz', allow_pickle=True)['acts'],
        'pythia-1.4b': np.load(_os_sb.path.join(_SBROOT, 'mechanism', 'acts_cities_pythia-1.4b.npz'), allow_pickle=True)['acts']}
out = {}
for m, A in acts.items():
    out[m] = {}
    for t, y in T.items():
        r2s, wrs, chosen = [], [], []
        for s in range(10):
            tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=s)
            inner = list(KFold(5, shuffle=True, random_state=s).split(tr))
            sc = []
            for L in range(A.shape[0]):
                X = A[L].astype(np.float64)
                sc.append(np.mean([r2_score(y[tr[b]], RidgeCV(alphas=ALPHAS).fit(X[tr[a]], y[tr[a]]).predict(X[tr[b]])) for a, b in inner]))
            L = int(np.argmax(sc)); chosen.append(L)
            X = A[L].astype(np.float64); p = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr]).predict(X[te])
            r2s.append(r2_score(y[te], p)); wrs.append(within_r(p, y[te], cont[te]))
        out[m][t] = {'r2': float(np.mean(r2s)), 'sd': float(np.std(r2s)), 'within_r': float(np.mean(wrs)),
                     'layers': chosen, 'modal_layer': Counter(chosen).most_common(1)[0][0]}
        print(m, t, out[m][t], flush=True)
json.dump(out, open('out_new/nested_cities.json', 'w'), indent=1)
print('ALLDONE')
