"""Ridge probes on G&T entity datasets: R^2, group-only baseline, within-group ordering.

Usage: python probe.py <dataset> <rep> [<rep> ...]
  rep = GloVe | Word2Vec | pythia-2.8b   (pythia runs every saved layer)
Evaluates on the G&T train/test split, restricted to items covered by GloVe (>=1 word)
so every representation is scored on the same items; Pythia is also scored on all items.
"""
import sys, json, warnings, os
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

DS = sys.argv[1]
REPS = sys.argv[2:]
ALPHAS = np.logspace(-2, 5, 29)
df = pd.read_csv(f'data/{DS}.csv')
test = df['is_test'].astype(str).str.lower().eq('true').values

if DS == 'world_place':
    targets = ['latitude', 'longitude']
    import country_converter as coco
    cc = coco.CountryConverter()
    ctry = df['country'].fillna('').str.replace('_', ' ')
    uniq = ctry.unique().tolist()
    conv = cc.convert(uniq, to='continent', not_found=None)
    conv = [c[0] if isinstance(c, list) else c for c in conv]
    cmap = dict(zip(uniq, conv))
    group = ctry.map(cmap).fillna('NA_unknown').values
else:
    targets = ['death_year']
    group = df['death_century'].values
Y = df[targets].values.astype(float)
ok_y = np.isfinite(Y).all(1)
cov = np.load(f'static_GloVe_{DS}.npz')['n_in_vocab'] > 0

def within_r(pred, true, g):
    d = pd.DataFrame({'p': pred, 't': true, 'g': g})
    d = d[d.groupby('g')['t'].transform('size') >= 2]
    p = d['p'] - d.groupby('g')['p'].transform('mean')
    t = d['t'] - d.groupby('g')['t'].transform('mean')
    return float(np.corrcoef(p, t)[0, 1])

def evaluate(X, name, mask):
    tr, te = mask & ~test & ok_y, mask & test & ok_y
    m = RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr])
    P = m.predict(X[te]).reshape(int(te.sum()), -1)
    row = {'dataset': DS, 'rep': name, 'subset': 'glove_covered' if mask is cov else 'all',
           'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'alpha': float(m.alpha_),
           'r2_avg': float(r2_score(Y[te], P))}
    for k, t in enumerate(targets):
        row[f'r2_{t}'] = float(r2_score(Y[te][:, k], P[:, k]))
        row[f'within_r_{t}'] = within_r(P[:, k], Y[te][:, k], group[te])
    return row

def group_baseline(mask):
    tr, te = mask & ~test & ok_y, mask & test & ok_y
    G = pd.get_dummies(pd.Series(group)).values.astype(float)
    m = RidgeCV(alphas=ALPHAS).fit(G[tr], Y[tr]); P = m.predict(G[te]).reshape(int(te.sum()), -1)
    row = {'dataset': DS, 'rep': 'group one-hot', 'subset': 'glove_covered' if mask is cov else 'all',
           'n_test': int(te.sum()), 'r2_avg': float(r2_score(Y[te], P))}
    for k, t in enumerate(targets):
        row[f'r2_{t}'] = float(r2_score(Y[te][:, k], P[:, k]))
        yt = Y[te][:, k]; gm = pd.Series(yt).groupby(group[te]).transform('mean')
        row[f'eta2_{t}'] = float(1 - np.sum((yt - gm) ** 2) / np.sum((yt - yt.mean()) ** 2))
    return row

rows = [group_baseline(cov), group_baseline(np.ones(len(df), bool))]
for rep in REPS:
    if rep in ('GloVe', 'Word2Vec'):
        X = np.load(f'static_{rep}_{DS}.npz')['vecs']
        rows.append(evaluate(X, rep, cov))
    else:
        meta = json.load(open(f'meta_{DS}_{rep}.json'))
        A = np.load(f'acts_{DS}_{rep}.npy', mmap_mode='r')
        want = [int(x) for x in os.environ.get('LAYERS', '').split(',') if x] or meta['layers']
        masks = (cov,) if os.environ.get('COVONLY') else (cov, np.ones(len(df), bool))
        for j, L in enumerate(meta['layers']):
            if L not in want: continue
            X = np.asarray(A[j], dtype=np.float32)
            for mask in masks:
                r = evaluate(X, f'{rep} L{L}', mask); r['layer'] = L; rows.append(r)
            print(rows[-len(masks)], flush=True)
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
out = f'results_{DS}.json'
old = json.load(open(out)) if os.path.exists(out) else []
old = [o for o in old if not any(o['rep'] == r['rep'] and o['subset'] == r['subset'] for r in rows)]
json.dump(old + rows, open(out, 'w'), indent=1)
