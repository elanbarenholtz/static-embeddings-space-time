"""Targeted decomposition on G&T data for a few representations.

For each rep: R^2 and within-group ordering r at two grain sizes
(places: continent, country; figures: century, decade), on the GloVe-covered
test items, plus Pythia reduced to 300 PCs (fit on train) to match GloVe's dimension.
Usage: python decompose.py <dataset> <pythia_layer>
"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')

DS, LAYER = sys.argv[1], int(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else 'pythia-2.8b'
SHORT = 'pythia' if MODEL.startswith('pythia') else 'llama'
ALPHAS = np.logspace(-2, 5, 29)
df = pd.read_csv(f'data/{DS}.csv')
test = df['is_test'].astype(str).str.lower().eq('true').values
if DS == 'world_place':
    targets = ['latitude', 'longitude']
    import country_converter as coco
    ctry = df['country'].fillna('').str.replace('_', ' ')
    uniq = ctry.unique().tolist()
    conv = coco.CountryConverter().convert(uniq, to='continent', not_found=None)
    conv = [c[0] if isinstance(c, list) else c for c in conv]
    groups = {'continent': ctry.map(dict(zip(uniq, conv))).fillna('unk').values,
              'country': ctry.values}
else:
    targets = ['death_year']
    groups = {'century': df['death_century'].values,
              'decade': (df['death_year'] // 10).values}
Y = df[targets].values.astype(float)
ok = np.isfinite(Y).all(1) & (np.load(f'static_GloVe_{DS}.npz')['n_in_vocab'] > 0)
tr, te = ok & ~test, ok & test

def within_r(pred, true, g):
    d = pd.DataFrame({'p': pred, 't': true, 'g': g})
    d = d[d.groupby('g')['t'].transform('size') >= 2]
    p = d['p'] - d.groupby('g')['p'].transform('mean')
    t = d['t'] - d.groupby('g')['t'].transform('mean')
    return float(np.corrcoef(p, t)[0, 1]), int(len(d))

meta = json.load(open(f'meta_{DS}_{MODEL}.json'))
A = np.load(f'acts_{DS}_{MODEL}.npy', mmap_mode='r')
reps = {
    'GloVe': np.load(f'static_GloVe_{DS}.npz')['vecs'],
    'Word2Vec': np.load(f'static_Word2Vec_{DS}.npz')['vecs'],
    f'{SHORT} L0': np.asarray(A[meta['layers'].index(0)], np.float32),
    f'{SHORT} L{LAYER}': np.asarray(A[meta['layers'].index(LAYER)], np.float32),
}
rows, preds = [], {}
for name, X in list(reps.items()) + [('pca300', None)]:
    if name == 'pca300':
        X = reps[f'{SHORT} L{LAYER}']
        pca = PCA(300, random_state=0).fit(X[tr])
        X = pca.transform(X); name = f'{SHORT} L{LAYER} (300 PCs)'
    m = RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr])
    P = m.predict(X[te]).reshape(int(te.sum()), -1)
    preds[name] = P
    row = {'rep': name, 'n_test': int(te.sum())}
    for k, t in enumerate(targets):
        row[f'r2_{t}'] = round(float(r2_score(Y[te][:, k], P[:, k])), 3)
        for gname, g in groups.items():
            r, n = within_r(P[:, k], Y[te][:, k], g[te])
            row[f'within_{gname}_{t}'] = round(r, 3)
    rows.append(row); print(row, flush=True)
np.savez(f'preds_{DS}.npz' if SHORT == 'pythia' else f'preds_{DS}_{SHORT}.npz', y=Y[te], idx=np.where(te)[0], **{k.replace(' ', '_'): v for k, v in preds.items()})
json.dump(rows, open(f'decompose_{DS}_{SHORT}.json', 'w'), indent=1)
