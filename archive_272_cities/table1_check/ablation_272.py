"""
Semantic subspace ablation, rerun on the 272-city dataset with a correctly
specified ridge probe (intercept fitted).

Same procedure as semantic_ablation.py:
  - for each semantic category, PCA the category word vectors, keep components
    covering 90% of variance (cap 20 dims)
  - project city embeddings onto that subspace and subtract
  - re-probe, measure R^2 drop vs the unablated baseline
  - compare against 100 random orthonormal subspaces of the same dimensionality

Difference: RidgeCV with intercept, and R^2 is the mean over 10 train/test
splits rather than a single split, so drops are not split lottery.
"""
import json, numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
N_SPLITS = 10
N_RANDOM = 100
ALPHAS = np.logspace(-2, 3, 20)

cities = [tuple(c) for c in json.load(open('cities272.json'))['cities']]
categories = json.load(open('categories.json'))

glove = {}
for fn in ['vecs272.txt', 'catvecs.txt']:
    for line in open(fn):
        p = line.rstrip('\n').split(' ')
        glove[p[0]] = np.array(p[1:], dtype=np.float64)


def gvec(name):
    v = [glove[w] for w in name.split() if w in glove]
    return np.mean(v, axis=0) if v else None


valid = [c for c in cities if gvec(c[0]) is not None]
X = np.array([gvec(c[0]) for c in valid])
targets = {
    'Latitude':    np.array([c[1] for c in valid], dtype=float),
    'Longitude':   np.array([c[2] for c in valid], dtype=float),
    'Temperature': np.array([c[3] for c in valid], dtype=float),
}
n = len(valid)
print(f'{n} cities, {X.shape[1]} dims\n', flush=True)


def probe(Xin, y):
    """mean R^2 over N_SPLITS train/test splits, RidgeCV with intercept"""
    out = []
    for s in range(N_SPLITS):
        Xtr, Xte, ytr, yte = train_test_split(Xin, y, test_size=0.2, random_state=s)
        m = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
        out.append(r2_score(yte, m.predict(Xte)))
    return float(np.mean(out))


def principal_subspace(words, var_target=0.90, cap=20):
    V = np.array([glove[w] for w in words if w in glove])
    Vc = V - V.mean(0)
    U, S, Vt = np.linalg.svd(Vc, full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2)
    k = int(np.searchsorted(np.cumsum(ev), var_target) + 1)
    k = min(k, cap, Vt.shape[0])
    Q, _ = np.linalg.qr(Vt[:k].T)          # orthonormal basis, d x k
    return Q, k, len(V)


def ablate(X, Q):
    return X - (X @ Q) @ Q.T


baseline = {t: probe(X, y) for t, y in targets.items()}
print('baseline (10-split mean R^2, intercept fitted):')
for t, v in baseline.items():
    print(f'  {t:12s} {v:.3f}')
print(flush=True)

# random subspace null, one distribution per dimensionality we need
dims_needed = {}
cat_info = {}
for cat, words in categories.items():
    Q, k, nw = principal_subspace(words)
    cat_info[cat] = dict(Q=Q, k=k, n_words=nw)
    dims_needed.setdefault(k, []).append(cat)

random_null = {}
d = X.shape[1]
for k in sorted(dims_needed):
    print(f'random control, k={k} ({N_RANDOM} draws)...', flush=True)
    drops = {t: [] for t in targets}
    for _ in range(N_RANDOM):
        A = rng.standard_normal((d, k))
        Qr, _ = np.linalg.qr(A)
        Xa = ablate(X, Qr)
        for t, y in targets.items():
            drops[t].append(baseline[t] - probe(Xa, y))
    random_null[k] = {t: (float(np.mean(v)), float(np.std(v))) for t, v in drops.items()}

results = {'n_cities': n, 'baseline': baseline, 'categories': {}}
print(f'\n{"category":22s} {"k":>3s} {"words":>6s} | '
      + '  '.join(f'{t:>26s}' for t in targets))
print('-' * 110)
for cat, info in cat_info.items():
    Xa = ablate(X, info['Q'])
    row = {'k': info['k'], 'n_words': info['n_words'], 'targets': {}}
    cells = []
    for t, y in targets.items():
        r2a = probe(Xa, y)
        drop = baseline[t] - r2a
        mu, sd = random_null[info['k']][t]
        z = (drop - mu) / sd if sd > 0 else float('nan')
        row['targets'][t] = dict(r2_ablated=r2a, drop=drop, z=z,
                                 random_drop_mean=mu, random_drop_sd=sd)
        cells.append(f'R2={r2a:+.3f} d={drop:+.3f} z={z:5.1f}')
    results['categories'][cat] = row
    print(f'{cat:22s} {info["k"]:3d} {info["n_words"]:6d} | ' + '  '.join(cells), flush=True)

json.dump(results, open('ablation_272_results.json', 'w'), indent=1)
print('\nwrote ablation_272_results.json')
