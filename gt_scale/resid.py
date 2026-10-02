"""Residualize Pythia-2.8B activations against static embeddings on G&T data.

A (activations at one layer) is split into P = S B (linear image of the static vectors,
B fit by ridge on training rows) and R = A - P. Each is probed as A is. Nulls:
variance-matched (remove a random subspace taking out the same fraction of variance)
and shuffled-static (static vectors assigned to the wrong items before fitting B).
Usage: python resid.py <dataset> <static: GloVe|Word2Vec> <layer>
"""
import sys, json, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
DS, SM, LAYER = sys.argv[1], sys.argv[2], int(sys.argv[3])
MODEL = sys.argv[4] if len(sys.argv) > 4 else 'pythia-2.8b'
ALPHAS = np.logspace(-2, 5, 29)
rng = np.random.default_rng(0)
df = pd.read_csv(f'data/{DS}.csv')
test = df['is_test'].astype(str).str.lower().eq('true').values
targets = ['latitude', 'longitude'] if DS == 'world_place' else ['death_year']
Y = df[targets].values.astype(float)
mask = np.isfinite(Y).all(1) & (np.load(f'static_GloVe_{DS}.npz')['n_in_vocab'] > 0)
St = np.load(f'static_{SM}_{DS}.npz')
mask &= St['n_in_vocab'] > 0
tr, te = mask & ~test, mask & test
S = St['vecs'].astype(np.float64)
meta = json.load(open(f'meta_{DS}_{MODEL}.json'))
A = np.load(f'acts_{DS}_{MODEL}.npy', mmap_mode='r')[meta['layers'].index(LAYER)]
A = np.asarray(A, dtype=np.float32)
A = A - A[tr].mean(0); S = S - S[tr].mean(0)

def probe(X):
    m = RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr])
    P = m.predict(X[te]).reshape(int(te.sum()), -1)
    return [round(float(r2_score(Y[te][:, k], P[:, k])), 3) for k in range(len(targets))]

def fit_map(Sx):
    alpha = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Sx[tr], A[tr, :256]).alpha_
    B = Ridge(alpha=alpha).fit(Sx[tr], A[tr])
    return (B.predict(Sx) - B.intercept_).astype(np.float32) + B.intercept_.astype(np.float32), alpha

def varfrac(R):
    return float(1 - R[tr].var(0).sum() / A[tr].var(0).sum())

out = {'dataset': DS, 'static': SM, 'layer': LAYER, 'n_train': int(tr.sum()), 'n_test': int(te.sum()), 'targets': targets}
out['static_direct'] = probe(S.astype(np.float32))
out['full'] = probe(A)
P, a = fit_map(S); R = A - P
out['map_alpha'] = float(a)
out['projection'] = probe(P)
out['residual'] = probe(R)
f = varfrac(R); out['var_removed'] = round(f, 3)
print(out, flush=True)
# variance-matched random subspace removal
Q, _ = np.linalg.qr(rng.standard_normal((A.shape[1], A.shape[1])).astype(np.float32))
proj_var = ((A[tr] @ Q) ** 2).mean(0) - (A[tr] @ Q).mean(0) ** 2
cum = np.cumsum(proj_var) / A[tr].var(0).sum()
k = int(np.searchsorted(cum, f)) + 1
Qk = Q[:, :k]
Rn = A - (A @ Qk) @ Qk.T
out['varmatched_k'] = k; out['varmatched_removed'] = round(varfrac(Rn), 3)
out['varmatched_null'] = probe(Rn)
# shuffled-static null: permute static vectors within train and within test
perm = np.arange(len(S))
for part in (tr, te):
    idx = np.where(part)[0]; perm[idx] = rng.permutation(idx)
Ps, _ = fit_map(S[perm]); Rs = A - Ps
out['shuffled_removed'] = round(varfrac(Rs), 3)
out['shuffled_null'] = probe(Rs)
print(out, flush=True)
out['model'] = MODEL
json.dump(out, open(f'resid_{DS}_{SM}_L{LAYER}_{MODEL}.json', 'w'), indent=1)
