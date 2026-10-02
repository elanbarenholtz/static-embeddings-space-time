"""Replicate G&T's own probe exactly: Ridge(alpha=d), standardized target, all items,
their split. Usage: python gtprobe.py <dataset> <acts_tag> <layer>"""
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
DS, TAG, L = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(f'data/{DS}.csv')
test = df['is_test'].astype(str).str.lower().eq('true').values
cols = ['longitude', 'latitude'] if DS == 'world_place' else ['death_year']
Y = df[cols].values.astype(float)
ok = np.isfinite(Y).all(1)
meta = json.load(open(f'meta_{TAG}.json'))
A = np.asarray(np.load(f'acts_{TAG}.npy', mmap_mode='r')[meta['layers'].index(L)], np.float32)
tr, te = ok & ~test, ok & test
mu, sd = Y[tr].mean(0), Y[tr].std(0)
m = Ridge(alpha=A.shape[1]).fit(A[tr], (Y[tr] - mu) / sd)
P = m.predict(A[te]).reshape(int(te.sum()), -1) * sd + mu
print(TAG, 'L', L, 'n_test', int(te.sum()), 'R2 (uniform avg)', round(float(r2_score(Y[te], P)), 3),
      'per-target', [round(float(r2_score(Y[te][:, k], P[:, k])), 3) for k in range(len(cols))], flush=True)
