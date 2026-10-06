"""
Does embedding similarity track geographic PROXIMITY, or administrative
CO-MEMBERSHIP (same country / same continent)?

The claim under test: cities near each other but in different countries are
LESS similar than cities far apart within the same country. If so, the
structure the probe exploits is administrative-linguistic, not spatial.

Pairs are not independent, so every p-value is a Mantel-style permutation
over CITY labels (10,000 draws), never an analytic test on pairs.
"""
import json, numpy as np

rng = np.random.default_rng(0)
cities = [tuple(c) for c in json.load(open('cities272.json'))['cities']]
emb = {}
for line in open('vecs272.txt'):
    p = line.rstrip('\n').split(' ')
    emb[p[0]] = np.array(p[1:], dtype=np.float64)

def gv(nm):
    v = [emb[w] for w in nm.split() if w in emb]
    return np.mean(v, 0) if v else None

rows = [c for c in cities if gv(c[0]) is not None]
X = np.array([gv(c[0]) for c in rows])
lat = np.array([c[1] for c in rows], float)
lon = np.array([c[2] for c in rows], float)
cont = np.array([c[4] for c in rows])
ctry = np.array([c[7] for c in rows], float)      # GDP-per-capita block = country proxy
n = len(rows)
print(f'{n} cities, GloVe')

# --- cosine similarity matrix
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
S = Xn @ Xn.T

# --- great-circle distance (km)
R = 6371.0
la, lo = np.radians(lat), np.radians(lon)
dla = la[:, None] - la[None, :]
dlo = lo[:, None] - lo[None, :]
h = np.sin(dla/2)**2 + np.cos(la)[:, None]*np.cos(la)[None, :]*np.sin(dlo/2)**2
D = 2*R*np.arcsin(np.sqrt(np.clip(h, 0, 1)))

iu = np.triu_indices(n, 1)
sim  = S[iu]
dist = D[iu]
same_ctry = (ctry[:, None] == ctry[None, :])[iu].astype(float)
same_cont = (cont[:, None] == cont[None, :])[iu].astype(float)
logd = np.log10(dist + 1)

print(f'{len(sim)} pairs | same-country {int(same_ctry.sum())} | '
      f'same-continent {int(same_cont.sum())}')

# ---------------------------------------------------------------- 1. the direct contrast
print('\n=== 1. NEAR-different-country  vs  FAR-same-country ===')
for near_km, far_km in [(500, 1000), (500, 2000), (1000, 2000)]:
    a = sim[(dist < near_km) & (same_ctry == 0)]
    b = sim[(dist > far_km) & (same_ctry == 1)]
    if len(a) < 5 or len(b) < 5:
        print(f'  <{near_km}km diff-country n={len(a)}, >{far_km}km same-country n={len(b)} -- too few')
        continue
    print(f'  <{near_km:5d} km, DIFFERENT country: mean sim {a.mean():.3f}  (n={len(a)})')
    print(f'  >{far_km:5d} km, SAME country:      mean sim {b.mean():.3f}  (n={len(b)})')
    print(f'      difference {b.mean()-a.mean():+.3f}\n')

# ---------------------------------------------------------------- 2. regression
def zs(v): return (v - v.mean()) / v.std()

def betas(y, cols):
    A = np.column_stack([np.ones(len(y))] + [zs(c) for c in cols])
    return np.linalg.lstsq(A, zs(y), rcond=None)[0][1:]

models = {
 'distance only':                      [logd],
 'co-membership only':                 [same_ctry, same_cont],
 'distance + co-membership':           [logd, same_ctry, same_cont],
}
print('=== 2. standardised betas predicting cosine similarity ===')
for name, cols in models.items():
    b = betas(sim, cols)
    labels = {'distance only': ['log dist'],
              'co-membership only': ['same country', 'same continent'],
              'distance + co-membership': ['log dist', 'same country', 'same continent']}[name]
    print(f'  {name:26s} ' + '  '.join(f'{l} {v:+.3f}' for l, v in zip(labels, b)))

# ---------------------------------------------------------------- 3. distance WITHIN strata
print('\n=== 3. does distance still predict once co-membership is fixed? ===')
for label, mask in [('same country',        same_ctry == 1),
                    ('same continent, different country',
                                            (same_cont == 1) & (same_ctry == 0)),
                    ('different continent', same_cont == 0)]:
    s, d_ = sim[mask], logd[mask]
    r = np.corrcoef(s, d_)[0, 1]
    print(f'  {label:36s} n={mask.sum():6d}   r(sim, log dist) = {r:+.3f}')

# ---------------------------------------------------------------- 4. Mantel permutation
def mantel(stat_fn, reps=10000):
    obs = stat_fn(np.arange(n))
    null = np.empty(reps)
    for i in range(reps):
        null[i] = stat_fn(rng.permutation(n))
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (reps + 1)
    return obs, float(np.mean(null)), float(np.std(null)), float(p)

def partial_dist_beta(perm):
    """beta on log-distance with co-membership controlled, under a city relabelling"""
    Sp = S[np.ix_(perm, perm)][iu]
    A = np.column_stack([np.ones(len(Sp)), zs(logd), zs(same_ctry), zs(same_cont)])
    return np.linalg.lstsq(A, zs(Sp), rcond=None)[0][1]

def ctry_beta(perm):
    Sp = S[np.ix_(perm, perm)][iu]
    A = np.column_stack([np.ones(len(Sp)), zs(logd), zs(same_ctry), zs(same_cont)])
    return np.linalg.lstsq(A, zs(Sp), rcond=None)[0][2]

print('\n=== 4. Mantel permutation (10,000 city relabellings) ===')
for nm, fn in [('log-distance beta (co-membership controlled)', partial_dist_beta),
               ('same-country beta (distance controlled)',      ctry_beta)]:
    o, m, s_, p = mantel(fn, 10000)
    print(f'  {nm:44s} obs {o:+.3f}   null {m:+.3f}±{s_:.3f}   p={p:.4f}')
