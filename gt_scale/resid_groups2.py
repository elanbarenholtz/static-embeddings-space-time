"""Group-conditioned residualization (review item 3).
For places the group is country, for figures century. On the G&T split, GloVe-covered items:
  full            probe on activations A
  -group          A minus its ridge image from group one-hot (country/century)
  -static         A minus its ridge image from static vectors S (as in resid.py)
  -group-static_perp  A minus group image, then minus the image of S_perp, where S_perp is S with
                  its group-predictable part removed (static structure beyond group membership)
  -group-random   same as previous but removing a random subspace of matched variance (null)
Also incremental R^2: probe on [A,S] vs A alone vs S alone.
Within-group r (country / century) reported for each probe.
Usage: python resid_groups.py <dataset> <layer> <model> [static=GloVe]"""
import sys, json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
DS, LAYER, MODEL = sys.argv[1], int(sys.argv[2]), sys.argv[3]
SM = sys.argv[4] if len(sys.argv) > 4 else 'GloVe'
ALPHAS = np.logspace(-2, 5, 29); rng = np.random.default_rng(0)
df = pd.read_csv(f'data/{DS}.csv'); test = df['is_test'].astype(str).str.lower().eq('true').values
if DS == 'world_place':
    T = ['latitude', 'longitude']; grp = df['country'].fillna('NA').values
else:
    T = ['death_year']; grp = df['death_century'].values
Y = df[T].values.astype(float)
St = np.load(f'static_{SM}_{DS}.npz')
mask = np.isfinite(Y).all(1) & (np.load(f'static_GloVe_{DS}.npz')['n_in_vocab'] > 0) & (St['n_in_vocab'] > 0)
tr, te = mask & ~test, mask & test
S = St['vecs'].astype(np.float32)
meta = json.load(open(f'meta_{DS}_{MODEL}.json'))
A = np.asarray(np.load(f'acts_{DS}_{MODEL}.npy', mmap_mode='r')[meta['layers'].index(LAYER)], np.float32)
A -= A[tr].mean(0); S -= S[tr].mean(0)
grp = np.asarray([str(g) for g in grp], dtype=object)
cats = sorted(set(grp[tr]))
cidx = {c: i for i, c in enumerate(cats)}
G = np.zeros((len(grp), len(cats)), np.float32)                          # unseen groups -> all zeros
for i, g in enumerate(grp):
    if g in cidx: G[i, cidx[g]] = 1.0

def within_r(p, t, g):
    d = pd.DataFrame({'p': p, 't': t, 'g': g}); d = d[d.groupby('g')['t'].transform('size') >= 2]
    return float(np.corrcoef(d.p - d.groupby('g').p.transform('mean'), d.t - d.groupby('g').t.transform('mean'))[0, 1])

PRED = {}
def probe(X, key=None):
    m = RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr]); P = m.predict(X[te]).reshape(int(te.sum()), -1)
    if key: PRED[key] = P
    return {'r2': [round(float(r2_score(Y[te][:, k], P[:, k])), 3) for k in range(len(T))],
            'within_r': [round(within_r(P[:, k], Y[te][:, k], grp[te]), 3) for k in range(len(T))]}

def image(Xsrc, Tgt):
    a = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Xsrc[tr], Tgt[tr, :256]).alpha_
    m = Ridge(alpha=a).fit(Xsrc[tr], Tgt[tr]); return (m.predict(Xsrc) - m.intercept_).astype(np.float32)

def varfrac(R, ref):
    return round(float(1 - R[tr].var(0).sum() / ref[tr].var(0).sum()), 3)

out = {'dataset': DS, 'model': MODEL, 'layer': LAYER, 'static': SM, 'n_test': int(te.sum())}
out['full'] = probe(A, 'full')
out['static_only'] = probe(S)
RG = A - image(G, A); out['minus_group'] = probe(RG, 'minus_group'); out['minus_group_var'] = varfrac(RG, A)
RS = A - image(S, A); out['minus_static'] = probe(RS); out['minus_static_var'] = varfrac(RS, A)
Sp = S - image(G, S)                                   # static structure beyond group membership
RGS = RG - image(Sp, RG); out['minus_group_then_static_perp'] = probe(RGS, 'static_perp')
f = varfrac(RGS, RG); out['static_perp_var_of_RG'] = f
Q, _ = np.linalg.qr(rng.standard_normal((A.shape[1], A.shape[1])).astype(np.float32))
pv = ((RG[tr] @ Q) ** 2).mean(0) - (RG[tr] @ Q).mean(0) ** 2
k = int(np.searchsorted(np.cumsum(pv) / RG[tr].var(0).sum(), f)) + 1
RGn = RG - (RG @ Q[:, :k]) @ Q[:, :k].T
out['minus_group_then_random_null'] = probe(RGn, 'random_null')
# incremental R^2 with standardized blocks
sA = A[tr].std(0).mean(); sS = S[tr].std(0).mean()
out['A_plus_S'] = probe(np.hstack([A / sA, S / sS]))
# cluster bootstrap over test-set groups (countries or centuries), 1000 resamples
gt = grp[te]; Yt = Y[te]; ug = np.unique(gt); gidx = {u: np.where(gt == u)[0] for u in ug}
brng = np.random.default_rng(1); B = {k: [] for k in ['minus_group', 'static_perp', 'random_null', 'diff']}
for b in range(1000):
    pick = np.concatenate([gidx[u] for u in brng.choice(ug, len(ug))])
    vals = {}
    for k in ['minus_group', 'static_perp', 'random_null']:
        vals[k] = [within_r(PRED[k][pick, j], Yt[pick, j], gt[pick]) for j in range(len(T))]
        B[k].append(vals[k])
    B['diff'].append([vals['static_perp'][j] - vals['random_null'][j] for j in range(len(T))])
out['bootstrap_within_r_95ci'] = {k: [[round(float(np.percentile(np.array(v)[:, j], 2.5)), 3), round(float(np.percentile(np.array(v)[:, j], 97.5)), 3)] for j in range(len(T))] for k, v in B.items()}
out['n_test_groups'] = int(len(ug))
print(json.dumps(out), flush=True)
json.dump(out, open(f'resid_groups2_{DS}_{MODEL}_L{LAYER}_{SM}.json', 'w'), indent=1)
