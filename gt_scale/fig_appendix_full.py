"""Appendix analyses for historical figures, rerun on the full released dataset
(replacing the 201-figure versions). Target: death year (G&T's target); birth year also reported.
1) eta^2 between eras / centuries; era- and century-centroid and within-era-residual probes (GloVe)
2) data-driven word correlations with death year (top-20k GloVe words)
3) modern-ancient composite
"""
import numpy as np, pandas as pd, json
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
ALPHAS = np.logspace(-2, 5, 29)
df = pd.read_csv('data/historical_figure.csv')
st = np.load('static_GloVe_historical_figure.npz'); X = st['vecs'].astype(np.float64)
ok = (st['n_in_vocab'] > 0) & np.isfinite(df['death_year'].values)
test = df['is_test'].astype(str).str.lower().eq('true').values
tr, te = ok & ~test, ok & test
y = df['death_year'].values.astype(float)
era = np.where(y < 500, 0, np.where(y < 1400, 1, 2))
cent = df['death_century'].values
out = {}
def eta2(yy, g):
    gm = pd.Series(yy).groupby(g).transform('mean').values
    return 1 - ((yy - gm) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
out['n_train'], out['n_test'] = int(tr.sum()), int(te.sum())
out['eta2_era_test'] = eta2(y[te], era[te]); out['eta2_century_test'] = eta2(y[te], cent[te])
def probe(Z):
    """Closed-form ridge; alpha picked on a held-out 20% of the training rows."""
    Z = np.asarray(Z, float); A = Z[tr]; b = y[tr]
    rng = np.random.default_rng(0); v = rng.random(len(A)) < 0.2
    def fit(A, b, a):
        mu, bm = A.mean(0), b.mean(); Ac = A - mu
        w = np.linalg.solve(Ac.T @ Ac + a * np.eye(A.shape[1]), Ac.T @ (b - bm))
        return lambda Q: (Q - mu) @ w + bm
    best = max(ALPHAS, key=lambda a: r2_score(b[v], fit(A[~v], b[~v], a)(A[v])))
    return r2_score(y[te], fit(A, b, best)(Z[te]))
out['glove_full'] = probe(X)
for name, g in [('era', era), ('century', cent)]:
    G = pd.get_dummies(pd.Series(g)).values.astype(float)
    out[f'{name}_onehot'] = probe(G)
    cen = {k: X[tr & (g == k)].mean(0) for k in np.unique(g[tr])}
    gm = X[tr].mean(0)
    C = np.stack([cen.get(k, gm) for k in g])
    out[f'{name}_centroid'] = probe(C)
    out[f'{name}_within_residual'] = probe(X - C)
print('probes done', out, flush=True)
# word correlations (GloVe top 20k)
names_tok = set(w.lower() for n in df['name'].astype(str) for w in n.replace('-', ' ').split())
words, W = [], []
with open('data/glove.6B.300d.txt', encoding='utf8') as f:
    for i, line in enumerate(f):
        if i >= 20000: break
        p = line.rstrip().split(' '); words.append(p[0]); W.append(np.asarray(p[1:], np.float32))
W = np.stack(W); widx = {w: i for i, w in enumerate(words)}
valid = np.array([len(w) >= 4 and w.isalpha() and w not in names_tok for w in words])
F = X[ok]; F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
yo = y[ok]; yz = (yo - yo.mean()) / yo.std()
r = np.zeros(len(words))
for s in range(0, len(words), 2000):
    S = F @ Wn[s:s + 2000].T
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-12)
    r[s:s + 2000] = Sz.T @ yz / len(yz)
order = np.argsort(r)
vi = [i for i in order if valid[i]]
out['n_valid_words'] = int(valid.sum())
out['earliest'] = [(words[i], round(float(r[i]), 3)) for i in vi[:20]]
out['latest'] = [(words[i], round(float(r[i]), 3)) for i in vi[::-1][:20]]
for a, b in [('modern', 'ancient'), ('industrial', 'medieval')]:
    comp = F @ Wn[widx[a]] - F @ Wn[widx[b]]
    out[f'composite_{a}_{b}_death'] = float(np.corrcoef(comp, yo)[0, 1])
    by = df['birth_year'].values[ok].astype(float); m = np.isfinite(by)
    out[f'composite_{a}_{b}_birth'] = float(np.corrcoef(comp[m], by[m])[0, 1])
out['n_corr'] = int(ok.sum())
json.dump(out, open('fig_appendix_full.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
