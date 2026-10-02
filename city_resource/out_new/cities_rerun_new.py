"""272-city reruns for the review.
(1) Dimensionality scaling and random-embedding control under the Table 1 protocol: ten random
    80/20 splits (seeds 0-9), RidgeCV over 20 alphas in [1e-2, 1e3], fitted intercept, mean R^2.
    (The earlier figure used a single 80/20 split, seed 42.)
(2) Leave-one-continent-out for GloVe and Word2Vec (latitude, longitude, temperature), R^2 pooled
    over held-out predictions, against the same ten-split random protocol.
Writes cities_rerun.json and regenerates fig_dimensionality_scaling.png, fig_random_control.png."""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__)))))
import json, os, zipfile, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
P = _SBROOT
ALPHAS = np.logspace(-2, 3, 20)
cities = json.load(open(_os_sb.path.join(_SBROOT, 'city_resource/out_new/expanded_data_new.json')))['cities']
names = [c[0] for c in cities]; Y = {'latitude': np.array([c[1] for c in cities]), 'longitude': np.array([c[2] for c in cities]),
                                     'temperature': np.array([c[3] for c in cities])}
cont = np.array([c[4] for c in cities])
need = {w for n in names for w in n.lower().split()}
def glove(dim):
    path = f'{P}/glove.6B.{dim}d.txt'
    if not os.path.exists(path):
        zipfile.ZipFile(f'{P}/glove.6B.zip').extract(f'glove.6B.{dim}d.txt', P)
    E = {}
    for line in open(path, encoding='utf8'):
        w, r = line.split(' ', 1)
        if w in need: E[w] = np.array(r.split(), np.float32)
    return E
def embed(E, d):
    X = np.zeros((len(names), d), np.float32)
    for i, n in enumerate(names):
        v = [E[w] for w in n.lower().split() if w in E]; X[i] = np.mean(v, 0)
    return X
def ten_split(X, y):
    r = []
    for s in range(10):
        a, b, ya, yb = train_test_split(X, y, test_size=0.2, random_state=s)
        r.append(r2_score(yb, RidgeCV(alphas=ALPHAS).fit(a, ya).predict(b)))
    return float(np.mean(r)), float(np.std(r))
out = {'dims': {}, 'random': {}, 'loco': {}}
rng = np.random.default_rng(0)
for d in [50, 100, 200, 300]:
    X = embed(glove(d), d)
    out['dims'][d] = {t: ten_split(X, y) for t, y in Y.items()}
    rr = {t: [] for t in Y}
    for i in range(20):
        Xr = rng.standard_normal((len(names), d)).astype(np.float32)
        for t, y in Y.items(): rr[t].append(ten_split(Xr, y)[0])
    out['random'][d] = {t: [float(np.mean(v)), float(np.std(v)), [float(x) for x in v]] for t, v in rr.items()}
    print(d, out['dims'][d], {t: out['random'][d][t][:2] for t in Y}, flush=True)
# LOCO for GloVe 300d and Word2Vec
import sys; sys.path.insert(0, f'{P}/review_v2')
from emb import _load_w2v, DATA
X300 = embed(glove(300), 300)
W = _load_w2v(f'{DATA}/w2v.gz', {w for n in names for w in n.split()} | {'_'.join(n.split()) for n in names} | need)
Xw = np.zeros((len(names), 300), np.float32)
for i, n in enumerate(names):
    ph = '_'.join(n.split())
    if ph in W and len(n.split()) > 1: Xw[i] = W[ph]; continue
    v = [W[w] if w in W else W.get(w.lower()) for w in n.split()]; v = [x for x in v if x is not None]; Xw[i] = np.mean(v, 0)
for nm, X in [('GloVe', X300), ('Word2Vec', Xw)]:
    out['loco'][nm] = {}
    for t, y in Y.items():
        pred = np.zeros_like(y, dtype=float)
        for c in np.unique(cont):
            te = cont == c; pred[te] = RidgeCV(alphas=ALPHAS).fit(X[~te], y[~te]).predict(X[te])
        out['loco'][nm][t] = {'loco_pooled_r2': round(float(r2_score(y, pred)), 3), 'random10': [round(v, 3) for v in ten_split(X, y)]}
    print(nm, out['loco'][nm], flush=True)
out['continent_counts'] = {c: int((cont == c).sum()) for c in np.unique(cont)}
json.dump(out, open(f_os_sb.path.join(_SBROOT, 'city_resource/out_new/cities_rerun.json'), 'w'), indent=1)
# figures
plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
dims = [50, 100, 200, 300]; col = {'latitude': '#2171b5', 'longitude': '#e6550d', 'temperature': '#31a354'}
fig, ax = plt.subplots(figsize=(6, 4))
for t in Y:
    m = [out['dims'][d][t][0] for d in dims]; s = [out['dims'][d][t][1] for d in dims]
    ax.errorbar(dims, m, yerr=s, marker='o', color=col[t], label=t.capitalize(), capsize=3)
    rm = np.array([out['random'][d][t][0] for d in dims]); rs = np.array([out['random'][d][t][1] for d in dims])
    ax.fill_between(dims, rm - 2 * rs, rm + 2 * rs, color=col[t], alpha=0.08)
ax.set_xlabel('GloVe dimensionality'); ax.set_ylabel('Probe $R^2$ (mean of ten splits)'); ax.set_xticks(dims)
ax.axhline(0, color='#999', lw=0.8); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(f_os_sb.path.join(_SBROOT, 'city_resource/out_new/fig_dimensionality_scaling.png'), dpi=300)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, t in zip(axs, Y):
    ax.boxplot([out['random'][d][t][2] for d in dims], labels=[str(d) for d in dims])
    ax.scatter(range(1, 5), [out['dims'][d][t][0] for d in dims], marker='D', color=col[t], zorder=3, label='GloVe')
    ax.set_title(t.capitalize()); ax.set_xlabel('Dimensionality'); ax.axhline(0, color='#999', lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)
axs[0].set_ylabel('Probe $R^2$ (mean of ten splits)')
fig.tight_layout(); fig.savefig(f_os_sb.path.join(_SBROOT, 'city_resource/out_new/fig_random_control.png'), dpi=300)
print('figures saved')
