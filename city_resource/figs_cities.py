"""Paper figures for the 272-city set, from the re-sourced data and the JSON outputs.
Usage: PYTHONPATH=shim python figs_cities.py <cities.json> <tag>"""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, os, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
P = _SBROOT
DATA, TAG = sys.argv[1], sys.argv[2]; OUT = f'out_{TAG}'
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})
rows = [[c[0].lower()] + list(c[1:]) for c in json.load(open(DATA))['cities']]
S = json.load(open(f'{OUT}/static_cities.json')); PY = json.load(open(f'{OUT}/pythia_cities.json'))
n = len(rows)
# ---- embeddings (as in table1_272.py)
need = {t for r in rows for t in r[0].split()}; G = {}
for line in open(f'{P}/gt_scale/data/glove.6B.300d.txt', encoding='utf8'):
    w, rest = line.split(' ', 1)
    if w in need: G[w] = np.array(rest.split(), np.float64)
    if len(G) == len(need): break
import gensim.downloader as api
W2 = api.load('word2vec-google-news-300')
def wv(nm):
    ph = '_'.join(w.capitalize() for w in nm.split())
    if ph in W2: return np.asarray(W2[ph], np.float64)
    f = [w.capitalize() if w.capitalize() in W2 else w for w in nm.split() if w.capitalize() in W2 or w in W2]
    return np.mean([W2[w] for w in f], 0)
X = {'GloVe': np.array([np.mean([G[w] for w in r[0].split() if w in G], 0) for r in rows]),
     'Word2Vec': np.array([wv(r[0]) for r in rows])}
lat = np.array([r[1] for r in rows]); lon = np.array([r[2] for r in rows]); cont = [r[4] for r in rows]
cmap = {'North America': '#e74c3c', 'Europe': '#3498db', 'Asia': '#f39c12', 'Africa': '#27ae60',
        'South America': '#9b59b6', 'Oceania': '#1abc9c'}
col = np.array([cmap[c] for c in cont])
# ---- Figure: geography (first of the ten splits)
tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=0)
fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
for ax, (m, F) in zip(axs, X.items()):
    pl = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(F[tr], lat[tr]).predict(F[te])
    pg = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(F[tr], lon[tr]).predict(F[te])
    ax.scatter(lon[tr], lat[tr], c=col[tr], s=22, alpha=0.25, edgecolors='none')
    for i, j in enumerate(te):
        ax.plot([lon[j], pg[i]], [lat[j], pl[i]], color='#94a3b8', lw=0.8, zorder=1)
    ax.scatter(lon[te], lat[te], c=col[te], s=60, edgecolors='white', linewidths=0.8, zorder=3)
    ax.scatter(pg, pl, c=col[te], s=55, marker='x', linewidths=1.6, zorder=4)
    ax.set_xlim(-180, 180); ax.set_ylim(-60, 80); ax.set_title(m, loc='left'); ax.set_xlabel('Longitude')
    ax.grid(alpha=0.15)
axs[0].set_ylabel('Latitude')
axs[1].legend(handles=[Line2D([0], [0], marker='o', ls='', color='#374151', label='actual (held out)'),
                       Line2D([0], [0], marker='x', ls='', color='#374151', label='predicted'),
                       Line2D([0], [0], marker='o', ls='', color='#cbd5e1', label='training cities')],
              frameon=False, loc='lower left', fontsize=9)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_geography.png', dpi=300)
# ---- Figure: bar plot of Table 1
order = ['Longitude', 'Latitude', 'Temperature', 'GDP per capita', 'Population', 'Elevation']
fig, ax = plt.subplots(figsize=(10, 4.8)); x = np.arange(len(order)); bw = 0.38
for k, (m, c) in enumerate([('GloVe', '#2563eb'), ('Word2Vec', '#ea580c')]):
    mu = [S['table1'][m][t][0] for t in order]; sd = [S['table1'][m][t][1] for t in order]
    ax.bar(x + (k - 0.5) * bw, mu, bw * 0.95, yerr=sd, color=c, label=m, capsize=3, error_kw=dict(ecolor='#64748b', lw=1))
    for xi, v, s in zip(x + (k - 0.5) * bw, mu, sd): ax.text(xi, v + s + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=9, color='#334155')
ax.axhline(0, color='#94a3b8', lw=1); ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha='right')
ax.set_ylabel('Probe $R^2$ (mean of 10 splits)'); ax.set_ylim(min(0, ax.get_ylim()[0]), 1.0); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_barplot.png', dpi=300)
# ---- Figure: semantic words for temperature
sem = S['semantic']['Temperature']; pos = sem['top_pos'][:15]; neg = sem['top_neg'][:15]
words = [w for w, _ in neg] + [w for w, _ in pos][::-1]; vals = [v for _, v in neg] + [v for _, v in pos][::-1]
fig, ax = plt.subplots(figsize=(8, 7.5)); y = np.arange(len(words))
ax.barh(y, vals, color=['#2563eb' if v < 0 else '#dc2626' for v in vals], height=0.7)
for yi, v in zip(y, vals): ax.text(v + (0.01 if v > 0 else -0.01), yi, f'{v:+.2f}', va='center', ha='left' if v > 0 else 'right', fontsize=9, color='#334155')
ax.set_yticks(y); ax.set_yticklabels(words); ax.axvline(0, color='#94a3b8', lw=1); ax.set_xlim(-0.72, 0.72)
ax.set_xlabel('Pearson $r$ between word–city cosine similarity and mean annual temperature')
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color='#dc2626'), plt.Rectangle((0, 0), 1, 1, color='#2563eb')],
          labels=['warmer cities', 'colder cities'], frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_semantic_272.png', dpi=300)
# ---- Figure: Pythia layers
T = ['Latitude', 'Longitude', 'Temperature']
fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, t in zip(axs, T):
    m = np.array([v['r2'] for v in PY['layers']['pythia-2.8b'][t]]); sd = np.array([v['sd'] for v in PY['layers']['pythia-2.8b'][t]])
    ax.plot(m, '-o', ms=3, color='#2563eb', lw=1.8); ax.fill_between(range(len(m)), m - sd, m + sd, color='#2563eb', alpha=0.12)
    ax.axhline(PY['static']['Word2Vec'][t]['r2'], ls='--', color='k', lw=1.3)
    ax.text(0.5, PY['static']['Word2Vec'][t]['r2'] + 0.015, 'static embedding (Word2Vec)', fontsize=8)
    ax.set_title(t, loc='left'); ax.set_xlabel('layer (0 = token embedding, before attention)'); ax.set_ylim(0, 1)
axs[0].set_ylabel('probe $R^2$ (random 80/20 splits)')
fig.tight_layout(); fig.savefig(f'{OUT}/llm_layers_pythia-2.8b.png', dpi=300)
# ---- Figure: residualization
fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
for row, s in enumerate(['GloVe', 'Word2Vec']):
    R = PY['resid']['pythia-2.8b'][s]
    for ax, t in zip(axs[row], T):
        ax.plot([r[t]['full'] for r in R], '-o', ms=3, color='#2563eb', lw=1.8, label='full activations')
        ax.plot([r[t]['proj'] for r in R], '-', color='#16a34a', lw=1.5, label=f'part linearly predictable from {s}')
        ax.plot([r[t]['null_random'] for r in R], '--', color='#94a3b8', lw=1.3, label='variance-matched null removal')
        ax.plot([r[t]['resid'] for r in R], '-s', ms=3, color='#dc2626', lw=1.8, label=f'residual after removing {s}')
        ax.axhline(PY['static'][s][t]['r2'], ls=':', color='#16a34a', lw=1.2); ax.axhline(0, color='#cbd5e1', lw=1)
        ax.set_title(f'{t} — {s} removed', loc='left', fontsize=11); ax.set_ylim(-0.25, 0.95)
        if row == 1: ax.set_xlabel('layer  (0 = token embedding, pre-attention)')
    axs[row][0].set_ylabel('probe $R^2$ (random 80/20)')
axs[0][0].legend(frameon=False, fontsize=8, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/residualise_pythia-2.8b.png', dpi=300)
print('figures done')
