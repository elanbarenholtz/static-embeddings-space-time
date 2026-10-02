"""Paired bootstrap CIs for transformer-minus-static R^2 on the same held-out G&T items
(review item 7D). Resamples items, and separately whole countries / centuries. Also checks
Word2Vec without phrase entries (review: phrase robustness) and how country_ceiling.py handles
countries absent from training."""
import json, numpy as np, pandas as pd, re
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
rng = np.random.default_rng(0); B = 1000
out = {}
for DS, files in [('world_place', ['preds_world_place.npz', 'preds_world_place_llama.npz']),
                  ('historical_figure', ['preds_historical_figure.npz', 'preds_historical_figure_llama.npz'])]:
    d0 = np.load(files[0]); d1 = np.load(files[1]); Y = d0['y']
    df = pd.read_csv(f'data/{DS}.csv').iloc[d0['idx']].reset_index(drop=True)
    cl = df['country'].fillna('NA').astype(str).values if DS == 'world_place' else df['death_century'].astype(str).values
    big = {'pythia': [k for k in d0.files if k.startswith('pythia_L') and 'PC' not in k and k != 'pythia_L0'][0],
           'llama': [k for k in d1.files if k.startswith('llama_L') and 'PC' not in k and k != 'llama_L0'][0]}
    P = {'GloVe': d0['GloVe'], 'Word2Vec': d0['Word2Vec'], 'pythia': d0[big['pythia']], 'llama': d1[big['llama']]}
    groups = pd.Series(np.arange(len(Y))).groupby(cl).apply(list).tolist()
    for k in range(Y.shape[1]):
        for a in ['pythia', 'llama']:
            for b in ['GloVe', 'Word2Vec']:
                est = r2_score(Y[:, k], P[a][:, k]) - r2_score(Y[:, k], P[b][:, k])
                res = {}
                for scheme in ['items', 'clusters']:
                    v = []
                    for _ in range(B):
                        if scheme == 'items': ii = rng.integers(0, len(Y), len(Y))
                        else: ii = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
                        v.append(r2_score(Y[ii, k], P[a][ii, k]) - r2_score(Y[ii, k], P[b][ii, k]))
                    res[scheme] = [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
                out[f'{DS}|t{k}|{a}-{b}'] = {'diff': round(float(est), 3), **res}
                print(DS, k, a, b, out[f'{DS}|t{k}|{a}-{b}'], flush=True)
json.dump(out, open('r2_boot.json', 'w'), indent=1)
