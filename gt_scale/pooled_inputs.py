"""No-context transformer baselines: the model's own input-token embeddings of each name,
mean-pooled and sum-pooled over the name's tokens (BOS excluded), alongside the last-token
input embedding (= layer-0 residual at the last token). Scored exactly like decompose.py:
G&T split, GloVe-covered test items, RidgeCV, R^2 and within-group r."""
import sys, json, numpy as np, pandas as pd, torch, warnings
from safetensors import safe_open
from transformers import AutoTokenizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import glob
warnings.filterwarnings('ignore')
ALPHAS = np.logspace(-2, 5, 29)
MODELS = {'pythia-2.8b': 'gpt_neox.embed_in.weight', 'llama-2-7b': 'model.embed_tokens.weight'}

def embed_matrix(mdir, key):
    for f in sorted(glob.glob(f'{mdir}/*.safetensors')):
        with safe_open(f, 'pt') as h:
            if key in h.keys():
                return h.get_tensor(key).float().numpy()
    raise KeyError(key)

def within_r(p, t, g):
    d = pd.DataFrame({'p': p, 't': t, 'g': g}); d = d[d.groupby('g')['t'].transform('size') >= 2]
    return float(np.corrcoef(d.p - d.groupby('g').p.transform('mean'), d.t - d.groupby('g').t.transform('mean'))[0, 1])

out = {}
for DS in ['world_place', 'historical_figure']:
    df = pd.read_csv(f'data/{DS}.csv'); names = df['name'].astype(str).tolist()
    test = df['is_test'].astype(str).str.lower().eq('true').values
    if DS == 'world_place':
        T = ['latitude', 'longitude']
        import country_converter as coco
        ctry = df['country'].fillna('').str.replace('_', ' '); uniq = ctry.unique().tolist()
        conv = coco.CountryConverter().convert(uniq, to='continent', not_found=None)
        conv = [c[0] if isinstance(c, list) else c for c in conv]
        groups = {'continent': ctry.map(dict(zip(uniq, conv))).fillna('unk').values, 'country': ctry.values}
    else:
        T = ['death_year']; groups = {'century': df['death_century'].values, 'decade': (df['death_year'] // 10).values}
    Y = df[T].values.astype(float)
    ok = np.isfinite(Y).all(1) & (np.load(f'static_GloVe_{DS}.npz')['n_in_vocab'] > 0)
    tr, te = ok & ~test, ok & test
    for M, key in MODELS.items():
        tok = AutoTokenizer.from_pretrained(f'./{M}')
        E = embed_matrix(f'./{M}', key)
        ids = tok(names, add_special_tokens=False)['input_ids']
        reps = {'last': np.stack([E[i[-1]] for i in ids]),
                'mean': np.stack([E[i].mean(0) for i in ids]),
                'sum': np.stack([E[i].sum(0) for i in ids])}
        for rname, X in reps.items():
            m = RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr]); P = m.predict(X[te]).reshape(int(te.sum()), -1)
            row = {}
            for k, t in enumerate(T):
                row[f'r2_{t}'] = round(float(r2_score(Y[te][:, k], P[:, k])), 3)
                for gname, g in groups.items():
                    row[f'within_{gname}_{t}'] = round(within_r(P[:, k], Y[te][:, k], g[te]), 3)
            out[f'{DS}|{M}|{rname}'] = row
            print(DS, M, rname, row, flush=True)
        del E
json.dump(out, open('pooled_inputs.json', 'w'), indent=1)
