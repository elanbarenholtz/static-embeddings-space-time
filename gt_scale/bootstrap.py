"""Bootstrap CIs for within-group ordering r on G&T test items (from preds_<ds>.npz).
Two schemes: resample items, and resample whole groups (countries / centuries).
Usage: python bootstrap.py <dataset> [model_tag]   (model_tag: pythia (default) or llama)
"""
import sys, json, numpy as np, pandas as pd
DS = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else 'pythia'
B = 1000
rng = np.random.default_rng(0)
d = np.load(f'preds_{DS}.npz' if TAG == 'pythia' else f'preds_{DS}_{TAG}.npz', allow_pickle=True)
df = pd.read_csv(f'data/{DS}.csv').iloc[d['idx']].reset_index(drop=True)
Y = d['y']
if DS == 'world_place':
    import country_converter as coco
    ctry = df['country'].fillna('').str.replace('_', ' ')
    uniq = ctry.unique().tolist()
    conv = coco.CountryConverter().convert(uniq, to='continent', not_found=None)
    conv = [c[0] if isinstance(c, list) else c for c in conv]
    groups = {'continent': ctry.map(dict(zip(uniq, conv))).fillna('unk').values, 'country': ctry.values}
    targets = ['latitude', 'longitude']; cluster = groups['country']
else:
    groups = {'century': df['death_century'].values, 'decade': (df['death_year'] // 10).values}
    targets = ['death_year']; cluster = groups['century']
reps = [k for k in d.files if k not in ('y', 'idx')]
big = [k for k in reps if k.split('_')[0] in ('pythia','llama') and 'L0' not in k and 'PC' not in k][0]

def wr(p, t, g):
    x = pd.DataFrame({'p': p, 't': t, 'g': g})
    x = x[x.groupby('g')['t'].transform('size') >= 2]
    a = x.p - x.groupby('g').p.transform('mean'); b = x.t - x.groupby('g').t.transform('mean')
    return np.corrcoef(a, b)[0, 1]

res = {}
cl_idx = pd.Series(np.arange(len(Y))).groupby(cluster).apply(list).tolist()
for scheme in ('items', 'clusters'):
    for gname, g in groups.items():
        for k, t in enumerate(targets):
            est = {r: wr(d[r][:, k], Y[:, k], g) for r in reps}
            boots = {r: [] for r in reps}
            for _ in range(B):
                if scheme == 'items':
                    ii = rng.integers(0, len(Y), len(Y))
                else:
                    pick = rng.integers(0, len(cl_idx), len(cl_idx))
                    ii = np.concatenate([cl_idx[j] for j in pick])
                gg = np.char.add(np.asarray(g[ii]).astype(str), np.char.add('_', np.arange(len(ii)).astype(str))) if False else g[ii]
                for r in reps:
                    boots[r].append(wr(d[r][ii, k], Y[ii, k], gg))
            out = {}
            for r in reps:
                b = np.array(boots[r]); out[r] = [round(est[r], 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
            for r in ('GloVe', 'Word2Vec'):
                diff = np.array(boots[big]) - np.array(boots[r])
                out[f'{big} - {r}'] = [round(est[big] - est[r], 3), round(float(np.percentile(diff, 2.5)), 3), round(float(np.percentile(diff, 97.5)), 3)]
            res[f'{scheme}|within_{gname}|{t}'] = out
            print(scheme, gname, t, out, flush=True)
json.dump(res, open(f'bootstrap_{DS}_{TAG}.json', 'w'), indent=1)
