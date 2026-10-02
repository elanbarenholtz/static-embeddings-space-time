"""Grouped holdout on the G&T entities, with per-fold reporting (review round 2).
Places: leave-one-continent-out over Africa, America (North and South), Asia, Europe, Oceania
(continent from country_converter); figures: leave-one-era-block-out, death year cut into five
contiguous blocks with equal counts. All GloVe-covered items with finite targets. Latitude and
longitude are fitted jointly (one multi-output ridge) and scored separately. Reports, for each
held-out group: n, R^2 within the group, MAE and median absolute error; and over all groups:
R^2 of the pooled held-out predictions. Random five-fold splits of the same items for comparison.
Usage: python holdout2.py '<json layers>'   e.g. '{"pythia-2.8b": {"world_place": 28, ...}, ...}'"""
import sys, json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
ALPHAS = np.logspace(-2, 5, 15)
LAY = json.loads(sys.argv[1])
REPS = {'GloVe': None, 'Word2Vec': None, **LAY}
out = {}
for DS in ['world_place', 'historical_figure']:
    df = pd.read_csv(f'data/{DS}.csv')
    if DS == 'world_place':
        T = ['latitude', 'longitude']
        import country_converter as coco
        ctry = df['country'].fillna('').str.replace('_', ' '); uniq = ctry.unique().tolist()
        conv = coco.CountryConverter().convert(uniq, to='continent', not_found=None)
        conv = [c[0] if isinstance(c, list) else c for c in conv]
        grp = ctry.map(dict(zip(uniq, conv))).fillna('unk').astype(str).values
    else:
        T = ['death_year']
        y = df['death_year'].values
        q = np.nanquantile(y, [0.2, 0.4, 0.6, 0.8]); grp = np.digitize(y, q).astype(str)
    Y = df[T].values.astype(float)
    ok = np.isfinite(Y).all(1) & (np.load(f'static_GloVe_{DS}.npz')['n_in_vocab'] > 0)
    if DS == 'world_place': ok &= np.isin(grp, ['Africa', 'America', 'Asia', 'Europe', 'Oceania'])
    idx = np.where(ok)[0]; g = grp[idx]; Yo = Y[idx]
    info = {}
    for k in np.unique(g):
        m = g == k; info[k] = {'n': int(m.sum())}
        if DS == 'historical_figure': info[k]['range'] = [float(Yo[m, 0].min()), float(Yo[m, 0].max())]
    out[f'{DS}|groups'] = info
    rand = np.random.default_rng(0).integers(0, 5, len(idx)).astype(str)
    for rep, layers in REPS.items():
        if layers is None:
            X = np.load(f'static_{rep}_{DS}.npz')['vecs'][idx].astype(np.float32)
        else:
            meta = json.load(open(f'meta_{DS}_{rep}.json'))
            X = np.asarray(np.load(f'acts_{DS}_{rep}.npy', mmap_mode='r')[meta['layers'].index(layers[DS])][idx], np.float32)
        res = {}
        for scheme, folds in [('grouped', g), ('random', rand)]:
            P = np.zeros_like(Yo); per = {}
            for f in np.unique(folds):
                te = folds == f; m = RidgeCV(alphas=ALPHAS).fit(X[~te], Yo[~te])
                P[te] = m.predict(X[te]).reshape(int(te.sum()), -1)
                per[f] = {T[k]: {'r2': round(float(r2_score(Yo[te][:, k], P[te][:, k])), 3),
                                 'mae': round(float(np.mean(np.abs(Yo[te][:, k] - P[te][:, k]))), 2),
                                 'medae': round(float(np.median(np.abs(Yo[te][:, k] - P[te][:, k]))), 2)} for k in range(len(T))}
            res[scheme] = {'pooled_r2': [round(float(r2_score(Yo[:, k], P[:, k])), 3) for k in range(len(T))],
                           'pooled_medae': [round(float(np.median(np.abs(Yo[:, k] - P[:, k]))), 2) for k in range(len(T))],
                           'per_fold': per if scheme == 'grouped' else None}
        out[f'{DS}|{rep}'] = res
        print(DS, rep, json.dumps(res), flush=True)
        json.dump(out, open('holdout2.json', 'w'), indent=1)
        del X
print('ALLDONE')
