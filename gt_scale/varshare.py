"""Share of test-set variance lying between vs within groups, and how much of each
representation's R^2 comes from between-group placement (R^2 of its predictions after
replacing each prediction with its group's mean prediction)."""
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
for DS, files in [('world_place', ['preds_world_place.npz', 'preds_world_place_llama.npz']),
                  ('historical_figure', ['preds_historical_figure.npz', 'preds_historical_figure_llama.npz'])]:
    d0 = np.load(files[0]); df = pd.read_csv(f'data/{DS}.csv').iloc[d0['idx']].reset_index(drop=True)
    if DS == 'world_place':
        import country_converter as coco
        ctry = df['country'].fillna('').str.replace('_', ' ')
        uniq = ctry.unique().tolist(); conv = coco.CountryConverter().convert(uniq, to='continent', not_found=None)
        conv = [c[0] if isinstance(c, list) else c for c in conv]
        groups = {'continent': ctry.map(dict(zip(uniq, conv))).fillna('unk').values, 'country': ctry.values}
        T = ['latitude', 'longitude']
    else:
        groups = {'century': df['death_century'].values, 'decade': (df['death_year'] // 10).values}
        T = ['death_year']
    Y = d0['y']
    for g, gv in groups.items():
        for k, t in enumerate(T):
            y = Y[:, k]; gm = pd.Series(y).groupby(gv).transform('mean').values
            print(DS, g, t, 'eta2 %.3f' % (1 - ((y - gm) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
    for f in files:
        d = np.load(f)
        for r in [x for x in d.files if x not in ('y', 'idx')]:
            for k, t in enumerate(T):
                p = d[r][:, k]; out = []
                for g, gv in groups.items():
                    pg = pd.Series(p).groupby(gv).transform('mean').values
                    out.append('%s-means R2 %.3f' % (g, r2_score(Y[:, k], pg)))
                print(DS, r, t, 'R2 %.3f' % r2_score(Y[:, k], p), '|', ' | '.join(out))
