"""How far does knowing only the country get you? Predict each test place by its
country's training-set mean coordinates; score R^2 and within-continent ordering."""
import numpy as np, pandas as pd, country_converter as coco
from sklearn.metrics import r2_score
df = pd.read_csv('data/world_place.csv')
test = df['is_test'].astype(str).str.lower().eq('true').values
ok = np.load('static_GloVe_world_place.npz')['n_in_vocab'] > 0
ctry = df['country'].fillna('').str.replace('_', ' ')
uniq = ctry.unique().tolist()
conv = coco.CountryConverter().convert(uniq, to='continent', not_found=None)
conv = [c[0] if isinstance(c, list) else c for c in conv]
cont = ctry.map(dict(zip(uniq, conv))).fillna('unk').values
tr, te = ok & ~test, ok & test
for t in ['latitude', 'longitude']:
    means = df[tr].groupby(ctry[tr])[t].mean()
    pred = ctry[te].map(means).fillna(df.loc[tr, t].mean()).values
    y = df.loc[te, t].values
    d = pd.DataFrame({'p': pred, 'y': y, 'g': cont[te]})
    d = d[d.groupby('g')['y'].transform('size') >= 2]
    p = d.p - d.groupby('g').p.transform('mean'); yy = d.y - d.groupby('g').y.transform('mean')
    print(t, 'country-mean R2', round(r2_score(y, pred), 3), 'within-continent r', round(np.corrcoef(p, yy)[0, 1], 3))
