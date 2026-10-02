"""
Two controls a TMLR reviewer will ask for.

(1) Grouped holdout. Random 80/20 splits let the probe see Lyon while testing on
    Marseille. We re-evaluate under leave-one-country-out and
    leave-one-continent-out, so no city from a held-out group is ever in training.

    The dataset has no country field, but GDP per capita is assigned nationally,
    so cities sharing a GDP value are (almost always) cities sharing a country.
    That gives 102 country blocks over 272 cities. This is a proxy and is
    labelled as such.

(2) Categorical baselines. How much of the "map" is just
    city -> country/continent -> position? We probe country one-hot and
    continent one-hot under the same protocols. Under leave-one-group-out these
    are 0 by construction (an unseen group is all-zeros), which is itself the
    point: a lookup table cannot generalise to a new country, so whatever the
    embeddings retain under that protocol is not a lookup table.
"""
import json, numpy as np
from collections import Counter
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import r2_score
import gensim.downloader as api

ALPHAS = np.logspace(-2, 3, 20)
cities = [tuple(c) for c in json.load(open('cities272.json'))['cities']]
glove = {}
for line in open('vecs272.txt'):
    p = line.rstrip('\n').split(' ')
    glove[p[0]] = np.array(p[1:], dtype=np.float64)
w2v = api.load('word2vec-google-news-300')


def gv(nm):
    v = [glove[w] for w in nm.split() if w in glove]
    return np.mean(v, 0) if v else None


def wv(nm):
    ph = '_'.join(w.capitalize() for w in nm.split())
    if ph in w2v: return w2v[ph]
    f = [w.capitalize() if w.capitalize() in w2v else w
         for w in nm.split() if w.capitalize() in w2v or w in w2v]
    return np.mean([w2v[w] for w in f], 0) if f else None


valid = [c for c in cities if gv(c[0]) is not None and wv(c[0]) is not None]
n = len(valid)
X_g = np.array([gv(c[0]) for c in valid])
X_w = np.array([wv(c[0]) for c in valid], dtype=float)
lat = np.array([c[1] for c in valid], float)
lon = np.array([c[2] for c in valid], float)
temp = np.array([c[3] for c in valid], float)
cont = [c[4] for c in valid]
gdp = [c[7] for c in valid]

country_id = {g: i for i, g in enumerate(sorted(set(gdp)))}
cont_id = {g: i for i, g in enumerate(sorted(set(cont)))}
g_country = np.array([country_id[g] for g in gdp])
g_cont = np.array([cont_id[c] for c in cont])
X_country = np.eye(len(country_id))[g_country]
X_cont = np.eye(len(cont_id))[g_cont]
print(f'{n} cities | {len(country_id)} country blocks | {len(cont_id)} continents')
print('country block sizes:', Counter(Counter(g_country).values()), flush=True)

targets = {'Latitude': lat, 'Longitude': lon, 'Temperature': temp}
FEATS = {'GloVe': X_g, 'Word2Vec': X_w,
         'Country one-hot': X_country, 'Continent one-hot': X_cont}


def random_split(X, y, n_splits=10):
    out = []
    for s in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s)
        out.append(r2_score(yte, RidgeCV(alphas=ALPHAS).fit(Xtr, ytr).predict(Xte)))
    return float(np.mean(out)), float(np.std(out))


def grouped(X, y, groups):
    """leave-one-group-out; pool predictions and score once (folds are tiny)"""
    pred = np.empty(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr]).predict(X[te])
    return float(r2_score(y, pred))


rows = {}
hdr = f'{"predictor":20s} {"target":12s} {"random 80/20":>16s} {"leave-country-out":>19s} {"leave-continent-out":>21s}'
print('\n' + hdr); print('-' * len(hdr))
for fname, X in FEATS.items():
    for tname, y in targets.items():
        rm, rs = random_split(X, y)
        gc = grouped(X, y, g_country)
        gk = grouped(X, y, g_cont)
        rows[f'{fname}|{tname}'] = dict(random_mean=rm, random_sd=rs,
                                        leave_country_out=gc, leave_continent_out=gk)
        print(f'{fname:20s} {tname:12s} {rm:10.3f}±{rs:5.3f} {gc:19.3f} {gk:21.3f}', flush=True)

json.dump({'n_cities': n, 'n_country_blocks': len(country_id), 'results': rows},
          open('holdout_results.json', 'w'), indent=1)
print('\nwrote holdout_results.json')
