"""Word2Vec with vs without phrase entries on the G&T entities (review: phrase robustness)."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from static_vecs import words, load_w2v
ALPHAS = np.logspace(-2, 5, 29); out = {}
need = set()
for ds in ['world_place', 'historical_figure']:
    for n in pd.read_csv(f'data/{ds}.csv')['name']:
        ws = words(n); need |= set(ws) | {w.lower() for w in ws} | {'_'.join(ws)}
E = load_w2v('data/w2v.gz', need)
for ds, T in [('world_place', ['latitude', 'longitude']), ('historical_figure', ['death_year'])]:
    df = pd.read_csv(f'data/{ds}.csv'); test = df['is_test'].astype(str).str.lower().eq('true').values
    Y = df[T].values.astype(float)
    ok = np.isfinite(Y).all(1) & (np.load(f'static_GloVe_{ds}.npz')['n_in_vocab'] > 0)
    nph = 0; X = {'with': np.zeros((len(df), 300), np.float32), 'without': np.zeros((len(df), 300), np.float32)}
    for i, name in enumerate(df['name']):
        ws = words(name); got = [E[w] if w in E else E.get(w.lower()) for w in ws]; got = [g for g in got if g is not None]
        avg = np.mean(got, 0) if got else np.zeros(300, np.float32)
        X['without'][i] = avg; ph = '_'.join(ws)
        if ph in E and len(ws) > 1: X['with'][i] = E[ph]; nph += ok[i]
        else: X['with'][i] = avg
    tr, te = ok & ~test, ok & test
    r = {'n_items_with_phrase_entry': int(nph), 'n_items': int(ok.sum())}
    for k, Xk in X.items():
        P = RidgeCV(alphas=ALPHAS).fit(Xk[tr], Y[tr]).predict(Xk[te]).reshape(int(te.sum()), -1)
        r[k] = [round(float(r2_score(Y[te][:, j], P[:, j])), 3) for j in range(len(T))]
    out[ds] = r; print(ds, r, flush=True)
json.dump(out, open('phrase_check.json', 'w'), indent=1)
