"""Training-only layer selection for the G&T entities (review round 2).
For each model and dataset, the G&T training partition (GloVe-covered items, as in Table 3) is
split once into inner-train (80%) and validation (20%), seed 0. Every saved layer is probed on
inner-train (RidgeCV, alphas chosen by GCV on inner-train) and scored on validation (R^2 averaged
over targets). The layer with the best validation R^2 is refit on the full training partition and
scored once on the test partition. Also reports test R^2 at every layer (for comparison with the
earlier test-selected best layer) and at the fixed 60%-depth layer (19)."""
import json, warnings, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
warnings.filterwarnings('ignore')
ALPHAS = np.logspace(-2, 5, 29)
out = {}
for DS in ['world_place', 'historical_figure']:
    df = pd.read_csv(f'data/{DS}.csv'); test = df['is_test'].astype(str).str.lower().eq('true').values
    T = ['latitude', 'longitude'] if DS == 'world_place' else ['death_year']
    Y = df[T].values.astype(float)
    ok = np.isfinite(Y).all(1) & (np.load(f'static_GloVe_{DS}.npz')['n_in_vocab'] > 0)
    tr_idx = np.where(ok & ~test)[0]; te_idx = np.where(ok & test)[0]
    rng = np.random.default_rng(0); perm = rng.permutation(len(tr_idx)); nv = len(tr_idx) // 5
    va_idx, it_idx = tr_idx[perm[:nv]], tr_idx[perm[nv:]]
    for MODEL in ['pythia-2.8b', 'llama-2-7b']:
        meta = json.load(open(f'meta_{DS}_{MODEL}.json')); A = np.load(f'acts_{DS}_{MODEL}.npy', mmap_mode='r')
        res = {'layers': meta['layers'], 'val': {}, 'test': {}}
        for j, L in enumerate(meta['layers']):
            X = np.asarray(A[j], np.float32)
            m = RidgeCV(alphas=ALPHAS).fit(X[it_idx], Y[it_idx])
            res['val'][L] = float(np.mean([r2_score(Y[va_idx][:, k], m.predict(X[va_idx]).reshape(len(va_idx), -1)[:, k]) for k in range(len(T))]))
            m2 = RidgeCV(alphas=ALPHAS).fit(X[tr_idx], Y[tr_idx]); P = m2.predict(X[te_idx]).reshape(len(te_idx), -1)
            res['test'][L] = [float(r2_score(Y[te_idx][:, k], P[:, k])) for k in range(len(T))]
            print(DS, MODEL, L, 'val', round(res['val'][L], 4), 'test', [round(v, 4) for v in res['test'][L]], flush=True)
            del X
        Ls = max(res['val'], key=res['val'].get)
        res['selected_layer'] = Ls; res['test_at_selected'] = res['test'][Ls]
        res['test_selected_layer'] = max(res['test'], key=lambda L: np.mean(res['test'][L]))
        res['test_at_19'] = res['test'][19]
        out[f'{DS}|{MODEL}'] = res
        print('SELECTED', DS, MODEL, Ls, res['test_at_selected'], 'test-best', res['test_selected_layer'], flush=True)
        json.dump(out, open('nested_layer.json', 'w'), indent=1)
print('ALLDONE')
