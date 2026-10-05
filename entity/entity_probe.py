"""Entity-level static baseline (Wikipedia2Vec 300d, enwiki 2018-04-20) on the released places and
historical figures. All representations are trained and tested on the same items: those with an
entity vector, a Wikipedia2Vec word vector and a GloVe vector, using the authors' split.
Probe: RidgeCV (alphas 1e-2..1e5). Reports R^2, within-country / within-century r, and median error."""
import re, json, numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
ALPHAS = np.logspace(-2, 5, 29); TOK = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
G = '../gt_scale'
e = np.load('w2v_entities.npz'); EV = dict(zip(e['keys'], e['vecs']))
w = np.load('w2v_words.npz'); WV = dict(zip(w['keys'], w['vecs']))
def _ld(p):
    try: return json.load(open(p))
    except Exception: return {}
T = _ld('figure_titles.json'); T.update({k: v for k, v in json.load(open('figure_titles_sparql.json')).items() if v})
def wavg(n):
    v = [WV[t.lower()] for t in TOK.findall(n) if t.lower() in WV]
    return np.mean(v, 0) if v else None
def within_r(y, p, g):
    yc = y - pd.Series(y).groupby(g).transform('mean').values; pc = p - pd.Series(p).groupby(g).transform('mean').values
    return float(np.corrcoef(yc, pc)[0, 1])
out = {}
for DS in ['world_place', 'historical_figure']:
    df = pd.read_csv(f'{G}/data/{DS}.csv'); test = df['is_test'].astype(str).str.lower().eq('true').values
    tg = ['latitude', 'longitude'] if DS == 'world_place' else ['death_year']
    Y = df[tg].values.astype(float)
    if DS == 'world_place': keys = df['name'].tolist(); grp = df['country'].fillna('?').values
    else:
        keys = [T.get(q) or n for q, n in zip(df['wiki_id'], df['name'])]; grp = df['death_century'].values
    ent = [EV.get(k.replace(' ', '_')) if isinstance(k, str) else None for k in keys]
    wrd = [wavg(n) for n in df['name']]
    gl = np.load(f'{G}/static_GloVe_{DS}.npz')
    ok = np.isfinite(Y).all(1) & np.array([x is not None for x in ent]) & np.array([x is not None for x in wrd]) & (gl['n_in_vocab'] > 0)
    tr, te = ok & ~test, ok & test
    res = {'n_items': int(len(df)), 'entity_coverage': float(np.mean([x is not None for x in ent])), 'n_train': int(tr.sum()), 'n_test': int(te.sum())}
    Z = lambda L: np.stack([x if x is not None else np.zeros(300, np.float32) for x in L])
    reps = {'Wikipedia2Vec entity': Z(ent), 'Wikipedia2Vec words (avg)': Z(wrd), 'GloVe (avg)': gl['vecs'],
            'fastText (avg)': np.load(f'{G}/static_fastText_{DS}.npz')['vecs']}
    sel = {'world_place': {'pythia-2.8b': 24, 'llama-2-7b': 24}, 'historical_figure': {'pythia-2.8b': 28, 'llama-2-7b': 24}}[DS]
    for M, L in sel.items():
        meta = json.load(open(f'{G}/meta_{DS}_{M}.json')); A = np.load(f'{G}/acts_{DS}_{M}.npy', mmap_mode='r')
        reps[f'{M} L{L}'] = A[meta['layers'].index(L)]
    for name, X in reps.items():
        X = np.asarray(X, np.float32)
        P = RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr]).predict(X[te]).reshape(int(te.sum()), -1)
        r = {'r2': [round(r2_score(Y[te][:, k], P[:, k]), 3) for k in range(len(tg))],
             'within_group_r': [round(within_r(Y[te][:, k], P[:, k], grp[te]), 3) for k in range(len(tg))]}
        if DS == 'world_place':
            la1, lo1, la2, lo2 = map(np.radians, (Y[te][:, 0], Y[te][:, 1], P[:, 0], P[:, 1]))
            a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
            r['median_km'] = int(np.median(6371 * 2 * np.arcsin(np.sqrt(a))))
        else:
            r['median_abs_years'] = int(np.median(np.abs(Y[te][:, 0] - P[:, 0])))
        res[name] = r; print(DS, name, r, flush=True)
        if DS == 'world_place': np.save(f'pred_{DS}_{name.split()[0]}_{name.split()[-1]}.npy', P)
    out[DS] = res
    json.dump(out, open('entity_probe.json', 'w'), indent=1)
print('ENTITYDONE')
