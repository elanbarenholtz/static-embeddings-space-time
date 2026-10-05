"""Robustness for the entity-level baseline, on the same item subset as entity_probe.py:
(1) paired cluster-bootstrap 95% intervals (1,000 resamples of test-set countries or centuries)
    for R^2 differences and within-group r; (2) error percentiles; (3) grouped holdouts
    (leave-one-continent-out for places; five equal-count death-year blocks for figures),
    transformers at the fixed layer 19 as in the paper."""
import re, json, numpy as np, pandas as pd, country_converter as coco
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
ALPHAS = np.logspace(-2, 5, 29); TOK = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*"); G = '../gt_scale'
e = np.load('w2v_entities.npz'); EV = dict(zip(e['keys'], e['vecs'])); w = np.load('w2v_words.npz'); WV = dict(zip(w['keys'], w['vecs']))
def _ld(p):
    try: return json.load(open(p))
    except Exception: return {}
T = _ld('figure_titles.json'); T.update({k: v for k, v in _ld('figure_titles_sparql.json').items() if v})
def wavg(n):
    v = [WV[t.lower()] for t in TOK.findall(n) if t.lower() in WV]; return np.mean(v, 0) if v else None
def wr(y, p, g):
    yc = y - pd.Series(y).groupby(g).transform('mean').values; pc = p - pd.Series(p).groupby(g).transform('mean').values
    return float(np.corrcoef(yc, pc)[0, 1])
rng = np.random.default_rng(0); out = {}
for DS in ['world_place', 'historical_figure']:
    df = pd.read_csv(f'{G}/data/{DS}.csv'); test = df['is_test'].astype(str).str.lower().eq('true').values
    tg = ['latitude', 'longitude'] if DS == 'world_place' else ['death_year']; Y = df[tg].values.astype(float)
    if DS == 'world_place':
        keys = df['name'].tolist(); grp = df['country'].fillna('?').values
        ctry = df['country'].fillna('').str.replace('_', ' '); uq = ctry.unique().tolist()
        conv = coco.CountryConverter().convert(uq, to='continent', not_found=None); conv = [c[0] if isinstance(c, list) else c for c in conv]
        hg = ctry.map(dict(zip(uq, conv))).fillna('unk').astype(str).values
    else:
        keys = [T.get(q) or n for q, n in zip(df['wiki_id'], df['name'])]; grp = df['death_century'].values
        yy = df['death_year'].values; hg = np.digitize(yy, np.nanquantile(yy, [0.2, 0.4, 0.6, 0.8])).astype(str)
    ent = [EV.get(k.replace(' ', '_')) if isinstance(k, str) else None for k in keys]; wrd = [wavg(n) for n in df['name']]
    gl = np.load(f'{G}/static_GloVe_{DS}.npz')
    ok = np.isfinite(Y).all(1) & np.array([x is not None for x in ent]) & np.array([x is not None for x in wrd]) & (gl['n_in_vocab'] > 0)
    tr, te = ok & ~test, ok & test
    Z = lambda L: np.stack([x if x is not None else np.zeros(300, np.float32) for x in L]).astype(np.float32)
    reps = {'entity': Z(ent), 'w2v_words': Z(wrd), 'GloVe': gl['vecs'].astype(np.float32)}
    sel = {'world_place': {'pythia-2.8b': 24, 'llama-2-7b': 24}, 'historical_figure': {'pythia-2.8b': 28, 'llama-2-7b': 24}}[DS]
    acts = {}
    for M in sel:
        meta = json.load(open(f'{G}/meta_{DS}_{M}.json')); A = np.load(f'{G}/acts_{DS}_{M}.npy', mmap_mode='r'); acts[M] = (meta, A)
        reps[M] = np.asarray(A[meta['layers'].index(sel[M])], np.float32)
    P = {k: RidgeCV(alphas=ALPHAS).fit(X[tr], Y[tr]).predict(X[te]).reshape(int(te.sum()), -1) for k, X in reps.items()}
    yt, gt = Y[te], grp[te]; res = {}
    # error percentiles
    for k, p in P.items():
        if DS == 'world_place':
            la1, lo1, la2, lo2 = map(np.radians, (yt[:, 0], yt[:, 1], p[:, 0], p[:, 1]))
            err = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
        else: err = np.abs(yt[:, 0] - p[:, 0])
        res[f'err_pct_{k}'] = [int(np.percentile(err, q)) for q in (10, 25, 50, 75, 90)]
    # paired cluster bootstrap
    groups = pd.Series(np.arange(len(gt))).groupby(gt).apply(list).tolist()
    def stats(ix):
        s = {}
        for k, p in P.items():
            s[f'r2_{k}'] = float(np.mean([r2_score(yt[ix, j], p[ix, j]) for j in range(len(tg))]))
            s[f'wr_{k}'] = float(np.mean([wr(yt[ix, j], p[ix, j], gt[ix]) for j in range(len(tg))]))
        return s
    base = stats(np.arange(len(gt))); B = []
    for _ in range(1000):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]); B.append(stats(ix))
    comps = [('entity', 'w2v_words'), ('entity', 'GloVe'), ('llama-2-7b', 'entity'), ('entity', 'pythia-2.8b')]
    for a, b in comps:
        for m in ['r2', 'wr']:
            d = [x[f'{m}_{a}'] - x[f'{m}_{b}'] for x in B]
            res[f'{m}_{a}_minus_{b}'] = [round(base[f'{m}_{a}'] - base[f'{m}_{b}'], 3), round(np.percentile(d, 2.5), 3), round(np.percentile(d, 97.5), 3)]
    res['base'] = {k: round(v, 3) for k, v in base.items()}
    # grouped holdouts (transformers at layer 19)
    hreps = {'entity': reps['entity'], 'w2v_words': reps['w2v_words'], 'GloVe': reps['GloVe']}
    for M, (meta, A) in acts.items(): hreps[M + '_L19'] = np.asarray(A[meta['layers'].index(19)], np.float32)
    idx = np.where(ok & (np.isin(hg, ['Africa', 'America', 'Asia', 'Europe', 'Oceania']) if DS == 'world_place' else True))[0]
    hgi = hg[idx]; ho = {}
    for k, X in hreps.items():
        pr = np.zeros((len(idx), len(tg)))
        for gname in np.unique(hgi):
            m = hgi == gname
            pr[m] = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(X[idx][~m], Y[idx][~m]).predict(X[idx][m]).reshape(int(m.sum()), -1)
        ho[k] = [round(r2_score(Y[idx][:, j], pr[:, j]), 3) for j in range(len(tg))]
        print(DS, 'holdout', k, ho[k], flush=True)
    res['grouped_holdout_pooled_r2'] = ho; res['holdout_groups'] = {k: int(v) for k, v in zip(*np.unique(hgi, return_counts=True))}
    out[DS] = res; print(DS, json.dumps({k: v for k, v in res.items() if k != 'base'}), flush=True)
    json.dump(out, open('entity_robust.json', 'w'), indent=1)
print('ROBUSTDONE')
