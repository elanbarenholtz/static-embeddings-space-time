"""Fully nested single-layer city analyses (review round 3): each outer split uses the layer its
own inner cross-validation selected (out_new/nested_cities.json) for residualization against the
static embeddings (with the variance-matched random null and shuffled-static null), the
200-PC comparison, error correlations, and neighbour availability. Results are aggregated over
the ten outer test folds."""
import json, os, sys, numpy as np
from scipy import stats
sys.argv = ['x', 'run_new/cities272_country.json', 'new']
src = open('pythia_cities.py').read()
exec(src[:src.index("res = {'n': n}")])
exec(src[src.index("# ---- residualization"):src.index("res['resid'] = {'pythia-2.8b': {}}")])
exec(src[src.index("def _corr("):src.index("res['err_corr'] = {}; res['neighbors'] = {}")])
from sklearn.decomposition import PCA
NC = json.load(open('out_new/nested_cities.json'))
acts = {'pythia-2.8b': A28['acts'], 'pythia-1.4b': A14['acts']}
out = {'layers': {m: {t: NC[m][t]['layers'] for t in targets} for m in NC}}
def resid_split(A, S, tr, te, y):
    Sp = S[PERM]
    P_, R_ = split_parts(A, S, tr); _, Rsh = split_parts(A, Sp, tr)
    vr = var_removed(A, R_); Arnd, _ = random_matched(A, vr, 7)
    sc = lambda F: r2_score(y[te], fit(F[tr], y[tr]).predict(F[te]))
    return {'full': sc(A), 'proj': sc(P_), 'resid': sc(R_), 'null_random': sc(Arnd), 'resid_shuffled': sc(Rsh), 'var_removed': vr}
for m in ['pythia-2.8b', 'pythia-1.4b']:
    out[m] = {}
    for t, y in targets.items():
        Ls = NC[m][t]['layers']; o = {s: [] for s in STATIC}; pc = []
        for k, (tr, te) in enumerate(folds):
            A = acts[m][Ls[k]].astype(np.float64)
            for s, S in STATIC.items(): o[s].append(resid_split(A, S, tr, te, y))
            if m == 'pythia-2.8b':
                p = PCA(200).fit(A[tr]); pc.append(r2_score(y[te], fit(p.transform(A[tr]), y[tr]).predict(p.transform(A[te]))))
        out[m][t] = {s: {k: float(np.mean([d[k] for d in v])) for k in v[0]} for s, v in o.items()}
        if pc: out[m][t]['pca200'] = float(np.mean(pc))
        print(m, t, json.dumps(out[m][t]), flush=True)
# error correlations and neighbour availability (2.8B), per-split layers
def fold_err_ps(Ls, S, y, within=False, rand=False):
    acc = []
    for k, (tr, te) in enumerate(folds):
        A = A28['acts'][Ls[k]].astype(np.float64); F2 = rng.normal(size=S.shape) if rand else S
        e1 = fit(A[tr], y[tr]).predict(A[te]) - y[te]; e2 = fit(F2[tr], y[tr]).predict(F2[te]) - y[te]
        if within:
            for c in range(len(conts)):
                mm = cid[te] == c
                if mm.sum() >= 5: acc.append((mm.sum(), _corr(e1[mm], e2[mm], y[te][mm])))
        else: acc.append((len(te), _corr(e1, e2, y[te])))
    acc = [(a, b) for a, b in acc if np.isfinite(b)]; w = np.array([a for a, _ in acc], float)
    return float(np.sum(w * np.array([b for _, b in acc])) / w.sum())
out['err_corr'] = {}; out['neighbors'] = {}
for t, y in targets.items():
    Ls = NC['pythia-2.8b'][t]['layers']
    out['err_corr'][t] = {s: {'r': fold_err_ps(Ls, S, y), 'within': fold_err_ps(Ls, S, y, True), 'random_floor': fold_err_ps(Ls, S, y, rand=True)} for s, S in STATIC.items()}
    err = [[] for _ in range(n)]; sim = [[] for _ in range(n)]
    for k, (tr, te) in enumerate(folds):
        A = A28['acts'][Ls[k]].astype(np.float64); An = A / np.linalg.norm(A, axis=1, keepdims=True); Sm = An @ An.T; np.fill_diagonal(Sm, -np.inf)
        p = fit(A[tr], y[tr]).predict(A[te])
        for j, c in enumerate(te): err[c].append(abs(p[j] - y[c])); sim[c].append(np.sort(Sm[c, tr])[::-1][:5].mean())
    keep = [c for c in range(n) if err[c]]
    e = np.array([np.mean(err[c]) for c in keep]); sv = np.array([np.mean(sim[c]) for c in keep]); g = cid[keep]
    D = np.column_stack([np.ones(len(e))] + [(g == c).astype(float) for c in range(1, len(conts))])
    re_ = e - D @ np.linalg.lstsq(D, e, rcond=None)[0]; rs_ = sv - D @ np.linalg.lstsq(D, sv, rcond=None)[0]
    r = float(np.corrcoef(re_, rs_)[0, 1]); df = len(e) - len(conts) - 1; tt = r * np.sqrt(df / (1 - r ** 2))
    out['neighbors'][t] = {'r': r, 'p': float(2 * stats.t.sf(abs(tt), df)), 'n_cities': len(e)}
    print(t, out['err_corr'][t], out['neighbors'][t], flush=True)
json.dump(out, open('out_new/per_split_aux.json', 'w'), indent=1)
print('ALLDONE')
