"""Static-embedding rerun of the Pain Axis representational tests, updated for the paper's v2.

Directions follow the authors' v2 audit code (v2_controls/cosine_constructions/src/run_audit.py):
  paper construction: S1/S2 pain = pain mean - pooled control mean (B,C1,C2,D,E of that set),
     denoised against that control cloud's PCs (50% variance); every other direction = its mean
     minus the pooled first-person neutral (D of S1_1P and S2_1P), denoised against that neutral cloud.
  common_neutral: every direction (pain included) against the pooled neutral mean and basis.
  common_controls: every direction against the pooled S1/S2 control mean (B,C1,C2,D,E) and basis.
Fear, NegEmotion, NegWorld, BodySens are pooled over S1_1P and S2_1P; Arousal, Random, Numb,
Sadness use their first-person sets.
Also: held-out AUC (5-fold over sentence sets, 20 fold assignments), held-out AUC of pain vs the
Arousal and Random sets, and z-scores of the standalone sets (1P and 3P averaged) on the S2
direction relative to the S2 first-person sentences, plus per-pain-category z.
"""
import json, numpy as np, os
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from emb import load, embed, MODELS
R = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f'{R}/painrepo/datasets/3.1_pain_and_control_datasets.json'))['datasets']
D.update(json.load(open(f'{R}/painrepo/datasets/3.1_sadness_dataset.json'))['datasets'])
PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']; CONTROL = ['B', 'C1', 'C2', 'D', 'E']
rng = np.random.default_rng(0)

def basis(cloud):
    X = cloud - cloud.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False)
    c = np.cumsum(S ** 2) / (S ** 2).sum(); return Vt[:int(np.searchsorted(c, .5)) + 1]

def pout(v, B):
    for d in B: v = v - (v @ d) * d
    return v

def pain_vec(X, cats):
    pm = X[np.isin(cats, PAIN)].mean(0); C = X[np.isin(cats, CONTROL)]; cm = C.mean(0)
    pca = PCA().fit(C - cm); n = min(int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .5)) + 1, len(pca.components_))
    return pout(pm - cm, pca.components_[:n])

def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

E_all = {m: load('review', m) for m in MODELS}
out = {}
for m in MODELS:
    E = E_all[m]; X = {}; C = {}
    for k, v in D.items():
        X[k], n_in = embed([s['prompt'] for s in v['sentences']], E, m)
        C[k] = np.array([s['category'] for s in v['sentences']])
        if (n_in == 0).any(): print(m, k, 'all-OOV sentences:', int((n_in == 0).sum()))
    rows = lambda ds, labs: X[ds][np.isin(C[ds], labs)]
    means = {'S1_pain': rows('S1_1P', PAIN).mean(0), 'S2_pain': rows('S2_1P', PAIN).mean(0)}
    for key, cat in [('Fear', 'B'), ('NegEmotion', 'C1'), ('NegWorld', 'C2'), ('BodySens', 'E')]:
        means[key] = np.concatenate([rows('S1_1P', [cat]), rows('S2_1P', [cat])]).mean(0)
    for key, ds in [('Arousal', 'Arousal_1P'), ('Random', 'Random_1P'), ('Numb', 'Numb_1P'), ('Sadness', 'SD_sadness_1P')]:
        means[key] = X[ds].mean(0)
    neutral = np.concatenate([rows('S1_1P', ['D']), rows('S2_1P', ['D'])])
    controls = np.concatenate([rows('S1_1P', CONTROL), rows('S2_1P', CONTROL)])
    res = {}
    for cname in ['paper', 'common_neutral', 'common_controls']:
        cloud = controls if cname == 'common_controls' else neutral
        Bs = basis(cloud); base = cloud.mean(0)
        V = {k: pout(v - base, Bs) for k, v in means.items()}
        if cname == 'paper':
            V['S1_pain'] = pain_vec(X['S1_1P'], C['S1_1P']); V['S2_pain'] = pain_vec(X['S2_1P'], C['S2_1P'])
        res[cname] = {f'{a}x{b}': round(cos(V[a], V[b]), 3) for a in V for b in V if a < b}
    # held-out separation
    def cv_auc(ds, extra=None, reps=20):
        Xs, cs = X[ds], C[ds]; sets = np.array([s['set'] for s in D[ds]['sentences']]); us = np.unique(sets)
        aucs = {'controls': [], 'Arousal': [], 'Random': []}
        for r in range(reps):
            fold = dict(zip(us, rng.permutation(len(us)) % 5))
            f = np.array([fold[s] for s in sets]); a1, a2, a3 = [], [], []
            for k in range(5):
                trm, tem = f != k, f == k
                v = pain_vec(Xs[trm], cs[trm]); p = Xs[tem] @ v; y = np.isin(cs[tem], PAIN)
                aucs['controls'].append(roc_auc_score(y, p))
                pp = p[y]
                for nm, ods in [('Arousal', 'Arousal_1P'), ('Random', 'Random_1P')]:
                    q = X[ods] @ v; aucs[nm].append(roc_auc_score(np.r_[np.ones(len(pp)), np.zeros(len(q))], np.r_[pp, q]))
        return {k: [round(float(np.mean(v)), 3), round(float(np.std([np.mean(v[i*5:(i+1)*5]) for i in range(reps)])), 3)] for k, v in aucs.items()}
    res['heldout_auc_S2'] = cv_auc('S2_1P'); res['heldout_auc_S1'] = cv_auc('S1_1P')
    v2 = pain_vec(X['S2_1P'], C['S2_1P']); ref = X['S2_1P'] @ v2; mu, sd = ref.mean(), ref.std()
    z = lambda A: (A @ v2 - mu) / sd
    res['z'] = {nm: round(float(np.mean([z(X[f'{nm}_1P']).mean(), z(X[f'{nm}_3P']).mean()])), 2)
                for nm in ['Numb', 'Arousal', 'Random']}
    res['z']['Sadness'] = round(float(np.mean([z(X['SD_sadness_1P']).mean(), z(X['SD_sadness_3P']).mean()])), 2)
    res['z']['Numb_1P_only'] = round(float(z(X['Numb_1P']).mean()), 2)
    res['z']['pain'] = round(float(z(rows('S2_1P', PAIN)).mean()), 2); res['z']['controls'] = round(float(z(rows('S2_1P', CONTROL)).mean()), 2)
    res['z_by_pain_cat'] = {c: round(float(z(rows('S2_1P', [c])).mean()), 2) for c in PAIN}
    out[m] = res
    print(m, json.dumps(res), flush=True)
json.dump(out, open(f'{R}/pain_v2.json', 'w'), indent=1)
