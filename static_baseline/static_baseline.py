#!/usr/bin/env python3
"""
static_baseline.py -- the static-embedding baseline for claims about LLM internal states.

Before attributing structure found in a language model's activations to the model, run
the identical analysis on static word embeddings of the same stimuli. Whatever the
static representation reproduces cannot be evidence of anything beyond word
co-occurrence; only the excess over this baseline is candidate evidence about the model.

Each stimulus is represented as the average of its words' vectors (GloVe, Word2Vec,
fastText). Two analyses are supported, matching the two common designs:

  direction  Binary contrast (e.g. pain vs control sentences). Difference-in-means
             direction, optionally denoised by projecting out the top principal
             components of the control cloud; held-out AUC of projections, with
             cross-validation over user-supplied groups (e.g. sentence sets) so that no
             group contributes to both fitting and evaluation.
  probe      Continuous target (e.g. latitude, birth year). Ridge regression with a
             fitted intercept and the regularisation chosen by internal CV; R^2 over
             repeated random splits, and optionally pooled R^2 under leave-one-group-out.

Input: a CSV with a text column and either a label column (direction) or a numeric
target column (probe). Optional group column.

Examples
  python static_baseline.py direction stimuli.csv --text prompt --label is_pain \\
         --group set --denoise 0.5
  python static_baseline.py probe cities.csv --text name --target latitude \\
         --group continent

Requires: numpy, pandas, scikit-learn, gensim (embeddings download on first use).
"""
import argparse, re, sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import KFold, GroupKFold, train_test_split, LeaveOneGroupOut

MODELS = {'glove': ('glove-wiki-gigaword-300', True),
          'word2vec': ('word2vec-google-news-300', False),
          'fasttext': ('fasttext-wiki-news-subwords-300', False)}
ALPHAS = np.logspace(-2, 3, 20)


def load(name):
    import gensim.downloader as api
    path, lower = MODELS[name]
    print(f'loading {name} ...', file=sys.stderr, flush=True)
    return api.load(path), lower


def embed(texts, kv, lower, strip=None):
    out, missing = [], 0
    for t in texts:
        t = str(t)
        if strip: t = re.sub(strip, '', t)
        toks = re.findall(r"[A-Za-z']+", t)
        ws = [w.lower() for w in toks] if lower else [w if w in kv else w.lower() for w in toks]
        ws = [w for w in ws if w in kv]
        if ws: out.append(np.mean([kv[w] for w in ws], 0))
        else:  out.append(np.zeros(kv.vector_size)); missing += 1
    if missing: print(f'  warning: {missing} stimuli had no in-vocabulary words', file=sys.stderr)
    return np.asarray(out, float)


# ---------------------------------------------------------------- direction analysis
def _basis(X, frac):
    p = PCA().fit(X - X.mean(0))
    k = min(int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), frac)) + 1, len(p.components_))
    return p.components_[:k]

def direction_vec(X, y, denoise):
    v = X[y == 1].mean(0) - X[y == 0].mean(0)
    if denoise:
        for b in _basis(X[y == 0], denoise): v = v - (v @ b) * b
    return v

def _auc(X, y, v):
    return roc_auc_score(y, X @ (v / np.linalg.norm(v)))

def run_direction(X, y, groups, denoise, seeds):
    res = {'insample_auc': _auc(X, y, direction_vec(X, y, denoise))}
    ug = np.unique(groups) if groups is not None else None
    per = []
    for s in range(seeds):
        if ug is not None:
            folds = KFold(5, shuffle=True, random_state=s).split(ug)
            folds = [(np.isin(groups, ug[a]), np.isin(groups, ug[b])) for a, b in folds]
        else:
            folds = [(np.isin(np.arange(len(y)), a), np.isin(np.arange(len(y)), b))
                     for a, b in KFold(5, shuffle=True, random_state=s).split(X)]
        per.append(np.mean([_auc(X[te], y[te], direction_vec(X[tr], y[tr], denoise))
                            for tr, te in folds if len(np.unique(y[te])) == 2]))
    res.update(heldout_auc_mean=float(np.mean(per)), heldout_auc_sd=float(np.std(per)))
    return res


# ---------------------------------------------------------------- probe analysis
def run_probe(X, y, groups, seeds):
    fit = lambda A, b: RidgeCV(alphas=ALPHAS).fit(A, b)
    r2 = [r2_score(y[te], fit(X[tr], y[tr]).predict(X[te]))
          for tr, te in (train_test_split(np.arange(len(y)), test_size=.2, random_state=s)
                         for s in range(seeds))]
    res = {'random_split_r2_mean': float(np.mean(r2)), 'random_split_r2_sd': float(np.std(r2))}
    if groups is not None:
        pred = np.empty(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            pred[te] = fit(X[tr], y[tr]).predict(X[te])
        res['leave_one_group_out_r2'] = float(r2_score(y, pred))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['direction', 'probe'])
    ap.add_argument('csv')
    ap.add_argument('--text', required=True, help='column holding the stimulus text')
    ap.add_argument('--label', help='direction mode: column with 1 = target class, 0 = control')
    ap.add_argument('--target', help='probe mode: numeric target column')
    ap.add_argument('--group', help='optional grouping column for cross-validation / holdout')
    ap.add_argument('--models', default='glove,word2vec,fasttext')
    ap.add_argument('--denoise', type=float, default=0.0,
                    help='direction mode: project out control PCs up to this variance fraction')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--strip', default=None, help='regex removed from every stimulus first')
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    groups = df[a.group].to_numpy() if a.group else None
    rows = []
    for m in a.models.split(','):
        kv, lower = load(m.strip())
        X = embed(df[a.text], kv, lower, a.strip)
        if a.mode == 'direction':
            r = run_direction(X, df[a.label].astype(int).to_numpy(), groups, a.denoise, a.seeds)
        else:
            r = run_probe(X, df[a.target].astype(float).to_numpy(), groups, a.seeds)
        rows.append({'embedding': m, **r}); del kv
    out = pd.DataFrame(rows)
    print('\nStatic-embedding baseline (no model):')
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nCompare these against the same metrics computed on model activations. Anything the\n'
          'static baseline reproduces is explained by word co-occurrence; only the excess is\n'
          'candidate evidence about the model.')

if __name__ == '__main__':
    main()
