"""Emotion analyses on static embeddings, with per-emotion detail and topic-bootstrap CIs.

Stories: replication corpus (Ogunlana 2026), 12 emotions, split by topic as shipped (25% of
topics held out, the same for every emotion). Neutral Person/AI dialogues from training topics
define the denoising basis (PCs reaching 50% of their variance); all directions, PCs and
standardizations are fit on training topics only.
  story direction: mean of an emotion's training stories minus mean of the other emotions'
  word direction: label vector minus mean of the other eleven label vectors
Scores: per-emotion one-vs-rest AUC on held-out topics (macro mean), and 12-way accuracy
(unit directions, scores z-scored on training stories, argmax). 95% CIs: 2000 bootstrap
resamples of held-out topics; paired CI for story minus word.

GoEmotions (Demszky et al. 2020): official train/test split; multi-label, so for each of the
27 emotions the positives are comments carrying that label and the negatives all other
comments. Direction = mean(train positives) - mean(train negatives), denoised against PCs of
train comments labeled only 'neutral'. Macro-mean test AUC over the 27 emotions. 'Emotion
words removed' drops every comment (train and test) containing a label word, a WordNet
synonym of it, or a derivationally related form.
"""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, glob, os, csv, numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score
from emb import load, embed, MODELS, words
R = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

def basis(cloud):
    X = cloud - cloud.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False)
    c = np.cumsum(S ** 2) / (S ** 2).sum(); return Vt[:int(np.searchsorted(c, .5)) + 1]

def pout(v, B):
    v = v - B.T @ (B @ v) if B.ndim == 2 and len(B) else v
    return v

# ---------------- stories
rows = []
for f in sorted(glob.glob(_os_sb.path.join(_SBROOT, 'review_v2', 'data', 'qwen-emotion-stories', 'corpus', '*.jsonl'))):
    for l in open(f):
        rows.append(json.loads(l))
EMO = sorted({r['emotion'] for r in rows if r['emotion'] != 'neutral'})
counts = {e: dict(Counter(r['split'] for r in rows if r['emotion'] == e)) for e in EMO + ['neutral']}
out = {'counts': counts}
BAL = []
for m in MODELS:
    E = load('review', m)
    X, nin = embed([r['text'] for r in rows], E, m)
    lab = np.array([r['emotion'] for r in rows]); split = np.array([r['split'] for r in rows]); topic = np.array([r['topic'] for r in rows])
    trn = split == 'train'
    B = basis(X[trn & (lab == 'neutral')])
    em = np.isin(lab, EMO)
    Lv, _ = embed(EMO, E, m)
    dirs = {}
    for e in EMO:
        s = X[trn & (lab == e)].mean(0) - X[trn & em & (lab != e)].mean(0)
        w = Lv[EMO.index(e)] - np.delete(Lv, EMO.index(e), 0).mean(0)
        dirs[('story', e)] = pout(s, B); dirs[('word', e)] = pout(w, B)
    te = ~trn & em; Xt = X[te]; yt = lab[te]; tt = topic[te]
    res = {}
    for kind in ['story', 'word']:
        D = np.stack([dirs[(kind, e)] / np.linalg.norm(dirs[(kind, e)]) for e in EMO])
        Str = X[trn & em] @ D.T; mu, sd = Str.mean(0), Str.std(0)
        S = (Xt @ D.T - mu) / sd
        res[kind] = {'S': S}
    def metrics(S, sel):
        aucs = [roc_auc_score(yt[sel] == e, S[sel, i]) for i, e in enumerate(EMO)]
        pred = np.array(EMO)[S[sel].argmax(1)]; acc = float(np.mean(pred == yt[sel]))
        BAL.append(float(np.mean([np.mean(pred[yt[sel] == e] == e) for e in EMO if (yt[sel] == e).any()])))
        return aucs, acc
    summ = {}
    full = np.ones(len(yt), bool)
    for kind in ['story', 'word']:
        a, acc = metrics(res[kind]['S'], full)
        summ[kind] = {'auc_macro': round(float(np.mean(a)), 3), 'acc': round(acc, 3), 'balanced_acc': round(BAL[-1], 3), 'auc_by_emotion': dict(zip(EMO, [round(x, 3) for x in a]))}
    ut = np.unique(tt); boots = {'story_auc': [], 'word_auc': [], 'story_acc': [], 'word_acc': [], 'diff_auc': [], 'diff_acc': []}
    per_em = {('story', e): [] for e in EMO}
    for b in range(2000):
        pick = rng.choice(ut, len(ut), replace=True)
        sel = np.concatenate([np.where(tt == t)[0] for t in pick])
        if any((yt[sel] == e).sum() == 0 for e in EMO): continue
        a1, c1 = metrics(res['story']['S'][sel], np.ones(len(sel), bool)) if False else (None, None)
        Ss, Sw = res['story']['S'][sel], res['word']['S'][sel]; ys = yt[sel]
        as_ = [roc_auc_score(ys == e, Ss[:, i]) for i, e in enumerate(EMO)]
        aw = [roc_auc_score(ys == e, Sw[:, i]) for i, e in enumerate(EMO)]
        cs = float(np.mean(np.array(EMO)[Ss.argmax(1)] == ys)); cw = float(np.mean(np.array(EMO)[Sw.argmax(1)] == ys))
        boots['story_auc'].append(np.mean(as_)); boots['word_auc'].append(np.mean(aw))
        boots['story_acc'].append(cs); boots['word_acc'].append(cw)
        boots['diff_auc'].append(np.mean(as_) - np.mean(aw)); boots['diff_acc'].append(cs - cw)
        for i, e in enumerate(EMO): per_em[('story', e)].append(as_[i])
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    summ['ci'] = {k: ci(v) for k, v in boots.items()}
    summ['story']['auc_by_emotion_ci'] = {e: ci(per_em[('story', e)]) for e in EMO}
    summ['n_boot'] = len(boots['story_auc']); summ['n_test'] = int(te.sum()); summ['n_test_topics'] = int(len(ut))
    summ['n_pcs'] = int(len(B))
    out[m] = {'stories': summ}
    print(m, json.dumps(summ), flush=True)

# ---------------- GoEmotions
from nltk.corpus import wordnet as wn
labels = open(f'{R}/data/emotions.txt').read().split()
def lexicon(lbl):
    s = {lbl}
    for syn in wn.synsets(lbl):
        for l in syn.lemmas():
            s.add(l.name().lower())
            for d in l.derivationally_related_forms(): s.add(d.name().lower())
    return {w for w in s if '_' not in w and len(w) > 2}
LEX = set()
for l in labels:
    if l != 'neutral': LEX |= lexicon(l)
def read(split):
    T, L = [], []
    for row in csv.reader(open(f'{R}/data/{split}.tsv'), delimiter='\t'):
        T.append(row[0]); L.append([labels[int(i)] for i in row[1].split(',')])
    return T, L
trT, trL = read('train'); teT, teL = read('test')
has_lex = lambda t: any(w.lower() in LEX for w in words(t))
EMO27 = [l for l in labels if l != 'neutral']
for m in MODELS:
    E = load('review', m)
    Xtr, ntr = embed(trT, E, m); Xte, nte = embed(teT, E, m)
    res = {}
    for cond in ['all', 'no_emotion_words']:
        ktr = np.array([n > 0 for n in ntr]); kte = np.array([n > 0 for n in nte])
        if cond == 'no_emotion_words':
            ktr &= ~np.array([has_lex(t) for t in trT]); kte &= ~np.array([has_lex(t) for t in teT])
        neu = ktr & np.array([l == ['neutral'] for l in trL])
        B = basis(Xtr[neu])
        aucs = {}
        for e in EMO27:
            ptr = np.array([e in l for l in trL]); pte = np.array([e in l for l in teL])
            if (ptr & ktr).sum() < 5 or (pte & kte).sum() < 5: continue
            v = pout(Xtr[ktr & ptr].mean(0) - Xtr[ktr & ~ptr].mean(0), B)
            aucs[e] = roc_auc_score(pte[kte], Xte[kte] @ v)
        res[cond] = {'auc_macro': round(float(np.mean(list(aucs.values()))), 3), 'n_emotions': len(aucs),
                     'n_train': int(ktr.sum()), 'n_test': int(kte.sum()),
                     'auc_by_emotion': {k: round(float(v), 3) for k, v in aucs.items()}}
    res['lexicon_size'] = len(LEX)
    out[m]['goemotions'] = res
    print(m, 'GoEmotions', json.dumps({c: {k: v for k, v in res[c].items() if k != 'auc_by_emotion'} for c in ['all', 'no_emotion_words']}), flush=True)
json.dump(out, open(f'{R}/emotion_v2b.json', 'w'), indent=1)
