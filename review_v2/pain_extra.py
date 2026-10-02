"""In-sample S2 AUC, the 'I feel:' suffix check, and the appendix pain figure (v2)."""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, re, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pain_v2 as pv   # reruns pain_v2 main block; results in pv.out
R = os.path.dirname(os.path.abspath(__file__))
from emb import load, embed, MODELS
res = {}
for m in MODELS:
    E = pv.E_all[m]; r = {}
    for ds in ['S2_1P', 'S1_1P']:
        sent = pv.D[ds]['sentences']; cats = np.array([s['category'] for s in sent])
        X, _ = embed([s['prompt'] for s in sent], E, m)
        v = pv.pain_vec(X, cats); r[f'insample_{ds}'] = round(float(roc_auc_score(np.isin(cats, pv.PAIN), X @ v)), 3)
        Xn, _ = embed([re.sub(r'\s*I feel:\s*$', '', s['prompt']) for s in sent], E, m)
        vn = pv.pain_vec(Xn, cats); r[f'insample_nosuffix_{ds}'] = round(float(roc_auc_score(np.isin(cats, pv.PAIN), Xn @ vn)), 3)
    res[m] = r; print(m, r, flush=True)
json.dump(res, open(f'{R}/pain_extra.json', 'w'), indent=1)
out = pv.out
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [1, 1.6]})
names = ['Keyword', 'GloVe', 'Word2Vec', 'fastText']
vals = [0.727] + [out[m]['heldout_auc_S2']['controls'][0] for m in MODELS]
errs = [0] + [out[m]['heldout_auc_S2']['controls'][1] for m in MODELS]
a.bar(range(4), vals, yerr=errs, color=['#bdbdbd', '#3987e5', '#3987e5', '#3987e5'], capsize=3)
a.axhspan(0.91, 1.00, color='#e6550d', alpha=0.15); a.text(1.5, 0.955, '25 LLMs (range)', ha='center', va='center', fontsize=9)
a.set_xticks(range(4)); a.set_xticklabels(names); a.set_ylim(0.5, 1.02); a.set_ylabel('Held-out AUC (S2)')
a.spines[['top', 'right']].set_visible(False); a.set_title('A', loc='left')
cons = ['paper', 'common_neutral', 'common_controls']; lab = ['Original', 'Common\nneutral', 'Common\ncontrols']
llm = {'Fear': [0.12, 0.58, 0.08], 'NegEmotion': [0.21, 0.77, 0.14]}
w = 0.2
for j, (key, nm) in enumerate([('Fear', 'fear'), ('NegEmotion', 'neg. emotion')]):
    for i, m in enumerate(MODELS):
        y = [out[m][c][f'{key}xS2_pain'] for c in cons]
        b.scatter(np.arange(3) + (j * 4) + (i - 1) * w, y, marker='o', color=['#3987e5', '#1c5cab', '#86b6ef'][i], label=m if j == 0 else None, zorder=3)
    b.scatter(np.arange(3) + j * 4, llm[key], marker='_', s=400, color='#e6550d', label='LLM mean' if j == 0 else None, zorder=2)
b.set_xticks(list(range(3)) + list(range(4, 7))); b.set_xticklabels(lab + lab, fontsize=8)
b.text(1, 0.95, 'S2 $\\times$ fear', ha='center'); b.text(5, 0.95, 'S2 $\\times$ negative emotion', ha='center')
b.set_ylim(-0.2, 1.0); b.set_ylabel('Cosine'); b.axhline(0, color='#999', lw=0.8)
b.legend(frameon=False, fontsize=8, loc='center right'); b.spines[['top', 'right']].set_visible(False); b.set_title('B', loc='left')
fig.tight_layout(); fig.savefig(_os_sb.path.join(_SBROOT, 'paper/figures/pain_static_baseline.png'), dpi=300)
print('figure saved')
