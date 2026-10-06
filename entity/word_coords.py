"""Which GloVe words' similarity to a place name tracks its latitude / longitude?  (released places)
Usage: python word_coords.py
For every candidate word (20,000 most frequent GloVe words, at least four letters, alphabetic, not on the
country/demonym/geographic exclusion list in exclusion_words.json, and not a token that occurs in at least three
place names), correlate cosine similarity to each place's averaged name vector with the place's latitude and longitude.
Writes word_coords_released.json; make_word_figure.py draws the figure from it."""
import json, re, numpy as np, pandas as pd
from collections import Counter
P = '..'
excl = set(json.load(open('exclusion_words.json'))['words'])
df = pd.read_csv(f'{P}/gt_scale/data/world_place.csv'); names = [str(n).lower() for n in df['name']]
lat = df['latitude'].values.astype(float); lon = df['longitude'].values.astype(float)
TOK = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
toks = [TOK.findall(n) for n in names]
cnt = Counter(t for ts in toks for t in set(ts)); namewords = {t for t, c in cnt.items() if c >= 3}
need = {t for ts in toks for t in ts}; G = {}; vocab, Wv = [], []
with open(f'{P}/gt_scale/data/glove.6B.300d.txt', encoding='utf8') as f:
    for i, line in enumerate(f):
        w, rest = line.split(' ', 1)
        if i < 20000: vocab.append(w); Wv.append(np.array(rest.split(), np.float32))
        if w in need: G[w] = np.array(rest.split(), np.float32)
Wv = np.stack(Wv)
ok_item = np.array([any(t in G for t in ts) for ts in toks])
X = np.stack([np.mean([G[t] for t in ts if t in G], 0) if o else np.zeros(300, np.float32) for ts, o in zip(toks, ok_item)])
ok_w = np.array([len(w) >= 4 and w.isalpha() and w not in excl and w not in namewords for w in vocab])
Wn = Wv[ok_w] / np.linalg.norm(Wv[ok_w], axis=1, keepdims=True); wk = np.array(vocab)[ok_w]
Xn = X[ok_item] / np.linalg.norm(X[ok_item], axis=1, keepdims=True)
out = {'n_items': int(ok_item.sum()), 'n_words': int(ok_w.sum())}
for t, y in [('latitude', lat), ('longitude', lon)]:
    y = y[ok_item]; yc = y - y.mean(); r = np.zeros(len(wk))
    for a in range(0, len(wk), 1000):
        S = Wn[a:a+1000] @ Xn.T; Sc = S - S.mean(1, keepdims=True)
        r[a:a+1000] = (Sc @ yc) / (np.linalg.norm(Sc, axis=1) * np.linalg.norm(yc))
    o = np.argsort(r)
    out[t] = {'top_pos': [(wk[i], round(float(r[i]), 3)) for i in o[::-1][:30]], 'top_neg': [(wk[i], round(float(r[i]), 3)) for i in o[:30]]}
    print(t, 'POS', [(w, round(x, 2)) for w, x in out[t]['top_pos'][:20]]); print(t, 'NEG', [(w, round(x, 2)) for w, x in out[t]['top_neg'][:20]], flush=True)
json.dump(out, open('word_coords_released.json', 'w'), indent=1)
