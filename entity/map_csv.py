"""Write held-out predictions on the entity-covered test places (same subset as entity_probe.py)."""
import re, json, numpy as np, pandas as pd
TOK = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*"); G = '../gt_scale'
e = np.load('w2v_entities.npz'); EV = set(e['keys']); w = np.load('w2v_words.npz'); WV = set(w['keys'])
df = pd.read_csv(f'{G}/data/world_place.csv'); test = df['is_test'].astype(str).str.lower().eq('true').values
Y = df[['latitude', 'longitude']].values.astype(float)
ent = np.array([n.replace(' ', '_') in EV for n in df['name']])
wrd = np.array([any(t.lower() in WV for t in TOK.findall(n)) for n in df['name']])
ok = np.isfinite(Y).all(1) & ent & wrd & (np.load(f'{G}/static_GloVe_world_place.npz')['n_in_vocab'] > 0)
te = ok & test
out = df.loc[te, ['name', 'country', 'latitude', 'longitude']].reset_index(drop=True)
for f, k in [('Wikipedia2Vec_entity', 'entity'), ('Wikipedia2Vec_(avg)', 'w2v_words'), ('GloVe_(avg)', 'GloVe'), ('fastText_(avg)', 'fastText'), ('pythia-2.8b_L24', 'pythia'), ('llama-2-7b_L24', 'llama')]:
    P = np.load(f'pred_world_place_{f}.npy'); assert len(P) == len(out), (f, len(P), len(out))
    out[k + '_lat'], out[k + '_lon'] = P[:, 0], P[:, 1]
out.to_csv('entity_map_preds.csv', index=False); print('ok', len(out))
