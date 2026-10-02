"""Extract from the GoogleNews binary every token the city scripts may look up."""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, os, sys, numpy as np
P = _SBROOT
sys.path.insert(0, f'{P}/review_v2'); from emb import _load_w2v, DATA
names = [c[0].lower() for c in json.load(open(_os_sb.path.join(_SBROOT, 'city_resource', 'expanded_data_sourced.json')))['cities']]
need = set()
for n in names:
    ws = n.split()
    need.add('_'.join(w.capitalize() for w in ws)); need.add('_'.join(ws))
    for w in ws: need |= {w, w.lower(), w.capitalize(), w.upper()}
os.makedirs('cache', exist_ok=True)
E = _load_w2v(f'{DATA}/w2v.gz', need)
ws = sorted(E); np.savez('cache/w2v_city_tokens.npz', words=np.array(ws), vecs=np.stack([E[w] for w in ws]))
print(len(ws), 'of', len(need))
