"""Stream the Wikipedia2Vec text file from stdin and keep only the entity and word vectors we need."""
import sys, csv, json, re, numpy as np
TOK = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")
P = [r['name'] for r in csv.DictReader(open('../gt_scale/data/world_place.csv'))]
H = list(csv.DictReader(open('../gt_scale/data/historical_figure.csv')))
def _ld(p):
    try: return json.load(open(p))
    except Exception: return {}
T = _ld('figure_titles.json'); T.update({k: v for k, v in json.load(open('figure_titles_sparql.json')).items() if v})
titles = set(P) | {r['name'] for r in H} | {t for t in T.values() if t}
need_e = {'ENTITY/' + t.replace(' ', '_') for t in titles}
need_w = {w.lower() for t in list(P) + [r['name'] for r in H] for w in TOK.findall(t)}
E, W = {}, {}
first = sys.stdin.readline(); print('header', first.strip(), flush=True)
for i, line in enumerate(sys.stdin):
    k, _, rest = line.rstrip('\n').partition(' ')
    if k in need_e: E[k[7:]] = np.array(rest.split(), np.float32)
    elif k in need_w: W[k] = np.array(rest.split(), np.float32)
    if i % 1000000 == 0: print(i, len(E), len(W), flush=True)
np.savez('w2v_entities.npz', keys=np.array(list(E)), vecs=np.stack(list(E.values())))
np.savez('w2v_words.npz', keys=np.array(list(W)), vecs=np.stack(list(W.values())))
print('FILTERDONE', len(E), len(W), 'of', len(need_e), len(need_w), flush=True)
