"""Wikidata ID -> English Wikipedia title via the SPARQL endpoint, 400 IDs per query."""
import csv, json, os, time, urllib.request, urllib.parse
H = list(csv.DictReader(open('../gt_scale/data/historical_figure.csv')))
ids = sorted({r['wiki_id'] for r in H if r['wiki_id'].startswith('Q')})
out = json.load(open('figure_titles_sparql.json')) if os.path.exists('figure_titles_sparql.json') else {}
todo = [q for q in ids if q not in out]
UA = {'User-Agent': 'static-baseline-research/0.1 (academic)', 'Accept': 'application/sparql-results+json'}
for i in range(0, len(todo), 2500):
    b = todo[i:i + 2500]
    q = ('SELECT ?item ?article WHERE { VALUES ?item { ' + ' '.join('wd:' + x for x in b) +
         ' } ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }')
    data = urllib.parse.urlencode({'query': q}).encode()
    for attempt in range(6):
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request('https://query.wikidata.org/sparql', data=data, headers={**UA, 'Content-Type': 'application/x-www-form-urlencoded'}), timeout=120)); break
        except Exception as e:
            print('retry', i, e, flush=True); time.sleep(65)
    else:
        continue
    for x in b: out.setdefault(x, None)
    for r in d['results']['bindings']:
        qid = r['item']['value'].rsplit('/', 1)[-1]
        out[qid] = urllib.parse.unquote(r['article']['value'].rsplit('/wiki/', 1)[-1]).replace('_', ' ')
    json.dump(out, open('figure_titles_sparql.json', 'w')); print(i + len(b), len(todo), flush=True)
    time.sleep(62)
print('SPARQLDONE', len(out), sum(v is not None for v in out.values()))
