"""Map historical-figure Wikidata IDs to English Wikipedia titles (sitelinks), cached."""
import csv, json, os, time, urllib.request, urllib.parse
H = list(csv.DictReader(open('../gt_scale/data/historical_figure.csv')))
ids = sorted({r['wiki_id'] for r in H if r['wiki_id'].startswith('Q')})
out = json.load(open('figure_titles.json')) if os.path.exists('figure_titles.json') else {}
todo = [q for q in ids if q not in out]
UA = {'User-Agent': 'static-baseline-research/0.1 (academic; contact via repository)'}
from concurrent.futures import ThreadPoolExecutor
def get(b):
    url = 'https://www.wikidata.org/w/api.php?' + urllib.parse.urlencode({'action': 'wbgetentities', 'ids': '|'.join(b), 'props': 'sitelinks', 'sitefilter': 'enwiki', 'format': 'json'})
    for attempt in range(6):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    return {}
batches = [todo[i:i + 50] for i in range(0, len(todo), 50)]
with ThreadPoolExecutor(4) as ex:
    for k, d in enumerate(ex.map(get, batches)):
        for q, e in d.get('entities', {}).items():
            out[q] = e.get('sitelinks', {}).get('enwiki', {}).get('title')
        if k % 20 == 0:
            json.dump(out, open('figure_titles.json', 'w')); print(k * 50, len(todo), flush=True)
json.dump(out, open('figure_titles.json', 'w')); print('TITLESDONE', len(out), sum(v is not None for v in out.values()))
