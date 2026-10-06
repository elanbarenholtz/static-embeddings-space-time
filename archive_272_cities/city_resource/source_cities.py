"""Re-source the 272-city table from documented public sources.

  coordinates      DBpedia (geo:lat / geo:long), as in Gurnee & Tegmark (2024); Wikidata P625 fallback
  population       DBpedia dbo:populationTotal; Wikidata P1082 (latest dated) fallback
  elevation        DBpedia dbo:elevation (m); Wikidata P2044 fallback
  year founded     DBpedia dbo:foundingDate / dbo:foundingYear (earliest); Wikidata P571 fallback
  GDP per capita   World Bank NY.GDP.PCAP.CD (current US$), 2023 or latest year >= 2015
  temperature      WorldClim 2.1 (1970-2000), mean of the 12 monthly mean temperatures, 2.5 arc-min

Stage 1 (this file, needs network): fetch everything and cache raw responses in raw/.
Writes cities_sourced.csv (values + per-field source) and raw/*.json caches.
Temperature is added by worldclim.py.
"""
import os as _os_sb
_SBROOT = _os_sb.environ.get('SB_ROOT', _os_sb.path.dirname(_os_sb.path.dirname(_os_sb.path.abspath(__file__))))
import json, os, re, sys, time, urllib.parse, urllib.request
from titles import TITLES

UA = {'User-Agent': 'static-baseline-research/1.0 (anonymous)'}
RAW = 'raw'; os.makedirs(RAW, exist_ok=True)


def get(url, tries=4):
    import subprocess
    for k in range(tries):
        r = subprocess.run(['curl', '-sS', '-L', '-m', '60', '-A', UA['User-Agent'], url], capture_output=True, text=True)
        if r.returncode == 0:
            try:
                return json.loads(r.stdout)
            except Exception as e:
                err = 'json: ' + r.stdout[:120]
        else:
            err = r.stderr[:200]
        print('retry', k, url[:90], err, flush=True); time.sleep(3 * (k + 1))
    raise RuntimeError(url)


def cached(name, fn):
    p = os.path.join(RAW, name)
    if os.path.exists(p):
        return json.load(open(p))
    d = fn(); json.dump(d, open(p, 'w'), ensure_ascii=False, indent=1); return d


old = json.load(open(_os_sb.path.join(_SBROOT, 'city_resource', 'expanded_data_sourced.json')))['cities']
names = [r[0] for r in old]
titles = {n: TITLES.get(n, n) for n in names}

# ---- 1. Wikipedia: resolve redirects, get Wikidata QID, flag disambiguation pages
def wp():
    out = {}
    tl = sorted(set(titles.values()))
    for i in range(0, len(tl), 50):
        q = urllib.parse.urlencode({'action': 'query', 'titles': '|'.join(tl[i:i + 50]), 'redirects': 1,
                                    'prop': 'pageprops', 'format': 'json', 'formatversion': 2})
        d = get('https://en.wikipedia.org/w/api.php?' + q)['query']
        m = {t: t for t in tl[i:i + 50]}
        for x in d.get('normalized', []): m = {k: (x['to'] if v == x['from'] else v) for k, v in m.items()}
        for x in d.get('redirects', []): m = {k: (x['to'] if v == x['from'] else v) for k, v in m.items()}
        pages = {p['title']: p for p in d['pages']}
        for t, f in m.items():
            p = pages.get(f, {})
            out[t] = {'final': f, 'qid': p.get('pageprops', {}).get('wikibase_item'),
                      'disambig': 'disambiguation' in p.get('pageprops', {}), 'missing': p.get('missing', False)}
        time.sleep(0.5)
    return out
WP = cached('wikipedia.json', wp)
bad = [(n, titles[n], WP[titles[n]]) for n in names if WP[titles[n]]['disambig'] or WP[titles[n]]['missing'] or not WP[titles[n]]['qid']]
if bad:
    print('UNRESOLVED:'); [print(' ', b) for b in bad]; sys.exit(1)

# ---- 2. DBpedia
PROPS = {'lat': 'http://www.w3.org/2003/01/geo/wgs84_pos#lat', 'long': 'http://www.w3.org/2003/01/geo/wgs84_pos#long',
         'populationTotal': 'http://dbpedia.org/ontology/populationTotal',
         'populationAsOf': 'http://dbpedia.org/ontology/populationAsOf',
         'elevation': 'http://dbpedia.org/ontology/elevation',
         'foundingDate': 'http://dbpedia.org/ontology/foundingDate',
         'foundingYear': 'http://dbpedia.org/ontology/foundingYear'}
def dbp():
    from concurrent.futures import ThreadPoolExecutor
    inv = {v: k for k, v in PROPS.items()}
    os.makedirs(f'{RAW}/dbpedia_parts', exist_ok=True)
    res_of = {n: 'http://dbpedia.org/resource/' + WP[titles[n]]['final'].replace(' ', '_') for n in names}
    def one(n):
        part = f"{RAW}/dbpedia_parts/{urllib.parse.quote(n, safe='')}.json"
        if os.path.exists(part): return n, json.load(open(part))
        res = res_of[n]
        q = 'SELECT ?p ?o WHERE { <%s> ?p ?o . FILTER(?p IN (%s)) }' % (res, ','.join('<%s>' % u for u in PROPS.values()))
        url = 'https://dbpedia.org/sparql?' + urllib.parse.urlencode({'default-graph-uri': 'http://dbpedia.org', 'query': q,
                                                                       'format': 'application/sparql-results+json'})
        b = get(url, tries=8)['results']['bindings']
        rec = {'resource': res}
        for x in b: rec.setdefault(inv[x['p']['value']], []).append(x['o']['value'])
        json.dump(rec, open(part, 'w'), ensure_ascii=False); print('dbpedia', n, flush=True)
        return n, rec
    with ThreadPoolExecutor(6) as ex:
        return dict(ex.map(one, names))
DB = cached('dbpedia.json', dbp)

# ---- 3. Wikidata
def wd_entities(qs):
    out = {}
    for i in range(0, len(qs), 50):
        q = urllib.parse.urlencode({'action': 'wbgetentities', 'ids': '|'.join(qs[i:i + 50]), 'props': 'claims|labels',
                                    'languages': 'en', 'format': 'json'})
        out.update(get('https://www.wikidata.org/w/api.php?' + q).get('entities', {})); time.sleep(1)
    for q in qs:
        if q not in out or 'claims' not in out[q]:
            out[q] = get(f'https://www.wikidata.org/wiki/Special:EntityData/{q}.json')['entities'].popitem()[1]; time.sleep(0.5)
    return out
def wdq():
    E = wd_entities(sorted({WP[titles[n]]['qid'] for n in names}))
    out = {}
    for n in names:
        q = WP[titles[n]]['qid']; e = E[q]
        out[n] = {'qid': q, 'label': e['labels'].get('en', {}).get('value'),
                  'claims': {p: e['claims'].get(p, []) for p in ['P625', 'P1082', 'P2044', 'P571', 'P1249', 'P17']}}
    return out
WD = cached('wikidata.json', wdq)

def best(stmts):
    s = [x for x in stmts if x['rank'] != 'deprecated' and x['mainsnak'].get('datavalue')]
    pref = [x for x in s if x['rank'] == 'preferred']
    return pref or s

def ctry_candidates(n):
    st = [x for x in WD[n]['claims']['P17'] if x['rank'] != 'deprecated' and x['mainsnak'].get('datavalue')
          and 'P582' not in x.get('qualifiers', {})]
    st.sort(key=lambda x: x['rank'] != 'preferred')
    out = []
    for x in st:
        c = x['mainsnak']['datavalue']['value']['id']
        if c not in out: out.append(c)
    return out
def ctry(n):
    c = ctry_candidates(n); return c[0] if c else None
def countries():
    E = wd_entities(sorted({c for n in names for c in ctry_candidates(n)}))
    out = {}
    for c, e in E.items():
        iso = best(e['claims'].get('P298', []))
        out[c] = {'label': e['labels'].get('en', {}).get('value'), 'iso3': iso[0]['mainsnak']['datavalue']['value'] if iso else None}
    out.setdefault('Q55', {'label': 'Netherlands'})['iso3'] = 'NLD'          # ISO 3166 code sits on the Kingdom item
    return out
CT = cached('countries.json', countries)

# ---- 4. World Bank
WB = cached('worldbank_gdppc.json', lambda: get(
    'https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.CD?format=json&date=2015:2023&per_page=20000')[1])
gdp = {}
for r in WB:
    if r['value'] is not None:
        k = r['countryiso3code']
        if k not in gdp or int(r['date']) > gdp[k][1]: gdp[k] = (r['value'], int(r['date']))

# ---- helpers
def year_of(s):
    m = re.match(r'^\+?(-?\d{1,5})', s.strip()); return int(m.group(1)) if m else None
def wd_pop(n):
    s = best(WD[n]['claims']['P1082'])
    def date(x):
        qs = x.get('qualifiers', {}).get('P585', [])
        return qs[0]['datavalue']['value']['time'] if qs and qs[0].get('datavalue') else ''
    if not s: return None, None
    x = max(s, key=lambda x: (x['rank'] == 'preferred', date(x)))
    return float(x['mainsnak']['datavalue']['value']['amount']), date(x)[1:11]
def wd_elev(n):
    s = best(WD[n]['claims']['P2044'])
    if not s: return None
    v = s[0]['mainsnak']['datavalue']['value']; a = float(v['amount'])
    return a * 0.3048 if v['unit'].endswith('Q3710') else a
def wd_found(n, prop='P571'):
    ys = [year_of(x['mainsnak']['datavalue']['value']['time']) for x in best(WD[n]['claims'].get(prop, []))]
    ys = [y for y in ys if y is not None]; return min(ys) if ys else None
def wd_coord(n):
    s = best(WD[n]['claims']['P625'])
    if not s: return None, None
    v = s[0]['mainsnak']['datavalue']['value']; return v['latitude'], v['longitude']

rows = []
for r in old:
    n, cont = r[0], r[4]; d = DB[n]; rec = {'name': n, 'continent': cont, 'wikipedia': WP[titles[n]]['final'],
                                           'dbpedia': d['resource'], 'qid': WD[n]['qid']}
    wlat, wlon = wd_coord(n)
    if 'lat' in d and 'long' in d:
        dl, dg = float(d['lat'][0]), float(d['long'][0])
        if wlat is not None and (abs(dl - wlat) > 0.2 or abs(dg - wlon) > 0.2):
            rec.update(lat=wlat, lon=wlon, src_coord='Wikidata P625 (DBpedia disagrees by >0.2 deg)')
        else:
            rec.update(lat=dl, lon=dg, src_coord='DBpedia')
    else:
        rec.update(lat=wlat, lon=wlon, src_coord='Wikidata P625')
    rec['wd_lat'], rec['wd_lon'] = wlat, wlon
    rec['dbp_lat'], rec['dbp_lon'] = (d.get('lat') or [None])[0], (d.get('long') or [None])[0]
    wp_, wdate = wd_pop(n)
    dpop = max(float(v) for v in d['populationTotal']) if 'populationTotal' in d else None
    if wp_ is not None:
        rec.update(population=wp_, src_pop='Wikidata P1082', pop_asof=wdate)
    else:
        rec.update(population=dpop, src_pop='DBpedia populationTotal', pop_asof=(d.get('populationAsOf') or [''])[0][:10])
    rec['dbp_population'] = dpop
    we = wd_elev(n)
    de = sorted(float(v) for v in d['elevation'])[len(d['elevation']) // 2] if 'elevation' in d else None
    if we is not None:
        rec.update(elevation=we, src_elev='Wikidata P2044')
    else:
        rec.update(elevation=de, src_elev='DBpedia elevation')
    rec['dbp_elevation'] = de
    wf, wr = wd_found(n), wd_found(n, 'P1249')
    dys = [year_of(v) for v in d.get('foundingDate', []) + d.get('foundingYear', [])]
    dys = [y for y in dys if y is not None]; df_ = min(dys) if dys else None
    if wf is not None:
        rec.update(year_founded=wf, src_found='Wikidata P571')
    elif df_ is not None:
        rec.update(year_founded=df_, src_found='DBpedia foundingDate/Year')
    elif wr is not None:
        rec.update(year_founded=wr, src_found='Wikidata P1249 (earliest record)')
    else:
        rec.update(year_founded=None, src_found='MISSING')
    rec['dbp_year_founded'], rec['wd_first_record'] = df_, wr
    cands = ctry_candidates(n); c, iso = (cands[0] if cands else None), None
    for cc in cands:
        i3 = CT.get(cc, {}).get('iso3')
        if i3 in gdp or i3 in ('TWN', 'PRK'): c, iso = cc, i3; break
    if n == 'Hong Kong': iso = 'HKG'
    rec['country'] = CT.get(c, {}).get('label') if c else None; rec['iso3'] = iso
    if iso in gdp:
        rec.update(gdp_pc=gdp[iso][0], gdp_year=gdp[iso][1], src_gdp='World Bank NY.GDP.PCAP.CD')
    else:
        rec.update(gdp_pc=None, gdp_year=None, src_gdp='MISSING')
    rec['wd_year_founded'] = wf
    rec.update(old_lat=r[1], old_lon=r[2], old_temp=r[3], old_year=r[5], old_elev=r[6], old_gdp=r[7], old_pop=r[8])
    rows.append(rec)

import csv
with open('cities_sourced.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print('wrote', len(rows))
for k in ['lat', 'population', 'elevation', 'year_founded', 'gdp_pc']:
    print(k, 'missing:', [x['name'] for x in rows if x[k] is None])
