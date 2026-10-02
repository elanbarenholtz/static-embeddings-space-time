"""Stage 2: annual mean temperature from WorldClim 2.1 (Fick & Hijmans 2017), 1970-2000 normals,
2.5 arc-minute grid: mean of the 12 monthly mean-temperature rasters at each city's coordinates.
Cells without data (sea) fall back to the mean of valid cells within a 5x5 window.
Then writes expanded_data_sourced.json in the format of the original expanded_data.json:
[name, lat, lon, temp, continent, year_founded, elevation, gdp_pc, population]."""
import csv, json, numpy as np, tifffile
rows = list(csv.DictReader(open('cities_sourced.csv')))
stack = []
for m in range(1, 13):
    a = tifffile.imread(f'raw/wc_tavg/wc2.1_2.5m_tavg_{m:02d}.tif').astype(np.float32)
    a[a < -1e4] = np.nan; stack.append(a)
A = np.nanmean(np.stack(stack), 0) if False else np.mean(np.stack(stack), 0)   # nan if any month nan
H, W = A.shape; res = 360.0 / W
assert abs(180.0 / H - res) < 1e-9
for r in rows:
    lat, lon = float(r['lat']), float(r['lon'])
    i = int((90 - lat) / res); j = int((lon + 180) / res)
    v = A[i, j]; how = 'cell'
    if np.isnan(v):
        win = A[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]; v = np.nanmean(win); how = 'window5x5'
    r['temp_c'] = round(float(v), 2); r['src_temp'] = 'WorldClim 2.1 tavg 1970-2000 2.5m (' + how + ')'
# fills for the few gaps left by DBpedia/Wikidata/World Bank
DEM = {'Geneva': 368.0, 'Singapore': 15.0, 'Nagoya': 23.0}     # Copernicus GLO-90 via Open-Meteo elevation API
for r in rows:
    if r['elevation'] in ('', 'None') and r['name'] in DEM:
        r['elevation'] = DEM[r['name']]; r['src_elev'] = 'Copernicus DEM GLO-90 (Open-Meteo elevation API)'
    if r['name'] == 'Taipei':
        r['gdp_pc'], r['gdp_year'], r['src_gdp'] = 32319.0, 2023, 'DGBAS (Taiwan), via CEIC'
    if r['name'] == 'Pyongyang':
        r['gdp_pc'], r['gdp_year'], r['src_gdp'] = 590.0, 2022, 'UN Statistics Division AMA (UN Statistical Yearbook 67)'
    r['year_founded'] = 0; r['src_found'] = 'dropped (not used)'
fields = list(rows[0].keys())
with open('cities_sourced.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
def num(x): return None if x in ('', 'None', None) else float(x)
out = [[r['name'], round(num(r['lat']), 4), round(num(r['lon']), 4), r['temp_c'], r['continent'],
        int(num(r['year_founded'])) if num(r['year_founded']) is not None else None,
        round(num(r['elevation']), 1) if num(r['elevation']) is not None else None,
        round(num(r['gdp_pc']), 1) if num(r['gdp_pc']) is not None else None,
        int(num(r['population'])) if num(r['population']) is not None else None] for r in rows]
json.dump({'cities': out}, open('expanded_data_sourced.json', 'w'), ensure_ascii=False, indent=0)
print('window fallbacks:', [r['name'] for r in rows if 'window' in r['src_temp']])
print('missing any:', [o[0] for o in out if any(v is None for v in o)])
