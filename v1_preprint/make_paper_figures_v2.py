"""
Publication figures for LaTeX paper. Clean plots only — no captions,
no editorializing, no R² in titles. All explanatory text in LaTeX.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import os, sys

# ============================================================
# STYLE
# ============================================================
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

# ============================================================
# DATA
# ============================================================
cities = [
    ("anchorage", 61.2, -149.9, 2, 290000, 76000, 30, 1914),
    ("seattle", 47.6, -122.3, 11, 750000, 76000, 56, 1851),
    ("portland", 45.5, -122.7, 12, 650000, 76000, 15, 1845),
    ("san francisco", 37.8, -122.4, 14, 870000, 76000, 16, 1776),
    ("los angeles", 34.1, -118.2, 18, 3900000, 76000, 71, 1781),
    ("san diego", 32.7, -117.2, 18, 1420000, 76000, 19, 1769),
    ("phoenix", 33.4, -112.1, 23, 1680000, 76000, 331, 1867),
    ("denver", 39.7, -105.0, 10, 715000, 76000, 1609, 1858),
    ("dallas", 32.8, -96.8, 19, 1340000, 76000, 131, 1841),
    ("houston", 29.8, -95.4, 21, 2300000, 76000, 15, 1836),
    ("austin", 30.3, -97.7, 20, 980000, 76000, 149, 1839),
    ("chicago", 41.9, -87.6, 10, 2700000, 76000, 176, 1833),
    ("detroit", 42.3, -83.0, 10, 640000, 76000, 183, 1701),
    ("minneapolis", 44.98, -93.3, 7, 430000, 76000, 264, 1856),
    ("miami", 25.8, -80.2, 25, 470000, 76000, 2, 1896),
    ("atlanta", 33.7, -84.4, 17, 500000, 76000, 320, 1837),
    ("boston", 42.4, -71.1, 11, 690000, 76000, 6, 1630),
    ("new york", 40.7, -74.0, 13, 8300000, 76000, 10, 1624),
    ("philadelphia", 40.0, -75.2, 13, 1580000, 76000, 12, 1682),
    ("washington", 38.9, -77.0, 14, 690000, 76000, 22, 1790),
    ("nashville", 36.2, -86.8, 15, 680000, 76000, 117, 1779),
    ("memphis", 35.1, -90.0, 17, 630000, 76000, 81, 1819),
    ("cleveland", 41.5, -81.7, 10, 380000, 76000, 199, 1796),
    ("pittsburgh", 40.4, -80.0, 11, 300000, 76000, 367, 1758),
    ("milwaukee", 43.0, -87.9, 9, 580000, 76000, 188, 1846),
    ("new orleans", 30.0, -90.1, 21, 380000, 76000, -0.5, 1718),
    ("kansas city", 39.1, -94.6, 13, 510000, 76000, 230, 1838),
    ("las vegas", 36.2, -115.1, 20, 650000, 76000, 610, 1905),
    ("salt lake city", 40.8, -111.9, 11, 200000, 76000, 1288, 1847),
    ("toronto", 43.7, -79.4, 9, 2800000, 52000, 76, 1793),
    ("montreal", 45.5, -73.6, 7, 1780000, 52000, 36, 1642),
    ("vancouver", 49.3, -123.1, 10, 680000, 52000, 0, 1886),
    ("mexico city", 19.4, -99.1, 17, 9200000, 20000, 2240, 1325),
    ("london", 51.5, -0.1, 11, 8980000, 46000, 11, 43),
    ("paris", 48.9, 2.3, 12, 2160000, 44000, 35, -250),
    ("berlin", 52.5, 13.4, 10, 3640000, 51000, 34, 1237),
    ("madrid", 40.4, -3.7, 15, 3200000, 38000, 650, 860),
    ("rome", 41.9, 12.5, 16, 2870000, 34000, 21, -753),
    ("amsterdam", 52.4, 4.9, 10, 870000, 57000, -2, 1275),
    ("brussels", 50.8, 4.4, 10, 1200000, 50000, 13, 979),
    ("vienna", 48.2, 16.4, 11, 1900000, 53000, 171, -500),
    ("prague", 50.1, 14.4, 10, 1300000, 40000, 235, 885),
    ("warsaw", 52.2, 21.0, 9, 1790000, 35000, 100, 1300),
    ("budapest", 47.5, 19.0, 12, 1750000, 33000, 96, 89),
    ("lisbon", 38.7, -9.1, 17, 550000, 32000, 2, -1200),
    ("dublin", 53.3, -6.3, 10, 550000, 85000, 8, 841),
    ("edinburgh", 55.9, -3.2, 9, 530000, 46000, 47, 640),
    ("stockholm", 59.3, 18.1, 7, 980000, 55000, 28, 1252),
    ("oslo", 59.9, 10.7, 6, 700000, 82000, 23, 1040),
    ("helsinki", 60.2, 24.9, 5, 660000, 49000, 26, 1550),
    ("copenhagen", 55.7, 12.6, 9, 800000, 58000, 14, 1167),
    ("moscow", 55.8, 37.6, 6, 12600000, 28000, 156, 1147),
    ("athens", 38.0, 23.7, 18, 660000, 28000, 70, -3000),
    ("istanbul", 41.0, 29.0, 15, 15500000, 28000, 40, -660),
    ("barcelona", 41.4, 2.2, 16, 1620000, 38000, 12, -230),
    ("munich", 48.1, 11.6, 9, 1470000, 51000, 519, 1158),
    ("zurich", 47.4, 8.5, 9, 420000, 87000, 408, 929),
    ("milan", 45.5, 9.2, 13, 1370000, 34000, 120, -400),
    ("marseille", 43.3, 5.4, 15, 870000, 44000, 12, -600),
    ("tokyo", 35.7, 139.7, 16, 13960000, 42000, 40, 1457),
    ("beijing", 39.9, 116.4, 13, 21540000, 17000, 43, -1045),
    ("shanghai", 31.2, 121.5, 17, 24870000, 17000, 4, 991),
    ("mumbai", 19.1, 72.9, 27, 12440000, 7000, 14, 1507),
    ("delhi", 28.6, 77.2, 25, 16780000, 7000, 216, -1500),
    ("bangkok", 13.8, 100.5, 28, 10540000, 16000, 2, 1782),
    ("singapore", 1.4, 103.9, 27, 5690000, 65000, 15, 1819),
    ("hong kong", 22.3, 114.2, 23, 7500000, 50000, 32, 1842),
    ("seoul", 37.6, 127.0, 13, 9770000, 33000, 38, -18),
    ("taipei", 25.0, 121.5, 23, 2650000, 33000, 9, 1709),
    ("osaka", 34.7, 135.5, 17, 2750000, 42000, 0, 1583),
    ("jakarta", -6.2, 106.8, 27, 10560000, 12000, 8, 397),
    ("manila", 14.6, 121.0, 28, 1850000, 8000, 16, 1571),
    ("hanoi", 21.0, 105.8, 24, 8050000, 8000, 12, 1010),
    ("dubai", 25.3, 55.3, 28, 3400000, 44000, 5, 1833),
    ("tehran", 35.7, 51.4, 17, 8700000, 13000, 1189, -3000),
    ("baghdad", 33.3, 44.4, 23, 7100000, 9000, 34, 762),
    ("kabul", 34.5, 69.2, 13, 4600000, 2000, 1791, -1500),
    ("karachi", 24.9, 67.0, 26, 14900000, 5000, 8, 1729),
    ("cairo", 30.0, 31.2, 22, 10230000, 12000, 75, 969),
    ("lagos", 6.5, 3.4, 27, 15400000, 5000, 41, 1472),
    ("nairobi", -1.3, 36.8, 18, 4400000, 4000, 1661, 1899),
    ("johannesburg", -26.2, 28.0, 16, 5780000, 13000, 1753, 1886),
    ("cape town", -33.9, 18.4, 17, 4620000, 13000, 0, 1652),
    ("casablanca", 33.6, -7.6, 18, 3720000, 7000, 27, 768),
    ("addis ababa", 9.0, 38.7, 16, 3600000, 2000, 2355, 1886),
    ("accra", 5.6, -0.2, 27, 2270000, 5000, 61, 1877),
    ("algiers", 36.8, 3.1, 18, 3600000, 11000, 0, 944),
    ("buenos aires", -34.6, -58.4, 17, 3060000, 22000, 25, 1536),
    ("santiago", -33.4, -70.6, 14, 6160000, 25000, 520, 1541),
    ("lima", -12.0, -77.0, 19, 10000000, 12000, 161, 1535),
    ("bogota", 4.7, -74.1, 14, 7410000, 14000, 2640, 1538),
    ("rio de janeiro", -22.9, -43.2, 24, 6750000, 15000, 11, 1565),
    ("sao paulo", -23.6, -46.6, 20, 12300000, 15000, 760, 1554),
    ("caracas", 10.5, -66.9, 22, 2940000, 16000, 900, 1567),
    ("quito", -0.2, -78.5, 14, 2800000, 11000, 2850, 1534),
    ("sydney", -33.9, 151.2, 18, 5300000, 52000, 3, 1788),
    ("melbourne", -37.8, 144.96, 15, 5000000, 52000, 31, 1835),
    ("auckland", -36.8, 174.8, 15, 1660000, 42000, 0, 1840),
    ("perth", -31.9, 115.9, 19, 2100000, 52000, 0, 1829),
]

regions = {
    'North America': ['anchorage', 'seattle', 'portland', 'san francisco', 'los angeles',
                      'san diego', 'phoenix', 'denver', 'dallas', 'houston', 'austin',
                      'chicago', 'detroit', 'minneapolis', 'miami', 'atlanta', 'boston',
                      'new york', 'philadelphia', 'washington', 'nashville', 'memphis',
                      'cleveland', 'pittsburgh', 'milwaukee', 'new orleans', 'kansas city',
                      'las vegas', 'salt lake city', 'toronto', 'montreal', 'vancouver',
                      'mexico city'],
    'Europe': ['london', 'paris', 'berlin', 'madrid', 'rome', 'amsterdam', 'brussels',
               'vienna', 'prague', 'warsaw', 'budapest', 'lisbon', 'dublin', 'edinburgh',
               'stockholm', 'oslo', 'helsinki', 'copenhagen', 'moscow', 'athens', 'istanbul',
               'barcelona', 'munich', 'zurich', 'milan', 'marseille'],
    'Asia': ['tokyo', 'beijing', 'shanghai', 'mumbai', 'delhi', 'bangkok', 'singapore',
             'hong kong', 'seoul', 'taipei', 'osaka', 'jakarta', 'manila', 'hanoi',
             'dubai', 'tehran', 'baghdad', 'kabul', 'karachi'],
    'Africa': ['cairo', 'lagos', 'nairobi', 'johannesburg', 'cape town', 'casablanca',
               'addis ababa', 'accra', 'algiers'],
    'South America': ['buenos aires', 'santiago', 'lima', 'bogota', 'rio de janeiro',
                      'sao paulo', 'caracas', 'quito'],
    'Oceania': ['sydney', 'melbourne', 'auckland', 'perth'],
}

color_map = {
    'North America': '#E74C3C', 'Europe': '#3498DB', 'Asia': '#F39C12',
    'Africa': '#27AE60', 'South America': '#9B59B6', 'Oceania': '#1ABC9C',
}

def get_color(name):
    for r, cl in regions.items():
        if name in cl:
            return color_map[r]
    return '#888888'

# ============================================================
# LOAD EMBEDDINGS
# ============================================================
GLOVE_PATH = "glove.6B.300d.txt"
if not os.path.exists(GLOVE_PATH):
    print(f"ERROR: {GLOVE_PATH} not found"); sys.exit(1)

print("Loading GloVe...")
needed_words = set()
for name, *_ in cities:
    for w in name.split():
        needed_words.add(w)
glove_embs = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if parts[0] in needed_words:
            glove_embs[parts[0]] = np.array([float(x) for x in parts[1:]])

print("Loading Word2Vec...")
import gensim.downloader as api
w2v_model = api.load('word2vec-google-news-300')

w2v_keys = {}
for name, *_ in cities:
    phrase = '_'.join(w.capitalize() for w in name.split())
    if phrase in w2v_model:
        w2v_keys[name] = ('phrase', phrase)
    else:
        found = []
        for w in name.split():
            if w.capitalize() in w2v_model: found.append(w.capitalize())
            elif w in w2v_model: found.append(w)
        w2v_keys[name] = ('words', found) if found else ('missing', None)

def glove_vec(name):
    vecs = [glove_embs[w] for w in name.split() if w in glove_embs]
    return np.mean(vecs, axis=0) if vecs else None

def w2v_vec(name):
    mode, key = w2v_keys[name]
    if mode == 'phrase': return w2v_model[key]
    elif mode == 'words' and key: return np.mean([w2v_model[w] for w in key], axis=0)
    return None

valid = []
for name, lat, lon, temp, pop, gdp, elev, year in cities:
    ge, we = glove_vec(name), w2v_vec(name)
    if ge is None or we is None: continue
    valid.append(dict(name=name, lat=lat, lon=lon, temp=temp, pop=pop,
                      gdp=gdp, elev=elev, year=year, glove=ge, w2v=we,
                      color=get_color(name)))

names = [v['name'] for v in valid]
colors = [v['color'] for v in valid]
X_g = np.array([v['glove'] for v in valid])
X_w = np.array([v['w2v'] for v in valid])
n = len(valid)
print(f"  {n} cities")

targets = {
    'Latitude': np.array([v['lat'] for v in valid]),
    'Longitude': np.array([v['lon'] for v in valid]),
    'Temperature': np.array([v['temp'] for v in valid]),
    'Year Founded': np.array([v['year'] for v in valid]),
    'Elevation': np.array([v['elev'] for v in valid]),
    'GDP per capita': np.log10(np.array([v['gdp'] for v in valid])),
    'Population': np.log10(np.array([v['pop'] for v in valid])),
}

# ============================================================
# PROBES
# ============================================================
lambdas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

def run_all_probes(X):
    d = X.shape[1]
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_folds, fold_size = 5, n // 5
    n_train = int(0.8 * n)
    tr, te = idx[:n_train], idx[n_train:]
    results = {}
    for tname, y in targets.items():
        best_lam, best_cv = None, -np.inf
        for lam in lambdas:
            fr2 = []
            for fold in range(n_folds):
                mask = np.zeros(n, dtype=bool)
                mask[fold * fold_size:(fold + 1) * fold_size] = True
                Xtr, Xte_ = X[idx[~mask]], X[idx[mask]]
                ytr, yte_ = y[idx[~mask]], y[idx[mask]]
                W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ ytr)
                yp = Xte_ @ W
                ss_r = np.sum((yte_ - yp)**2)
                ss_t = np.sum((yte_ - yte_.mean())**2)
                fr2.append(1 - ss_r / ss_t if ss_t > 0 else 0)
            cv = np.mean(fr2)
            if cv > best_cv: best_cv, best_lam = cv, lam
        W = np.linalg.solve(X[tr].T @ X[tr] + best_lam * np.eye(d), X[tr].T @ y[tr])
        yp_te = X[te] @ W
        yp_all = X @ W
        ss_r = np.sum((y[te] - yp_te)**2)
        ss_t = np.sum((y[te] - y[te].mean())**2)
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        mae = np.mean(np.abs(y[te] - yp_te))
        results[tname] = dict(r2=r2, mae=mae, lam=best_lam,
                              yp_all=yp_all, y=y, tr=tr, te=te)
    return results

print("Running probes...")
G = run_all_probes(X_g)
W = run_all_probes(X_w)
tr, te = G['Latitude']['tr'], G['Latitude']['te']
te_set = set(te)

for t in targets:
    print(f"  {t:<18s} GloVe={G[t]['r2']:+.3f}  W2V={W[t]['r2']:+.3f}")

# ============================================================
# ANNOTATION HELPER
# ============================================================
def annotate_city(ax, x, y, name, is_test, fontsize=9.5):
    """Label a city with context-aware offset to reduce overlap."""
    dx, dy = 6, 6
    n_lower = name.lower()
    # Per-city manual nudges for legibility
    nudges = {
        'anchorage': (6, -12), 'buenos aires': (-8, -12),
        'moscow': (6, -12), 'helsinki': (6, -12),
        'sydney': (-42, -12), 'tokyo': (-32, 6),
        'beijing': (6, -12), 'mumbai': (-42, 6),
        'singapore': (-48, -12), 'bangkok': (6, -12),
        'miami': (6, -10), 'cairo': (6, -10),
        'nairobi': (6, -10),
    }
    if n_lower in nudges:
        dx, dy = nudges[n_lower]

    weight = 'bold' if is_test else 'normal'
    alpha = 1.0 if is_test else 0.65
    style = 'normal' if is_test else 'italic'

    ax.annotate(name.title(), (x, y), fontsize=fontsize,
                fontweight=weight, fontstyle=style, alpha=alpha,
                zorder=10, xytext=(dx, dy), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                         alpha=0.75, edgecolor='none'))


# ============================================================
# SHARED LEGEND BUILDERS
# ============================================================
def make_map_legend():
    """Region patches + train/test markers for Figures 1 and 3."""
    els = [Patch(facecolor=color_map[r], edgecolor='white', label=r)
           for r in regions]
    els.append(Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#999', markersize=7, alpha=0.4,
                      label='Training data (known to probe)'))
    els.append(Line2D([0], [0], marker='D', color='w',
                      markerfacecolor='#999', markeredgecolor='white',
                      markersize=7, alpha=0.9,
                      label='Held-out test predictions'))
    return els


# ============================================================
# FIGURE 1: GEOGRAPHY (3 panels)
# ============================================================
geo_labels = ['anchorage', 'new york', 'los angeles', 'miami', 'london',
              'moscow', 'beijing', 'tokyo', 'cairo', 'mumbai', 'nairobi',
              'buenos aires', 'sydney']

print("\nFigure 1...")
fig1, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(20, 7))


def plot_geo_actual(ax, lons, lats):
    for i in range(n):
        ax.scatter(lons[i], lats[i], c=colors[i], s=48, alpha=0.85,
                   marker='o', edgecolors='white', linewidth=0.5, zorder=5)
    for i in range(n):
        if names[i] in geo_labels:
            annotate_city(ax, lons[i], lats[i], names[i], True)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, alpha=0.15, linewidth=0.5)


def plot_geo_probe(ax, lons, lats):
    for i in tr:
        ax.scatter(lons[i], lats[i], c=colors[i], s=30, alpha=0.4,
                   marker='o', edgecolors='white', linewidth=0.5, zorder=3)
    for i in te:
        ax.scatter(lons[i], lats[i], c=colors[i], s=60, alpha=0.9,
                   marker='D', edgecolors='white', linewidth=0.5, zorder=6)
    for i in range(n):
        if names[i] in geo_labels:
            annotate_city(ax, lons[i], lats[i], names[i], i in te_set)
    ax.set_xlabel('Predicted Longitude', fontsize=11)
    ax.set_ylabel('Predicted Latitude', fontsize=11)
    ax.grid(True, alpha=0.15, linewidth=0.5)


plot_geo_actual(ax_a, targets['Longitude'], targets['Latitude'])
ax_a.set_title('A. Actual Geography', fontsize=12, fontweight='bold')

plot_geo_probe(ax_b, G['Longitude']['yp_all'], G['Latitude']['yp_all'])
ax_b.set_title('B. GloVe 300d', fontsize=12, fontweight='bold')

plot_geo_probe(ax_c, W['Longitude']['yp_all'], W['Latitude']['yp_all'])
ax_c.set_title('C. Word2Vec 300d', fontsize=12, fontweight='bold')

fig1.legend(handles=make_map_legend(), loc='lower center', ncol=4,
            fontsize=9.5, frameon=True, fancybox=True, edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.06))
fig1.tight_layout()
fig1.subplots_adjust(bottom=0.12)
fig1.savefig('fig1_geography.png', dpi=300, bbox_inches='tight')
print("  Saved fig1_geography.png")


# ============================================================
# FIGURE 2: BAR CHART
# ============================================================
print("Figure 2...")
target_order = ['Longitude', 'Latitude', 'Temperature', 'Year Founded',
                'Elevation', 'GDP per capita', 'Population']

g_r2 = [G[t]['r2'] for t in target_order]
w_r2 = [W[t]['r2'] for t in target_order]
clip_lo = -0.5
g_disp = [max(v, clip_lo) for v in g_r2]
w_disp = [max(v, clip_lo) for v in w_r2]

fig2, ax2 = plt.subplots(figsize=(10, 6))
x = np.arange(len(target_order))
bw = 0.35

bars_g = ax2.bar(x - bw/2, g_disp, bw, color='#4472C4', edgecolor='white',
                 linewidth=1.2, alpha=0.88, label='GloVe 300d (2014)')
bars_w = ax2.bar(x + bw/2, w_disp, bw, color='#ED7D31', edgecolor='white',
                 linewidth=1.2, alpha=0.88, label='Word2Vec 300d (2013)')

for bars, r2_actual in [(bars_g, g_r2), (bars_w, w_r2)]:
    for bar, r2a in zip(bars, r2_actual):
        h = bar.get_height()
        clipped = r2a < clip_lo
        if clipped:
            ax2.text(bar.get_x() + bar.get_width()/2, clip_lo - 0.02,
                     f'{r2a:.2f}', ha='center', va='top', fontsize=8,
                     fontweight='bold', color='#C0392B')
        elif h >= 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                     f'{r2a:.2f}', ha='center', va='bottom', fontsize=8,
                     fontweight='bold')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, h - 0.02,
                     f'{r2a:.2f}', ha='center', va='top', fontsize=8,
                     fontweight='bold')

ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='--', zorder=1)
ax2.set_xticks(x)
ax2.set_xticklabels(target_order, fontsize=10, rotation=20, ha='right')
ax2.set_ylabel('Test R\u00B2', fontsize=12)
ax2.set_ylim(-0.5, 1.0)
ax2.grid(True, alpha=0.15, axis='y', linewidth=0.5)
ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)

fig2.tight_layout()
fig2.savefig('fig2_barplot.png', dpi=300, bbox_inches='tight')
print("  Saved fig2_barplot.png")


# ============================================================
# FIGURE 3: TEMPERATURE (2 panels)
# ============================================================
temp_labels = ['anchorage', 'helsinki', 'moscow',
               'miami', 'cairo', 'mumbai', 'bangkok',
               'sydney']

print("Figure 3...")
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6.5))


train_color = '#888888'
test_color = '#444444'

def plot_temp(ax, res):
    y_act = res['y']
    y_pred = res['yp_all']

    # Reference line
    lo = min(y_act.min(), y_pred.min()) - 2
    hi = max(y_act.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1, zorder=1)

    # Actual: open circles on diagonal
    for i in range(n):
        ax.scatter(y_act[i], y_act[i], facecolors='none',
                   edgecolors=train_color, s=50, alpha=0.35, linewidth=1.2, zorder=2)

    # Predicted: filled triangles
    for i in tr:
        ax.scatter(y_act[i], y_pred[i], c=train_color, s=45, alpha=0.35,
                   marker='^', edgecolors='white', linewidth=0.4, zorder=4)
    for i in te:
        ax.scatter(y_act[i], y_pred[i], c=test_color, s=80, alpha=0.8,
                   marker='^', edgecolors='white', linewidth=0.5, zorder=6)

    # Connecting lines + labels for labeled cities
    for i in range(n):
        if names[i] in temp_labels:
            ax.plot([y_act[i], y_act[i]], [y_act[i], y_pred[i]],
                    color=test_color if i in te_set else train_color,
                    alpha=0.55, linewidth=1.3, zorder=3)
            annotate_city(ax, y_act[i], y_pred[i], names[i], i in te_set)

    ax.set_xlabel('Actual Temperature (\u00B0C)', fontsize=13)
    ax.set_ylabel('Predicted Temperature (\u00B0C)', fontsize=13)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')


plot_temp(ax3a, G['Temperature'])
ax3a.set_title('A. GloVe 300d', fontsize=14, fontweight='bold')

plot_temp(ax3b, W['Temperature'])
ax3b.set_title('B. Word2Vec 300d', fontsize=14, fontweight='bold')

# Minimal legend
from matplotlib.lines import Line2D as L2D
temp_legend = [
    L2D([0], [0], marker='o', color='w', markerfacecolor='none',
        markeredgecolor='#666', markersize=8, alpha=0.5, label='Actual temperature'),
    L2D([0], [0], marker='^', color='w',
        markerfacecolor='#666', markersize=8, alpha=0.9, label='Predicted temperature'),
]
fig3.legend(handles=temp_legend, loc='lower center', ncol=2,
            fontsize=11, frameon=True, fancybox=True, edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.04))
fig3.tight_layout()
fig3.subplots_adjust(bottom=0.08)
fig3.savefig('fig3_temperature.png', dpi=300, bbox_inches='tight')
print("  Saved fig3_temperature.png")

print("\nDone.")
