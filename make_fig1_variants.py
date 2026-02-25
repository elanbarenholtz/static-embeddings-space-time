"""
Generate two variants of the geography figure for the paper.
V1: 2-panel overlay (actual faint + predicted bold)
V2: 2-panel error vectors (arrows from actual to predicted)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
import os, sys

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
# DATA (same as make_paper_figures_v2.py)
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

def get_region(name):
    for r, cl in regions.items():
        if name in cl:
            return r
    return 'Other'

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
                      color=get_color(name), region=get_region(name)))

names = [v['name'] for v in valid]
colors = [v['color'] for v in valid]
X_g = np.array([v['glove'] for v in valid])
X_w = np.array([v['w2v'] for v in valid])
n = len(valid)
print(f"  {n} cities")

targets = {
    'Latitude': np.array([v['lat'] for v in valid]),
    'Longitude': np.array([v['lon'] for v in valid]),
}

# ============================================================
# PROBES
# ============================================================
lambdas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

def run_probes(X):
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
        yp_all = X @ W
        results[tname] = dict(yp_all=yp_all, tr=tr, te=te)
    return results

print("Running probes...")
G = run_probes(X_g)
W = run_probes(X_w)
tr, te = G['Latitude']['tr'], G['Latitude']['te']
te_set = set(te)

lat_true = targets['Latitude']
lon_true = targets['Longitude']
lat_pred_g = G['Latitude']['yp_all']
lon_pred_g = G['Longitude']['yp_all']
lat_pred_w = W['Latitude']['yp_all']
lon_pred_w = W['Longitude']['yp_all']

# anchor cities for labeling
anchor_cities = ['anchorage', 'new york', 'moscow',
                 'cairo', 'singapore',
                 'buenos aires', 'sydney']

# ============================================================
# VERSION 1: 2-panel overlay (actual faint + predicted bold)
# ============================================================
print("\nVersion 1: Overlay figure...")
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, lon_pred, lat_pred, title in [
    (ax1, lon_pred_g, lat_pred_g, 'A. GloVe 300d'),
    (ax2, lon_pred_w, lat_pred_w, 'B. Word2Vec 300d'),
]:
    # Actual points: faint open circles
    for i in range(n):
        ax.scatter(lon_true[i], lat_true[i], facecolors='none',
                   edgecolors=colors[i], s=35, alpha=0.25, linewidth=1.0, zorder=2)

    # Predicted points: bold filled triangles
    for i in tr:
        ax.scatter(lon_pred[i], lat_pred[i], c=colors[i], s=40, alpha=0.5,
                   marker='^', edgecolors='white', linewidth=0.4, zorder=4)
    for i in te:
        ax.scatter(lon_pred[i], lat_pred[i], c=colors[i], s=65, alpha=0.9,
                   marker='^', edgecolors='white', linewidth=0.5, zorder=6)

    # Thin connecting lines + labels for anchor cities only
    for i in range(n):
        if names[i] in anchor_cities:
            ax.plot([lon_true[i], lon_pred[i]], [lat_true[i], lat_pred[i]],
                    color=colors[i], alpha=0.55, linewidth=1.3, zorder=3)
            ax.annotate(names[i].title(), (lon_pred[i], lat_pred[i]),
                        fontsize=9, fontweight='bold' if i in te_set else 'normal',
                        alpha=0.85, zorder=10, xytext=(5, 5),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  alpha=0.7, edgecolor='none'))

    ax.set_xlim(-170, 185)
    ax.set_ylim(-45, 70)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.12, linewidth=0.5)

# Legend
legend_els = [Patch(facecolor=color_map[r], edgecolor='white', label=r)
              for r in regions]
legend_els.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                         markeredgecolor='#666', markersize=7, alpha=0.5,
                         label='Actual location'))
legend_els.append(Line2D([0], [0], marker='^', color='w',
                         markerfacecolor='#666', markersize=7, alpha=0.9,
                         label='Predicted location'))

fig1.legend(handles=legend_els, loc='lower center', ncol=4,
            fontsize=11, frameon=True, fancybox=True, edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.08))
fig1.tight_layout()
fig1.subplots_adjust(bottom=0.14)
fig1.savefig('fig1_v1_overlay.png', dpi=300, bbox_inches='tight')
fig1.savefig('paper/figures/fig1_v1_overlay.png', dpi=300, bbox_inches='tight')
print("  Saved fig1_v1_overlay.png")


# ============================================================
# VERSION 2: 2-panel error vectors (arrows actual -> predicted)
# ============================================================
print("\nVersion 2: Error vectors figure...")
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, lon_pred, lat_pred, title in [
    (ax1, lon_pred_g, lat_pred_g, 'A. GloVe 300d'),
    (ax2, lon_pred_w, lat_pred_w, 'B. Word2Vec 300d'),
]:
    # Draw arrows from actual to predicted
    for i in range(n):
        dx = lon_pred[i] - lon_true[i]
        dy = lat_pred[i] - lat_true[i]
        ax.annotate('', xy=(lon_pred[i], lat_pred[i]),
                    xytext=(lon_true[i], lat_true[i]),
                    arrowprops=dict(arrowstyle='->', color=colors[i],
                                    alpha=0.35, lw=0.8, shrinkA=0, shrinkB=0))

    # Actual points: small dots
    for i in range(n):
        ax.scatter(lon_true[i], lat_true[i], c=colors[i], s=20, alpha=0.7,
                   marker='o', edgecolors='white', linewidth=0.3, zorder=5)

    # Predicted points: small x's
    for i in range(n):
        ax.scatter(lon_pred[i], lat_pred[i], c=colors[i], s=25, alpha=0.6,
                   marker='x', linewidth=1.0, zorder=5)

    ax.set_xlim(-170, 185)
    ax.set_ylim(-45, 70)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.12, linewidth=0.5)

# Legend
legend_els2 = [Patch(facecolor=color_map[r], edgecolor='white', label=r)
               for r in regions]
legend_els2.append(Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#666', markersize=6,
                          label='Actual location'))
legend_els2.append(Line2D([0], [0], marker='x', color='w',
                          markeredgecolor='#666', markersize=7, markeredgewidth=1.5,
                          label='Predicted location'))
legend_els2.append(Line2D([0], [0], color='#666', alpha=0.5, lw=1,
                          label='Prediction error'))

fig2.legend(handles=legend_els2, loc='lower center', ncol=5,
            fontsize=9, frameon=True, fancybox=True, edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.08))
fig2.tight_layout()
fig2.subplots_adjust(bottom=0.14)
fig2.savefig('fig1_v2_arrows.png', dpi=300, bbox_inches='tight')
fig2.savefig('paper/figures/fig1_v2_arrows.png', dpi=300, bbox_inches='tight')
print("  Saved fig1_v2_arrows.png")

print("\nDone!")
