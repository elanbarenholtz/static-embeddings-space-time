"""
US Cities Probe: Finer-grained spatial resolution within a single country.

Gurnee et al. (2023) tested world, US, and NYC places separately.
Our world-cities result is partly driven by continent-level clustering.
This tests whether static embeddings resolve spatial gradients WITHIN
the US — a much harder test where continent/region words can't help.

~120 US cities spanning the full lat/lon range, probing for:
  - Latitude, Longitude
  - Mean annual temperature
  - Region (as a sanity check)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
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
# US CITIES DATASET
# (name, lat, lon, mean_annual_temp_C, state, region)
# ~120 cities spanning AK to FL, ME to HI
# ============================================================
us_cities = [
    # Northeast
    ("boston", 42.36, -71.06, 11, "MA", "Northeast"),
    ("providence", 41.82, -71.41, 11, "RI", "Northeast"),
    ("hartford", 41.76, -72.68, 10, "CT", "Northeast"),
    ("bridgeport", 41.18, -73.19, 11, "CT", "Northeast"),
    ("albany", 42.65, -73.75, 9, "NY", "Northeast"),
    ("buffalo", 42.89, -78.88, 8, "NY", "Northeast"),
    ("rochester", 43.16, -77.61, 9, "NY", "Northeast"),
    ("syracuse", 43.05, -76.15, 9, "NY", "Northeast"),
    ("pittsburgh", 40.44, -80.00, 11, "PA", "Northeast"),
    ("philadelphia", 39.95, -75.17, 13, "PA", "Northeast"),
    ("newark", 40.74, -74.17, 12, "NJ", "Northeast"),
    ("portland", 43.66, -70.26, 7, "ME", "Northeast"),
    ("burlington", 44.48, -73.21, 7, "VT", "Northeast"),
    ("concord", 43.21, -71.54, 8, "NH", "Northeast"),

    # Midwest
    ("chicago", 41.88, -87.63, 10, "IL", "Midwest"),
    ("detroit", 42.33, -83.05, 10, "MI", "Midwest"),
    ("cleveland", 41.50, -81.69, 10, "OH", "Midwest"),
    ("columbus", 39.96, -82.99, 12, "OH", "Midwest"),
    ("cincinnati", 39.10, -84.51, 12, "OH", "Midwest"),
    ("indianapolis", 39.77, -86.16, 12, "IN", "Midwest"),
    ("milwaukee", 43.04, -87.91, 8, "WI", "Midwest"),
    ("madison", 43.07, -89.40, 8, "WI", "Midwest"),
    ("minneapolis", 44.98, -93.27, 7, "MN", "Midwest"),
    ("duluth", 46.79, -92.10, 4, "MN", "Midwest"),
    ("fargo", 46.88, -96.79, 5, "ND", "Midwest"),
    ("sioux", 43.55, -96.73, 8, "SD", "Midwest"),
    ("omaha", 41.26, -95.94, 10, "NE", "Midwest"),
    ("wichita", 37.69, -97.34, 14, "KS", "Midwest"),
    ("topeka", 39.05, -95.68, 13, "KS", "Midwest"),
    ("springfield", 39.78, -89.65, 12, "IL", "Midwest"),
    ("peoria", 40.69, -89.59, 11, "IL", "Midwest"),

    # South
    ("miami", 25.76, -80.19, 25, "FL", "South"),
    ("tampa", 27.95, -82.46, 23, "FL", "South"),
    ("orlando", 28.54, -81.38, 23, "FL", "South"),
    ("jacksonville", 30.33, -81.66, 21, "FL", "South"),
    ("tallahassee", 30.44, -84.28, 20, "FL", "South"),
    ("savannah", 32.08, -81.09, 19, "GA", "South"),
    ("atlanta", 33.75, -84.39, 17, "GA", "South"),
    ("charleston", 32.78, -79.93, 19, "SC", "South"),
    ("charlotte", 35.23, -80.84, 16, "NC", "South"),
    ("raleigh", 35.77, -78.64, 16, "NC", "South"),
    ("richmond", 37.54, -77.43, 14, "VA", "South"),
    ("norfolk", 36.85, -76.29, 15, "VA", "South"),
    ("nashville", 36.16, -86.78, 15, "TN", "South"),
    ("memphis", 35.15, -90.05, 17, "TN", "South"),
    ("knoxville", 35.96, -83.92, 15, "TN", "South"),
    ("birmingham", 33.52, -86.80, 17, "AL", "South"),
    ("montgomery", 32.37, -86.30, 18, "AL", "South"),
    ("mobile", 30.69, -88.04, 20, "AL", "South"),
    ("jackson", 32.30, -90.18, 18, "MS", "South"),
    ("baton rouge", 30.45, -91.19, 20, "LA", "South"),
    ("shreveport", 32.53, -93.75, 18, "LA", "South"),
    ("little rock", 34.75, -92.29, 17, "AR", "South"),
    ("louisville", 38.25, -85.76, 14, "KY", "South"),
    ("lexington", 38.05, -84.50, 13, "KY", "South"),

    # Southwest / Texas
    ("dallas", 32.78, -96.80, 19, "TX", "Southwest"),
    ("houston", 29.76, -95.37, 21, "TX", "Southwest"),
    ("austin", 30.27, -97.74, 20, "TX", "Southwest"),
    ("san antonio", 29.42, -98.49, 20, "TX", "Southwest"),
    ("el paso", 31.76, -106.44, 18, "TX", "Southwest"),
    ("lubbock", 33.58, -101.85, 16, "TX", "Southwest"),
    ("amarillo", 35.22, -101.83, 14, "TX", "Southwest"),
    ("phoenix", 33.45, -112.07, 23, "AZ", "Southwest"),
    ("tucson", 32.22, -110.93, 21, "AZ", "Southwest"),
    ("albuquerque", 35.08, -106.65, 14, "NM", "Southwest"),
    ("santa fe", 35.69, -105.94, 12, "NM", "Southwest"),
    ("las vegas", 36.17, -115.14, 20, "NV", "Southwest"),
    ("reno", 39.53, -119.81, 11, "NV", "Southwest"),

    # West
    ("los angeles", 34.05, -118.24, 18, "CA", "West"),
    ("san francisco", 37.77, -122.42, 14, "CA", "West"),
    ("san diego", 32.72, -117.16, 18, "CA", "West"),
    ("sacramento", 38.58, -121.49, 16, "CA", "West"),
    ("fresno", 36.74, -119.77, 18, "CA", "West"),
    ("bakersfield", 35.37, -119.02, 19, "CA", "West"),
    ("oakland", 37.80, -122.27, 15, "CA", "West"),
    ("portland", 45.52, -122.68, 12, "OR", "West"),
    ("eugene", 44.05, -123.09, 11, "OR", "West"),
    ("seattle", 47.61, -122.33, 11, "WA", "West"),
    ("tacoma", 47.25, -122.44, 11, "WA", "West"),
    ("spokane", 47.66, -117.43, 9, "WA", "West"),
    ("boise", 43.62, -116.21, 11, "ID", "West"),

    # Mountain / Plains
    ("denver", 39.74, -104.99, 10, "CO", "Mountain"),
    ("colorado springs", 38.83, -104.82, 10, "CO", "Mountain"),
    ("boulder", 40.01, -105.27, 10, "CO", "Mountain"),
    ("salt lake city", 40.76, -111.89, 11, "UT", "Mountain"),
    ("provo", 40.23, -111.66, 11, "UT", "Mountain"),
    ("billings", 45.78, -108.50, 8, "MT", "Mountain"),
    ("missoula", 46.87, -114.00, 7, "MT", "Mountain"),
    ("cheyenne", 41.14, -104.82, 8, "WY", "Mountain"),
    ("casper", 42.87, -106.31, 7, "WY", "Mountain"),

    # Alaska / Hawaii (extremes)
    ("anchorage", 61.22, -149.90, 2, "AK", "Alaska"),
    ("fairbanks", 64.84, -147.72, -3, "AK", "Alaska"),
    ("juneau", 58.30, -134.42, 5, "AK", "Alaska"),
    ("honolulu", 21.31, -157.86, 25, "HI", "Hawaii"),
]

# Note: "portland" appears twice (ME and OR). We'll handle by using
# the first occurrence only since GloVe can't distinguish them.
# Actually, let's just keep both — same embedding, different ground truth.
# This is a known limitation we should mention.

# Region colors
region_colors = {
    'Northeast': '#3498DB',
    'Midwest': '#F39C12',
    'South': '#E74C3C',
    'Southwest': '#9B59B6',
    'West': '#27AE60',
    'Mountain': '#95A5A6',
    'Alaska': '#1ABC9C',
    'Hawaii': '#E67E22',
}


# ============================================================
# LOAD EMBEDDINGS
# ============================================================
GLOVE_PATH = "glove.6B.300d.txt"
if not os.path.exists(GLOVE_PATH):
    print(f"ERROR: {GLOVE_PATH} not found"); sys.exit(1)

needed = set()
for name, *_ in us_cities:
    for w in name.split():
        needed.add(w.lower())

print("Loading GloVe...")
glove = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if parts[0] in needed:
            glove[parts[0]] = np.array([float(x) for x in parts[1:]])
print(f"  {len(glove)}/{len(needed)} words found")

print("Loading Word2Vec...")
import gensim.downloader as api
w2v_model = api.load('word2vec-google-news-300')


# ============================================================
# BUILD DATASET
# ============================================================
def glove_vec(name):
    words = name.lower().split()
    vecs = [glove[w] for w in words if w in glove]
    return np.mean(vecs, axis=0) if vecs else None

def w2v_vec(name):
    # Try phrase first (e.g., "San_Francisco"), then individual words
    phrase = '_'.join(w.capitalize() for w in name.split())
    if phrase in w2v_model:
        return w2v_model[phrase]
    vecs = []
    for w in name.split():
        cap = w.capitalize()
        if cap in w2v_model: vecs.append(w2v_model[cap])
        elif w in w2v_model: vecs.append(w2v_model[w])
    return np.mean(vecs, axis=0) if vecs else None


# Deduplicate portland — keep only the first (ME) since embeddings are identical
seen_names = set()
valid = []
for name, lat, lon, temp, state, region in us_cities:
    if name in seen_names:
        print(f"  Skipping duplicate: {name} ({state}) — same embedding as first occurrence")
        continue
    seen_names.add(name)
    gv = glove_vec(name)
    wv = w2v_vec(name)
    if gv is None and wv is None:
        print(f"  MISSING both: {name} ({state})")
        continue
    if gv is None:
        print(f"  GloVe missing: {name} ({state})")
    if wv is None:
        print(f"  W2V missing: {name} ({state})")
    valid.append(dict(name=name, lat=lat, lon=lon, temp=temp,
                      state=state, region=region,
                      glove=gv, w2v=wv,
                      color=region_colors.get(region, '#888')))

both_valid = [v for v in valid if v['glove'] is not None and v['w2v'] is not None]
print(f"\n  {len(both_valid)} cities with both embeddings (of {len(us_cities)} total)")

valid = both_valid
names = [v['name'] for v in valid]
lats = np.array([v['lat'] for v in valid])
lons = np.array([v['lon'] for v in valid])
temps = np.array([v['temp'] for v in valid])
colors = [v['color'] for v in valid]
regions_list = [v['region'] for v in valid]

X_g = np.array([v['glove'] for v in valid])
X_w = np.array([v['w2v'] for v in valid])
n = len(valid)


# ============================================================
# PROBE
# ============================================================
lambdas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

def run_probe(X, y):
    d = X.shape[1]
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    n_folds, fold_size = 5, n // 5
    n_train = int(0.8 * n)
    tr, te = idx[:n_train], idx[n_train:]

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

    return dict(r2=r2, mae=mae, lam=best_lam, yp_all=yp_all, y=y,
                tr=tr, te=te)


targets = {
    'Latitude': lats,
    'Longitude': lons,
    'Temperature': temps,
}

print("\n" + "=" * 70)
print("US CITIES PROBE RESULTS")
print("=" * 70)

G = {}
W = {}
for tname, y in targets.items():
    G[tname] = run_probe(X_g, y)
    W[tname] = run_probe(X_w, y)
    print(f"  {tname:<14s}  GloVe: R²={G[tname]['r2']:+.3f}, MAE={G[tname]['mae']:.1f}  "
          f"W2V: R²={W[tname]['r2']:+.3f}, MAE={W[tname]['mae']:.1f}")

tr = G['Latitude']['tr']
te = G['Latitude']['te']
te_set = set(te)


# ============================================================
# FIGURES
# ============================================================
print("\nGenerating figures...")

label_cities = ['anchorage', 'fairbanks', 'honolulu', 'miami', 'seattle',
                'los angeles', 'phoenix', 'denver', 'chicago', 'boston',
                'minneapolis', 'dallas', 'atlanta', 'detroit']

def annotate_us(ax, x, y, name, is_test, fontsize=7):
    dx, dy = 5, 5
    nudges = {
        'anchorage': (5, -10), 'fairbanks': (5, -10),
        'honolulu': (5, -10), 'miami': (5, -10),
        'seattle': (-40, 5), 'los angeles': (-60, -10),
        'san francisco': (-65, 5), 'boston': (5, -10),
        'chicago': (5, -10), 'detroit': (5, -10),
        'minneapolis': (-55, -10),
    }
    if name in nudges:
        dx, dy = nudges[name]
    weight = 'bold' if is_test else 'normal'
    alpha = 1.0 if is_test else 0.65
    ax.annotate(name.title(), (x, y), fontsize=fontsize,
                fontweight=weight, alpha=alpha, zorder=10,
                xytext=(dx, dy), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                         alpha=0.75, edgecolor='none'))


# --- Fig 1: Geography map — actual vs predicted (3 panels) ---
fig1, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(21, 7))

def plot_actual(ax):
    for i in range(n):
        ax.scatter(lons[i], lats[i], c=colors[i], s=48, alpha=0.85,
                   marker='o', edgecolors='white', linewidth=0.5, zorder=5)
    for i in range(n):
        if names[i] in label_cities:
            annotate_us(ax, lons[i], lats[i], names[i], True)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.grid(True, alpha=0.15, linewidth=0.5)

def plot_predicted(ax, res_lon, res_lat):
    pred_lon = res_lon['yp_all']
    pred_lat = res_lat['yp_all']
    for i in tr:
        ax.scatter(pred_lon[i], pred_lat[i], c=colors[i], s=30, alpha=0.4,
                   marker='o', edgecolors='white', linewidth=0.5, zorder=3)
    for i in te:
        ax.scatter(pred_lon[i], pred_lat[i], c=colors[i], s=60, alpha=0.9,
                   marker='D', edgecolors='white', linewidth=0.5, zorder=6)
    for i in range(n):
        if names[i] in label_cities:
            annotate_us(ax, pred_lon[i], pred_lat[i], names[i], i in te_set)
    ax.set_xlabel('Predicted Longitude', fontsize=11)
    ax.set_ylabel('Predicted Latitude', fontsize=11)
    ax.grid(True, alpha=0.15, linewidth=0.5)

plot_actual(ax_a)
ax_a.set_title('A. Actual Geography', fontsize=12, fontweight='bold')

plot_predicted(ax_b, G['Longitude'], G['Latitude'])
ax_b.set_title(f'B. GloVe 300d', fontsize=12, fontweight='bold')

plot_predicted(ax_c, W['Longitude'], W['Latitude'])
ax_c.set_title(f'C. Word2Vec 300d', fontsize=12, fontweight='bold')

# Legend
legend_els = [Patch(facecolor=region_colors[r], edgecolor='white', label=r)
              for r in region_colors]
legend_els.append(Line2D([0], [0], marker='o', color='w',
                          markerfacecolor='#999', markersize=7, alpha=0.4,
                          label='Training'))
legend_els.append(Line2D([0], [0], marker='D', color='w',
                          markerfacecolor='#999', markeredgecolor='white',
                          markersize=7, alpha=0.9, label='Held-out test'))

fig1.legend(handles=legend_els, loc='lower center', ncol=5,
            fontsize=9, frameon=True, fancybox=True, edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.06))
fig1.tight_layout()
fig1.subplots_adjust(bottom=0.12)
fig1.savefig('us_cities_geography.png', dpi=300, bbox_inches='tight')
print("  Saved us_cities_geography.png")


# --- Fig 2: Bar chart ---
fig2, ax2 = plt.subplots(figsize=(8, 5))

target_order = ['Latitude', 'Longitude', 'Temperature']
g_r2 = [G[t]['r2'] for t in target_order]
w_r2 = [W[t]['r2'] for t in target_order]

x = np.arange(len(target_order))
bw = 0.35
bars_g = ax2.bar(x - bw/2, g_r2, bw, color='#4472C4', edgecolor='white',
                 linewidth=1.2, alpha=0.88, label='GloVe 300d')
bars_w = ax2.bar(x + bw/2, w_r2, bw, color='#ED7D31', edgecolor='white',
                 linewidth=1.2, alpha=0.88, label='Word2Vec 300d')

for bars, r2s in [(bars_g, g_r2), (bars_w, w_r2)]:
    for bar, r2 in zip(bars, r2s):
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        y_pos = max(h, 0) + 0.02 if h >= 0 else h - 0.02
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{r2:.2f}', ha='center', va=va, fontsize=9, fontweight='bold')

ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='--', zorder=1)
ax2.set_xticks(x)
ax2.set_xticklabels(target_order, fontsize=11)
ax2.set_ylabel('Test R²', fontsize=12)
ax2.set_ylim(-0.5, 1.0)
ax2.grid(True, alpha=0.15, axis='y', linewidth=0.5)
ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)

fig2.tight_layout()
fig2.savefig('us_cities_r2_barplot.png', dpi=300, bbox_inches='tight')
print("  Saved us_cities_r2_barplot.png")


# --- Fig 3: Temperature scatter (2 panels) ---
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6.5))

def plot_temp_scatter(ax, res, title):
    y_act = res['y']
    y_pred = res['yp_all']
    lo = min(y_act.min(), y_pred.min()) - 2
    hi = max(y_act.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1, zorder=1)
    for i in tr:
        ax.scatter(y_act[i], y_pred[i], c=colors[i], s=30, alpha=0.4,
                   marker='o', edgecolors='white', linewidth=0.5, zorder=3)
    for i in te:
        ax.scatter(y_act[i], y_pred[i], c=colors[i], s=60, alpha=0.9,
                   marker='D', edgecolors='white', linewidth=0.5, zorder=6)
    for i in range(n):
        if names[i] in label_cities:
            annotate_us(ax, y_act[i], y_pred[i], names[i], i in te_set)
    ax.set_xlabel('Actual Temperature (°C)', fontsize=11)
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

plot_temp_scatter(ax3a, G['Temperature'], 'A. GloVe 300d')
plot_temp_scatter(ax3b, W['Temperature'], 'B. Word2Vec 300d')

fig3.legend(handles=legend_els, loc='lower center', ncol=5,
            fontsize=9, frameon=True, fancybox=True, edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.06))
fig3.tight_layout()
fig3.subplots_adjust(bottom=0.14)
fig3.savefig('us_cities_temperature.png', dpi=300, bbox_inches='tight')
print("  Saved us_cities_temperature.png")


# ============================================================
# SEMANTIC SIMILARITY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("US CITIES: SEMANTIC SIMILARITY ANALYSIS")
print("=" * 70)

us_semantic_words = {
    'cold_north': ['snow', 'ice', 'cold', 'freezing', 'winter', 'blizzard',
                   'frost', 'frozen'],
    'warm_south': ['hot', 'humid', 'tropical', 'heat', 'sunny', 'sweltering',
                   'hurricane', 'palm'],
    'coastal': ['ocean', 'beach', 'coast', 'coastal', 'harbor', 'port',
                'maritime', 'surf'],
    'inland': ['plains', 'prairie', 'farmland', 'ranch', 'wheat', 'corn',
               'rural', 'heartland'],
    'mountain_west': ['mountain', 'mining', 'ski', 'wilderness', 'canyon',
                      'altitude', 'hiking', 'rocky'],
}

# Load semantic vecs
sem_needed = set()
for words in us_semantic_words.values():
    sem_needed.update(words)

sem_glove = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if parts[0] in sem_needed:
            sem_glove[parts[0]] = np.array([float(x) for x in parts[1:]])

print(f"  Found {len(sem_glove)}/{len(sem_needed)} semantic words")

X_g_norms = np.linalg.norm(X_g, axis=1, keepdims=True)
X_g_normed = X_g / (X_g_norms + 1e-10)

for cat, words in us_semantic_words.items():
    print(f"\n  {cat}:")
    for w in words:
        if w not in sem_glove: continue
        wn = sem_glove[w] / (np.linalg.norm(sem_glove[w]) + 1e-10)
        sims = X_g_normed @ wn
        r_lat, p_lat = pearsonr(sims, lats)
        r_lon, p_lon = pearsonr(sims, lons)
        r_temp, p_temp = pearsonr(sims, temps)
        sig_lat = '*' if p_lat < 0.05 else ''
        sig_lon = '*' if p_lon < 0.05 else ''
        sig_temp = '*' if p_temp < 0.05 else ''
        print(f"    {w:<14s}  r(lat)={r_lat:+.3f}{sig_lat:<2s}  "
              f"r(lon)={r_lon:+.3f}{sig_lon:<2s}  "
              f"r(temp)={r_temp:+.3f}{sig_temp:<2s}")

# Composite: cold-warm vs latitude
cold_words = us_semantic_words['cold_north']
warm_words = us_semantic_words['warm_south']

cold_scores = np.zeros(n)
warm_scores = np.zeros(n)
nc, nw = 0, 0
for w in cold_words:
    if w in sem_glove:
        wn = sem_glove[w] / (np.linalg.norm(sem_glove[w]) + 1e-10)
        cold_scores += X_g_normed @ wn
        nc += 1
for w in warm_words:
    if w in sem_glove:
        wn = sem_glove[w] / (np.linalg.norm(sem_glove[w]) + 1e-10)
        warm_scores += X_g_normed @ wn
        nw += 1
if nc: cold_scores /= nc
if nw: warm_scores /= nw
diff_scores = cold_scores - warm_scores

r_diff, p_diff = pearsonr(diff_scores, lats)
print(f"\n  Cold−warm composite vs latitude: r={r_diff:.3f} (p={p_diff:.4f})")

r_diff_t, p_diff_t = pearsonr(diff_scores, temps)
print(f"  Cold−warm composite vs temperature: r={r_diff_t:.3f} (p={p_diff_t:.4f})")

# Coastal-inland vs longitude
coastal_words = us_semantic_words['coastal']
inland_words = us_semantic_words['inland']

coastal_scores = np.zeros(n)
inland_scores = np.zeros(n)
nc2, ni = 0, 0
for w in coastal_words:
    if w in sem_glove:
        wn = sem_glove[w] / (np.linalg.norm(sem_glove[w]) + 1e-10)
        coastal_scores += X_g_normed @ wn
        nc2 += 1
for w in inland_words:
    if w in sem_glove:
        wn = sem_glove[w] / (np.linalg.norm(sem_glove[w]) + 1e-10)
        inland_scores += X_g_normed @ wn
        ni += 1
if nc2: coastal_scores /= nc2
if ni: inland_scores /= ni
coast_inland_diff = coastal_scores - inland_scores

r_ci_lon, p_ci_lon = pearsonr(coast_inland_diff, lons)
print(f"  Coastal−inland composite vs longitude: r={r_ci_lon:.3f} (p={p_ci_lon:.4f})")


# --- Fig 4: Semantic composites vs geography (2 panels) ---
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: cold-warm vs latitude
for i in range(n):
    ax4a.scatter(lats[i], diff_scores[i], c=colors[i], s=55, alpha=0.85,
                 edgecolors='white', linewidth=0.5, zorder=5)
m, b = np.polyfit(lats, diff_scores, 1)
xs = np.linspace(lats.min() - 2, lats.max() + 2, 100)
ax4a.plot(xs, m * xs + b, color='#333', alpha=0.4, linewidth=2,
          linestyle='--', zorder=2)
for i in range(n):
    if names[i] in label_cities:
        annotate_us(ax4a, lats[i], diff_scores[i], names[i], True)
ax4a.set_xlabel('Actual Latitude', fontsize=11)
ax4a.set_ylabel('Cold − Warm Similarity Score', fontsize=11)
ax4a.set_title(f'A. Cold−warm gradient (r = {r_diff:.3f})',
               fontsize=12, fontweight='bold')
ax4a.grid(True, alpha=0.15, linewidth=0.5)

# Panel B: coastal-inland vs longitude
for i in range(n):
    ax4b.scatter(lons[i], coast_inland_diff[i], c=colors[i], s=55, alpha=0.85,
                 edgecolors='white', linewidth=0.5, zorder=5)
m2, b2 = np.polyfit(lons, coast_inland_diff, 1)
xs2 = np.linspace(lons.min() - 2, lons.max() + 2, 100)
ax4b.plot(xs2, m2 * xs2 + b2, color='#333', alpha=0.4, linewidth=2,
          linestyle='--', zorder=2)
for i in range(n):
    if names[i] in label_cities:
        annotate_us(ax4b, lons[i], coast_inland_diff[i], names[i], True)
ax4b.set_xlabel('Actual Longitude', fontsize=11)
ax4b.set_ylabel('Coastal − Inland Similarity Score', fontsize=11)
ax4b.set_title(f'B. Coastal−inland gradient (r = {r_ci_lon:.3f})',
               fontsize=12, fontweight='bold')
ax4b.grid(True, alpha=0.15, linewidth=0.5)

fig4.legend(handles=[Patch(facecolor=region_colors[r], edgecolor='white', label=r)
                      for r in region_colors],
            loc='lower center', ncol=4, fontsize=9, framealpha=0.9,
            bbox_to_anchor=(0.5, -0.04))
fig4.tight_layout()
fig4.subplots_adjust(bottom=0.1)
fig4.savefig('us_cities_semantic_gradients.png', dpi=300, bbox_inches='tight')
print("  Saved us_cities_semantic_gradients.png")


print("\nDone.")
