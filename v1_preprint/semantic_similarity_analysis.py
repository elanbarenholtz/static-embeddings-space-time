"""
Semantic Similarity Analysis

The nearest-neighbor analysis showed city neighborhoods are dominated by
other city/country names, NOT climate words. But the linear probe recovers
a continuous latitude gradient — clusters alone can't explain that.

Hypothesis: semantically meaningful words (cold, tropical, northern, etc.)
aren't nearest neighbors but DO have graded similarity to cities that
correlates with latitude. The signal is diffuse across many dimensions,
not concentrated in the top-50 neighbors.

This script:
1. Computes cosine similarity between each city and a set of semantically
   meaningful words (climate, direction, etc.)
2. Tests whether those similarities correlate with actual latitude
3. Compares: do semantic-word similarities predict latitude BETTER than
   the average latitude of a city's nearest-neighbor cities?
4. Checks how deep in the neighbor list semantic words appear
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, pearsonr
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
# CITY DATA
# ============================================================
cities = [
    ("anchorage", 61.2, -149.9, 2), ("seattle", 47.6, -122.3, 11),
    ("portland", 45.5, -122.7, 12), ("san francisco", 37.8, -122.4, 14),
    ("los angeles", 34.1, -118.2, 18), ("san diego", 32.7, -117.2, 18),
    ("phoenix", 33.4, -112.1, 23), ("denver", 39.7, -105.0, 10),
    ("dallas", 32.8, -96.8, 19), ("houston", 29.8, -95.4, 21),
    ("austin", 30.3, -97.7, 20), ("chicago", 41.9, -87.6, 10),
    ("detroit", 42.3, -83.0, 10), ("minneapolis", 44.98, -93.3, 7),
    ("miami", 25.8, -80.2, 25), ("atlanta", 33.7, -84.4, 17),
    ("boston", 42.4, -71.1, 11), ("new york", 40.7, -74.0, 13),
    ("philadelphia", 40.0, -75.2, 13), ("washington", 38.9, -77.0, 14),
    ("nashville", 36.2, -86.8, 15), ("memphis", 35.1, -90.0, 17),
    ("cleveland", 41.5, -81.7, 10), ("pittsburgh", 40.4, -80.0, 11),
    ("milwaukee", 43.0, -87.9, 9), ("new orleans", 30.0, -90.1, 21),
    ("kansas city", 39.1, -94.6, 13), ("las vegas", 36.2, -115.1, 20),
    ("salt lake city", 40.8, -111.9, 11), ("toronto", 43.7, -79.4, 9),
    ("montreal", 45.5, -73.6, 7), ("vancouver", 49.3, -123.1, 10),
    ("mexico city", 19.4, -99.1, 17),
    ("london", 51.5, -0.1, 11), ("paris", 48.9, 2.3, 12),
    ("berlin", 52.5, 13.4, 10), ("madrid", 40.4, -3.7, 15),
    ("rome", 41.9, 12.5, 16), ("amsterdam", 52.4, 4.9, 10),
    ("brussels", 50.8, 4.4, 10), ("vienna", 48.2, 16.4, 11),
    ("prague", 50.1, 14.4, 10), ("warsaw", 52.2, 21.0, 9),
    ("budapest", 47.5, 19.0, 12), ("lisbon", 38.7, -9.1, 17),
    ("dublin", 53.3, -6.3, 10), ("edinburgh", 55.9, -3.2, 9),
    ("stockholm", 59.3, 18.1, 7), ("oslo", 59.9, 10.7, 6),
    ("helsinki", 60.2, 24.9, 5), ("copenhagen", 55.7, 12.6, 9),
    ("moscow", 55.8, 37.6, 6), ("athens", 38.0, 23.7, 18),
    ("istanbul", 41.0, 29.0, 15), ("barcelona", 41.4, 2.2, 16),
    ("munich", 48.1, 11.6, 9), ("zurich", 47.4, 8.5, 9),
    ("milan", 45.5, 9.2, 13), ("marseille", 43.3, 5.4, 15),
    ("tokyo", 35.7, 139.7, 16), ("beijing", 39.9, 116.4, 13),
    ("shanghai", 31.2, 121.5, 17), ("mumbai", 19.1, 72.9, 27),
    ("delhi", 28.6, 77.2, 25), ("bangkok", 13.8, 100.5, 28),
    ("singapore", 1.4, 103.9, 27), ("hong kong", 22.3, 114.2, 23),
    ("seoul", 37.6, 127.0, 13), ("taipei", 25.0, 121.5, 23),
    ("osaka", 34.7, 135.5, 17), ("jakarta", -6.2, 106.8, 27),
    ("manila", 14.6, 121.0, 28), ("hanoi", 21.0, 105.8, 24),
    ("dubai", 25.3, 55.3, 28), ("tehran", 35.7, 51.4, 17),
    ("baghdad", 33.3, 44.4, 23), ("kabul", 34.5, 69.2, 13),
    ("karachi", 24.9, 67.0, 26),
    ("cairo", 30.0, 31.2, 22), ("lagos", 6.5, 3.4, 27),
    ("nairobi", -1.3, 36.8, 18), ("johannesburg", -26.2, 28.0, 16),
    ("cape town", -33.9, 18.4, 17), ("casablanca", 33.6, -7.6, 18),
    ("addis ababa", 9.0, 38.7, 16), ("accra", 5.6, -0.2, 27),
    ("algiers", 36.8, 3.1, 18),
    ("buenos aires", -34.6, -58.4, 17), ("santiago", -33.4, -70.6, 14),
    ("lima", -12.0, -77.0, 19), ("bogota", 4.7, -74.1, 14),
    ("rio de janeiro", -22.9, -43.2, 24), ("sao paulo", -23.6, -46.6, 20),
    ("caracas", 10.5, -66.9, 22), ("quito", -0.2, -78.5, 14),
    ("sydney", -33.9, 151.2, 18), ("melbourne", -37.8, 144.96, 15),
    ("auckland", -36.8, 174.8, 15), ("perth", -31.9, 115.9, 19),
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
        if name in cl: return r
    return 'Unknown'

def get_color(name):
    for r, cl in regions.items():
        if name in cl: return color_map[r]
    return '#888888'

# ============================================================
# SEMANTIC PROBE WORDS
# ============================================================
# Words whose similarity to cities might carry latitude/temperature signal
semantic_words = {
    'cold_climate': [
        'snow', 'ice', 'cold', 'freezing', 'frost', 'winter', 'arctic',
        'frozen', 'blizzard', 'glacier', 'tundra', 'polar',
    ],
    'warm_climate': [
        'tropical', 'hot', 'humid', 'monsoon', 'desert', 'heat',
        'drought', 'sunny', 'palm', 'jungle', 'rainforest', 'sweltering',
    ],
    'northern_direction': [
        'north', 'northern', 'nordic', 'scandinavian', 'boreal',
    ],
    'southern_direction': [
        'south', 'southern', 'austral',
    ],
    'european_identity': [
        'european', 'eu', 'nato', 'parliament', 'chancellor', 'monarchy',
    ],
    'developing_world': [
        'poverty', 'developing', 'slum', 'cholera', 'malaria', 'refugee',
    ],
}

# ============================================================
# LOAD GLOVE
# ============================================================
GLOVE_PATH = "glove.6B.300d.txt"
if not os.path.exists(GLOVE_PATH):
    print(f"ERROR: {GLOVE_PATH} not found"); sys.exit(1)

# Collect words we need
all_needed = set()
for name, *_ in cities:
    for w in name.split():
        all_needed.add(w)
for cat_words in semantic_words.values():
    for w in cat_words:
        all_needed.add(w)

# Also load full vocab for neighbor-city analysis
print("Loading GloVe embeddings...")
glove = {}
all_words_list = []
all_vecs_list = []

with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec = np.array([float(x) for x in parts[1:]])
        if word in all_needed:
            glove[word] = vec
        all_words_list.append(word)
        all_vecs_list.append(vec)

all_vecs_arr = np.array(all_vecs_list)
norms = np.linalg.norm(all_vecs_arr, axis=1, keepdims=True)
all_vecs_normed = all_vecs_arr / (norms + 1e-10)
word_to_idx = {w: i for i, w in enumerate(all_words_list)}
vocab_size = len(all_words_list)
print(f"  Loaded {vocab_size} total, {len(glove)} needed words")

# ============================================================
# BUILD CITY DATA
# ============================================================
def city_emb(name):
    vecs = [glove[w] for w in name.split() if w in glove]
    return np.mean(vecs, axis=0) if vecs else None

valid = []
for name, lat, lon, temp in cities:
    e = city_emb(name)
    if e is None: continue
    valid.append(dict(name=name, lat=lat, lon=lon, temp=temp,
                      emb=e, color=get_color(name), region=get_region(name)))

names = [v['name'] for v in valid]
lats = np.array([v['lat'] for v in valid])
lons = np.array([v['lon'] for v in valid])
temps = np.array([v['temp'] for v in valid])
colors = [v['color'] for v in valid]
X = np.array([v['emb'] for v in valid])
n = len(valid)
print(f"  {n} cities")

# Normalize city embeddings
X_norms = np.linalg.norm(X, axis=1, keepdims=True)
X_normed = X / (X_norms + 1e-10)


# ============================================================
# ANALYSIS 1: Cosine similarity to semantic words vs latitude
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: Cosine similarity to semantic words vs latitude")
print("=" * 70)

all_word_results = []

for cat_name, words in semantic_words.items():
    print(f"\n  {cat_name}:")
    for word in words:
        if word not in glove:
            print(f"    {word}: NOT IN VOCAB")
            continue
        w_vec = glove[word]
        w_normed = w_vec / (np.linalg.norm(w_vec) + 1e-10)
        sims = X_normed @ w_normed
        r_lat, p_lat = pearsonr(sims, lats)
        r_temp, p_temp = pearsonr(sims, temps)
        sig_lat = '*' if p_lat < 0.05 else ''
        sig_temp = '*' if p_temp < 0.05 else ''
        print(f"    {word:<16s}  r(lat)={r_lat:+.3f}{sig_lat:<2s}  "
              f"r(temp)={r_temp:+.3f}{sig_temp:<2s}  "
              f"mean_sim={sims.mean():.3f}  range=[{sims.min():.3f}, {sims.max():.3f}]")
        all_word_results.append(dict(
            word=word, category=cat_name,
            r_lat=r_lat, p_lat=p_lat,
            r_temp=r_temp, p_temp=p_temp,
            mean_sim=sims.mean(), sims=sims,
        ))


# ============================================================
# ANALYSIS 2: Composite scores (average sim to cold words minus warm words)
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: Composite semantic scores")
print("=" * 70)

def composite_score(city_emb_normed, pos_words, neg_words):
    """Average similarity to pos_words minus average similarity to neg_words."""
    pos_sims = []
    for w in pos_words:
        if w in glove:
            wn = glove[w] / (np.linalg.norm(glove[w]) + 1e-10)
            pos_sims.append(city_emb_normed @ wn)
    neg_sims = []
    for w in neg_words:
        if w in glove:
            wn = glove[w] / (np.linalg.norm(glove[w]) + 1e-10)
            neg_sims.append(city_emb_normed @ wn)
    pos_mean = np.mean(pos_sims) if pos_sims else 0
    neg_mean = np.mean(neg_sims) if neg_sims else 0
    return pos_mean - neg_mean

composites = {
    'cold - warm': (semantic_words['cold_climate'], semantic_words['warm_climate']),
    'northern - southern': (semantic_words['northern_direction'], semantic_words['southern_direction']),
    'european - developing': (semantic_words['european_identity'], semantic_words['developing_world']),
}

composite_scores = {}
for comp_name, (pos_words, neg_words) in composites.items():
    scores = np.array([composite_score(X_normed[i], pos_words, neg_words)
                       for i in range(n)])
    r_lat, p_lat = pearsonr(scores, lats)
    r_temp, p_temp = pearsonr(scores, temps)
    composite_scores[comp_name] = scores
    print(f"  {comp_name:<30s}  r(lat)={r_lat:+.3f} (p={p_lat:.4f})  "
          f"r(temp)={r_temp:+.3f} (p={p_temp:.4f})")


# ============================================================
# ANALYSIS 3: Where do semantic words rank in the neighbor list?
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: Rank of semantic words in neighbor lists")
print("=" * 70)

# For a few showcase cities, find what rank climate words appear at
showcase = ['helsinki', 'moscow', 'london', 'tokyo', 'cairo',
            'mumbai', 'bangkok', 'lagos', 'singapore', 'sydney']

climate_words_all = set(semantic_words['cold_climate'] + semantic_words['warm_climate'] +
                        semantic_words['northern_direction'] + semantic_words['southern_direction'])

for city_name in showcase:
    v = next(v for v in valid if v['name'] == city_name)
    e_normed = v['emb'] / (np.linalg.norm(v['emb']) + 1e-10)
    sims = all_vecs_normed @ e_normed
    sorted_idx = np.argsort(sims)[::-1]

    print(f"\n  {city_name.title()} (lat {v['lat']:+.1f}°):")
    found = []
    for rank, idx in enumerate(sorted_idx):
        w = all_words_list[idx]
        if w in climate_words_all:
            found.append((rank, w, float(sims[idx])))
            if len(found) >= 10:
                break
    if found:
        for rank, w, sim in found:
            print(f"    rank {rank:>6d}  {w:<16s}  sim={sim:.4f}")
    else:
        print(f"    No climate words found in top 10000")


# ============================================================
# ANALYSIS 4: Average latitude of nearest-neighbor cities
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: Average latitude of neighbor cities vs actual latitude")
print("=" * 70)

# Build lookup of city names to lat/lon (including cities not in our main list)
# We'll use our 99 cities as the reference set
city_lat_lookup = {v['name']: v['lat'] for v in valid}
city_lon_lookup = {v['name']: v['lon'] for v in valid}

# For each city, find which other cities in our set are most similar
# and compute the average latitude of the top-K neighbors
for K in [5, 10, 20]:
    neighbor_lats = []
    for i, v in enumerate(valid):
        e_normed = X_normed[i]
        sims = X_normed @ e_normed  # similarity to all other cities
        sims[i] = -1  # exclude self
        top_k_idx = np.argsort(sims)[::-1][:K]
        avg_lat = np.mean([lats[j] for j in top_k_idx])
        neighbor_lats.append(avg_lat)
    neighbor_lats = np.array(neighbor_lats)
    r, p = pearsonr(neighbor_lats, lats)
    print(f"  Top-{K:>2d} city neighbors avg lat vs actual lat: r={r:.3f} (p={p:.2e})")


# ============================================================
# ANALYSIS 5: How much variance do semantic sims explain beyond clusters?
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: Semantic sims as predictors — incremental over clusters")
print("=" * 70)

from numpy.linalg import lstsq

# Feature set 1: average lat of top-10 neighbor cities (cluster proxy)
neighbor_avg_lat = []
for i in range(n):
    sims_to_others = X_normed @ X_normed[i]
    sims_to_others[i] = -1
    top_idx = np.argsort(sims_to_others)[::-1][:10]
    neighbor_avg_lat.append(np.mean(lats[top_idx]))
neighbor_avg_lat = np.array(neighbor_avg_lat)

# Feature set 2: similarities to all semantic words
sem_features = []
sem_feature_names = []
for cat_name, words in semantic_words.items():
    for word in words:
        if word not in glove: continue
        w_normed = glove[word] / (np.linalg.norm(glove[word]) + 1e-10)
        sem_features.append(X_normed @ w_normed)
        sem_feature_names.append(word)
sem_features = np.array(sem_features).T  # (n_cities, n_words)

def r2_ols(X_feat, y):
    X_aug = np.column_stack([np.ones(len(y)), X_feat])
    w, _, _, _ = lstsq(X_aug, y, rcond=None)
    yp = X_aug @ w
    ss_res = np.sum((y - yp)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res / ss_tot

r2_cluster = r2_ols(neighbor_avg_lat.reshape(-1, 1), lats)
r2_semantic = r2_ols(sem_features, lats)
r2_both = r2_ols(np.column_stack([neighbor_avg_lat, sem_features]), lats)

print(f"  Predicting latitude (OLS, in-sample):")
print(f"    Cluster proxy only (avg neighbor lat):   R² = {r2_cluster:.3f}")
print(f"    Semantic word sims only ({sem_features.shape[1]} words): R² = {r2_semantic:.3f}")
print(f"    Both combined:                           R² = {r2_both:.3f}")
print(f"    Increment from adding semantic words:    ΔR² = {r2_both - r2_cluster:.3f}")

# Same for temperature
r2_cluster_t = r2_ols(neighbor_avg_lat.reshape(-1, 1), temps)
neighbor_avg_temp = []
for i in range(n):
    sims_to_others = X_normed @ X_normed[i]
    sims_to_others[i] = -1
    top_idx = np.argsort(sims_to_others)[::-1][:10]
    neighbor_avg_temp.append(np.mean(temps[top_idx]))
neighbor_avg_temp = np.array(neighbor_avg_temp)

r2_cluster_t = r2_ols(neighbor_avg_temp.reshape(-1, 1), temps)
r2_semantic_t = r2_ols(sem_features, temps)
r2_both_t = r2_ols(np.column_stack([neighbor_avg_temp, sem_features]), temps)

print(f"\n  Predicting temperature (OLS, in-sample):")
print(f"    Cluster proxy only (avg neighbor temp):  R² = {r2_cluster_t:.3f}")
print(f"    Semantic word sims only ({sem_features.shape[1]} words): R² = {r2_semantic_t:.3f}")
print(f"    Both combined:                           R² = {r2_both_t:.3f}")
print(f"    Increment from adding semantic words:    ΔR² = {r2_both_t - r2_cluster_t:.3f}")


# ============================================================
# FIGURES
# ============================================================
print("\nGenerating figures...")

label_cities = ['anchorage', 'helsinki', 'moscow', 'london', 'new york',
                'tokyo', 'miami', 'cairo', 'mumbai', 'bangkok',
                'singapore', 'lagos', 'sydney', 'buenos aires']

def annotate_select(ax, xs, ys):
    for i in range(n):
        if names[i] in label_cities:
            dx, dy = 6, 6
            nl = names[i]
            if nl in ('singapore', 'bangkok', 'lagos', 'mumbai'):
                dx, dy = -8, -10
            if nl == 'buenos aires':
                dx = -60
            if nl == 'anchorage':
                dy = -10
            ax.annotate(nl.title(), (xs[i], ys[i]),
                       fontsize=7.5, fontweight='bold', zorder=10,
                       xytext=(dx, dy), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                                alpha=0.75, edgecolor='none'))


# --- Fig 1: Individual semantic word correlations with latitude ---
# Sort by absolute r(lat)
sorted_results = sorted(all_word_results, key=lambda x: -abs(x['r_lat']))

fig1, ax1 = plt.subplots(figsize=(12, 7))
word_labels = [r['word'] for r in sorted_results]
r_vals = [r['r_lat'] for r in sorted_results]
bar_colors = []
for r in sorted_results:
    if r['category'] == 'cold_climate': bar_colors.append('#4472C4')
    elif r['category'] == 'warm_climate': bar_colors.append('#ED7D31')
    elif r['category'] == 'northern_direction': bar_colors.append('#70AD47')
    elif r['category'] == 'southern_direction': bar_colors.append('#A855F7')
    elif r['category'] == 'european_identity': bar_colors.append('#FFD700')
    elif r['category'] == 'developing_world': bar_colors.append('#FF6B6B')
    else: bar_colors.append('#999')

sig_markers = ['*' if r['p_lat'] < 0.05 else '' for r in sorted_results]

bars = ax1.barh(range(len(word_labels)), r_vals, color=bar_colors,
                edgecolor='white', linewidth=0.8, alpha=0.85)

for i, (bar, sig) in enumerate(zip(bars, sig_markers)):
    if sig:
        w = bar.get_width()
        ax1.text(w + 0.01 * np.sign(w), i, '*', fontsize=14,
                 fontweight='bold', va='center', color='black')

ax1.set_yticks(range(len(word_labels)))
ax1.set_yticklabels(word_labels, fontsize=9)
ax1.set_xlabel('Pearson r with Latitude', fontsize=12)
ax1.axvline(x=0, color='black', linewidth=0.5)
ax1.grid(True, alpha=0.15, axis='x', linewidth=0.5)
ax1.invert_yaxis()

from matplotlib.patches import Patch as P
legend_els = [
    P(facecolor='#4472C4', label='Cold/climate'),
    P(facecolor='#ED7D31', label='Warm/tropical'),
    P(facecolor='#70AD47', label='Northern direction'),
    P(facecolor='#A855F7', label='Southern direction'),
    P(facecolor='#FFD700', label='European identity'),
    P(facecolor='#FF6B6B', label='Developing world'),
]
ax1.legend(handles=legend_els, fontsize=9, loc='lower right', framealpha=0.9)

fig1.tight_layout()
fig1.savefig('semantic_word_lat_correlations.png', dpi=300, bbox_inches='tight')
print("  Saved semantic_word_lat_correlations.png")


# --- Fig 2: Composite scores vs latitude (3 panels) ---
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (comp_name, scores) in enumerate(composite_scores.items()):
    ax = axes[idx]
    r, p = pearsonr(scores, lats)
    for i in range(n):
        ax.scatter(lats[i], scores[i], c=colors[i], s=55, alpha=0.85,
                   edgecolors='white', linewidth=0.5, zorder=5)
    # Regression line
    m, b = np.polyfit(lats, scores, 1)
    xs = np.linspace(lats.min() - 3, lats.max() + 3, 100)
    ax.plot(xs, m * xs + b, color='#333', alpha=0.4, linewidth=2,
            linestyle='--', zorder=2)
    annotate_select(ax, lats, scores)
    ax.set_xlabel('Actual Latitude', fontsize=11)
    ax.set_ylabel(f'Similarity score: {comp_name}', fontsize=11)
    ax.set_title(f'{comp_name}\nr = {r:.3f} (p = {p:.4f})',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.15, linewidth=0.5)

legend_elements = [Patch(facecolor=color_map[r], edgecolor='white', label=r)
                   for r in regions]
fig2.legend(handles=legend_elements, loc='lower center', ncol=6,
            fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
fig2.tight_layout()
fig2.subplots_adjust(bottom=0.1)
fig2.savefig('semantic_composite_vs_latitude.png', dpi=300, bbox_inches='tight')
print("  Saved semantic_composite_vs_latitude.png")


# --- Fig 3: Scatter of best individual semantic words vs latitude ---
# Pick top 4 words by abs(r_lat)
top4 = sorted_results[:4]
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5.5))

for idx, wr in enumerate(top4):
    ax = axes3[idx]
    sims = wr['sims']
    r, p = wr['r_lat'], wr['p_lat']
    for i in range(n):
        ax.scatter(lats[i], sims[i], c=colors[i], s=50, alpha=0.85,
                   edgecolors='white', linewidth=0.5, zorder=5)
    m, b = np.polyfit(lats, sims, 1)
    xs = np.linspace(lats.min() - 3, lats.max() + 3, 100)
    ax.plot(xs, m * xs + b, color='#333', alpha=0.4, linewidth=2,
            linestyle='--', zorder=2)
    annotate_select(ax, lats, sims)
    ax.set_xlabel('Actual Latitude', fontsize=11)
    ax.set_ylabel(f'Cosine sim to "{wr["word"]}"', fontsize=11)
    ax.set_title(f'"{wr["word"]}" (r = {r:.3f})',
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.15, linewidth=0.5)

fig3.legend(handles=legend_elements, loc='lower center', ncol=6,
            fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
fig3.tight_layout()
fig3.subplots_adjust(bottom=0.12)
fig3.savefig('semantic_top_words_vs_latitude.png', dpi=300, bbox_inches='tight')
print("  Saved semantic_top_words_vs_latitude.png")


# --- Fig 4: Neighbor-city avg latitude vs actual latitude ---
fig4, ax4 = plt.subplots(figsize=(8, 7))

for i in range(n):
    ax4.scatter(lats[i], neighbor_avg_lat[i], c=colors[i], s=55, alpha=0.85,
                edgecolors='white', linewidth=0.5, zorder=5)

r_nl, p_nl = pearsonr(neighbor_avg_lat, lats)
m, b = np.polyfit(lats, neighbor_avg_lat, 1)
xs = np.linspace(lats.min() - 3, lats.max() + 3, 100)
ax4.plot(xs, m * xs + b, color='#333', alpha=0.4, linewidth=2,
         linestyle='--', zorder=2)
# Perfect prediction line
ax4.plot(xs, xs, color='red', alpha=0.3, linewidth=1.5,
         linestyle=':', zorder=1, label='Perfect prediction')
annotate_select(ax4, lats, neighbor_avg_lat)

ax4.set_xlabel('Actual Latitude', fontsize=12)
ax4.set_ylabel('Avg Latitude of Top-10 Nearest City Neighbors', fontsize=12)
ax4.set_title(f'City cluster proxy: r = {r_nl:.3f}', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.15, linewidth=0.5)
ax4.legend(handles=legend_elements + [plt.Line2D([0],[0], color='red', alpha=0.3,
           linestyle=':', label='y = x')],
           loc='lower right', fontsize=9, framealpha=0.9)

fig4.tight_layout()
fig4.savefig('semantic_neighbor_city_latitude.png', dpi=300, bbox_inches='tight')
print("  Saved semantic_neighbor_city_latitude.png")


print("\nDone.")
