"""
Semantic Ablation Analysis

For each semantic category (direction words, climate terms, region names,
country names, economic terms, cultural/language terms), we:
  1. Collect GloVe vectors for words in that category
  2. Compute the principal subspace they span (top-k PCs)
  3. Project city embeddings into that subspace and subtract it out
  4. Re-run the ridge probe on the ablated embeddings
  5. Measure R² drop — this tells us how much of the geographic (or other)
     signal was carried by co-occurrence with that word category

This decomposes the distributional signal into interpretable sources.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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

# ============================================================
# SEMANTIC CATEGORIES TO ABLATE
# ============================================================
categories = {
    'Cardinal\ndirections': [
        'north', 'south', 'east', 'west',
        'northern', 'southern', 'eastern', 'western',
        'northeast', 'northwest', 'southeast', 'southwest',
        'northward', 'southward', 'eastward', 'westward',
    ],
    'Climate &\nweather': [
        'cold', 'warm', 'hot', 'cool', 'freezing', 'mild',
        'tropical', 'arctic', 'temperate', 'humid', 'arid', 'dry',
        'monsoon', 'snow', 'rain', 'sunny', 'winter', 'summer',
        'desert', 'ice', 'frost', 'heat', 'climate', 'weather',
        'equatorial', 'subtropical', 'polar',
    ],
    'Region &\ncontinent': [
        'europe', 'european', 'asia', 'asian', 'africa', 'african',
        'america', 'american', 'oceania', 'pacific', 'atlantic',
        'mediterranean', 'caribbean', 'scandinavian', 'nordic',
        'latin', 'middle', 'southeast', 'western', 'eastern',
        'central', 'continental', 'hemisphere', 'equator',
        'arctic', 'antarctic', 'tropical', 'subtropical',
    ],
    'Country\nnames': [
        'united', 'states', 'canada', 'canadian', 'mexico', 'mexican',
        'britain', 'british', 'england', 'english', 'france', 'french',
        'germany', 'german', 'spain', 'spanish', 'italy', 'italian',
        'japan', 'japanese', 'china', 'chinese', 'india', 'indian',
        'russia', 'russian', 'brazil', 'brazilian', 'australia', 'australian',
        'egypt', 'egyptian', 'nigeria', 'nigerian', 'kenya', 'kenyan',
        'turkey', 'turkish', 'iran', 'iranian', 'iraq', 'iraqi',
        'korea', 'korean', 'thailand', 'pakistan', 'indonesian',
        'south', 'zealand', 'colombian', 'argentina', 'chilean',
        'peruvian', 'venezuelan', 'swedish', 'norwegian', 'finnish',
        'dutch', 'belgian', 'austrian', 'czech', 'polish', 'hungarian',
        'portuguese', 'irish', 'scottish', 'greek', 'danish',
    ],
    'Economic\nterms': [
        'rich', 'poor', 'wealthy', 'poverty', 'developed', 'developing',
        'industrial', 'economy', 'economic', 'gdp', 'trade', 'market',
        'financial', 'business', 'commercial', 'investment', 'growth',
        'income', 'prosperity', 'infrastructure', 'modern', 'urban',
        'metropolitan', 'cosmopolitan', 'capital', 'port', 'hub',
    ],
    'Cultural &\nlanguage': [
        'english', 'french', 'spanish', 'chinese', 'arabic', 'hindi',
        'japanese', 'german', 'russian', 'portuguese', 'korean',
        'turkish', 'persian', 'dutch', 'italian', 'swedish',
        'muslim', 'christian', 'buddhist', 'hindu', 'catholic',
        'colonial', 'imperial', 'ancient', 'historic', 'medieval',
        'cultural', 'heritage', 'tradition', 'cuisine', 'language',
    ],
}

# ============================================================
# LOAD FULL GLOVE (we need category words too)
# ============================================================
GLOVE_PATH = "glove.6B.300d.txt"
if not os.path.exists(GLOVE_PATH):
    print(f"ERROR: {GLOVE_PATH} not found"); sys.exit(1)

# Collect all words we need
all_needed = set()
for name, *_ in cities:
    for w in name.split():
        all_needed.add(w)
for cat_words in categories.values():
    for w in cat_words:
        all_needed.add(w)

print("Loading GloVe embeddings...")
glove = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if parts[0] in all_needed:
            glove[parts[0]] = np.array([float(x) for x in parts[1:]])

print(f"  Loaded {len(glove)} vectors")

# Report category coverage
for cat_name, words in categories.items():
    found = [w for w in words if w in glove]
    label = cat_name.replace('\n', ' ')
    print(f"  {label}: {len(found)}/{len(words)} words found")

# ============================================================
# BUILD CITY EMBEDDINGS & TARGETS
# ============================================================
def city_emb(name):
    vecs = [glove[w] for w in name.split() if w in glove]
    return np.mean(vecs, axis=0) if vecs else None

valid = []
for name, lat, lon, temp, pop, gdp, elev, year in cities:
    e = city_emb(name)
    if e is None: continue
    valid.append(dict(name=name, lat=lat, lon=lon, temp=temp,
                      pop=pop, gdp=gdp, elev=elev, year=year, emb=e))

names = [v['name'] for v in valid]
X = np.array([v['emb'] for v in valid])
n, d = X.shape
print(f"  {n} cities, {d} dims")

targets = {
    'Latitude': np.array([v['lat'] for v in valid]),
    'Longitude': np.array([v['lon'] for v in valid]),
    'Temperature': np.array([v['temp'] for v in valid]),
}

# ============================================================
# PROBE FUNCTION
# ============================================================
lambdas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

def probe_r2(X_in, y):
    """Run ridge probe with CV, return test R²."""
    d_ = X_in.shape[1]
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
            Xtr, Xte_ = X_in[idx[~mask]], X_in[idx[mask]]
            ytr, yte_ = y[idx[~mask]], y[idx[mask]]
            W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d_), Xtr.T @ ytr)
            yp = Xte_ @ W
            ss_r = np.sum((yte_ - yp)**2)
            ss_t = np.sum((yte_ - yte_.mean())**2)
            fr2.append(1 - ss_r / ss_t if ss_t > 0 else 0)
        cv = np.mean(fr2)
        if cv > best_cv: best_cv, best_lam = cv, lam

    W = np.linalg.solve(X_in[tr].T @ X_in[tr] + best_lam * np.eye(d_),
                         X_in[tr].T @ y[tr])
    yp = X_in[te] @ W
    ss_r = np.sum((y[te] - yp)**2)
    ss_t = np.sum((y[te] - y[te].mean())**2)
    return 1 - ss_r / ss_t if ss_t > 0 else 0

# ============================================================
# COMPUTE BASELINES
# ============================================================
print("\nBaseline (full embeddings):")
baselines = {}
for tname, y in targets.items():
    r2 = probe_r2(X, y)
    baselines[tname] = r2
    print(f"  {tname}: R² = {r2:.3f}")

# ============================================================
# ABLATION: Remove each category's subspace
# ============================================================
def get_category_subspace(cat_words, n_components=None):
    """Compute PCA basis for the subspace spanned by category words."""
    vecs = [glove[w] for w in cat_words if w in glove]
    if len(vecs) < 2:
        return None
    V = np.array(vecs)
    V_centered = V - V.mean(axis=0)
    U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)
    # Keep enough components to capture 90% of variance, or n_components
    if n_components is None:
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        n_components = max(1, np.searchsorted(cumvar, 0.90) + 1)
        n_components = min(n_components, len(vecs), 20)  # cap at 20
    basis = Vt[:n_components]  # shape (k, 300)
    return basis


def ablate_subspace(X_in, basis):
    """Remove the projection of X onto the subspace spanned by basis."""
    # basis: (k, d), each row is a unit-ish direction
    # Orthonormalize
    Q, _ = np.linalg.qr(basis.T)  # Q: (d, k)
    proj = X_in @ Q @ Q.T  # projection onto subspace
    return X_in - proj


print("\n" + "="*70)
print("ABLATION RESULTS")
print("="*70)
print(f"\n{'Category':<22s}", end='')
for tname in targets:
    print(f" {tname:>12s} (Δ)", end='')
print(f" {'dims removed':>14s}")
print("-"*70)

ablation_results = {}

for cat_name, cat_words in categories.items():
    label = cat_name.replace('\n', ' ')
    basis = get_category_subspace(cat_words)
    if basis is None:
        print(f"  {label}: not enough words, skipping")
        continue

    X_ablated = ablate_subspace(X, basis)
    n_dims = basis.shape[0]

    ablation_results[cat_name] = {'n_dims': n_dims}
    print(f"  {label:<20s}", end='')
    for tname, y in targets.items():
        r2_abl = probe_r2(X_ablated, y)
        drop = baselines[tname] - r2_abl
        ablation_results[cat_name][tname] = {
            'r2': r2_abl, 'drop': drop,
            'pct_drop': drop / baselines[tname] * 100 if baselines[tname] > 0 else 0,
        }
        print(f"   {r2_abl:+.3f} ({drop:+.3f})", end='')
    print(f"   {n_dims:>10d}")

print("="*70)

# Also run with ALL categories ablated simultaneously
print("\nAll categories ablated simultaneously:")
all_bases = []
for cat_words in categories.values():
    basis = get_category_subspace(cat_words)
    if basis is not None:
        all_bases.append(basis)
combined_basis = np.vstack(all_bases)
X_all_ablated = ablate_subspace(X, combined_basis)
total_dims = combined_basis.shape[0]
print(f"  Total subspace dimensions removed: {total_dims}")
ablation_results['All combined'] = {'n_dims': total_dims}
for tname, y in targets.items():
    r2_abl = probe_r2(X_all_ablated, y)
    drop = baselines[tname] - r2_abl
    ablation_results['All combined'][tname] = {
        'r2': r2_abl, 'drop': drop,
        'pct_drop': drop / baselines[tname] * 100 if baselines[tname] > 0 else 0,
    }
    print(f"  {tname}: R² = {r2_abl:.3f} (drop: {drop:+.3f}, "
          f"{drop/baselines[tname]*100:.0f}% of signal)" if baselines[tname] > 0
          else f"  {tname}: R² = {r2_abl:.3f}")


# ============================================================
# FIGURE: ABLATION BAR CHART
# ============================================================
print("\nGenerating figures...")

cat_names = list(categories.keys())
target_names = list(targets.keys())
target_colors = {'Latitude': '#4472C4', 'Longitude': '#ED7D31', 'Temperature': '#70AD47'}

# --- Figure A: R² drop per category ---
fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.arange(len(cat_names))
n_targets = len(target_names)
total_w = 0.7
bw = total_w / n_targets

for ti, tname in enumerate(target_names):
    drops = []
    for cat in cat_names:
        if cat in ablation_results and tname in ablation_results[cat]:
            drops.append(ablation_results[cat][tname]['drop'])
        else:
            drops.append(0)
    offset = (ti - (n_targets - 1) / 2) * bw
    bars = ax.bar(x + offset, drops, bw, color=target_colors[tname],
                  edgecolor='white', linewidth=1, alpha=0.85, label=tname)
    # Value labels
    for bar, drop in zip(bars, drops):
        if abs(drop) > 0.005:
            y_pos = drop + 0.008 if drop >= 0 else drop - 0.008
            va = 'bottom' if drop >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{drop:+.3f}', ha='center', va=va, fontsize=7.5,
                    fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(cat_names, fontsize=10)
ax.set_ylabel('R\u00B2 drop when category ablated', fontsize=12)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.15, axis='y', linewidth=0.5)

fig.tight_layout()
fig.savefig('ablation_r2_drop.png', dpi=300, bbox_inches='tight')
print("  Saved ablation_r2_drop.png")


# --- Figure B: Percentage of signal explained ---
fig2, ax2 = plt.subplots(figsize=(12, 6.5))

for ti, tname in enumerate(target_names):
    pcts = []
    for cat in cat_names:
        if cat in ablation_results and tname in ablation_results[cat]:
            pcts.append(ablation_results[cat][tname]['pct_drop'])
        else:
            pcts.append(0)
    offset = (ti - (n_targets - 1) / 2) * bw
    bars = ax2.bar(x + offset, pcts, bw, color=target_colors[tname],
                   edgecolor='white', linewidth=1, alpha=0.85, label=tname)
    for bar, pct in zip(bars, pcts):
        if abs(pct) > 1:
            y_pos = pct + 1 if pct >= 0 else pct - 1
            va = 'bottom' if pct >= 0 else 'top'
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                     f'{pct:.0f}%', ha='center', va=va, fontsize=7.5,
                     fontweight='bold')

ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(cat_names, fontsize=10)
ax2.set_ylabel('% of R\u00B2 explained by category', fontsize=12)
ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax2.grid(True, alpha=0.15, axis='y', linewidth=0.5)

fig2.tight_layout()
fig2.savefig('ablation_pct_explained.png', dpi=300, bbox_inches='tight')
print("  Saved ablation_pct_explained.png")


# --- Figure C: Waterfall — cumulative ablation ---
print("\nCumulative ablation (removing categories one at a time):")
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

# Order categories by average R² drop across geography targets
cat_importance = []
for cat in cat_names:
    avg_drop = np.mean([ablation_results[cat][t]['drop']
                        for t in ['Latitude', 'Longitude']
                        if t in ablation_results[cat]])
    cat_importance.append((cat, avg_drop))
cat_importance.sort(key=lambda x: -x[1])
sorted_cats = [c[0] for c in cat_importance]

for ti, tname in enumerate(target_names):
    ax = axes3[ti]
    # Cumulatively ablate categories in order of importance
    X_cum = X.copy()
    r2_values = [baselines[tname]]
    labels = ['Full']

    for cat in sorted_cats:
        basis = get_category_subspace(categories[cat.replace('\n', ' ')]
                                       if cat.replace('\n', ' ') in categories
                                       else categories[cat])
        if basis is not None:
            X_cum = ablate_subspace(X_cum, basis)
        r2_val = probe_r2(X_cum, targets[tname])
        r2_values.append(r2_val)
        labels.append(f'− {cat}')

    # Plot as step-down
    bar_colors = [target_colors[tname]] + ['#D5D5D5'] * len(sorted_cats)
    bar_colors_actual = []
    for i, r2 in enumerate(r2_values):
        if i == 0:
            bar_colors_actual.append(target_colors[tname])
        else:
            bar_colors_actual.append('#AAAAAA')

    ax.bar(range(len(r2_values)), r2_values, color=bar_colors_actual,
           edgecolor='white', linewidth=1, alpha=0.85)

    for i, r2 in enumerate(r2_values):
        ax.text(i, r2 + 0.02, f'{r2:.2f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel('Test R\u00B2', fontsize=11)
    ax.set_title(tname, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15, axis='y', linewidth=0.5)
    ax.set_ylim(min(0, min(r2_values) - 0.1), max(r2_values) + 0.15)

fig3.tight_layout()
fig3.savefig('ablation_cumulative.png', dpi=300, bbox_inches='tight')
print("  Saved ablation_cumulative.png")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY: What drives the geographic signal in GloVe?")
print("="*70)
for cat in sorted_cats:
    label = cat.replace('\n', ' ')
    lat_pct = ablation_results[cat]['Latitude']['pct_drop']
    lon_pct = ablation_results[cat]['Longitude']['pct_drop']
    temp_pct = ablation_results[cat]['Temperature']['pct_drop']
    print(f"  {label:<22s}  Lat: {lat_pct:+5.1f}%   Lon: {lon_pct:+5.1f}%   "
          f"Temp: {temp_pct:+5.1f}%")

all_lat = ablation_results['All combined']['Latitude']
all_lon = ablation_results['All combined']['Longitude']
all_temp = ablation_results['All combined']['Temperature']
print(f"\n  All combined:          "
      f"Lat: {all_lat['pct_drop']:+5.1f}%   Lon: {all_lon['pct_drop']:+5.1f}%   "
      f"Temp: {all_temp['pct_drop']:+5.1f}%")
print(f"  Residual R² after all: "
      f"Lat: {all_lat['r2']:.3f}    Lon: {all_lon['r2']:.3f}    "
      f"Temp: {all_temp['r2']:.3f}")

print("\nDone.")
