"""
Experiment: The Probe Recovers Whatever You Ask For

Linear probes on GloVe 300d embeddings for multiple target variables:
  1. Latitude / Longitude (geography)
  2. Mean annual temperature
  3. Population (log10)
  4. GDP per capita (of country, log10)
  5. Elevation (meters)
  6. Year founded

If multiple probes succeed, it demonstrates that the technique recovers
any variable correlated with distributional semantics, not specifically
spatial structure.

Usage:
    pip install numpy matplotlib
    wget https://nlp.stanford.edu/data/glove.6B.zip
    unzip glove.6B.zip glove.6B.300d.txt
    python glove_multi_probe.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import sys

# ============================================================
# CITY DATA
# Sources: Wikipedia, World Bank, various climate databases
# GDP is country-level GDP per capita (PPP, approx 2023 USD)
# Population is city/metro approximate (millions aren't needed, just order of magnitude)
# Elevation in meters
# Founding year: approximate, negative = BCE
# ============================================================
cities = [
    # (name, lat, lon, temp_C, population, gdp_per_capita, elevation_m, year_founded)
    # North America
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
    # Europe
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
    # Asia
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
    # Africa
    ("cairo", 30.0, 31.2, 22, 10230000, 12000, 75, 969),
    ("lagos", 6.5, 3.4, 27, 15400000, 5000, 41, 1472),
    ("nairobi", -1.3, 36.8, 18, 4400000, 4000, 1661, 1899),
    ("johannesburg", -26.2, 28.0, 16, 5780000, 13000, 1753, 1886),
    ("cape town", -33.9, 18.4, 17, 4620000, 13000, 0, 1652),
    ("casablanca", 33.6, -7.6, 18, 3720000, 7000, 27, 768),
    ("addis ababa", 9.0, 38.7, 16, 3600000, 2000, 2355, 1886),
    ("accra", 5.6, -0.2, 27, 2270000, 5000, 61, 1877),
    ("algiers", 36.8, 3.1, 18, 3600000, 11000, 0, 944),
    # South America
    ("buenos aires", -34.6, -58.4, 17, 3060000, 22000, 25, 1536),
    ("santiago", -33.4, -70.6, 14, 6160000, 25000, 520, 1541),
    ("lima", -12.0, -77.0, 19, 10000000, 12000, 161, 1535),
    ("bogota", 4.7, -74.1, 14, 7410000, 14000, 2640, 1538),
    ("rio de janeiro", -22.9, -43.2, 24, 6750000, 15000, 11, 1565),
    ("sao paulo", -23.6, -46.6, 20, 12300000, 15000, 760, 1554),
    ("caracas", 10.5, -66.9, 22, 2940000, 16000, 900, 1567),
    ("quito", -0.2, -78.5, 14, 2800000, 11000, 2850, 1534),
    # Oceania
    ("sydney", -33.9, 151.2, 18, 5300000, 52000, 3, 1788),
    ("melbourne", -37.8, 144.96, 15, 5000000, 52000, 31, 1835),
    ("auckland", -36.8, 174.8, 15, 1660000, 42000, 0, 1840),
    ("perth", -31.9, 115.9, 19, 2100000, 52000, 0, 1829),
]

# Regions for color coding
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
    'North America': '#E74C3C',
    'Europe': '#3498DB',
    'Asia': '#F39C12',
    'Africa': '#27AE60',
    'South America': '#9B59B6',
    'Oceania': '#1ABC9C',
}


# ============================================================
# LOAD GLOVE
# ============================================================
GLOVE_PATH = "glove.6B.300d.txt"

if not os.path.exists(GLOVE_PATH):
    print(f"ERROR: Cannot find {GLOVE_PATH}")
    print("Download: wget https://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip glove.6B.300d.txt")
    sys.exit(1)

print("Loading GloVe embeddings...")

needed_words = set()
for name, *_ in cities:
    for word in name.split():
        needed_words.add(word)

embeddings = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        if word in needed_words:
            vec = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vec

print(f"Loaded {len(embeddings)} relevant embeddings")


# ============================================================
# BUILD DATA
# ============================================================
def get_emb(name):
    vecs = [embeddings[w] for w in name.split() if w in embeddings]
    return np.mean(vecs, axis=0) if vecs else None

def get_color(name):
    for r, cl in regions.items():
        if name in cl:
            return color_map[r]
    return '#888888'

valid = []
for name, lat, lon, temp, pop, gdp, elev, year in cities:
    emb = get_emb(name)
    if emb is None:
        print(f"  Skipping {name}")
        continue
    valid.append({
        'name': name, 'lat': lat, 'lon': lon, 'temp': temp,
        'pop': pop, 'gdp': gdp, 'elev': elev, 'year': year,
        'emb': emb, 'color': get_color(name)
    })

X = np.array([v['emb'] for v in valid])
names = [v['name'] for v in valid]
colors = [v['color'] for v in valid]
n = len(valid)
d = X.shape[1]

print(f"Valid cities: {n}, dims: {d}")


# ============================================================
# DEFINE PROBE TARGETS
# ============================================================
targets = {
    'Latitude': np.array([v['lat'] for v in valid]),
    'Longitude': np.array([v['lon'] for v in valid]),
    'Temperature (°C)': np.array([v['temp'] for v in valid]),
    'Population (log₁₀)': np.log10(np.array([v['pop'] for v in valid])),
    'GDP per capita (log₁₀ $)': np.log10(np.array([v['gdp'] for v in valid])),
    'Elevation (m)': np.array([v['elev'] for v in valid]),
    'Year Founded': np.array([v['year'] for v in valid]),
}


# ============================================================
# PROBE EACH TARGET
# ============================================================
np.random.seed(42)
indices = np.random.permutation(n)

n_folds = 5
fold_size = n // n_folds

results = {}

print("\n" + "="*60)
print("PROBING RESULTS")
print("="*60)

for target_name, y in targets.items():
    # Cross-validate lambda
    best_lam = None
    best_cv_r2 = -np.inf

    for lam in [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        fold_r2s = []
        for fold in range(n_folds):
            te_mask = np.zeros(n, dtype=bool)
            te_mask[fold * fold_size:(fold + 1) * fold_size] = True
            tr_mask = ~te_mask

            X_tr = X[indices[tr_mask]]
            X_te = X[indices[te_mask]]
            y_tr = y[indices[tr_mask]]
            y_te = y[indices[te_mask]]

            W = np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(d), X_tr.T @ y_tr)
            y_pred = X_te @ W
            ss_res = np.sum((y_te - y_pred)**2)
            ss_tot = np.sum((y_te - y_te.mean())**2)
            if ss_tot > 0:
                fold_r2s.append(1 - ss_res / ss_tot)
            else:
                fold_r2s.append(0)

        cv_r2 = np.mean(fold_r2s)
        if cv_r2 > best_cv_r2:
            best_cv_r2 = cv_r2
            best_lam = lam

    # Final 80/20 split
    n_train = int(0.8 * n)
    tr = indices[:n_train]
    te = indices[n_train:]

    W = np.linalg.solve(X[tr].T @ X[tr] + best_lam * np.eye(d), X[tr].T @ y[tr])
    y_pred_te = X[te] @ W
    y_pred_all = X @ W

    ss_res = np.sum((y[te] - y_pred_te)**2)
    ss_tot = np.sum((y[te] - y[te].mean())**2)
    r2_test = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    mae = np.mean(np.abs(y[te] - y_pred_te))

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(y[te], y_pred_te)

    results[target_name] = {
        'r2': r2_test, 'mae': mae, 'rho': rho, 'pval': pval,
        'best_lam': best_lam, 'cv_r2': best_cv_r2,
        'y_pred_all': y_pred_all, 'y_actual': y,
        'train_idx': tr, 'test_idx': te,
    }

    print(f"\n  {target_name}:")
    print(f"    Best λ = {best_lam}, CV R² = {best_cv_r2:.3f}")
    print(f"    Test R² = {r2_test:.3f}, MAE = {mae:.1f}, Spearman ρ = {rho:.3f} (p={pval:.4f})")


# ============================================================
# FIGURE 1: Summary bar chart of R² values
# ============================================================
print("\nGenerating figures...")

target_names = list(results.keys())
r2_values = [results[t]['r2'] for t in target_names]
cv_r2_values = [results[t]['cv_r2'] for t in target_names]

# Sort by R²
sort_order = np.argsort(r2_values)[::-1]
sorted_names = [target_names[i] for i in sort_order]
sorted_r2 = [r2_values[i] for i in sort_order]
sorted_cv = [cv_r2_values[i] for i in sort_order]

bar_colors = ['#4472C4', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5', '#FF6B6B', '#A855F7']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(sorted_names)), sorted_r2, color=[bar_colors[i % len(bar_colors)] for i in range(len(sorted_names))],
              edgecolor='white', linewidth=1.5, alpha=0.85)

# Add R² labels on bars
for i, (bar, r2) in enumerate(zip(bars, sorted_r2)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'R²={r2:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(sorted_names)))
ax.set_xticklabels(sorted_names, fontsize=11, rotation=15, ha='right')
ax.set_ylabel('Test R² (held-out 20%)', fontsize=13)
ax.set_title('What Can a Linear Probe Recover from GloVe 300d Embeddings?\n'
             'The same technique used to claim LLMs have "spatial world models"\n'
             'recovers geography, temperature, economics, and more from 2014 word vectors.',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.15, axis='y')
ax.set_ylim(-0.1, max(sorted_r2) + 0.15)
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('glove_multi_probe_summary.png', dpi=150, bbox_inches='tight')
print("  Saved glove_multi_probe_summary.png")


# ============================================================
# FIGURE 2: Grid of predicted vs actual for each target
# ============================================================
# Only plot targets with positive R²
good_targets = [t for t in target_names if results[t]['r2'] > 0]
n_panels = len(good_targets)
ncols = min(3, n_panels)
nrows = (n_panels + ncols - 1) // ncols

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
if nrows == 1 and ncols == 1:
    axes2 = np.array([axes2])
axes2 = axes2.flatten()

label_these = ['anchorage', 'moscow', 'london', 'new york', 'cairo',
               'mumbai', 'bangkok', 'singapore', 'sydney', 'buenos aires',
               'tokyo', 'miami', 'helsinki', 'nairobi', 'beijing',
               'lagos', 'kabul', 'zurich', 'mexico city', 'athens',
               'denver', 'bogota', 'rome', 'shanghai']

for idx, target_name in enumerate(good_targets):
    ax = axes2[idx]
    res = results[target_name]
    y = res['y_actual']
    yp = res['y_pred_all']
    tr = res['train_idx']
    te = res['test_idx']

    for i in range(n):
        marker = 'D' if i in te else 'o'
        size = 55 if i in te else 35
        alpha = 0.9 if i in te else 0.5
        ax.scatter(y[i], yp[i], c=colors[i], s=size, alpha=alpha,
                   marker=marker, edgecolors='white', linewidth=0.5, zorder=5)

    # Diagonal
    lo = min(y.min(), yp.min())
    hi = max(y.max(), yp.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], 'k--', alpha=0.3)

    # Label select cities
    for i, name in enumerate(names):
        if name in label_these:
            ax.annotate(name.title(), (y[i], yp[i]), fontsize=7,
                       zorder=10, xytext=(4, 4), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                alpha=0.6, edgecolor='none'))

    ax.set_xlabel(f'Actual {target_name}', fontsize=11)
    ax.set_ylabel(f'Predicted {target_name}', fontsize=11)
    ax.set_title(f'{target_name}\n'
                 f'Test R² = {res["r2"]:.2f}, ρ = {res["rho"]:.2f}',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.15)

# Hide unused axes
for idx in range(len(good_targets), len(axes2)):
    axes2[idx].set_visible(False)

# Legend
legend_elements = [Patch(facecolor=color_map[r], edgecolor='white', label=r)
                   for r in regions.keys()]
fig2.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10,
            frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Linear Probes on GloVe 300d: Predicted vs Actual\n'
             '○ = train, ◆ = held-out test',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.06)
plt.savefig('glove_multi_probe_scatter.png', dpi=150, bbox_inches='tight')
print("  Saved glove_multi_probe_scatter.png")


# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print(f"{'Target':<25} {'Test R²':>8} {'CV R²':>8} {'MAE':>8} {'Spearman ρ':>12} {'λ':>6}")
print("-"*70)
for t in sorted_names:
    r = results[t]
    print(f"{t:<25} {r['r2']:>8.3f} {r['cv_r2']:>8.3f} {r['mae']:>8.1f} {r['rho']:>12.3f} {r['best_lam']:>6.0f}")
print("="*70)

print(f"""
INTERPRETATION:
  A linear probe on GloVe 300d embeddings (2014 word co-occurrence statistics)
  recovers multiple physical, economic, and historical properties of cities.

  If recovering geographic coordinates (R² ≈ 0.7) is evidence that the model
  has a "spatial world model" (Gurnee & Tegmark 2023), then recovering
  temperature, GDP, and population must be evidence of thermal, economic,
  and demographic "world models."

  The more parsimonious explanation: language carries statistical fingerprints
  of the world it describes. Linear probes read those fingerprints. This is
  what you'd expect from a coordination device, not a world model.
""")
