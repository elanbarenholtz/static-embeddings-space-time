"""
Temporal Probe: Can static word embeddings encode when a person lived?

Replicating the temporal arm of Gurnee et al. (2023) "Language Models
Represent Space and Time" — but with GloVe/Word2Vec instead of Llama-2.

They probed for:
  1. Historical figures (death year)
  2. Art/entertainment (release year)
  3. News headlines (publication year)

We do #1: historical figures, using name embeddings from GloVe & Word2Vec,
probing for approximate era / birth year.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
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
# HISTORICAL FIGURES DATASET
# (name, birth_year, death_year, domain)
# Chosen for: fame (likely in embeddings), spread across eras
# ============================================================
figures = [
    # ============================================================
    # NAMING CONVENTION: use last names (or most distinctive single
    # name) to avoid ambiguity with cities, common words, etc.
    # Removed: washington, hamilton, florence, columbus, franklin,
    #          smith, victoria, king, marco, alexander, napoleon,
    #          watson (too common / city names / ambiguous)
    # ============================================================

    # Ancient / Classical (before 500 AD)
    ("aristotle", -384, -322, "philosophy"),
    ("plato", -428, -348, "philosophy"),
    ("socrates", -470, -399, "philosophy"),
    ("caesar", -100, -44, "military"),
    ("cicero", -106, -43, "philosophy"),
    ("cleopatra", -69, -30, "politics"),
    ("augustus", -63, 14, "politics"),
    ("nero", 37, 68, "politics"),
    ("euclid", -325, -265, "science"),
    ("archimedes", -287, -212, "science"),
    ("hippocrates", -460, -370, "science"),
    ("virgil", -70, -19, "literature"),
    ("ovid", -43, 17, "literature"),
    ("homer", -800, -701, "literature"),
    ("confucius", -551, -479, "philosophy"),
    ("hannibal", -247, -183, "military"),
    ("herodotus", -484, -425, "literature"),
    ("thucydides", -460, -400, "literature"),
    ("sophocles", -496, -406, "literature"),
    ("euripides", -480, -406, "literature"),
    ("ptolemy", 100, 170, "science"),
    ("seneca", -4, 65, "philosophy"),
    ("tacitus", 56, 120, "literature"),
    ("plutarch", 46, 119, "literature"),

    # Medieval (500–1400)
    ("charlemagne", 747, 814, "politics"),
    ("aquinas", 1225, 1274, "philosophy"),
    ("dante", 1265, 1321, "literature"),
    ("chaucer", 1343, 1400, "literature"),
    ("fibonacci", 1170, 1250, "science"),
    ("saladin", 1137, 1193, "military"),
    ("genghis", 1162, 1227, "military"),
    ("giotto", 1267, 1337, "art"),
    ("avicenna", 980, 1037, "science"),
    ("maimonides", 1138, 1204, "philosophy"),
    ("petrarch", 1304, 1374, "literature"),
    ("boccaccio", 1313, 1375, "literature"),
    ("abelard", 1079, 1142, "philosophy"),
    ("hildegard", 1098, 1179, "religion"),
    ("khayyam", 1048, 1131, "literature"),

    # Renaissance / Early Modern (1400–1700)
    ("gutenberg", 1400, 1468, "science"),
    ("leonardo", 1452, 1519, "art"),
    ("michelangelo", 1475, 1564, "art"),
    ("raphael", 1483, 1520, "art"),
    ("luther", 1483, 1546, "religion"),
    ("copernicus", 1473, 1543, "science"),
    ("machiavelli", 1469, 1527, "philosophy"),
    ("shakespeare", 1564, 1616, "literature"),
    ("galileo", 1564, 1642, "science"),
    ("kepler", 1571, 1630, "science"),
    ("descartes", 1596, 1650, "philosophy"),
    ("rembrandt", 1606, 1669, "art"),
    ("pascal", 1623, 1662, "science"),
    ("newton", 1643, 1727, "science"),
    ("locke", 1632, 1704, "philosophy"),
    ("bach", 1685, 1750, "music"),
    ("cervantes", 1547, 1616, "literature"),
    ("hobbes", 1588, 1679, "philosophy"),
    ("cromwell", 1599, 1658, "politics"),
    ("spinoza", 1632, 1677, "philosophy"),
    ("leibniz", 1646, 1716, "science"),
    ("vermeer", 1632, 1675, "art"),
    ("caravaggio", 1571, 1610, "art"),
    ("montaigne", 1533, 1592, "philosophy"),
    ("velazquez", 1599, 1660, "art"),
    ("vivaldi", 1678, 1741, "music"),
    ("handel", 1685, 1759, "music"),

    # Enlightenment / 18th century (1700–1800)
    ("voltaire", 1694, 1778, "philosophy"),
    ("rousseau", 1712, 1778, "philosophy"),
    ("kant", 1724, 1804, "philosophy"),
    ("hume", 1711, 1776, "philosophy"),
    ("mozart", 1756, 1791, "music"),
    ("beethoven", 1770, 1827, "music"),
    ("jefferson", 1743, 1826, "politics"),
    ("goethe", 1749, 1832, "literature"),
    ("haydn", 1732, 1809, "music"),
    ("diderot", 1713, 1784, "philosophy"),
    ("lavoisier", 1743, 1794, "science"),
    ("euler", 1707, 1783, "science"),
    ("gauss", 1777, 1855, "science"),
    ("schiller", 1759, 1805, "literature"),
    ("bolivar", 1783, 1830, "military"),
    ("nelson", 1758, 1805, "military"),

    # 19th century (1800–1900)
    ("darwin", 1809, 1882, "science"),
    ("marx", 1818, 1883, "philosophy"),
    ("lincoln", 1809, 1865, "politics"),
    ("dickens", 1812, 1870, "literature"),
    ("tolstoy", 1828, 1910, "literature"),
    ("dostoevsky", 1821, 1881, "literature"),
    ("nietzsche", 1844, 1900, "philosophy"),
    ("wagner", 1813, 1883, "music"),
    ("chopin", 1810, 1849, "music"),
    ("verdi", 1813, 1901, "music"),
    ("tchaikovsky", 1840, 1893, "music"),
    ("pasteur", 1822, 1895, "science"),
    ("maxwell", 1831, 1879, "science"),
    ("mendel", 1822, 1884, "science"),
    ("twain", 1835, 1910, "literature"),
    ("whitman", 1819, 1892, "literature"),
    ("bismarck", 1815, 1898, "politics"),
    ("garibaldi", 1807, 1882, "politics"),
    ("monet", 1840, 1926, "art"),
    ("renoir", 1841, 1919, "art"),
    ("rodin", 1840, 1917, "art"),
    ("cezanne", 1839, 1906, "art"),
    ("brahms", 1833, 1897, "music"),
    ("faraday", 1791, 1867, "science"),
    ("boltzmann", 1844, 1906, "science"),
    ("kierkegaard", 1813, 1855, "philosophy"),
    ("schopenhauer", 1788, 1860, "philosophy"),
    ("chekhov", 1860, 1904, "literature"),
    ("ibsen", 1828, 1906, "literature"),
    ("zola", 1840, 1902, "literature"),
    ("flaubert", 1821, 1880, "literature"),
    ("balzac", 1799, 1850, "literature"),
    ("puccini", 1858, 1924, "music"),
    ("dvorak", 1841, 1904, "music"),
    ("liszt", 1811, 1886, "music"),
    ("degas", 1834, 1917, "art"),
    ("gauguin", 1848, 1903, "art"),
    ("courbet", 1819, 1877, "art"),
    ("delacroix", 1798, 1863, "art"),

    # Early 20th century (1900–1950)
    ("einstein", 1879, 1955, "science"),
    ("freud", 1856, 1939, "science"),
    ("curie", 1867, 1934, "science"),
    ("bohr", 1885, 1962, "science"),
    ("planck", 1858, 1947, "science"),
    ("turing", 1912, 1954, "science"),
    ("churchill", 1874, 1965, "politics"),
    ("roosevelt", 1882, 1945, "politics"),
    ("gandhi", 1869, 1948, "politics"),
    ("lenin", 1870, 1924, "politics"),
    ("stalin", 1878, 1953, "politics"),
    ("mussolini", 1883, 1945, "politics"),
    ("kafka", 1883, 1924, "literature"),
    ("hemingway", 1899, 1961, "literature"),
    ("fitzgerald", 1896, 1940, "literature"),
    ("orwell", 1903, 1950, "literature"),
    ("picasso", 1881, 1973, "art"),
    ("matisse", 1869, 1954, "art"),
    ("stravinsky", 1882, 1971, "music"),
    ("debussy", 1862, 1918, "music"),
    ("heisenberg", 1901, 1976, "science"),
    ("dirac", 1902, 1984, "science"),
    ("schrodinger", 1887, 1961, "science"),
    ("rutherford", 1871, 1937, "science"),
    ("hubble", 1889, 1953, "science"),
    ("joyce", 1882, 1941, "literature"),
    ("woolf", 1882, 1941, "literature"),
    ("proust", 1871, 1922, "literature"),
    ("eliot", 1888, 1965, "literature"),
    ("faulkner", 1897, 1962, "literature"),
    ("kandinsky", 1866, 1944, "art"),
    ("mondrian", 1872, 1944, "art"),
    ("duchamp", 1887, 1968, "art"),
    ("prokofiev", 1891, 1953, "music"),
    ("shostakovich", 1906, 1975, "music"),
    ("bartok", 1881, 1945, "music"),
    ("ravel", 1875, 1937, "music"),
    ("heidegger", 1889, 1976, "philosophy"),
    ("russell", 1872, 1970, "philosophy"),
    ("dewey", 1859, 1952, "philosophy"),

    # Mid/Late 20th century (1950–2000)
    ("kennedy", 1917, 1963, "politics"),
    ("mandela", 1918, 2013, "politics"),
    ("thatcher", 1925, 2013, "politics"),
    ("reagan", 1911, 2004, "politics"),
    ("gorbachev", 1931, 2022, "politics"),
    ("feynman", 1918, 1988, "science"),
    ("hawking", 1942, 2018, "science"),
    ("crick", 1916, 2004, "science"),
    ("lennon", 1940, 1980, "music"),
    ("presley", 1935, 1977, "music"),
    ("hendrix", 1942, 1970, "music"),
    ("marley", 1945, 1981, "music"),
    ("warhol", 1928, 1987, "art"),
    ("hitchcock", 1899, 1980, "art"),
    ("kubrick", 1928, 1999, "art"),
    ("vonnegut", 1922, 2007, "literature"),
    ("tolkien", 1892, 1973, "literature"),
    ("borges", 1899, 1986, "literature"),
    ("nabokov", 1899, 1977, "literature"),
    ("foucault", 1926, 1984, "philosophy"),
    ("sartre", 1905, 1980, "philosophy"),
    ("wittgenstein", 1889, 1951, "philosophy"),
    ("chomsky", 1928, 2024, "science"),
    ("sagan", 1934, 1996, "science"),
    ("fellini", 1920, 1993, "art"),
    ("bergman", 1918, 2007, "art"),
    ("kurosawa", 1910, 1998, "art"),
    ("coltrane", 1926, 1967, "music"),
    ("morrison", 1931, 2019, "literature"),
    ("solzhenitsyn", 1918, 2008, "literature"),
    ("marquez", 1927, 2014, "literature"),
    ("beckett", 1906, 1989, "literature"),
    ("derrida", 1930, 2004, "philosophy"),
]

# Domain colors
domain_colors = {
    'philosophy': '#9B59B6',
    'military': '#E74C3C',
    'politics': '#3498DB',
    'science': '#27AE60',
    'literature': '#F39C12',
    'art': '#E67E22',
    'music': '#1ABC9C',
    'religion': '#95A5A6',
    'exploration': '#D35400',
}

# ============================================================
# LOAD EMBEDDINGS
# ============================================================
GLOVE_PATH = "glove.6B.300d.txt"
if not os.path.exists(GLOVE_PATH):
    print(f"ERROR: {GLOVE_PATH} not found"); sys.exit(1)

# Collect needed words
needed = set()
for name, *_ in figures:
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
    # Try capitalized forms
    words = name.split()
    vecs = []
    for w in words:
        cap = w.capitalize()
        if cap in w2v_model:
            vecs.append(w2v_model[cap])
        elif w in w2v_model:
            vecs.append(w2v_model[w])
    return np.mean(vecs, axis=0) if vecs else None

valid = []
for name, birth, death, domain in figures:
    gv = glove_vec(name)
    wv = w2v_vec(name)
    if gv is None and wv is None:
        continue
    valid.append(dict(name=name, birth=birth, death=death, domain=domain,
                      midlife=(birth + death) / 2,
                      glove=gv, w2v=wv,
                      color=domain_colors.get(domain, '#888')))

# Report what we have
print(f"\n  {len(valid)} figures with at least one embedding")
glove_valid = [v for v in valid if v['glove'] is not None]
w2v_valid = [v for v in valid if v['w2v'] is not None]
both_valid = [v for v in valid if v['glove'] is not None and v['w2v'] is not None]
print(f"  GloVe: {len(glove_valid)}, Word2Vec: {len(w2v_valid)}, Both: {len(both_valid)}")

# Check which names are missing
for name, birth, death, domain in figures:
    gv = glove_vec(name)
    wv = w2v_vec(name)
    if gv is None and wv is None:
        print(f"  MISSING: {name}")
    elif gv is None:
        print(f"  GloVe missing: {name}")
    elif wv is None:
        print(f"  W2V missing: {name}")

# Use the intersection for fair comparison
valid = both_valid
names = [v['name'] for v in valid]
births = np.array([v['birth'] for v in valid])
deaths = np.array([v['death'] for v in valid])
midlives = np.array([v['midlife'] for v in valid])
colors = [v['color'] for v in valid]
domains = [v['domain'] for v in valid]

X_g = np.array([v['glove'] for v in valid])
X_w = np.array([v['w2v'] for v in valid])
n = len(valid)
print(f"\n  Using {n} figures with both embeddings")
print(f"  Year range: {births.min()} to {births.max()}")

# ============================================================
# PROBE
# ============================================================
lambdas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

def run_probe(X, y, label=""):
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
                tr=tr, te=te, W=W)

targets = {
    'Birth Year': births,
    'Death Year': deaths,
    'Midlife Year': midlives,
}

print("\n" + "=" * 70)
print("PROBE RESULTS")
print("=" * 70)

results_g = {}
results_w = {}
for tname, y in targets.items():
    rg = run_probe(X_g, y, f"GloVe-{tname}")
    rw = run_probe(X_w, y, f"W2V-{tname}")
    results_g[tname] = rg
    results_w[tname] = rw
    print(f"  {tname:<16s}  GloVe: R²={rg['r2']:+.3f}, MAE={rg['mae']:.0f}yr (λ={rg['lam']})  "
          f"W2V: R²={rw['r2']:+.3f}, MAE={rw['mae']:.0f}yr (λ={rw['lam']})")

# Use birth year as primary for figures
primary = 'Birth Year'
tr = results_g[primary]['tr']
te = results_g[primary]['te']
te_set = set(te)


# ============================================================
# FIGURES
# ============================================================
print("\nGenerating figures...")

label_figs = ['aristotle', 'homer', 'caesar', 'shakespeare', 'newton',
              'confucius', 'shakespeare', 'newton', 'einstein']

def annotate_fig(ax, x, y, name, is_test, fontsize=9.5):
    dx, dy = 6, 6
    nudges = {
        'aristotle': (-50, -10), 'plato': (-35, -10),
        'caesar': (6, -10), 'einstein': (-50, 6),
        'darwin': (6, -10), 'mozart': (-40, -10),
        'homer': (6, -10), 'confucius': (-55, 6),
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


# --- Fig 1: Actual vs Predicted birth year (2 panels: GloVe, W2V) ---
fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(16, 7))

train_color = '#888888'
test_color = '#444444'

def plot_temporal_scatter(ax, res, title):
    y_act = res['y']
    y_pred = res['yp_all']
    lo = min(y_act.min(), y_pred.min()) - 50
    hi = max(y_act.max(), y_pred.max()) + 50
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1, zorder=1)

    # Actual positions: open circles on the diagonal
    for i in range(n):
        ax.scatter(y_act[i], y_act[i], facecolors='none',
                   edgecolors=train_color, s=50, alpha=0.35, linewidth=1.2, zorder=2)

    # Predicted positions: filled triangles
    for i in tr:
        ax.scatter(y_act[i], y_pred[i], c=train_color, s=45, alpha=0.35,
                   marker='^', edgecolors='white', linewidth=0.4, zorder=4)
    for i in te:
        ax.scatter(y_act[i], y_pred[i], c=test_color, s=80, alpha=0.8,
                   marker='^', edgecolors='white', linewidth=0.5, zorder=6)

    # Connecting lines + labels for labeled figures
    for i in range(n):
        if names[i] in label_figs:
            ax.plot([y_act[i], y_act[i]], [y_act[i], y_pred[i]],
                    color=test_color if i in te_set else train_color,
                    alpha=0.55, linewidth=1.3, zorder=3)
            annotate_fig(ax, y_act[i], y_pred[i], names[i], i in te_set)

    ax.set_xlabel('Actual Birth Year', fontsize=13)
    ax.set_ylabel('Predicted Birth Year', fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

plot_temporal_scatter(ax1a, results_g[primary],
                      f'A. GloVe 300d (R² = {results_g[primary]["r2"]:.3f})')
plot_temporal_scatter(ax1b, results_w[primary],
                      f'B. Word2Vec 300d (R² = {results_w[primary]["r2"]:.3f})')

# Legend — just actual vs predicted
legend_els = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#666', markersize=8, alpha=0.5,
           label='Actual birth year'),
    Line2D([0], [0], marker='^', color='w',
           markerfacecolor='#666', markersize=8, alpha=0.9,
           label='Predicted birth year'),
]

fig1.legend(handles=legend_els, loc='lower center', ncol=2,
            fontsize=11, frameon=True, fancybox=True, edgecolor='#cccccc',
            bbox_to_anchor=(0.5, -0.04))
fig1.tight_layout()
fig1.subplots_adjust(bottom=0.08)
fig1.savefig('temporal_birth_year_scatter.png', dpi=300, bbox_inches='tight')
print("  Saved temporal_birth_year_scatter.png")


# --- Fig 2: Bar chart R² for birth/death/midlife ---
fig2, ax2 = plt.subplots(figsize=(8, 5))

target_order = list(targets.keys())
g_r2 = [results_g[t]['r2'] for t in target_order]
w_r2 = [results_w[t]['r2'] for t in target_order]

x = np.arange(len(target_order))
bw = 0.35
bars_g = ax2.bar(x - bw/2, g_r2, bw, color='#4472C4', edgecolor='white',
                 linewidth=1.2, alpha=0.88, label='GloVe 300d')
bars_w = ax2.bar(x + bw/2, w_r2, bw, color='#ED7D31', edgecolor='white',
                 linewidth=1.2, alpha=0.88, label='Word2Vec 300d')

for bars, r2s in [(bars_g, g_r2), (bars_w, w_r2)]:
    for bar, r2 in zip(bars, r2s):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2,
                 max(h, 0) + 0.02,
                 f'{r2:.2f}', ha='center', va='bottom', fontsize=9,
                 fontweight='bold')

ax2.axhline(y=0, color='black', linewidth=0.8, linestyle='--', zorder=1)
ax2.set_xticks(x)
ax2.set_xticklabels(target_order, fontsize=11)
ax2.set_ylabel('Test R²', fontsize=12)
ax2.set_ylim(-0.5, 1.0)
ax2.grid(True, alpha=0.15, axis='y', linewidth=0.5)
ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)

fig2.tight_layout()
fig2.savefig('temporal_r2_barplot.png', dpi=300, bbox_inches='tight')
print("  Saved temporal_r2_barplot.png")


# --- Fig 3: Residual analysis — which eras are well/poorly predicted? ---
fig3, ax3 = plt.subplots(figsize=(12, 6))

res = results_g[primary]
residuals = res['yp_all'] - res['y']

for i in range(n):
    marker = 'D' if i in te_set else 'o'
    alpha = 0.9 if i in te_set else 0.4
    s = 60 if i in te_set else 30
    ax3.scatter(res['y'][i], residuals[i], c=colors[i], s=s, alpha=alpha,
                marker=marker, edgecolors='white', linewidth=0.5, zorder=5)

ax3.axhline(y=0, color='black', linewidth=0.8, linestyle='--', zorder=1)
ax3.set_xlabel('Actual Birth Year', fontsize=11)
ax3.set_ylabel('Prediction Error (predicted − actual, years)', fontsize=11)
ax3.set_title('GloVe: Prediction Residuals by Era', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.15, linewidth=0.5)

# Label outliers
for i in range(n):
    if abs(residuals[i]) > 400 or names[i] in label_figs:
        annotate_fig(ax3, res['y'][i], residuals[i], names[i], i in te_set)

fig3.tight_layout()
fig3.savefig('temporal_residuals.png', dpi=300, bbox_inches='tight')
print("  Saved temporal_residuals.png")


# ============================================================
# SEMANTIC SIMILARITY ANALYSIS (like spatial version)
# ============================================================
print("\n" + "=" * 70)
print("TEMPORAL SEMANTIC SIMILARITY ANALYSIS")
print("=" * 70)

temporal_words = {
    'ancient': ['ancient', 'classical', 'antiquity', 'greek', 'roman',
                'mythology', 'philosopher', 'toga', 'senate', 'empire'],
    'medieval': ['medieval', 'feudal', 'knight', 'crusade', 'castle',
                 'monastery', 'plague', 'king', 'pope', 'cathedral'],
    'renaissance': ['renaissance', 'reformation', 'printing', 'humanism',
                    'exploration', 'colonial', 'baroque', 'enlightenment'],
    'industrial': ['industrial', 'factory', 'steam', 'railroad', 'telegraph',
                   'revolution', 'coal', 'empire', 'victorian', 'colonial'],
    'modern': ['modern', 'nuclear', 'computer', 'digital', 'internet',
               'television', 'satellite', 'atomic', 'electronic', 'global'],
    'contemporary': ['contemporary', 'smartphone', 'social', 'streaming',
                     'startup', 'viral', 'blockchain', 'algorithm'],
}

# Load needed semantic word vectors
sem_needed = set()
for words in temporal_words.values():
    sem_needed.update(words)

print("Loading semantic word vectors...")
sem_glove = {}
with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if parts[0] in sem_needed:
            sem_glove[parts[0]] = np.array([float(x) for x in parts[1:]])

print(f"  Found {len(sem_glove)}/{len(sem_needed)} words")

# Normalize city embeddings
X_g_norms = np.linalg.norm(X_g, axis=1, keepdims=True)
X_g_normed = X_g / (X_g_norms + 1e-10)

print("\n  Individual word correlations with birth year:")
word_results = []
for cat, words in temporal_words.items():
    print(f"\n  {cat}:")
    for w in words:
        if w not in sem_glove: continue
        wn = sem_glove[w] / (np.linalg.norm(sem_glove[w]) + 1e-10)
        sims = X_g_normed @ wn
        r, p = pearsonr(sims, births)
        sig = '*' if p < 0.05 else ''
        print(f"    {w:<16s}  r(birth)={r:+.3f}{sig}")
        word_results.append(dict(word=w, category=cat, r=r, p=p, sims=sims))

# Composite: ancient vs modern
ancient_words = temporal_words['ancient']
modern_words = temporal_words['modern']

def composite(X_normed, pos_words, neg_words, emb_dict):
    scores = np.zeros(len(X_normed))
    n_pos, n_neg = 0, 0
    for w in pos_words:
        if w in emb_dict:
            wn = emb_dict[w] / (np.linalg.norm(emb_dict[w]) + 1e-10)
            scores += X_normed @ wn
            n_pos += 1
    for w in neg_words:
        if w in emb_dict:
            wn = emb_dict[w] / (np.linalg.norm(emb_dict[w]) + 1e-10)
            scores -= X_normed @ wn
            n_neg += 1
    if n_pos: scores /= n_pos
    # normalize neg contribution
    return scores

modern_minus_ancient = composite(X_g_normed, modern_words, ancient_words, sem_glove)
r_comp, p_comp = pearsonr(modern_minus_ancient, births)
print(f"\n  Composite (modern - ancient): r(birth) = {r_comp:.3f} (p={p_comp:.4f})")


# --- Fig 4: Top semantic word correlations ---
sorted_wr = sorted(word_results, key=lambda x: -abs(x['r']))

fig4, ax4 = plt.subplots(figsize=(10, 8))

cat_colors = {
    'ancient': '#8B4513', 'medieval': '#DAA520', 'renaissance': '#B8860B',
    'industrial': '#696969', 'modern': '#4472C4', 'contemporary': '#27AE60',
}

wlabels = [r['word'] for r in sorted_wr]
rvals = [r['r'] for r in sorted_wr]
bcolors = [cat_colors.get(r['category'], '#999') for r in sorted_wr]
sigs = ['*' if r['p'] < 0.05 else '' for r in sorted_wr]

bars = ax4.barh(range(len(wlabels)), rvals, color=bcolors,
                edgecolor='white', linewidth=0.8, alpha=0.85)

for i, (bar, sig) in enumerate(zip(bars, sigs)):
    if sig:
        w = bar.get_width()
        ax4.text(w + 0.01 * np.sign(w), i, '*', fontsize=14,
                 fontweight='bold', va='center')

ax4.set_yticks(range(len(wlabels)))
ax4.set_yticklabels(wlabels, fontsize=9)
ax4.set_xlabel('Pearson r with Birth Year', fontsize=12)
ax4.axvline(x=0, color='black', linewidth=0.5)
ax4.grid(True, alpha=0.15, axis='x', linewidth=0.5)
ax4.invert_yaxis()

legend_els = [Patch(facecolor=cat_colors[c], label=c.title())
              for c in cat_colors]
ax4.legend(handles=legend_els, fontsize=9, loc='lower right', framealpha=0.9)

fig4.tight_layout()
fig4.savefig('temporal_semantic_correlations.png', dpi=300, bbox_inches='tight')
print("  Saved temporal_semantic_correlations.png")


# --- Fig 5: Modern-ancient composite vs birth year ---
fig5, ax5 = plt.subplots(figsize=(10, 7))

for i in range(n):
    ax5.scatter(births[i], modern_minus_ancient[i], c=colors[i], s=55,
                alpha=0.85, edgecolors='white', linewidth=0.5, zorder=5)

m, b = np.polyfit(births, modern_minus_ancient, 1)
xs = np.linspace(births.min() - 50, births.max() + 50, 100)
ax5.plot(xs, m * xs + b, color='#333', alpha=0.4, linewidth=2,
         linestyle='--', zorder=2)

for i in range(n):
    if names[i] in label_figs:
        annotate_fig(ax5, births[i], modern_minus_ancient[i], names[i], True)

ax5.set_xlabel('Actual Birth Year', fontsize=12)
ax5.set_ylabel('Modern − Ancient Similarity Score', fontsize=11)
ax5.set_title(f'Temporal semantic gradient (r = {r_comp:.3f})',
              fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.15, linewidth=0.5)

legend_els = [Patch(facecolor=domain_colors[d], edgecolor='white', label=d.title())
              for d in sorted(set(domains))]
ax5.legend(handles=legend_els, loc='lower right', fontsize=9, framealpha=0.9)

fig5.tight_layout()
fig5.savefig('temporal_semantic_gradient.png', dpi=300, bbox_inches='tight')
print("  Saved temporal_semantic_gradient.png")


print("\nDone.")
