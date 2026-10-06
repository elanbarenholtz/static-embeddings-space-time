"""
Data-driven semantic analysis: compute correlation between every GloVe word
and latitude/temperature, then show the top/bottom correlated words.
No hand-picking — let the data speak.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
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
    ("london", 51.5, -0.1, 11), ("paris", 48.9, 2.3, 12),
    ("berlin", 52.5, 13.4, 10), ("madrid", 40.4, -3.7, 15),
    ("rome", 41.9, 12.5, 16), ("lisbon", 38.7, -9.1, 17),
    ("amsterdam", 52.4, 4.9, 10), ("brussels", 50.8, 4.4, 10),
    ("vienna", 48.2, 16.4, 10), ("prague", 50.1, 14.4, 9),
    ("warsaw", 52.2, 21.0, 8), ("budapest", 47.5, 19.0, 11),
    ("stockholm", 59.3, 18.1, 7), ("oslo", 59.9, 10.7, 5),
    ("helsinki", 60.2, 24.9, 5), ("copenhagen", 55.7, 12.6, 9),
    ("moscow", 55.8, 37.6, 5), ("athens", 38.0, 23.7, 18),
    ("istanbul", 41.0, 29.0, 14), ("barcelona", 41.4, 2.2, 16),
    ("munich", 48.1, 11.6, 8), ("zurich", 47.4, 8.5, 9),
    ("milan", 45.5, 9.2, 13), ("marseille", 43.3, 5.4, 15),
    ("tokyo", 35.7, 139.7, 16), ("beijing", 39.9, 116.4, 12),
    ("shanghai", 31.2, 121.5, 16), ("mumbai", 19.1, 72.9, 27),
    ("delhi", 28.6, 77.2, 25), ("bangkok", 13.8, 100.5, 28),
    ("singapore", 1.3, 103.8, 27), ("hong kong", 22.3, 114.2, 23),
    ("seoul", 37.6, 127.0, 12), ("taipei", 25.0, 121.5, 23),
    ("osaka", 34.7, 135.5, 16), ("jakarta", -6.2, 106.8, 27),
    ("manila", 14.6, 121.0, 27), ("hanoi", 21.0, 105.9, 24),
    ("dubai", 25.3, 55.3, 27), ("tehran", 35.7, 51.4, 17),
    ("baghdad", 33.3, 44.4, 23), ("kabul", 34.5, 69.2, 12),
    ("karachi", 24.9, 67.0, 26),
    ("cairo", 30.0, 31.2, 22), ("lagos", 6.5, 3.4, 27),
    ("nairobi", -1.3, 36.8, 18), ("johannesburg", -26.2, 28.0, 16),
    ("cape town", -33.9, 18.4, 17), ("casablanca", 33.6, -7.6, 18),
    ("addis ababa", 9.0, 38.7, 16), ("accra", 5.6, -0.2, 27),
    ("algiers", 36.8, 3.1, 18),
    ("buenos aires", -34.6, -58.4, 17), ("santiago", -33.4, -70.7, 14),
    ("lima", -12.0, -77.0, 19), ("bogota", 4.7, -74.1, 14),
    ("rio de janeiro", -22.9, -43.2, 24), ("sao paulo", -23.6, -46.6, 19),
    ("caracas", 10.5, -66.9, 22), ("quito", -0.2, -78.5, 14),
    ("sydney", -33.9, 151.2, 18), ("melbourne", -37.8, 145.0, 15),
    ("auckland", -36.9, 174.8, 15), ("perth", -31.9, 115.9, 19),
]

# ============================================================
# LOAD ALL OF GLOVE
# ============================================================
GLOVE_PATH = "glove.6B.300d.txt"
print("Loading full GloVe vocabulary...")

# First pass: get city embeddings
city_words = set()
for name, *_ in cities:
    for w in name.split():
        city_words.add(w)

all_words = []
all_vecs = []
city_embs = {}

with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec = np.array([float(x) for x in parts[1:]])
        all_words.append(word)
        all_vecs.append(vec)
        if word in city_words:
            city_embs[word] = vec

all_vecs = np.array(all_vecs)
print(f"  Loaded {len(all_words)} words")

# Build city vectors
def city_vec(name):
    vecs = [city_embs[w] for w in name.split() if w in city_embs]
    return np.mean(vecs, axis=0) if vecs else None

valid = []
for name, lat, lon, temp in cities:
    v = city_vec(name)
    if v is not None:
        valid.append(dict(name=name, lat=lat, lon=lon, temp=temp, vec=v))

n = len(valid)
print(f"  {n} cities with embeddings")

city_vecs = np.array([v['vec'] for v in valid])
lats = np.array([v['lat'] for v in valid])
temps = np.array([v['temp'] for v in valid])

# Normalize all vectors for cosine similarity
city_norms = city_vecs / np.linalg.norm(city_vecs, axis=1, keepdims=True)
all_norms = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)

# ============================================================
# COMPUTE CORRELATION OF EVERY WORD WITH LATITUDE
# ============================================================
print("\nComputing correlations for all words...")

# For each word, compute cosine similarity to all cities, then correlate with latitude
# Do this in batches to avoid memory issues
batch_size = 10000
n_words = len(all_words)
r_lat = np.zeros(n_words)
p_lat = np.zeros(n_words)
r_temp = np.zeros(n_words)
p_temp = np.zeros(n_words)

for start in range(0, n_words, batch_size):
    end = min(start + batch_size, n_words)
    # (n_cities, 300) @ (300, batch) -> (n_cities, batch)
    sims = city_norms @ all_norms[start:end].T
    for j in range(end - start):
        r, p = stats.pearsonr(sims[:, j], lats)
        r_lat[start + j] = r
        p_lat[start + j] = p
        r, p = stats.pearsonr(sims[:, j], temps)
        r_temp[start + j] = r
        p_temp[start + j] = p
    if (end // batch_size) % 10 == 0:
        print(f"  {end}/{n_words}...")

print(f"  Done. Computed {n_words} correlations.")

# ============================================================
# FILTER: exclude city names, country names, very short words, numbers
# ============================================================
city_name_words = set()
for name, *_ in cities:
    for w in name.split():
        city_name_words.add(w.lower())

# Common country/demonym words to exclude (they'd dominate and aren't interesting)
country_words = set([
    'china', 'chinese', 'japan', 'japanese', 'india', 'indian', 'brazil', 'brazilian',
    'russia', 'russian', 'france', 'french', 'germany', 'german', 'italy', 'italian',
    'spain', 'spanish', 'portugal', 'portuguese', 'mexico', 'mexican', 'canada', 'canadian',
    'australia', 'australian', 'england', 'english', 'british', 'britain', 'uk',
    'korea', 'korean', 'thailand', 'thai', 'vietnam', 'vietnamese', 'indonesia', 'indonesian',
    'pakistan', 'pakistani', 'iran', 'iranian', 'iraq', 'iraqi', 'turkey', 'turkish',
    'egypt', 'egyptian', 'nigeria', 'nigerian', 'kenya', 'kenyan', 'ethiopia', 'ethiopian',
    'colombia', 'colombian', 'argentina', 'argentine', 'peru', 'peruvian', 'chile', 'chilean',
    'sweden', 'swedish', 'norway', 'norwegian', 'finland', 'finnish', 'denmark', 'danish',
    'netherlands', 'dutch', 'belgium', 'belgian', 'austria', 'austrian', 'switzerland', 'swiss',
    'poland', 'polish', 'czech', 'hungary', 'hungarian', 'greece', 'greek',
    'africa', 'african', 'asia', 'asian', 'europe', 'americas', 'oceania',
    'saudi', 'arabian', 'arab', 'emirates', 'qatar', 'kuwait',
    'malaysia', 'malaysian', 'philippines', 'filipino', 'singapore', 'taiwanese',
    'zealand', 'zealand',
])

MAX_RANK = 20000  # Only consider top 20K most frequent words (common vocabulary)

# Additional geographic / proper-noun words to exclude
geo_extra = set([
    'wellington', 'bathurst', 'darwin', 'bergen', 'petersburg', 'natal',
    'townsville', 'kimberley', 'barbados', 'samoa', 'guyana', 'trinidad',
    'fiji', 'tonga', 'suriname', 'belize', 'bermuda', 'bahamas', 'jamaica',
    'haiti', 'cuba', 'panama', 'costa', 'rica', 'honduras', 'guatemala',
    'nicaragua', 'salvador', 'dominican', 'puerto', 'rico',
    'hawaii', 'alaska', 'siberia', 'siberian', 'arctic', 'antarctic',
    'sahara', 'saharan', 'mediterranean', 'caribbean', 'pacific', 'atlantic',
    'scandinavian', 'nordic', 'baltic', 'balkan', 'caucasus',
    'moreton', 'hawker', 'centurion', 'shire',
    'cairns', 'antigua', 'mackay', 'queensland', 'papua', 'guinea',
    'indies', 'viscount', 'vale', 'bowler',  # place/cricket-adjacent
    'mindanao', 'gujarat', 'maldives', 'philippine', 'malay',
    'leningrad', 'carnegie', 'alpine',
    # Common surnames/names that appear lowercase in GloVe
    'sergei', 'eisner', 'schaeuble', 'sikorski', 'heinz', 'schroeder',
    'fritz', 'wilhelm', 'ludwig', 'hans', 'karl', 'otto',
    'polanski', 'nokia', 'muralitharan', 'thais', 'filipinos',
    'markus', 'alexei', 'elisabeth', 'adler', 'friedrich', 'carl',
    'macapagal', 'mehd', 'xfdws',
    'bangladesh', 'bangladeshis',
    # More place names / demonyms
    'lucia', 'zimbabwe', 'laguna', 'verde', 'mara', 'andes',
    'myanmar', 'bavarian', 'arroyo', 'ringgit', 'rahman',
    'sergey', 'rudolf', 'emil', 'kirsten', 'sked',
    'botswana', 'maharashtra', 'barker', 'willy', 'berg',
    'asians', 'para',
])

def is_valid_word(word, idx):
    """Filter out words that would trivially dominate."""
    if idx >= MAX_RANK:  # Frequency filter: GloVe is sorted by frequency
        return False
    if len(word) < 4:
        return False
    if word.lower() in city_name_words:
        return False
    if word.lower() in country_words:
        return False
    if word.lower() in geo_extra:
        return False
    if word[0].isupper():  # Proper nouns
        return False
    if any(c.isdigit() for c in word):
        return False
    if not word.isalpha():
        return False
    return True

valid_mask = np.array([is_valid_word(w, i) for i, w in enumerate(all_words)])
print(f"\n  {valid_mask.sum()} words after filtering (from {n_words})")

# ============================================================
# TOP/BOTTOM CORRELATED WORDS WITH LATITUDE
# ============================================================
N_SHOW = 15

valid_indices = np.where(valid_mask)[0]
valid_r_lat = r_lat[valid_indices]

# Top positive (north-associated)
top_pos_idx = valid_indices[np.argsort(valid_r_lat)[-N_SHOW:][::-1]]
# Top negative (south/equatorial-associated)
top_neg_idx = valid_indices[np.argsort(valid_r_lat)[:N_SHOW]]

print(f"\nTop {N_SHOW} words most positively correlated with latitude (north-associated):")
for i in top_pos_idx:
    print(f"  {all_words[i]:<20s}  r={r_lat[i]:+.3f}  p={p_lat[i]:.1e}")

print(f"\nTop {N_SHOW} words most negatively correlated with latitude (south/equatorial):")
for i in top_neg_idx:
    print(f"  {all_words[i]:<20s}  r={r_lat[i]:+.3f}  p={p_lat[i]:.1e}")

# ============================================================
# FIGURE: Combined top positive + negative
# ============================================================
show_idx = np.concatenate([top_neg_idx[::-1], top_pos_idx[::-1]])
show_words = [all_words[i] for i in show_idx]
show_r = [r_lat[i] for i in show_idx]
show_p = [p_lat[i] for i in show_idx]
show_colors = ['#3474B4' if r >= 0 else '#D94040' for r in show_r]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(range(len(show_words)), show_r, color=show_colors,
               edgecolor='white', linewidth=0.8, alpha=0.85)

for i, (bar, p) in enumerate(zip(bars, show_p)):
    if p < 0.05:
        w = bar.get_width()
        ax.text(w + 0.01 * np.sign(w), i, '*', fontsize=14,
                fontweight='bold', va='center', color='black')

ax.set_yticks(range(len(show_words)))
ax.set_yticklabels(show_words, fontsize=12)
ax.set_xlabel('Pearson r (city–word cosine similarity vs. actual latitude)', fontsize=13)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.15, axis='x', linewidth=0.5)
ax.invert_yaxis()
ax.tick_params(axis='x', labelsize=11)

fig.tight_layout()
fig.savefig('semantic_data_driven_lat.png', dpi=300, bbox_inches='tight')
fig.savefig('paper/figures/semantic_data_driven_lat.png', dpi=300, bbox_inches='tight')
print("\n  Saved semantic_data_driven_lat.png")

# ============================================================
# Also print top words for TEMPERATURE
# ============================================================
valid_r_temp = r_temp[valid_indices]

top_pos_temp = valid_indices[np.argsort(valid_r_temp)[-N_SHOW:][::-1]]
top_neg_temp = valid_indices[np.argsort(valid_r_temp)[:N_SHOW]]

print(f"\nTop {N_SHOW} words most positively correlated with temperature (warm):")
for i in top_pos_temp:
    print(f"  {all_words[i]:<20s}  r={r_temp[i]:+.3f}  p={p_temp[i]:.1e}")

print(f"\nTop {N_SHOW} words most negatively correlated with temperature (cold):")
for i in top_neg_temp:
    print(f"  {all_words[i]:<20s}  r={r_temp[i]:+.3f}  p={p_temp[i]:.1e}")

# ============================================================
# FIGURE: Temperature top positive + negative
# ============================================================
# Sort all 30 words by absolute r, descending
show_idx_t = np.concatenate([top_pos_temp, top_neg_temp])
abs_r = np.array([abs(r_temp[i]) for i in show_idx_t])
show_idx_t = show_idx_t[np.argsort(abs_r)[::-1]]
show_words_t = [all_words[i] for i in show_idx_t]
show_r_t = [r_temp[i] for i in show_idx_t]
show_p_t = [p_temp[i] for i in show_idx_t]
show_colors_t = ['#D94040' if r >= 0 else '#3474B4' for r in show_r_t]  # red=warm, blue=cold

fig2, ax2 = plt.subplots(figsize=(10, 8))
bars2 = ax2.barh(range(len(show_words_t)), show_r_t, color=show_colors_t,
                 edgecolor='white', linewidth=0.8, alpha=0.85)

for i, (bar, p) in enumerate(zip(bars2, show_p_t)):
    if p < 0.05:
        w = bar.get_width()
        ax2.text(w + 0.01 * np.sign(w), i, '*', fontsize=14,
                 fontweight='bold', va='center', color='black')

ax2.set_yticks(range(len(show_words_t)))
ax2.set_yticklabels(show_words_t, fontsize=14)
ax2.set_xlabel('Pearson r (city–word cosine similarity vs. mean annual temperature)', fontsize=14)
ax2.axvline(x=0, color='black', linewidth=0.5)
ax2.grid(True, alpha=0.15, axis='x', linewidth=0.5)
ax2.invert_yaxis()
ax2.tick_params(axis='x', labelsize=12)

fig2.tight_layout()
fig2.savefig('semantic_data_driven_temp.png', dpi=300, bbox_inches='tight')
fig2.savefig('paper/figures/semantic_data_driven_temp.png', dpi=300, bbox_inches='tight')
print("\n  Saved semantic_data_driven_temp.png")

print("\nDone.")
