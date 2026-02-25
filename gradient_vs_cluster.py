"""
Test 1: Region-centroid baselines (continent-centroid and k-means-centroid)
Test 2: Within-region R² for latitude and longitude
"""

import numpy as np
import os, sys

# ============================================================
# DATA (same cities/regions as other scripts)
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

regions = {
    'North America': ['anchorage', 'seattle', 'portland', 'san francisco', 'los angeles',
                      'san diego', 'phoenix', 'denver', 'dallas', 'houston', 'austin',
                      'chicago', 'detroit', 'minneapolis', 'miami', 'atlanta', 'boston',
                      'new york', 'philadelphia', 'washington', 'nashville', 'memphis'],
    'Europe': ['london', 'paris', 'berlin', 'madrid', 'rome', 'lisbon', 'amsterdam',
               'brussels', 'vienna', 'prague', 'warsaw', 'budapest',
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

def get_region(name):
    for r, cl in regions.items():
        if name in cl:
            return r
    return 'Other'

# ============================================================
# LOAD EMBEDDINGS
# ============================================================
GLOVE_PATH = "glove.6B.300d.txt"
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
for name, lat, lon, temp in cities:
    ge, we = glove_vec(name), w2v_vec(name)
    if ge is None or we is None: continue
    valid.append(dict(name=name, lat=lat, lon=lon, temp=temp,
                      glove=ge, w2v=we, region=get_region(name)))

names = [v['name'] for v in valid]
X_g = np.array([v['glove'] for v in valid])
X_w = np.array([v['w2v'] for v in valid])
region_labels = [v['region'] for v in valid]
n = len(valid)
print(f"  {n} cities\n")

lat = np.array([v['lat'] for v in valid])
lon = np.array([v['lon'] for v in valid])
temp = np.array([v['temp'] for v in valid])

# Same train/test split as other scripts
rng = np.random.RandomState(42)
idx = rng.permutation(n)
n_train = int(0.8 * n)
tr, te = idx[:n_train], idx[n_train:]

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

# ============================================================
# Ridge regression probe (same as main analysis)
# ============================================================
lambdas = [0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

def ridge_probe(X, y, tr, te):
    d = X.shape[1]
    n_folds, fold_size = 5, len(tr) // 5
    best_lam, best_cv = None, -np.inf
    for lam in lambdas:
        fr2 = []
        for fold in range(n_folds):
            mask = np.zeros(len(tr), dtype=bool)
            mask[fold * fold_size:(fold + 1) * fold_size] = True
            Xtr, Xte_ = X[tr[~mask]], X[tr[mask]]
            ytr, yte_ = y[tr[~mask]], y[tr[mask]]
            W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ ytr)
            yp = Xte_ @ W
            ss_r = np.sum((yte_ - yp)**2)
            ss_t = np.sum((yte_ - yte_.mean())**2)
            fr2.append(1 - ss_r / ss_t if ss_t > 0 else 0)
        cv = np.mean(fr2)
        if cv > best_cv: best_cv, best_lam = cv, lam
    W = np.linalg.solve(X[tr].T @ X[tr] + best_lam * np.eye(d), X[tr].T @ y[tr])
    yp = X[te] @ W
    return r2(y[te], yp)

# Full probe R² for reference
print("=" * 60)
print("FULL PROBE R² (reference)")
print("=" * 60)
for target_name, y in [('Latitude', lat), ('Longitude', lon), ('Temperature', temp)]:
    g_r2 = ridge_probe(X_g, y, tr, te)
    w_r2 = ridge_probe(X_w, y, tr, te)
    print(f"  {target_name:<15s}  GloVe={g_r2:.3f}  W2V={w_r2:.3f}")

# ============================================================
# TEST 1: Region-centroid baselines
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: REGION-CENTROID BASELINES")
print("=" * 60)

# 1a. Continent-centroid: predict test city's lat/lon as mean of its region's training cities
print("\n1a. Continent-centroid (region membership → region mean of training cities)")
for target_name, y in [('Latitude', lat), ('Longitude', lon), ('Temperature', temp)]:
    # Compute region centroids from training data only
    region_centroids = {}
    for r in regions:
        r_train = [i for i in tr if region_labels[i] == r]
        if r_train:
            region_centroids[r] = np.mean(y[r_train])

    # Predict test cities
    y_pred = np.array([region_centroids.get(region_labels[i], np.mean(y[tr])) for i in te])
    score = r2(y[te], y_pred)
    print(f"  {target_name:<15s}  R²={score:.3f}")

# 1b. K-means centroid in embedding space
from sklearn.cluster import KMeans

print("\n1b. K-means centroid in embedding space (k=6, matching number of continents)")
for emb_name, X in [('GloVe', X_g), ('Word2Vec', X_w)]:
    print(f"\n  {emb_name}:")
    km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X[tr])
    train_clusters = km.labels_
    test_clusters = km.predict(X[te])

    for target_name, y in [('Latitude', lat), ('Longitude', lon), ('Temperature', temp)]:
        # Compute cluster centroids for target from training data
        cluster_centroids = {}
        for c in range(6):
            c_train = [j for j, ci in enumerate(train_clusters) if ci == c]
            if c_train:
                cluster_centroids[c] = np.mean(y[tr[c_train]])

        y_pred = np.array([cluster_centroids.get(c, np.mean(y[tr])) for c in test_clusters])
        score = r2(y[te], y_pred)
        print(f"    {target_name:<15s}  R²={score:.3f}")

# ============================================================
# TEST 2: Within-region R²
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: WITHIN-REGION R² (probe trained and tested within each region)")
print("=" * 60)

for emb_name, X in [('GloVe', X_g), ('Word2Vec', X_w)]:
    print(f"\n{emb_name}:")
    for r_name in ['North America', 'Europe', 'Asia']:
        r_idx = np.array([i for i in range(n) if region_labels[i] == r_name])
        nr = len(r_idx)
        if nr < 10:
            print(f"  {r_name}: too few cities ({nr})")
            continue

        # Within-region train/test split
        rng2 = np.random.RandomState(42)
        perm = rng2.permutation(nr)
        n_tr = int(0.8 * nr)
        r_tr, r_te = r_idx[perm[:n_tr]], r_idx[perm[n_tr:]]

        if len(r_te) < 3:
            print(f"  {r_name}: too few test cities ({len(r_te)})")
            continue

        for target_name, y in [('Latitude', lat), ('Longitude', lon)]:
            score = ridge_probe(X, y, r_tr, r_te)
            print(f"  {r_name:<20s} {target_name:<12s} R²={score:.3f}  (n_train={len(r_tr)}, n_test={len(r_te)})")

print("\nDone.")
