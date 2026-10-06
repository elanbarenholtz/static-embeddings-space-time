#!/bin/zsh
# City set: run from city_resource/. Requires the external data listed in README.md.
cd "$(dirname "$0")"
R=${SB_ROOT:-$(cd .. && pwd)}
P=${PYTHON:-python3}
export PYTHONPATH=$PWD/shim
# optional: rebuild the sourced table from the cached API responses and WorldClim
# $P source_cities.py && $P worldclim.py
$P build_w2v_cache.py
mkdir -p run_new out_new
# one-hot baselines, grouped holdouts, distance analysis and random-subspace ablation (original scripts)
cp $R/table1_check/{vecs272.txt,catvecs.txt,categories.json,holdout.py,ablation_272.py} run_new/
cp $R/mechanism/distance_vs_membership.py run_new/
(cd run_new && $P holdout.py && $P distance_vs_membership.py && $P ablation_272.py)
$P out_new/cities_rerun_new.py           # dimensionality and random-embedding control
$P static_cities.py run_new/cities272_country.json new
$P pythia_cities.py run_new/cities272_country.json new
$P nested_cities.py                      # training-only layer selection
$P per_split_aux.py                     # single-layer analyses, each split at its own selected layer
$P neighbors_citylevel.py
$P ablation_matched.py                   # ablation against matched random word sets
$P figs_cities.py run_new/cities272_country.json new
