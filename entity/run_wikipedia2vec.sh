#!/bin/zsh
# Entity-level baseline: run from entity/ after gt_scale/run_entities.sh (it reads the static vectors, activations and layer choices there).
# Requires entity/enwiki_300d.txt.bz2 (see README.md).
cd "$(dirname "$0")"
P=${PYTHON:-python3}
$P fetch_titles.py                  # Wikidata ID -> enwiki title for the historical figures
$P fetch_titles_sparql.py           # SPARQL fill for titles the first pass missed
bzcat enwiki_300d.txt.bz2 | $P filter_w2v.py    # -> w2v_entities.npz, w2v_words.npz (items used only)
$P entity_probe.py                  # -> entity_probe.json (same items, same split, all representations)
$P entity_robust.py                 # -> entity_robust.json (leave-one-continent-out, death-year blocks, matched dimensionality)
$P map_csv.py                       # -> entity_map_preds.csv
$P make_entity_map_figure.py        # -> fig_entity_map_row.png
# Coverage check (items with and without an entity vector); append rows to coverage_check.jsonl, one call per piece
for DS in historical_figure world_place; do
  $P coverage_check.py $DS GloVe fastText
  $P coverage_check.py --lm $DS pythia llama
done
# Which words carry the signal (released places, GloVe): -> word_coords_released.json, fig_word_coords.png
# needs gt_scale/data/glove.6B.300d.txt and gt_scale/data/world_place.csv; exclusion list in exclusion_words.json
$P word_coords.py
$P make_word_figure.py
# Semantic ablation on the released places (word categories vs matched random word sets): -> semantic_ablation_released.json
$P semantic_ablation_released.py
