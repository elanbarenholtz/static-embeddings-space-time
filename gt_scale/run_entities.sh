#!/bin/zsh
# Released entities: run from gt_scale/. Requires the external data listed in README.md.
cd "$(dirname "$0")"
P=${PYTHON:-python3}
for DS in world_place historical_figure; do
  $P static_vecs.py $DS
  for M in pythia-2.8b llama-2-7b; do $P extract_safe.py $DS ./$M; done
done
$P static_table.py                       # static baseline table (GloVe, Word2Vec, fastText)
$P nested_layer.py                       # training-only layer selection -> nested_layer.json
# selected layers: places 24 (both models); figures 28 (Pythia-2.8B), 24 (Llama-2-7B)
$P decompose.py world_place 24 pythia-2.8b;       $P decompose.py historical_figure 28 pythia-2.8b
$P decompose.py world_place 24 llama-2-7b;        $P decompose.py historical_figure 24 llama-2-7b
for DS in world_place historical_figure; do $P bootstrap.py $DS pythia; $P bootstrap.py $DS llama; done
$P r2_boot.py; $P varshare.py; $P country_ceiling.py; $P pooled_inputs.py; $P phrase_check.py
for S in GloVe Word2Vec; do
  $P resid.py world_place $S 24 pythia-2.8b;  $P resid.py historical_figure $S 28 pythia-2.8b
  $P resid.py world_place $S 24 llama-2-7b;   $P resid.py historical_figure $S 24 llama-2-7b
done
$P resid_groups2.py world_place 24 pythia-2.8b;  $P resid_groups2.py historical_figure 28 pythia-2.8b
$P resid_groups2.py world_place 24 llama-2-7b;   $P resid_groups2.py historical_figure 24 llama-2-7b
$P holdout2.py '{"pythia-2.8b": {"world_place": 24, "historical_figure": 28}, "llama-2-7b": {"world_place": 24, "historical_figure": 24}}'
$P holdout2_L19.py '{"pythia-2.8b": {"world_place": 19, "historical_figure": 19}, "llama-2-7b": {"world_place": 19, "historical_figure": 19}}'   # transformers at the fixed 60%-depth layer
$P fig_appendix_full.py
