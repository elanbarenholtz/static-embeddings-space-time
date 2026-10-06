# World Properties without World Models

Code, derived data and result files for *World Properties without World Models: Distributional Associations and the Interpretation of Decoding Results from Language Models*.
`paper/` holds the LaTeX source and PDF of the current version. The code for the first preprint version is kept, unchanged, in `v1_preprint/`. An earlier revision also analyzed a set of 272 cities; that analysis is not part of the current version and is kept for the record in `archive_272_cities/` (see the note there).

Every analysis in the current version is reproduced by the code below. Directory names follow the analysis layout used during the work; run each script from its own directory. Set `SB_ROOT` to the root of this package if you run scripts from elsewhere.

| Directory | Contents |
|---|---|
| `static_baseline/` | `static_baseline.py`: runs the baseline on any labeled stimulus set (example stimulus files included). |
| `gt_scale/` | Released entities of Gurnee & Tegmark (places, historical figures): static vectors, activation extraction, probes, training-only layer selection, decompositions, bootstrap intervals, static and group removals, input-embedding averages, grouped holdouts. |
| `entity/` | Entity-level static baseline (Wikipedia2Vec) on the released places and figures: title matching, vector filtering, probes, robustness checks, the coverage check, and the map figure; and the word-level latitude/longitude analysis (`word_coords.py`, `make_word_figure.py`, `exclusion_words.json`) and the semantic ablation (`semantic_ablation_released.py`, `semantic_categories.json`). Result files are included. |
| `review_v2/` | Pain axis (v2 stimuli and controls) and emotion analyses. |
| `results/` | The JSON outputs from which the reported numbers are taken. |

## External data (not redistributed)

* GloVe 6B 300d (`glove.6B.300d.txt`) and GoogleNews Word2Vec (`GoogleNews-vectors-negative300.bin.gz`, saved as `w2v.gz`) in `gt_scale/data/`; GloVe 50/100/200d in the package root for the dimensionality analysis.
* fastText `wiki-news-300d-1M.vec` in `review_v2/data/`.
* Gurnee & Tegmark entity datasets (`world_place.csv`, `historical_figure.csv`, with their `is_test` split) in `gt_scale/data/`.
* Pain Axis repository (v2) cloned to `review_v2/painrepo/`.
* Emotion replication corpus (Qwen emotion stories; Hugging Face `foogunlana/qwen-emotion-stories`, revision `be8f6def898bdd53771ebc0fed1ab5c88c5c0fa7`) in `review_v2/data/qwen-emotion-stories/`; GoEmotions (`train.tsv`, `test.tsv`, `dev.tsv`, `emotions.txt`) in `review_v2/data/`.
* Wikipedia2Vec 300d English vectors, enwiki 2018-04-20 (`enwiki_20180420_300d.txt.bz2`, from https://wikipedia2vec.github.io/wikipedia2vec/pretrained/), saved as `entity/enwiki_300d.txt.bz2`. `entity/filter_w2v.py` reads it from standard input and writes `w2v_entities.npz` and `w2v_words.npz`, restricted to the items used (the full vector file is not redistributed).
* Natural Earth 110m coastline (`ne_110m_coastline.zip`, public domain) is included in `entity/` for the map figure.
* Model weights for Pythia-2.8B and Llama-2-7B from Hugging Face (only needed to re-extract activations).

## Order of runs

1. **Released entities** (`gt_scale/run_entities.sh`): static vectors, activations, the static table, layer selection, then every analysis at the selected layers.
2. **Entity-level baseline and word-level analysis** (`entity/run_wikipedia2vec.sh`): run after step 1, since it reads the static vectors, activations and layer choices in `gt_scale/`. The word-level commands (`word_coords.py`, `make_word_figure.py`, `semantic_ablation_released.py`) need only GloVe and the released place table.
3. **Pain and emotion** (`review_v2/`): `make_cache.py`, then `pain_v2.py`, `pain_extra.py`, `emotion_v2b.py`.
