# Archive: the 272-city analysis

Code, sourced data and results from an earlier revision of the paper, which also analyzed a set of 272 cities (coordinates, temperature, elevation, GDP per capita), with Pythia-1.4B and Pythia-2.8B probes, a pairwise-distance analysis and a semantic ablation on that set. The current version of the paper does not use it: all spatial and temporal results there are on the entities released by Gurnee & Tegmark. The material is kept unchanged for the record.

| Folder | Contents |
|---|---|
| `city_resource/` | Sourcing (DBpedia, Wikidata, World Bank, WorldClim), city analyses, Pythia analyses on saved activations, matched-word-set ablation, figures. |
| `table1_check/`, `mechanism/` | One-hot baselines and grouped holdouts, semantic ablation, pairwise distance analysis, activation extraction. |
| `results_city_resource/` | JSON outputs for the city analyses. |
| `data_driven_semantic.py` | Word-level analysis on the 272 cities. |
| `paper_previous_version/` | LaTeX source and PDF of the revision that used them. |

These scripts expect the original layout (they were written to run from the repository root with the folders `city_resource/`, `table1_check/`, `mechanism/` next to `gt_scale/`). To rerun them, copy those folders back to the top level.
