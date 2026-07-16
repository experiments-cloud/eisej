# Operational Strain Signatures — Data & Code

Complete pipeline for data extraction, cleaning, unsupervised ground truth
generation, temporal stability analysis, and predictive modeling used in the
manuscript *"Detecting Operational Strain Signatures in Software
Development: A Behavioral Telemetry Approach with Trait-State Validation"*,
submitted to e-Informatica Software Engineering Journal.

## Structure

```
eisej/
├── scripts/
│   ├── 01_extract_5repos.py                    # Sec. 3.1 — react, tensorflow, vscode, linux, bitcoin
│   ├── 01b_extract_2repos_community.py         # Sec. 3.1 — neovim, svelte
│   ├── 02_clean_and_feature_engineer.py        # Sec. 3.2, 3.3
│   ├── 03_groundtruth_windows.py               # Sec. 3.4 (windowing + aggregation)
│   ├── 03b_groundtruth_compare_methods.py      # Sec. 3.4 (percentile / Isolation Forest / GMM comparison)
│   ├── 04_trait_state_stability.py             # Sec. 3.6, 4.2
│   ├── 05_oracle_modeling_cv.py                # Sec. 3.7, 4.3
│   └── exploratory_superseded/                 # Discarded methodological attempts,
│                                               #   documented in the paper (Sec. 3.4, 3.7)
│                                               #   as part of the correction process.
│                                               #   NOT part of the final pipeline.
├── data/
│   ├── raw/                                    # Output of 01 + 01b
│   └── processed/                              # Output of 02, 03, 03b, 04, 05
└── requirements.txt
```

## Execution order (reproduction from scratch)

```bash
pip install -r requirements.txt

export GITHUB_TOKEN=your_token_here   # scope: none (classic) or "Public Repositories read-only" (fine-grained)

python scripts/01_extract_5repos.py
python scripts/01b_extract_2repos_community.py
# manually consolidate the two output parquet files into data/raw/github_raw_final.parquet
# (concat + drop_duplicates on 'sha'; see Section 3.1 of the paper)

python scripts/02_clean_and_feature_engineer.py
python scripts/03_groundtruth_windows.py
python scripts/03b_groundtruth_compare_methods.py
python scripts/04_trait_state_stability.py
python scripts/05_oracle_modeling_cv.py

python scripts/figures/generar_figura1.py   # ... through generar_figura6.py
```

**Note on exact reproducibility:** `05_oracle_modeling_cv.py` fixes the
random seed in NumPy, scikit-learn, and PyTorch (`torch.manual_seed(42)`).
Without this, LSTM training is not deterministic (see the limitation
documented in Section 4.3 of the paper). The published results correspond to
the seeded version, verified by duplicate runs with identical outputs.

## `exploratory_superseded/` folder

Contains two scripts representing methodological decisions **discarded**
during the development of the study, included here for transparency rather
than as part of the final pipeline:

- `commit_level_kmeans_FAILED.py`: first ground truth attempt, clustering
  with *k*-means over individual commits instead of windows. Failed due to
  binary-variable dominance and the absence of a differentiated minority
  cluster (documented in Section 3.4 of the paper).
- `single_split_modeling_SUPERSEDED.py`: first version of oracle modeling
  using a single `GroupShuffleSplit`, later replaced by 5-fold cross-
  validation (`05_oracle_modeling_cv.py`) for greater statistical robustness
  (Section 4.3 of the paper).

## Datasets included in `data/`

| File | Rows | Description |
|---|---|---|
| `raw/github_raw_final.parquet` | 21,638 | Raw commits after bot/spurious-identity filtering at extraction time |
| `processed/github_features_final.parquet` | 17,428 | Cleaned corpus with the features described in Section 3.3 |
| `processed/github_windows_groundtruth.parquet` | 15,564 | 5-commit windows with Isolation Forest labels |
| `processed/trait_state_author_profiles.parquet` | 238 | Per-author temporal stability profiles |
| `processed/oracle_oof_predictions.npz` | 15,098 | Out-of-fold predictions (RF and LSTM) underlying Table 9 and Figures 4-5 |

## Data NOT included

The full text content of commit messages is retained in the parquet files
(`commit_message`) as extracted from the public GitHub API, subject to
GitHub's terms of service. `author_id` was not anonymized, as it corresponds
to usernames already public in public repositories.
