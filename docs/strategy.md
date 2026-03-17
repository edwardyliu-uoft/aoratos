# Aoratos Recommender Model Strategy (Implementation-Ready)

## Executive summary table

| Model                                                                    |                   Expected Lift vs Current Baseline | Effort |   Risk   | Data Requirements                                                       |        Metadata Usage         | Why It Might Win                                                                                    |
| ------------------------------------------------------------------------ | --------------------------------------------------: | :----: | :------: | ----------------------------------------------------------------------- | :---------------------------: | --------------------------------------------------------------------------------------------------- |
| 1. Regularized Biased MF (Funk-SVD + time bins)                          |                           **Medium** (RMSE ↓ ~3–6%) |   M    | Low-Med  | `customer_id`, `movie_id`, `rating`, `date`                             |           Optional            | Adds latent user-item interaction terms and simple temporal drift on top of bias terms.             |
| 2. SVD++ (implicit user-history signal)                                  |                             **High** (RMSE ↓ ~5–9%) |  M-L   |   Med    | all above + per-user interaction history                                |           Optional            | Uses implicit feedback from watched/rated sets; historically strong on Netflix-style explicit data. |
| 3. LightGBM hybrid residual model (regression + optional rank objective) | **Medium-High** (RMSE ↓ ~4–8%; ranking lift likely) |   M    |   Med    | baseline/MF predictions + engineered user/item/time features + metadata |     Optional (high-value)     | Learns nonlinear interactions and can optimize ranking-aware objectives when grouped by user.       |
| 4. Factorization Machine / DeepFM hybrid                                 |         **High** (RMSE ↓ ~6–10%; cold-start better) |   L    | Med-High | sparse IDs + metadata categorical/text-derived features                 | **Required** for full benefit | Unifies collaborative and side features in one model; strong for sparse high-cardinality features.  |
| 5. SASRec-style sequence model (time-ordered per-user history)           |    **Conditional High** (ranking ↑; RMSE uncertain) |   L    |   High   | reliable timestamp order and sufficient sequence length                 |           Optional            | Captures short/long-term preference drift from event order; likely best for top-K personalization.  |

---

## Codebase & data audit summary

### Architecture summary
- Data ingestion/parsing: `src/aoratos/data/parsers.py`
- Raw -> parquet compression: `src/aoratos/data/compress.py`
- Dataset build/read/savepoint: `src/aoratos/data/builders.py`, `src/aoratos/data/reader.py`, `src/aoratos/data/savepoints.py`
- Metadata enrichment: `src/aoratos/data/supplement.py`
- Models + metrics interface: `src/aoratos/models/base.py`, `src/aoratos/models/baseline.py`, `src/aoratos/models/metrics.py`

### Baseline bottlenecks
- No latent interaction terms beyond additive biases.
- Cold-start fallback collapses to global mean + whichever side bias exists.
- Time feature (`date`) and enriched metadata are not used by model.
- No ranking-aware objective/metrics pipeline in model layer.

### Assumptions
- **Assumption:** `date` values are reliable enough for temporal feature engineering and sequence ordering.  
- **Assumption:** enriched metadata coverage after `supplement()` is high enough to train side-feature models.
- **Assumption:** baseline score will continue to be tracked as RMSE primary metric.

### Risks
- **Risk:** metadata null-rate can reduce hybrid gains if TMDB matching is sparse/ambiguous.
- **Risk:** sequence models can underperform if user histories are too short/noisy.
- **Risk:** ranking optimization may improve top-K while not improving RMSE.

### Missing Information
- Exact current baseline RMSE/MAE on canonical split (single source of truth run).
- Metadata coverage stats (`tmdb_id` match rate, null rates for genre/director/actor/description).
- Resource envelope (CPU/GPU availability, max training wall time).
- Acceptance threshold for statistically significant “beats baseline”.

---

## Research evidence summary

| Source                                                                  |      Year | Type  | Claim used                                                                                      | Why relevant here                                                  |
| ----------------------------------------------------------------------- | --------: | ----- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Koren et al., *Matrix Factorization Techniques for Recommender Systems* |      2009 | Paper | MF with biases became standard strong baseline for explicit rating prediction at Netflix scale. | Directly aligned to current bias-only model extension path.        |
| Koren, *Factorization Meets the Neighborhood*                           |      2008 | Paper | Combining latent factors with neighborhood/implicit signals improves Netflix-style predictions. | Supports SVD++-style upgrade over bias model.                      |
| Rendle, *BPR*                                                           | 2009/2012 | Paper | Pairwise optimization improves ranking from implicit preferences.                               | Relevant if adding top-K objective branch.                         |
| XGBoost Learning-to-Rank docs                                           |      2025 | Docs  | `rank:ndcg`, grouped `qid`, and pair-construction strategies are production-ready.              | Practical tree-based ranking path with known API behavior.         |
| LightGBM docs                                                           |      2026 | Docs  | Supports `lambdarank` / `rank_xendcg` and ranking metrics (`ndcg`, `map`).                      | Efficient CPU-friendly ranker for sparse tabular features.         |
| CatBoost ranking docs                                                   |      2026 | Docs  | Multiple ranking objectives/metrics with group-aware training.                                  | Alternative tree ranker with robust categorical handling.          |
| SASRec                                                                  |      2018 | Paper | Self-attention sequence modeling captures short/long-term user dynamics efficiently.            | Justifies sequence-aware branch when timestamp order is available. |
| BERT4Rec                                                                |      2019 | Paper | Bidirectional masked-item modeling improves sequential recommendation quality.                  | Advanced sequence baseline option after SASRec-style model.        |

Notes:
- External browsing was available.
- Recent (2021+) high-quality recommender papers are broad and often implicit-feedback oriented; for this repository, production docs (XGBoost/LightGBM/CatBoost) provide stronger immediate implementation guidance with explicit grouped ranking APIs.

---

## Full model playbooks

## Model 1 — Regularized Biased MF (Funk-SVD + time bins)

### A) Why it can beat baseline
- Baseline only models additive user/item offsets; MF adds user-item interaction via latent vectors.
- Time bins for user and movie drift can use `date` signal already present in ratings.
- Preconditions: enough repeated user/movie interactions; stable date parsing.
- **Metadata Usage:** `Optional`.
  - Optional fields: `genre`, `director`, `actor` as additional bias terms (e.g., genre_bias).

### B) LLM-executable implementation guide
1. **Add files**
   - `src/aoratos/models/mf.py`
   - `tests/models/test_mf.py`
2. **Modify files**
   - `src/aoratos/models/__init__.py` (export class)
   - `src/aoratos/models/types.py` (if extra typed config needed)
3. **Data prep**
   - Read `train` and `test` from `aoratos.data.read`.
   - Parse `date` into integer day index + optional coarse bins (month/quarter).
   - Build contiguous user/movie index maps.
4. **Model design**
   - Class: `MatrixFactorizationModel(BaseModel)`
   - Params: `n_factors`, `lr`, `reg`, `reg_bias`, `n_epochs`, `use_time_bias`, `clip_predictions`.
   - Prediction: `mu + b_u + b_i + <p_u, q_i> + optional_time_bias`.
5. **Training loop**
   - SGD over observed ratings; shuffle each epoch.
   - Early stopping on validation RMSE with patience.
6. **Hyperparameters**
   - Start: `n_factors=64`, `lr=0.01`, `reg=0.02`, `reg_bias=0.01`, `n_epochs=25`.
   - Search: `n_factors [32, 64, 128, 192]`, `lr [0.003, 0.02]`, `reg [0.005, 0.08]`, epochs `[15, 40]`.
7. **Validation**
   - Keep existing train/test protocol; optionally carve 10% train-valid temporal holdout.
8. **Inference integration**
   - Keep `predict(X)` requiring `customer_id`, `movie_id`.
   - Cold-start fallback to available biases/global mean.
9. **Tests to add**
   - Determinism with fixed seed.
   - Monotonic reduction of train RMSE in first few epochs.
   - Cold-start fallback behavior.
10. **Failure modes + debug checklist**
   - Divergence: lower `lr`, increase `reg`.
   - Overfit: reduce factors, add early stopping.
   - Date leakage: ensure features computed only from train context.

### C) Evaluation plan
- Metrics: RMSE, MAE primary.
- Ranking metrics (optional): NDCG@10/20, MAP@10, Recall@10 by user-grouped candidate ranking from held-out test rows.
- Baseline comparison: paired per-user/per-item error deltas; bootstrap CI on RMSE delta.
- Ablation: `+latent`, `+time`, `+metadata-bias`.
- Go/no-go: RMSE improves by **>=1.5% relative** and CI excludes zero.

### D) Delivery estimate
- Effort: **M**
- Runtime: moderate CPU (minutes to a few hours depending epochs/data slices).
- Dependency risk: low (numpy/pandas only).
- Maintainability: high (fits existing BaseModel pattern).

---

## Model 2 — SVD++ (implicit user-history enhancement)

### A) Why it can beat baseline
- Adds implicit-feedback term from user interaction history beyond explicit rating value.
- Historically effective on Netflix-style sparse explicit datasets.
- Preconditions: sufficient per-user history; manageable memory for implicit factors.
- **Metadata Usage:** `Optional`.
  - Add side-biases or side-factor priors from `genre/director/actor`.

### B) LLM-executable implementation guide
1. **Add files**
   - `src/aoratos/models/svdpp.py`
   - `tests/models/test_svdpp.py`
2. **Modify files**
   - `src/aoratos/models/__init__.py`
3. **Data prep**
   - Build user->set(items) interaction map from train.
   - Precompute normalization term `|N(u)|^{-1/2}`.
4. **Model design**
   - `SVDPPModel(BaseModel)` with parameters for explicit + implicit factors.
   - Prediction: `mu + b_u + b_i + q_i^T (p_u + |N(u)|^{-1/2} * sum(y_j))`.
5. **Training loop**
   - SGD with efficient cached implicit sum vectors per user (updated incrementally per epoch).
6. **Hyperparameters**
   - Start: `n_factors=64`, `lr=0.007`, `reg=0.02`, `reg_imp=0.03`, `n_epochs=20`.
   - Search: factors `[64, 96, 128]`, `lr [0.003, 0.015]`, `reg [0.01, 0.08]`.
7. **Validation protocol**
   - Same as Model 1; track train/valid RMSE and memory footprint.
8. **Inference API**
   - Use train-derived implicit history for known users.
   - For unseen user: fallback to MF/bias term only.
9. **Tests**
   - Fit/predict schema checks.
   - Known-user and unknown-user path correctness.
   - Stable score outputs for fixed seed.
10. **Debug checklist**
   - Memory pressure from interaction maps.
   - Slow training from repeated history sums (cache aggressively).
   - Numerical instability from high learning rates.

### C) Evaluation plan
- Metrics: RMSE/MAE first; ranking metrics optional.
- Baseline comparison: baseline vs MF vs SVD++ with paired bootstrap CI.
- Ablation: remove implicit term (`y_j`) and compare.
- Go/no-go: beat baseline by **>=2.5% RMSE relative** and beat MF by **>=0.8%**.

### D) Delivery estimate
- Effort: **M-L**
- Runtime: higher than MF (history terms increase cost).
- Dependency risk: low (no new heavy dependencies required).
- Maintainability: medium (more complex training internals).

---

## Model 3 — LightGBM hybrid residual model

### A) Why it can beat baseline
- Learns nonlinear interactions among user/item/popularity/time/metadata signals.
- Residual-learning setup (`target = rating - baseline_pred`) is robust and incremental.
- Can also support grouped ranking objective for top-K metrics.
- Preconditions: stable feature engineering and user-group construction for ranking branch.
- **Metadata Usage:** `Optional` (high impact when coverage is good).
  - Consume `genre`, `director`, `actor` as categorical encodings; `description` via hashed TF-IDF/embedding summary.

### B) LLM-executable implementation guide
1. **Dependencies**
   - Add `lightgbm` to `pyproject.toml`.
2. **Add files**
   - `src/aoratos/models/lgbm_hybrid.py`
   - `src/aoratos/models/features.py`
   - `tests/models/test_lgbm_hybrid.py`
3. **Feature engineering**
   - User stats: count, mean rating, variance, recency.
   - Item stats: count, mean rating, variance, recency.
   - Interaction features: user_mean - item_mean, user_count*item_count bins.
   - Temporal: year, month, age-from-release.
   - Metadata: category encodings and optional text-derived dimensions.
   - Add baseline/MF predictions as features.
4. **Model design**
   - `LightGBMHybridModel(BaseModel)` supporting two modes:
     - `objective='regression'` for RMSE.
     - `objective='lambdarank'` for ranking.
5. **Training**
   - Regression first (quick win), then optional ranking fine-tune by user groups.
6. **Hyperparameters**
   - Start (regression): `num_leaves=63`, `learning_rate=0.05`, `n_estimators=1200`, `min_data_in_leaf=100`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`.
   - Search: leaves `[31, 63, 127]`, lr `[0.02, 0.1]`, min_leaf `[50, 300]`, l2 `[0, 20]`.
7. **Validation**
   - Regression: RMSE/MAE.
   - Ranking: NDCG@10/20, MAP@10 with user-grouped evaluation.
8. **Inference API**
   - Preserve `predict(X)` returning rating estimate (for ranking mode, calibrate score to rating scale where needed).
9. **Tests**
   - Feature generation null-safe behavior.
   - Group sorting correctness for ranking mode.
   - End-to-end fit/predict smoke test on tiny dataset.
10. **Failure modes**
   - Leakage from aggregate features computed on full train+test.
   - Poor ranking due to wrong group boundaries.
   - Categorical explosion without frequency capping.

### C) Evaluation plan
- Compare to baseline, MF/SVD++ using same test split.
- Add ranking benchmark with user-grouped candidate sets.
- Ablations: no metadata, no temporal, no latent input features.
- Go/no-go: RMSE improves by **>=2% relative** OR NDCG@10 improves by **>=5% relative** with RMSE degradation <=0.5%.

### D) Delivery estimate
- Effort: **M**
- Runtime: moderate-high CPU depending features.
- Dependency risk: medium (new native dependency).
- Maintainability: medium-high if feature builder is modularized.

---

## Model 4 — Factorization Machine / DeepFM hybrid

### A) Why it can beat baseline
- FM models second-order feature interactions in sparse one-hot spaces efficiently.
- DeepFM adds nonlinear high-order interactions; helpful with metadata and cold-start.
- Preconditions: robust sparse encoding pipeline and stable training infra.
- **Metadata Usage:** `Required` for full expected benefit.
  - Use `genre`, `director`, `actor`, `year`, and optional `description` embeddings.

### B) LLM-executable implementation guide
1. **Dependencies**
   - Add `torch` (and optionally `scipy` for sparse ops).
2. **Add files**
   - `src/aoratos/models/deepfm.py`
   - `src/aoratos/models/feature_store.py`
   - `src/aoratos/models/dataloaders.py`
   - `tests/models/test_deepfm.py`
3. **Data prep**
   - Build categorical vocabularies for IDs and metadata fields.
   - Numerical features standardized (`user_count`, `item_count`, temporal bins).
4. **Model design**
   - Shared embedding table(s); FM interaction block + MLP block; fusion head predicts rating.
5. **Training loop**
   - Mini-batch Adam, MSE objective (+optional pairwise rank loss regularizer).
6. **Hyperparameters**
   - Start: `embed_dim=32`, `mlp=[256,128,64]`, `dropout=0.2`, `lr=1e-3`, `batch=8192`, `epochs=8`.
   - Search: embed `[16,32,64]`, lr `[3e-4,3e-3]`, dropout `[0.1,0.4]`.
7. **Validation**
   - Same RMSE/MAE; optional ranking metrics from predicted scores.
8. **Inference API**
   - Keep `predict(X)` with internal feature lookup maps.
9. **Tests**
   - Vocabulary unknown-token handling.
   - Batch collation and shape checks.
   - One-epoch overfit test on toy data.
10. **Failure modes**
   - OOV rate high at test time.
   - Instability from very sparse rare categories.
   - GPU/CPU training throughput bottlenecks.

### C) Evaluation plan
- Baseline + MF + LightGBM comparison with same splits.
- Ablation: IDs only vs IDs+metadata vs IDs+metadata+text.
- Go/no-go: RMSE improvement **>=3% relative** and cold-start slice (low-history items) MAE improves **>=5%**.

### D) Delivery estimate
- Effort: **L**
- Runtime: high (especially with text features).
- Dependency risk: medium-high (`torch` stack).
- Maintainability: medium if feature-store abstractions are clean.

---

## Model 5 — SASRec-style sequence model (temporal dynamics)

### A) Why it can beat baseline
- Uses chronological user interaction sequences to capture evolving taste drift.
- Proposal explicitly calls out temporal dynamics as a research question.
- Preconditions: valid timestamp ordering and sufficient per-user sequence length.
- **Metadata Usage:** `Optional`.
  - Append item metadata embeddings to item token embeddings.

### B) LLM-executable implementation guide
1. **Dependencies**
   - Add `torch` (if not already added by Model 4).
2. **Add files**
   - `src/aoratos/models/sasrec.py`
   - `src/aoratos/models/sequence_data.py`
   - `tests/models/test_sasrec.py`
3. **Data prep**
   - Sort each user’s interactions by `date`.
   - Create fixed-length windows (`max_len=100`) and next-item targets.
   - For explicit ratings use either:
     - item ranking objective, then map score -> rating via calibration; or
     - multitask head (next-item + rating regression).
4. **Model design**
   - Item embedding + positional embedding + stacked self-attention blocks.
5. **Training loop**
   - Cross-entropy/BPR for next-item; optional MSE head for rating.
6. **Hyperparameters**
   - Start: `d_model=128`, `n_heads=4`, `n_layers=2`, `dropout=0.2`, `lr=1e-3`, `batch=2048`.
   - Search: depth `[2,4]`, dim `[64,128,256]`, len `[50,100,200]`.
7. **Validation**
   - Primary: NDCG@10, Recall@10 on next-item task.
   - Secondary: calibrated RMSE if rating head added.
8. **Inference integration**
   - Add API for per-user top-K candidate scoring.
   - Optional adapter converting ranked score to rating estimate.
9. **Tests**
   - Sequence builder ordering correctness.
   - Padding/mask correctness.
   - Reproducible forward pass dimensions.
10. **Failure modes**
   - Sparse/short histories reduce gains.
   - Drift from train/test temporal mismatch.
   - Expensive candidate scoring for full catalog.

### C) Evaluation plan
- Ranking-focused by design: NDCG@K, MAP@K, Recall@K.
- Compare against LightGBM ranker and MF-derived ranking.
- Ablation: no positional embedding, no metadata embedding, shorter sequence length.
- Go/no-go: NDCG@10 improves **>=7% relative** and at least neutral RMSE impact after calibration.

### D) Delivery estimate
- Effort: **L**
- Runtime: high (GPU recommended).
- Dependency risk: high (new DL stack + training complexity).
- Maintainability: medium-low unless training framework standardized.

---

## Recommended execution order (quick wins first)

1. **Model 1 (Biased MF + time bins)** — fastest high-confidence RMSE gain with minimal dependency risk.
2. **Model 3 (LightGBM hybrid residual)** — strong practical lift; introduces ranking-friendly infrastructure.
3. **Model 2 (SVD++)** — likely top RMSE candidate among CF families.
4. **Model 4 (DeepFM hybrid)** — strongest metadata/cold-start upside once feature stack is mature.
5. **Model 5 (SASRec)** — pursue if ranking and sequence behavior become top priority.

---

## Practical baseline-comparison protocol (applies to all models)

1. Fix one canonical split build and random seed policy.
2. Run baseline and candidate models under same data/version snapshot.
3. Report RMSE/MAE and ranking metrics with bootstrap confidence intervals.
4. Run ablations for every added signal category.
5. Ship only models that clear their model-specific go/no-go thresholds and pass reproducibility checks.
