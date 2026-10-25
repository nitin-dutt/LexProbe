# LexProbe — LLM Legal Contract Risk Analysis & MLOps Pipeline

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-PEFT%20%7C%20Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving%20Layer-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Evidently AI](https://img.shields.io/badge/Evidently%20AI-Data%20Drift%20Monitoring-FF4B4B)](https://www.evidentlyai.com/)
[![Cloudflare Tunnel](https://img.shields.io/badge/Cloudflare-Edge%20Routing-F38020?logo=cloudflare&logoColor=white)](https://www.cloudflare.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end LLM engineering project that flags high-risk clauses in commercial contracts, quotes the supporting text, and tracks the whole lifecycle from versioned data to served model.

LexProbe fine-tunes **Llama-3.1-8B-Instruct** with **4-bit QLoRA on a single 8 GB laptop GPU** to detect **seven risky clause types** in commercial contracts (Governing Law, Limitation of Liability, Termination for Convenience, Auto-Renewal, Non-Compete, Liquidated Damages, Anti-Assignment) and to quote the supporting text. The deployed model is **v6b**, scored in a single pass per chunk. Every choice (threshold, calibration, variant) is made on validation data only, and results are reported on a frozen, leak-free test split of 76 contracts the models never saw:

* **Realistic full-document test** (every chunk of the 76 held-out contracts, where only ~1.5% of cases are real clauses): contract-level macro-F1 **0.771 (tuned zero-shot) → 0.867** (+0.096, 95% CI +0.039 to +0.151); chunk-level F1 0.794; 0.88 falsely flagged chunks per contract.
* **Ranking quality** (threshold-free average precision over the 7 clauses): **0.844 vs 0.710 for the untrained model** (the difference is outside the noise).
* **Evidence:** of 326 quotes on the sampled test, 100.0% are found in the source and 99.7% are exact substrings (325/326); span-overlap F1 against the experts' highlight is 0.901 (untrained model: 0.776).
* **Sampled test** (1 positive : 2 negatives, an easier test): macro-F1 0.881 → 0.947.

**Honest limits:** Liquidated Damages stays weak (contract F1 0.57; only 12 test positives). With 76 test contracts, differences of a few points between model versions (v4, v5, v6, v6b) are within noise, so the claim is "clearly better than untrained, versions statistically alike". The adapter was trained in stages (v4 → v5 → v6 → v6b), and 47 positive plus 49 negative **hand-written, AI-authored** Liquidated Damages examples (reviewed externally three times) were added in the last stage; their effect cannot be measured with 12 test examples. Of the six pre-set targets, two are met (contract F1 ≥ 0.86 and beating tuned zero-shot with a CI above zero) and four are not (chunk F1 0.794 < 0.81, weakest clause 0.57 < 0.73, 0.88 false alarms per contract > 0.5, quotes 99.7% exact rather than 100%). **History:** the first model (v4) scored macro-F1 0.875 → 0.960 on the sampled test but only 0.607 chunk-F1 on every chunk; that gap is explained in the sections below, and it is why the realistic test is the headline.

The model is served through a **FastAPI** microservice, with **MLflow** experiment tracking and **DVC** versioning of data and the adapter. A Dockerfile, Cloudflare Tunnel instructions and an offline Evidently drift demo are included but have not been validated end to end.

---

## 📌 Table of Contents
- [Project Objective & Motivation](#-project-objective--motivation)
- [System Architecture](#-system-architecture)
- [How It Works (Step-by-Step Pipeline)](#-how-it-works-step-by-step-pipeline)
- [Empirical Evaluation & Metrics](#-empirical-evaluation--metrics)
- [Production Data Drift Monitoring](#-production-data-drift-monitoring)
- [Repository Structure](#-repository-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Docker Deployment](#-docker-deployment)
- [Future Roadmap & Next Steps](#-future-roadmap--next-steps)
- [License](#-license)

---

## 🎯 Project Objective & Motivation

### The Problem
Commercial contracts are dense, highly idiosyncratic, and full of financial and operational liability. Reviewing them for risk clauses such as liability caps, unilateral termination rights or automatic renewals is slow, costly and easy to get wrong.

Off-the-shelf zero-shot prompting of a general model is a reasonable start, but on our leak-free test set the zero-shot `Llama-3.1-8B-Instruct` baseline raises **62 false alarms on 536 clause-free excerpts** (macro precision 0.833) in a balanced sample, and on every chunk of the held-out contracts it falsely flags **1,451 of 17,689 clause-free cases** (about 19 chunks per contract, macro precision 0.173). In a review queue, every false alarm costs reviewer time.

### The Objective
1. **Accurate first-pass triage:** fine-tune `Llama-3.1-8B-Instruct` to decide whether each of 7 high-risk clause types (6 originally; Anti-Assignment was added in dataset profile v4) is present in a contract excerpt, and quote the supporting text.
2. **Consumer hardware:** train with 4-bit QLoRA on a single **8 GB laptop GPU** (RTX 5070 Laptop).
3. **Measurement you can defend:** leak-free splits by near-duplicate contract group, a frozen DVC-versioned test set, paired significance testing, and verification that quoted evidence actually appears in the source.
4. **Production path:** FastAPI serving of the blend with verified evidence quotes, MLflow tracking and DVC versioning. A Dockerfile, Cloudflare Tunnel instructions and an offline drift demo are included but not yet validated end to end.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 LEXPROBE SYSTEM ARCHITECTURE                           │
└────────────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                     DATA (dataset v3, DVC-pinned, manifest + SHA-256)            │
  │                                                                                  │
  │   Stanford CUAD v1 (510 contracts) ── exact CUAD category → 6 risk labels        │
  │     ├── Whitespace normalisation with offset map (0 span-mapping errors)         │
  │     ├── Near-duplicate grouping (8-gram Jaccard ≥ 0.5) → group-level split       │
  │     │     357 / 77 / 76 contracts  (train / val / test), zero group overlap      │
  │     └── 1,500-char boundary-aware chunks: evidence inside the prompt for 100%    │
  │         of positives; per-split seeded sampling, 50% hard negatives              │
  └──────────────────────────────────────────┬───────────────────────────────────────┘
                                             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                     TRAINING (config-driven, MLflow-tracked)                     │
  │                                                                                  │
  │   4-bit NF4 QLoRA · r=16, α=32 · all attention + MLP projections                 │
  │   Official Llama-3.1 chat template · loss on answer tokens only                  │
  │   bf16 head + non-paged 8-bit AdamW → fits 8 GB (peak 6.97 GiB @ 768 tokens)     │
  │   Val decision-F1 every 100 steps, best checkpoint kept, resumable checkpoints   │
  │   820 steps / 2 epochs in 3 h 01 m                                               │
  └──────────────────────────────────────────┬───────────────────────────────────────┘
                                             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ EVALUATION (frozen test split, DVC-pinned; every choice made on validation only) │
  │                                                                                  │
  │ Baseline = same 4-bit model with the adapter disabled (identical inputs)         │
  │ Sampled test (804 queries): macro-F1 0.875 -> 0.960 · false alarms 62 -> 2       │
  │ Every chunk (2,995, 1.5% positive): v4 alone chunk-F1 0.607; contract-F1 0.720*  │
  │   *val-tuned threshold; tuned zero-shot 0.739 (not significantly different)      │
  │ Quotes verbatim: 71.4% -> 100% (254/254) · span-F1 vs gold 0.790 -> 0.911        │
  └──────────────────────────────────────────────────────────────────────────────────┘
                                             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ DEPLOYED: v6b, seven clauses, ONE pass per chunk (configs/model_v6b.json)        │
  │                                                                                  │
  │ Per-clause score shift + validation-chosen threshold; stable in 5 fold seeds     │
  │ Test contract-F1 0.771 -> 0.867 (+0.096, 95% CI +0.039..+0.151) · chunk 0.794    │
  │ 0.88 falsely flagged chunks/contract · 0.97 s per chunk · peak 5.7 GiB VRAM      │
  │ Adapter pinned in DVC (models/.../lora-v6b/final_adapter.dvc)                    │
  └──────────────────────────────────────────────────────────────────────────────────┘
                                             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │ SERVING: FastAPI (/analyze-contract, /upload: PDF/DOCX/TXT), whole documents,    │
  │                                                                                  │
  │ grounded quotes with offsets; SCORING=single (v6b, default), blend, or v4.       │
  │ The Dockerfile, Cloudflare Tunnel, Evidently drift demo: included, not validated │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

*The data, training and evaluation boxes describe the original v3/v4 pipeline. v6b reuses it with dataset profile v4 (adds Anti-Assignment, same contracts per split) and continues the adapter in stages v4 → v5 → v6 → v6b; see "Final model: v6b" below.*

---

## ⚙️ How It Works (Step-by-Step Pipeline)

### Step 1: Leak-Free, Versioned Dataset (`src/prepare_data.py`, `src/chunking.py`)
* **Labels:** exact CUAD categories mapped to 6 risk labels: `Governing Law`, `Auto-Renewal` (CUAD *Renewal Term*), `Limitation of Liability` (CUAD *Cap On Liability*), `Liquidated Damages`, `Non-Compete`, `Termination for Convenience`. Semantically opposite categories (e.g. *Uncapped Liability*) are excluded rather than merged.
* **Leak-free split:** contracts are grouped by near-duplicate text (6 near-copy pairs found in CUAD) and split **by group** 70/15/15, so a contract and its copy can never sit on both sides.
* **Evidence-visible chunks:** contracts are normalised and cut into ≤1,500-character windows on paragraph/sentence boundaries with overlap. A chunk is positive only if a gold clause lies inside it, and the stored chunk is exactly what the model sees. A build-time assertion checks that 100% of positives contain their evidence.
* **Frozen evaluation:** val/test sample negatives with their own seeded RNGs, independent of training choices. The build refuses to overwrite an existing dataset, writes a manifest with SHA-256 hashes, and is pinned with DVC.
* **Dataset v3:** train 3,279 queries · val 792 · **test 804 (268 positive / 536 negative)**.

### Step 2: QLoRA Fine-Tuning on an 8 GB GPU (`src/finetune.py`, `configs/qlora_v4.yaml`)
* **Backbone:** `meta-llama/Llama-3.1-8B-Instruct`, 4-bit NF4 with double quantisation, bf16 compute.
* **Adapter:** LoRA r=16, α=32, dropout 0.05 on `q,k,v,o,gate,up,down` projections (42M trainable parameters, 0.52%).
* **Prompting:** one shared prompt builder (`src/prompting.py`) using the official Llama-3.1 chat template for training, evaluation and serving.
* **Loss:** completion-only. Prompt tokens are masked, so the model learns the `Found: <quote>` / `None` answer.
* **Fitting 8 GB:** keeping embeddings and LM head in bf16 (instead of the usual fp32 upcast) and using a non-paged 8-bit optimizer stopped training from spilling into system RAM: one 512-token step went from 4.95 s to 1.12 s in single-step timing probes (not a full-run comparison). It also removed a paged-optimizer checkpoint crash.
* **Model selection:** decision macro-F1 on a fixed validation subset every 100 steps; the best checkpoint (step 500, val F1 0.947) is kept.
* **Reproducibility:** each run logs its config, dataset hashes and git commit to MLflow and writes `training_summary.json` next to the adapter. The adapter itself is pinned with DVC (`models/llama-3.1-8b-cuad-lora-v4/final_adapter.dvc`).

### Step 3: Evaluation (`src/evaluate.py`, `src/compare_results.py`, `src/evaluate_documents.py`, `src/document_eval.py`)
* **Fair baseline:** the same 4-bit model with the adapter **disabled**, same prompts, same inputs.
* **Decision:** first-token logit of `Found` vs `None`, one forward pass per query.
* **Statistics:** paired bootstrap over contract groups (2,000 draws) for macro-F1 CIs and the difference, plus an exact McNemar test.
* **Evidence:** for predicted positives the model generates the quote; we check it appears in the source chunk and score span-overlap F1 against the gold clause.
* **Realistic full-document evaluation:** a deployed system sees every chunk of a contract, where ~98.5% of (chunk, clause) cases are negative, so a sampled test set overstates precision. `evaluate_documents.py` scores all 17,970 (chunk, clause) pairs of the 76 test contracts. The system prompt and excerpt are identical for all 6 clauses, so their KV cache is computed once and reused (cached margins match the plain forward pass within bf16 rounding with 0 decision flips; 2.7–3.5× faster). `document_eval.py` reports chunk-level and contract-level metrics, thresholds chosen **on the validation split only**, paired group-bootstrap CIs, and threshold-free average precision.

* **Recovery Stage 1 (`src/calibration_experiments.py`):** can the gap be closed without retraining? Candidates (raw margin, prior-shift correction, per-clause logistic calibration, per-clause blend of the v4 and zero-shot margins) × contract rule × threshold objective are scored with **5-fold out-of-fold validation scores** (folds = near-duplicate contract groups). The success gate was written down before the experiment: paired group-bootstrap 95% CI of the contract-level macro-F1 gain over the validation-tuned zero-shot model must have a lower bound above 0, and one v4-based variant is chosen on validation alone.

### Step 4: Serving Layer (`api/`, `src/inference.py`, `src/blend.py`, `src/chunk_scoring.py`)
* **Full-document analysis:** the API normalises the uploaded text, splits the **whole** document with the training chunker, scores every (chunk, clause) pair, then generates one quote per flagged clause. In the default mode (`SCORING=single`, v6b) each chunk is scored in one pass; the earlier `SCORING=blend` scores each chunk twice, adapter on and off, and combines the two first-token margins with the per-clause weights in `configs/blend_v1.json`; `SCORING=v4` serves the v4 score alone. The system prompt and excerpt are shared by the clause questions, so their KV cache is computed once per pass.
* **Scoring modes:** `SCORING=single` (default) serves v6b alone (7 clauses, one pass per chunk, `configs/model_v6b.json`: per-clause prior shift and threshold −2.0); `SCORING=blend` serves the earlier v4 + zero-shot blend (6 clauses, `configs/blend_v1.json`); `SCORING=v4` serves v4 alone. The 7-clause mode selects its dataset profile from the config, so start it in a fresh process.
* **Web interface (`api/static/`):** three screens. **Start:** a large drag-and-drop upload area (a whole-page drop overlay too) with a file chip, next to a tall paste panel, plus two sample contracts. **Analysing:** a progress screen on its own. **Results:** only the results, with an **Analyse another contract** button at the top right (the browser Back button works too). Results show a card per flagged clause (quote, a green "verified verbatim in the source" mark or an amber warning, section heading, exact character range, one-click **Copy citation**, a short "how was this decided" note) beside the **source document** with every citation highlighted and numbered. **Download report** builds a self-contained HTML file in the browser (summary, findings with citations, the contract with each citation marked as `[n Clause]`, a not-found list, a disclaimer); it prints cleanly to PDF and nothing is stored on the server. Also: **Download PDF** uses the browser's print dialog and includes the contract with every citation marked; light and dark themes, keyboard focus styles and a phone layout. Model output is only ever inserted as text (never as HTML), the page and the report have no script, and the server sends a strict Content-Security-Policy with `X-Frame-Options: DENY` and `nosniff`. The page's pure JavaScript (section detection, highlight segments, citation text, report) has its own node tests; two bugs found in dry runs (every clause labelled with the same all-caps notice; a loading bar visible before upload) each have a regression test.
* **Verified evidence:** each quote (always generated by the fine-tuned adapter) is checked against the document. The response includes `grounded` (found verbatim or not) and the quote's character offsets, and the UI marks unverified quotes for manual review.
* **Endpoints:** `/analyze-contract` (JSON text), `/upload` (PDF via `pdfplumber`, DOCX via `python-docx`, TXT), `/health` (model loaded, adapter, scoring mode, threshold), `/clause-types`. Every analysis response reports `scoring` and `decision_threshold`.
* **Guardrails:** 2 MB upload limit enforced while streaming (HTTP 413), 15,000-character text limit, Pydantic validation (`min_length=50`), one inference at a time on the GPU (HTTP 503 when busy), internal errors returned as a request ID without stack details, model output HTML-escaped in the UI.
* **Verified on the GPU through the API path:** on 3 whole test contracts (174 chunk × clause scores) the live v4, zero-shot and blended scores equal the saved evaluation scores exactly (0 decision flips) and the live decisions match the offline ones; a 14,451-character contract takes about 42 s end to end including quotes (61 s with the earlier v4-only path); peak VRAM 5.7 GiB. This is a wiring check; the measured quality is the test result below.
* **v6b verified on the GPU through the API path (2026-10-23):** on 3 whole test contracts (203 chunk × clause scores) the live margins equal the saved evaluation scores exactly (max difference 0.000, 0 decision flips) and the live decisions match the offline ones on 21 of 21 clause cases; one request for an 8,080-character contract took 32 s (median 0.97 s per chunk, one pass), peak VRAM 5.7 GiB.
* *Not yet:* authentication / rate limiting for public exposure.

### Step 5: Public Edge Routing (Cloudflare Tunnel)
* `cloudflared tunnel --url http://localhost:8000` exposes the local service over HTTPS through an outbound-only tunnel, with no public IP or open inbound ports. (The tunnel hides the server's address; it does not authenticate users. Not exercised in the verification runs.)

### Step 6: Drift Monitoring (Evidently AI)
* An offline demo (`drift_monitoring.py`): embeds text chunks with `sentence-transformers/all-MiniLM-L6-v2` and runs Evidently's drift test between reference contract chunks and a synthetic batch. It is not connected to live traffic (see the drift section).

---

## 📊 Empirical Evaluation & Metrics

### Fine-tuned v4 vs zero-shot baseline — frozen test split (dataset v3)
**804 queries** drawn from **69 contracts (68 near-duplicate groups)** of the 76-contract held-out test split, never seen in training or validation: **268 positive / 536 negative** (chunk, clause) pairs.

| Clause | Pos/Neg | Base P | Base R | Base F1 | **v4 P** | **v4 R** | **v4 F1** | ΔF1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Governing Law | 74/148 | 0.851 | 1.000 | 0.919 | **1.000** | 1.000 | **1.000** | +0.081 |
| Limitation of Liability | 82/164 | 0.695 | 1.000 | 0.820 | **1.000** | 0.939 | **0.969** | +0.149 |
| Auto-Renewal | 34/68 | 1.000 | 0.971 | 0.985 | 0.971 | 1.000 | **0.986** | +0.000 |
| Non-Compete | 33/66 | 0.882 | 0.909 | 0.896 | **1.000** | 0.848 | **0.918** | +0.023 |
| Termination for Convenience | 33/66 | 0.839 | 0.788 | 0.812 | **0.964** | 0.818 | **0.885** | +0.073 |
| Liquidated Damages ⚠️ | 12/24 | 0.733 | 0.917 | 0.815 | 1.000 | 1.000 | 1.000 | +0.185 |
| **Macro average** | **268/536** | 0.833 | 0.931 | **0.875** | **0.989** | **0.934** | **0.960** | **+0.085** |

⚠️ Liquidated Damages has only 12 positive test queries (7 contracts), so treat its per-class numbers as indicative only.

**Significance (paired bootstrap over contract groups, 2,000 draws):**
* Baseline macro-F1 **0.875** (95% CI 0.822–0.904) → v4 **0.960** (95% CI 0.936–0.980).
* Improvement **+0.085** (95% CI **+0.051 to +0.149**), one-sided p < 0.001.
* McNemar exact test: v4 alone correct on 68 queries vs baseline alone on 12, **p = 1.2 × 10⁻¹⁰**.

| Errors on 804 test queries | Baseline | v4 | Change |
|---|:---:|:---:|:---:|
| False positives (false alarms) | 62 | **2** | **−96.8%** |
| False negatives (missed clauses) | 12 | 16 | +4 |
| Total errors | 74 | **18** | **−75.7%** |
| Accuracy | 90.8% | **97.8%** | +7.0 pts |

**Evidence quality** (for every predicted positive, the model generates the supporting quote; we check it against the source chunk):

| | Baseline | v4 |
|---|:---:|:---:|
| Quotes generated | 318 | 254 |
| Quote appears **verbatim** in the contract | 71.4% | **100.0%** |
| Quote grounded (verbatim or ≥ 90% overlap) | 72.3% | **100.0%** |
| Span-overlap F1 vs gold clause (true positives) | 0.790 | **0.911** |

The zero-shot model paraphrases or invents about 28% of its "citations". Every v4 quote is a verbatim excerpt (median 210 characters), so each flag can be checked against the contract.

### Full-document evaluation: the realistic test
**76 held-out test contracts, every chunk scored for all 6 clause types:** 2,995 chunks × 6 = 17,970 cases, of which **268 are positive and 17,689 negative (1.5% prevalence)**; 13 cases where a clause is cut by a chunk edge are excluded. A clause counts as flagged when the first-token margin exceeds a threshold. Thresholds are selected on the **validation** split only, separately for each model.

| System @ threshold | Chunk P | Chunk R | Chunk F1 | Falsely flagged chunks / contract | Contract F1 | Wrong clause flags / contract (of 6) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline @ 0 (default) | 0.173 | 0.931 | 0.281 | 19.1 | 0.645 | 2.05 |
| **v4 @ 0 (default)** | 0.509 | 0.934 | **0.607** | 4.8 | 0.733 | 1.37 |
| Baseline @ +7.0 (tuned on val) | 0.539 | 0.735 | 0.576 | 3.3 | 0.739 | 0.53 |
| **v4 @ +4.5 (tuned on val)** | 0.767 | 0.659 | **0.672** | 0.5 | 0.720 | 0.26 |

* **Default threshold:** v4 beats the baseline by **+0.326 chunk macro-F1** (95% CI +0.294 to +0.356) and **+0.088 contract macro-F1** (95% CI +0.057 to +0.122).
* **Both tuned on validation:** v4 is still ahead at chunk level (**+0.097**, 95% CI +0.046 to +0.141) but **not significantly different at contract level** (**−0.019**, 95% CI −0.094 to +0.054). v4 does leave fewer wrong flags (0.26 vs 0.53 per contract) and more contracts with no wrong flag (75% vs 55%).
* **Threshold-free ranking (average precision):** chunk level 0.693 → **0.783** (+0.090, 95% CI +0.034 to +0.143); contract level 0.821 → 0.842 (+0.021, 95% CI −0.044 to +0.079, not significant). The gains come from **Limitation of Liability (0.756 → 0.947), Termination for Convenience (0.379 → 0.719) and Governing Law (0.932 → 0.981)**; there is **no ranking gain on Auto-Renewal, Liquidated Damages or Non-Compete**.
* **Localisation:** at the default threshold the highest-scoring chunk (the one the API quotes) lies on the clause for 96.8% of correctly flagged contract-clauses (baseline 91.9%).
* **Caveats:** CUAD annotations are incomplete, so some "false alarms" may be real clauses nobody labelled (not yet reviewed by hand). Liquidated Damages has only 12 positive chunks in 7 contracts, so its numbers are unreliable. One global threshold hurts Termination for Convenience (recall 0.18 at +4.5 vs 0.82 at 0), so per-clause calibration is a next step.

Reproduce: `python src/evaluate_documents.py --adapter <adapter>` (add `--split val` for the threshold-selection scores), then `python src/document_eval.py --baseline … --system … --val-baseline … --val …` (see `reports/eval/document_level.md`).

### Recovery: calibrated blend of the two models (the earlier deployed model, 6 clauses)
The default threshold of the fine-tuned model is off by about +4 log-odds, because it was trained at 1.5 negatives per positive while real contracts have 41–204 (the shift is measured per clause from the training data). Rather than retrain, Stage 1 tested what can be fixed on the saved scores. The pre-registered gate: the contract-level macro-F1 gain over the validation-tuned zero-shot model must have a 95% CI lower bound above 0. All selection used validation only (out-of-fold for fitted calibrators); the test split was scored once for the pre-selected variant.

| System (all thresholds chosen on validation) | Test contract F1 | Test chunk F1 |
|---|:---:|:---:|
| Zero-shot, tuned threshold (approved reference) | 0.739 | 0.576 |
| v4 alone, tuned threshold | 0.720 | 0.672 |
| Zero-shot, per-clause calibrated (best baseline variant) | 0.773 | 0.673 |
| **Blend of v4 + zero-shot (earlier deployed model)** | **0.828** | **0.781** |

* **Primary gate: +0.089 contract macro-F1 (95% CI +0.018 to +0.156): PASS.** Stress test against the best calibrated zero-shot variant (0.773): +0.055 (CI +0.006 to +0.107): PASS, narrowly.
* **Secondary:** chunk P/R/F1 0.807 / 0.770 / 0.781 (+0.205 over the tuned zero-shot, CI +0.146 to +0.249); contract P/R/F1 0.847 / 0.820 / 0.828; 0.62 falsely flagged chunks and 0.21 wrongly flagged clause types (of 6) per contract; excluding Liquidated Damages, contract F1 0.894. Average precision (threshold-free): chunk level 0.693 → 0.813 (+0.120, CI +0.075 to +0.156); contract level +0.039 (CI −0.005 to +0.083, not significant).
* **The same variant won under 5 other fold assignments** (validation contract F1 0.816–0.833 vs 0.730–0.780 for calibrated v4 alone).

| Clause | Weight on v4 score | Weight on zero-shot score | Test contract F1 | Test chunk F1 |
|---|:---:|:---:|:---:|:---:|
| Governing Law | 0.56 | 0.12 | 1.000 | 0.987 |
| Limitation of Liability | 0.60 | 0.04 | 0.966 | 0.846 |
| Termination for Convenience | 0.99 | 0.08 | 0.868 | 0.783 |
| Non-Compete | 0.40 | 0.18 | 0.762 | 0.638 |
| Auto-Renewal | **0.07** | **0.46** | 0.873 | 0.833 |
| Liquidated Damages ⚠️ | 0.26 | 0.30 | 0.500 | 0.600 |

**Caveats:** the blend needs two forward passes (about 1.8× compute); for Auto-Renewal it relies mostly on the zero-shot score, and Liquidated Damages (81 train / 12 test positives) stays weak. Part of the gain is calibration, not fine-tuning: calibrating the zero-shot model alone lifts it from 0.739 to 0.773. The weights were fitted on 77 validation contracts, the confidence-interval lower bounds are close to zero, and the test split has now been scored once for the final candidate. Test numbers of the other variants were displayed for understanding only and not used to choose.

**What the errors are (hand review of the 42 most confident false alarms of the fine-tuned model, read against CUAD's clause definitions):** 6 were real clauses the dataset missed, 4 were plausible (late fees, demurrage, a no-hire clause headed "Non-Competition"), **28 were neighbouring concepts** (a plain term clause flagged as renewal, exclusivity flagged as non-compete, termination for cause or by mutual agreement flagged as termination for convenience), and 4 were unrelated. These are the highest-confidence errors, so the shares are not representative of all false alarms.

Reproduce: `python src/calibration_experiments.py --export-blend configs/blend_v1.json` (writes `reports/eval/recovery_stage1.md` and the config).

### Final model: v6b, seven clauses, one pass (deployed)
**Provenance (full honesty):** v4 (fresh QLoRA, dataset v3) → v5 (continued 869 steps on a harder-negative mix) → v6 (continued on dataset profile v4, which adds Anti-Assignment; same contracts per split, so the old test results stay comparable) → **v6b** (continued 700 steps with 47 positive and 49 negative hand-written Liquidated Damages examples added to training only). The synthetic data was first generated by the local 8B model, rejected after two external reviews, then written by hand and reviewed a third time; it is clearly AI-written text, not real data. Validation and test contain only real CUAD chunks.

**Scoring:** one forward pass per chunk. Score = first-token margin (`Found` − `None`) minus a per-clause constant that corrects for training on a harder mix than real contracts (computed from training data only); a clause is flagged when any chunk's score exceeds −2.0 (chosen on validation, stable in 5 other fold assignments). It reproduces the reported test result from saved scores and the exported config alone (`tests/test_single_model.py`).

| Test split, 76 contracts, every chunk, 7 clauses | Contract F1 | Chunk F1 |
|---|:---:|:---:|
| Zero-shot, tuned threshold (approved reference) | 0.771 | – |
| Best calibrated zero-shot variant | 0.799 | – |
| **v6b (deployed)** | **0.867** | **0.794** |

* **Gate (fixed in advance): +0.096 contract macro-F1 over the tuned zero-shot (95% CI +0.039 to +0.151): PASS.** Stress test against the best calibrated zero-shot variant: +0.068 (CI +0.016 to +0.126): PASS.
* **Per clause, contract F1:** Anti-Assignment 0.991 · Governing Law 1.000 · Limitation of Liability 0.966 · Termination for Convenience 0.885 · Auto-Renewal 0.873 · Non-Compete 0.783 · **Liquidated Damages 0.571 ⚠️ (12 test positives)**.
* **Secondary:** chunk P/R/F1 0.790 / 0.814 / 0.794; contract P/R/F1 0.867 / 0.875 / 0.867; 0.88 falsely flagged chunks and 0.22 wrongly flagged clause types per contract; excluding Liquidated Damages, contract F1 0.916.
* **Ranking quality (threshold-free average precision, 95% ranges from resampling contracts):** v6b 0.844 (0.788 to 0.888) vs untrained 0.710 (0.643 to 0.769). On the six original clauses: v5 0.829, v6 0.813, v6b 0.837, untrained 0.692; the differences between the three adapters are inside the noise (v6b vs v5 +0.008, range −0.017 to +0.033), while all three are clearly better than untrained (about +0.12 to +0.14). Liquidated Damages ranking: untrained 0.584, v5 0.708, v6 0.680, v6b 0.709 (12 test positives: no gain can be attributed to the hand-written data).
* **Evidence (sampled test, 1:2):** 326 quotes, 100.0% grounded, 99.7% exact substring (325/326), span-F1 0.901 vs 0.776 untrained; macro-F1 0.881 → 0.947 (+0.067, CI +0.032 to +0.120).

**Pre-set targets (written down on 2026-10-18 before this work):** contract F1 ≥ 0.86: **met** (0.867). Beat tuned zero-shot with a CI above zero: **met**. Chunk F1 ≥ 0.81: 0.794, not met. Weakest clause ≥ 0.73: 0.571 (Liquidated Damages), not met. ≤ 0.5 falsely flagged chunks per contract: 0.88, not met. Quotes 100% verbatim: 99.7% exact (100% grounded), not met.

**What this does and does not show:** the system is clearly better than the untrained model at ranking and at contract-level detection, and quotes are almost always verbatim. It is not shown to be better than v4, v5 or v6 beyond noise, Liquidated Damages is unreliable, and 76 test contracts cannot resolve differences of a few points.

Reproduce (CPU, saved scores): `LEXPROBE_PROFILE=v4 python src/calibration_experiments.py --adapter-system lora_v6b --tag v6b --train-queries data/train_mixes/v4h/train.json`, then `LEXPROBE_PROFILE=v4 python src/single_model.py --report reports/eval_v4/recovery_stage1_v6b.json --adapter models/llama-3.1-8b-cuad-lora-v6b/final_adapter --out configs/model_v6b.json`; model comparison with ranges: `LEXPROBE_PROFILE=v4 python src/compare_models.py --split test --out reports/eval_v4/model_comparison ...`.

### Key Takeaways
1. **Fine-tuning mostly buys precision.** Zero-shot Llama already finds most clauses (recall 0.931) but over-flags them. v4 keeps recall (0.934) and, on the sampled test, cuts false alarms from 62 to 2, which matters most for a reviewer's queue.
2. **The trade-off is honest and visible:** v4 misses 4 more clauses than the baseline (16 vs 12): Termination for Convenience 6, Non-Compete 5, Limitation of Liability 5. In a high-recall deployment the Found/None logit margin can be thresholded lower.
3. **The sampled test overstates real-world precision.** With 1.5% positives, v4's macro-F1 is 0.607 (not 0.960) at the default threshold, and a clause is wrongly flagged in about 1.4 of 6 clause types per contract. Evaluating on every chunk was the most informative experiment in the project.
4. **Fine-tuning's measured wins are calibration, fewer false alarms, verbatim evidence and better ranking on three clause types.** A zero-shot model with a validation-tuned threshold reaches the same contract-level F1 (0.739 vs 0.720, not significantly different), so the headline gain should not be read as a general detection advantage.
5. **The data pipeline was the biggest lever.** An earlier iteration showed the model only the first 600 characters of 8,000-character chunks, so the evidence was visible for just 4% of training positives. Fixing it made the task well-posed for both models. Results from that iteration (macro-F1 0.574 → 0.616) are retired and not comparable.
6. **The earlier deployed model was a calibrated blend, and the honest story is about measurement.** The fine-tuned model alone ties a tuned zero-shot model at contract level; combining the two with validation-fitted per-clause weights passes the pre-registered gate (+0.089, CI +0.018 to +0.156). The headline 0.960 came from a sampled test, 0.607 from the realistic one, and 0.828 was the realistic contract-level result of that blend (6 clauses); the current model, v6b, is covered in item 8.
7. **Most remaining errors are category confusion between neighbouring clause types**, not label noise: of 42 hand-read high-confidence false alarms, 28 were neighbouring concepts and 6 clearly real clauses the dataset missed.
8. **One model, seven clauses, one pass.** v6b replaces the two-pass blend: it ranks clauses clearly better than the untrained model (average precision 0.844 vs 0.710) and reaches contract-level F1 0.867, at about 0.97 s per chunk instead of 1.36 s.
9. **Differences of a few points between model versions are noise on 76 test contracts.** Comparing v5, v6 and v6b by ranking quality with resampled ranges showed they are statistically alike; an earlier reading of 0.877 vs 0.822 contract-F1 as a real regression was a threshold-selection artefact. Reporting ranges, not single numbers, is part of the method.
10. **Synthetic data needs a human read.** Statistics (cue-word shares) matched while meaning was broken, and only external review found incomplete provisions, template artefacts and a formatting leak. The final hand-written set is small, AI-authored and its effect is unmeasurable here.

---

## 📡 Production Data Drift Monitoring

Legal NLP services encounter distribution shifts: users may upload resumes, financial tables or IT tickets into a contract endpoint. `drift_monitoring.py` is an **offline demonstration** of how that could be detected, **not** a monitor connected to live traffic:
* **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional vectors).
* **Reference:** 100 chunks from the test split. **Synthetic "production" batch:** 50 other contract chunks plus 50 out-of-domain items made from 10 unique short sentences repeated 5 times (so the demo is easy by construction).
* **Detector:** Evidently's embedding drift test.
* **Status:** an earlier run reported ROC-AUC 0.739, but that report was not saved and could not be reproduced during the project audit, so **no drift result is claimed here**. The API does not log requests and nothing runs on a schedule, so turning this into real monitoring would need request logging, a rolling comparison window and alerting.

---

## 📂 Repository Structure

```
LexProbe/
├── api/
│   ├── extractor.py               # Multi-format parser (PDF via pdfplumber, DOCX, TXT)
│   ├── main.py                    # FastAPI service (/, /health, /analyze-contract, /upload)
│   ├── model.py                   # loads 4-bit base + adapter; single (v6b, default), blend or v4 scoring, quotes
│   └── static/                    # Web UI: index.html + style.css + app.js (no inline script), sample contracts
│       ├── app.js                 # Pure functions (sections, highlights, citations, report) + the page; model output only as text
│       └── samples/               # Two public CUAD contracts for the 'Try a sample' buttons
├── configs/
│   ├── model_v6b.json             # Deployed v6b: per-clause shift, threshold, provenance (7 clauses)
│   ├── blend_v1.json              # Earlier blend (v4 + zero-shot, 6 clauses), kept as SCORING=blend
│   ├── qlora_v4.yaml              # Training config for the first adapter (fresh run)
│   ├── qlora_v5.yaml / smoke_v5.yaml   # Continuation on a harder-negative mix
│   ├── qlora_v6b.yaml / qlora_v7.yaml  # Latest continuation (deployed) and a planned clean from-scratch run
│   └── smoke_test.yaml            # 20-step pipeline smoke test
├── models/
│   ├── llama-3.1-8b-cuad-lora-v4/final_adapter.dvc    # DVC pointer to the v4 adapter
│   └── llama-3.1-8b-cuad-lora-v6b/final_adapter.dvc   # DVC pointer to the deployed v6b adapter
├── data/
│   ├── raw/CUAD_v1.json.dvc       # DVC pointer to the raw CUAD dataset
│   ├── processed.dvc              # DVC pointer to dataset v3 (chunks, queries, manifest)
│   └── (processed_v4/, train_mixes/, synthetic/ are rebuilt locally, git-ignored)
├── src/
│   ├── download_data.py           # CUAD v1 download
│   ├── chunking.py                # Whitespace normalisation (offset-preserving) + chunking
│   ├── inference.py               # Full-document analysis: chunk → score → quote → ground
│   ├── blend.py                   # Loads/validates the blend config; blended scores
│   ├── single_model.py            # v6b config: per-clause shift + threshold; exports it from the validated report
│   ├── train_mix.py               # Training mixes: harder negatives, oversampled weak clauses, optional synthetic rows
│   ├── answer_loss.py             # Same loss, logits only at answer positions (about 0.5 GB less GPU memory)
│   ├── compare_models.py          # Models compared by ranking quality (AP) with 95% ranges from contract resampling
│   ├── run_report.py              # Builds a run report (timeline, incidents, checkpoint scores) from logs
│   ├── synthetic_ld.py            # 8B-generated synthetic data and its checks: a documented negative result
│   ├── synthetic_hand.py          # Checks and trims the hand-written Liquidated Damages examples
│   ├── chunk_scoring.py           # GPU scoring of one chunk for all clauses (shared KV cache)
│   ├── calibration_experiments.py # Recovery Stage 1: calibration/blend on saved scores, gate, export
│   ├── prepare_data.py            # Near-duplicate group split, labelling, sampling, manifest
│   ├── prompting.py               # Single prompt builder (official Llama-3.1 chat template)
│   ├── training_data.py           # Completion-only tokenisation, token-budget checks, val metrics
│   ├── finetune.py                # Config-driven QLoRA training with MLflow provenance
│   ├── evaluate.py                # Test evaluation: baseline (adapter disabled) vs fine-tuned
│   ├── evaluate_documents.py      # Scores every chunk × clause of a split (KV-cache reuse)
│   ├── document_eval.py           # Full-document metrics, val-tuned thresholds, AP, bootstrap CIs
│   ├── eval_metrics.py            # P/R/F1, evidence grounding, span F1, paired bootstrap, McNemar
│   ├── compare_results.py         # Significance report → reports/eval/comparison.md
│   └── mlflow_cleanup.py          # Marks orphaned MLflow runs
├── tests/                         # pytest suite (data, prompts, labels, metrics, blend, v6b config, API, front end, synthetic data), 150 tests
│   └── js/                        # Node tests for the page's pure JavaScript and the HTML report (test_app.js, test_report.js; run by tests/test_frontend.py)
├── reports/eval/, eval_v4/        # Test-set results, comparison reports (eval_v4 = 7-clause profile)
├── drift_monitoring.py            # Evidently AI embedding drift report
├── Dockerfile / .dockerignore     # Container definition (secrets passed at runtime)
└── requirements.txt
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and set up a virtual environment:

```powershell
git clone https://github.com/nitin-dutt/LexProbe.git
cd LexProbe

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_read_token
```

### 2. Run the FastAPI Serving Engine
Launch the server (loads the 4-bit base model and the v6b adapter on the GPU at startup and serves v6b in one pass; `SCORING=blend` serves the earlier 6-clause blend, `SCORING=v4` the v4 adapter alone). `dvc pull` first to fetch the adapter:

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
* **Interactive Web App:** Open `http://localhost:8000/` in your browser.
* **Interactive OpenAPI Docs:** Visit `http://localhost:8000/docs`.

### 3. Expose Edge Endpoint via Cloudflare Tunnel
To share your local GPU inference securely with remote clients or reviewers:

```powershell
cloudflared tunnel --url http://localhost:8000
```
Copy the generated temporary URL (e.g., `https://example-random-name.trycloudflare.com`) to access the app remotely.

### 4. Run Drift Monitoring
Generate an interactive data drift report comparing reference contracts against production traffic:

```powershell
python drift_monitoring.py
```
Open `drift_report.html` in your browser to inspect 2D semantic density shifts and classifier metrics.

### 5. Reproduce the Results

```powershell
pip install -r requirements-dev.txt
pytest tests -q                                            # data / prompt / metric invariants

dvc pull                                                   # data v3 + v4, training mixes, v4 and v6b adapters (or: python src/download_data.py)
python src/prepare_data.py                                 # rebuild dataset v3 (refuses to overwrite)
python src/finetune.py --config configs/qlora_v4.yaml      # ~3 h on an 8 GB RTX 5070 Laptop GPU

python src/evaluate.py --adapter models/llama-3.1-8b-cuad-lora-v4/final_adapter
python src/compare_results.py reports/eval/baseline_test.json reports/eval/lora_v4_test.json

# realistic full-document evaluation (~2 h of GPU for test + validation, both models)
python src/evaluate_documents.py --adapter models/llama-3.1-8b-cuad-lora-v4/final_adapter
python src/evaluate_documents.py --adapter models/llama-3.1-8b-cuad-lora-v4/final_adapter --split val
python src/document_eval.py --baseline reports/eval/doc_baseline_test.json --system reports/eval/doc_lora_v4_test.json \
    --val-baseline reports/eval/doc_baseline_val.json --val reports/eval/doc_lora_v4_val.json

# v6b (7 clauses): dataset profile v4, training mix, continued training, scoring (all steps are config-driven)
python src/prepare_data.py --profile v4                   # dataset v4 -> data/processed_v4 (v3 untouched)
python src/train_mix.py --profile v4 --synthetic data/synthetic/hand_v1.json --output-dir data/train_mixes/v4h
python run_overnight.py --config configs/qlora_v6b.yaml   # ~3.5 h: continue v6, score val/test, quotes
python study/verify_v6b_api.py                            # the API path reproduces the saved scores exactly (GPU, 5 min)

# recovery Stage 1 (CPU, saved scores): calibration + blend, exports configs/blend_v1.json
python src/calibration_experiments.py --export-blend configs/blend_v1.json

mlflow ui                                                  # runs, configs, data hashes, curves
```

---

## 🐳 Docker Deployment

To build and run the containerized inference service (the image has not yet been built or validated end to end):

```bash
# Build Docker image
docker build -t lexprobe:latest .

# Run with NVIDIA GPU pass-through (HF token passed at runtime, never baked in)
docker run --gpus all -p 8000:8000 -e HF_TOKEN="your_huggingface_token" lexprobe:latest
```

---

## 🔮 Future Roadmap & Next Steps

Planned enhancements:

### 0. Close the remaining target gaps (Liquidated Damages, false alarms, verbatim quotes)
* **Goal:** targets were agreed on 2026-10-18; two of six are met. Next: a clean from-scratch run (`configs/qlora_v7.yaml`), more *real* Liquidated Damages data (CUAD has only 61 contracts with it), a less saturated validation metric for checkpoint choice, and constrained decoding so quotes are always exact substrings.

### 1. Model Context Protocol (MCP) Server Extension
* **Goal:** Package LexProbe's inference logic into an MCP Server (`mcp_server/server.py`).
* **Tool Schema:** Expose tools such as `analyze_contract_clause(text: str, clause_type: str)` and `audit_master_agreement(file_path: str)`.
* **Workflow:** Enables direct invocation by AI-assisted IDEs (Claude Desktop, Cursor, Antigravity) to audit vendor agreements and software licenses directly from a developer's workspace.

### 2. Long-Context Hierarchical Agentic RAG
* **Goal:** Handle 100+ page master service agreements without 15,000-character truncation.
* **Mechanism:** Implement hierarchical chunking (section/sub-clause level) with hybrid vector-keyword retrieval (BM25 + BGE-large) and an agentic cross-clause conflict resolver (e.g., detecting contradictions between an explicit limitation of liability and an uncapped indemnity clause).

### 3. Automated Risk Scoring & Mitigation Suggestions
* **Goal:** Move beyond binary clause detection to actionable risk triage.
* **Mechanism:** Add a multi-tier liability scoring model (Low / Medium / High / Critical) that outputs automated attorney redline suggestions (e.g., "Add mutual 30-day notice requirement to unilateral termination clause").

### 4. Continuous Active Learning & Automated Retraining
* **Goal:** Close the MLOps loop from drift detection to model update.
* **Mechanism:** Connect Evidently AI drift threshold triggers to a webhook that flags out-of-distribution contract types, collects attorney corrections via the UI, and queues automated LoRA adapter retraining jobs.

### 5. Multi-Jurisdiction Legal Taxonomy
* **Goal:** Expand beyond US commercial contracts.
* **Mechanism:** Fine-tune specialized LoRA adapter heads for UK common law, EU GDPR compliance, and Indian Companies Act contractual requirements.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
