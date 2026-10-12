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

LexProbe fine-tunes **Llama-3.1-8B-Instruct** with **4-bit QLoRA on a single 8 GB laptop GPU**. On a frozen, leak-free test set it raises macro-F1 from **0.875 (zero-shot) to 0.960** (+0.085, 95% CI +0.051…+0.149, p < 0.001), cuts false alarms from **62 to 2**, and every quoted citation (254/254) is verbatim from the contract. The model is served through a **FastAPI** microservice behind **Cloudflare Tunnel**, with **MLflow** experiment tracking, **DVC** data versioning and **Evidently AI** drift monitoring.

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

Off-the-shelf zero-shot prompting of a general model is a reasonable start, but on our leak-free test set the zero-shot `Llama-3.1-8B-Instruct` baseline raises **62 false alarms on 536 clause-free excerpts** (macro precision 0.833). In a review queue, every false alarm costs reviewer time.

### The Objective
1. **Accurate first-pass triage:** fine-tune `Llama-3.1-8B-Instruct` to decide whether each of 6 high-risk clause types is present in a contract excerpt, and quote the supporting text.
2. **Consumer hardware:** train with 4-bit QLoRA on a single **8 GB laptop GPU** (RTX 5070 Laptop).
3. **Measurement you can defend:** leak-free splits by near-duplicate contract group, a frozen DVC-versioned test set, paired significance testing, and verification that quoted evidence actually appears in the source.
4. **Production path:** FastAPI serving, Docker, Cloudflare Tunnel, MLflow tracking and Evidently AI drift monitoring.

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
  │                     EVALUATION (frozen test split, 804 queries)                  │
  │                                                                                  │
  │   Baseline = same 4-bit model with the adapter disabled (identical inputs)       │
  │   Macro-F1 0.875 → 0.960 (+0.085, 95% CI +0.051…+0.149, p < 0.001)               │
  │   False positives 62 → 2 · total errors 74 → 18                                   │
  │   Quotes verbatim in source: 71.4% → 100% · span-F1 vs gold 0.790 → 0.911        │
  └──────────────────────────────────────────┬───────────────────────────────────────┘
                                             ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │   SERVING & MONITORING: FastAPI (/analyze-contract, /upload: PDF/DOCX/TXT),      │
  │   Docker, Cloudflare Tunnel, Evidently AI embedding-drift report                 │
  └──────────────────────────────────────────────────────────────────────────────────┘
```

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
* **Fitting 8 GB:** keeping embeddings and LM head in bf16 (instead of the usual fp32 upcast) and using a non-paged 8-bit optimizer took training from spilling into system RAM to **1.12 s/step at 512 tokens (vs 4.95 s), a 3–4× speed-up**. It also removed a paged-optimizer checkpoint crash.
* **Model selection:** decision macro-F1 on a fixed validation subset every 100 steps; the best checkpoint (step 500, val F1 0.947) is kept.
* **Reproducibility:** each run logs its config, dataset hashes and git commit to MLflow and writes `training_summary.json` next to the adapter.

### Step 3: Evaluation (`src/evaluate.py`, `src/compare_results.py`)
* **Fair baseline:** the same 4-bit model with the adapter **disabled**, same prompts, same inputs.
* **Decision:** first-token logit of `Found` vs `None`, one forward pass per query.
* **Statistics:** paired bootstrap over contract groups (2,000 draws) for macro-F1 CIs and the difference, plus an exact McNemar test.
* **Evidence:** for predicted positives the model generates the quote; we check it appears in the source chunk and score span-overlap F1 against the gold clause.

### Step 4: Serving Layer & Application Security
* **FastAPI service:** `/analyze-contract` (JSON) and `/upload` (PDF via `pdfplumber`, DOCX via `python-docx`, TXT).
* **Guardrails:** 2 MB upload limit (HTTP 413), 15,000-character text limit, Pydantic validation (`min_length=50`).
* **UI:** dark-mode web app (`api/static/index.html`) with per-clause badges and evidence cards.
* *Migration in progress:* the API still loads the earlier v2 adapter. Switching it to v4 with chunked full-document analysis is the next step.

### Step 5: Public Edge Routing (Cloudflare Tunnel)
* `cloudflared tunnel --url http://localhost:8000` exposes the local service over HTTPS through an outbound-only tunnel, with no public IP or open inbound ports.

### Step 6: Drift Monitoring (Evidently AI)
* Embeds documents with `sentence-transformers/all-MiniLM-L6-v2` and runs Evidently's domain-classifier drift test between reference contracts and a batch of incoming documents, producing `drift_report.html`.

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

### Key Takeaways
1. **Fine-tuning mostly buys precision.** Zero-shot Llama already finds most clauses (recall 0.931) but over-flags them. v4 keeps recall (0.934) and cuts false alarms from 62 to 2, which matters most for a reviewer's queue.
2. **The trade-off is honest and visible:** v4 misses 4 more clauses than the baseline (16 vs 12): Termination for Convenience 6, Non-Compete 5, Limitation of Liability 5. In a high-recall deployment the Found/None logit margin can be thresholded lower.
3. **The data pipeline was the biggest lever.** An earlier iteration showed the model only the first 600 characters of 8,000-character chunks, so the evidence was visible for just 4% of training positives. Fixing it made the task well-posed for both models. Results from that iteration (macro-F1 0.574 → 0.616) are retired and not comparable.

---

## 📡 Production Data Drift Monitoring

Legal NLP services deployed to production encounter distribution shifts: users may accidentally upload employee resumes, financial tables, medical records, or IT service tickets into a contract analysis endpoint.

LexProbe implements a scheduled **batch distribution monitor**:
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense semantic vectors).
* **Reference Distribution:** 100 authentic commercial contracts sampled from the CUAD test distribution.
* **Simulated Production Drift Batch:** 50 authentic legal contracts mixed with 50 out-of-domain anomaly documents (resumes, sports articles, IT manuals, culinary recipes, patient discharge notes).
* **Detection Engine:** Evidently AI fits a Gradient Boosted Tree classifier across 5-fold cross-validation to assess domain separability.
* **Results & Thresholds:**
  * Identical distributions (no drift) yield a classifier ROC-AUC $\approx 0.50$ (random guessing).
  * Our drift run achieved an **ROC-AUC of 0.739**, clearly surpassing the warning threshold ($0.55$) and triggering an automated alert.
  * Visual inspection is available in the generated `drift_report.html` interactive dashboard.

---

## 📂 Repository Structure

```
LexProbe/
├── api/
│   ├── extractor.py               # Multi-format parser (PDF via pdfplumber, DOCX, TXT)
│   ├── main.py                    # FastAPI service (/, /health, /analyze-contract, /upload)
│   ├── model.py                   # 4-bit base model + LoRA adapter loader and inference
│   └── static/index.html          # Dark-mode UI with per-clause evidence cards
├── configs/
│   ├── qlora_v4.yaml              # Training config for the current adapter
│   ├── qlora_v2_historical.yaml   # Record of the earlier (retired) v2 setup
│   └── smoke_test.yaml            # 20-step pipeline smoke test
├── data/
│   ├── raw/CUAD_v1.json.dvc       # DVC pointer to the raw CUAD dataset
│   └── processed.dvc              # DVC pointer to dataset v3 (chunks, queries, manifest)
├── src/
│   ├── download_data.py           # CUAD v1 download
│   ├── chunking.py                # Whitespace normalisation (offset-preserving) + chunking
│   ├── prepare_data.py            # Near-duplicate group split, labelling, sampling, manifest
│   ├── prompting.py               # Single prompt builder (official Llama-3.1 chat template)
│   ├── training_data.py           # Completion-only tokenisation, token-budget checks, val metrics
│   ├── finetune.py                # Config-driven QLoRA training with MLflow provenance
│   ├── evaluate.py                # Test evaluation: baseline (adapter disabled) vs fine-tuned
│   ├── eval_metrics.py            # P/R/F1, evidence grounding, span F1, paired bootstrap, McNemar
│   ├── compare_results.py         # Significance report → reports/eval/comparison.md
│   └── mlflow_cleanup.py          # Marks orphaned MLflow runs
├── tests/                         # pytest suite (data invariants, prompts, labels, metrics)
├── reports/eval/                  # Test-set results and comparison report
├── drift_monitoring.py            # Evidently AI embedding drift report
├── Dockerfile / .dockerignore     # Container definition (secrets passed at runtime)
└── requirements.txt
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and set up a virtual environment:

```powershell
git clone https://github.com/<your-username>/LexProbe.git
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
Launch the server (loads the 4-bit QLoRA adapter on GPU at startup):

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

dvc pull                                                   # or: python src/download_data.py
python src/prepare_data.py                                 # rebuild dataset v3 (refuses to overwrite)
python src/finetune.py --config configs/qlora_v4.yaml      # ~3 h on an 8 GB RTX 5070 Laptop GPU

python src/evaluate.py --adapter models/llama-3.1-8b-cuad-lora-v4/final_adapter
python src/compare_results.py reports/eval/baseline_test.json reports/eval/lora_v4_test.json

mlflow ui                                                  # runs, configs, data hashes, curves
```

---

## 🐳 Docker Deployment

To build and run the containerized inference service:

```bash
# Build Docker image
docker build -t lexprobe:latest .

# Run with NVIDIA GPU pass-through (HF token passed at runtime, never baked in)
docker run --gpus all -p 8000:8000 -e HF_TOKEN="your_huggingface_token" lexprobe:latest
```

---

## 🔮 Future Roadmap & Next Steps

Planned enhancements:

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
