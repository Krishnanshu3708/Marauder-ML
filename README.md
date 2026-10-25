<div align="center">

![Marauder ML](assets/header.webp)

# Marauder ML

### *I Solemnly Swear I Am Qualified*
**Intelligent Resume Matching, TF-IDF Text Similarity & Skill Gap Analysis Engine**

[![Machine Learning](https://img.shields.io/badge/ML-Supervised_Classification-blue.svg?style=for-the-badge)](#machine-learning-architecture--evaluation)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](#tech-stack)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](#tech-stack)
[![Next.js 16](https://img.shields.io/badge/Next.js_16-black?style=for-the-badge&logo=next.js&logoColor=white)](#tech-stack)

A self-hosted resume intelligence platform and Machine Learning harness featuring supervised candidate qualification classification, TF-IDF cosine similarity, automated skill gap discovery, deterministic ATS scoring, and LLM-powered resume tailoring.

![Marauder ML Demo](assets/Resume_Matcher_Demo_2.gif)

</div>

<br>

## Table of Contents
- [Overview](#overview)
- [Machine Learning Architecture & Evaluation](#machine-learning-architecture--evaluation)
  - [Dataset Specification](#dataset-specification)
  - [Empirical Evaluation Results](#empirical-evaluation-results)
  - [Confusion Matrix & Cross-Validation](#confusion-matrix--cross-validation)
- [Marauder ML Studio](#marauder-ml-studio)
- [Core Application Features](#core-application-features)
  - [Resume Builder & Multi-Template PDF Export](#resume-builder--multi-template-pdf-export)
  - [Deterministic ATS Score & Parsing Check](#deterministic-ats-score--parsing-check)
- [Quick Start & Installation](#quick-start--installation)
- [Machine Learning CLI Commands](#machine-learning-cli-commands)
- [Supported AI Providers](#supported-ai-providers)
- [Docker Deployment](#docker-deployment)
- [Tech Stack](#tech-stack)

---

## Overview

**Marauder ML** is an end-to-end Machine Learning and resume intelligence system designed to evaluate candidate-job alignment, discover critical skill gaps, and tailor resumes to specific job descriptions. 

Combining local supervised classification models (Scikit-Learn) with an interactive Next.js web application, Marauder ML runs 100% locally and privately on your own machine.

> **Detailed Academic Report**: See [**`ML_DOCUMENTATION.md`**](ML_DOCUMENTATION.md) for full dataset schemas, domain breakdowns, and pipeline implementation details.

---

## Machine Learning Architecture & Evaluation

The core Machine Learning subsystem performs binary qualification classification and semantic similarity scoring between candidate resumes and job descriptions.

### Dataset Specification

The curated dataset (`apps/backend/data/ml_dataset/expanded_dataset.csv`) contains **360 balanced resume–job description pairs** across 8 distinct technical domains (45 pairs each):
1. **Artificial Intelligence & Machine Learning**
2. **Data Science & Data Analytics**
3. **Frontend & Backend Web Development**
4. **Python & Core Software Development**
5. **Cybersecurity & Information Security**
6. **Cloud Computing & DevOps / SRE**
7. **Database Engineering & Business Intelligence**
8. **General IT & Technical Support**

- **Train/Test Split**: Stratified 80% Training (288 rows) / 20% Held-Out Testing (72 rows).
- **Leakage Prevention**: TF-IDF vocabulary is fitted strictly on the training set and independently transformed inside each cross-validation fold via `sklearn.pipeline.Pipeline`.

### Empirical Evaluation Results

The supervised Logistic Regression model was trained on `train.csv` (288 rows) and tested on the held-out `test.csv` (72 rows), compared against a naive majority-class `DummyClassifier` baseline:

| Metric | Supervised Logistic Regression | Dummy Baseline (Most Frequent) | Empirical Lift |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.7917 (79.2%)** | 0.5139 (51.4%) | **+0.2778 (+27.8%)** |
| **Precision (Class 1)** | **0.7895 (79.0%)** | 0.5139 (51.4%) | **+0.2756** |
| **Recall (Class 1)** | **0.8108 (81.1%)** | 1.0000 (100.0%) | -0.1892 (Baseline trivially predicts all 1s) |
| **F1-Score (Class 1)** | **0.8000 (80.0%)** | 0.6789 (67.9%) | **+0.1211 (+12.1%)** |
| **Macro F1-Score** | **0.7913 (79.1%)** | 0.3394 (33.9%) | **+0.4519 (+45.2%)** |
| **ROC-AUC** | **0.8479 (84.8%)** | 0.5000 (Random) | **+0.3479** |

### Confusion Matrix & Cross-Validation

#### Confusion Matrix on Held-Out Test Set (72 rows)
```
                    Predicted Mismatch (0)    Predicted Match (1)
Actual Mismatch (0)        TN = 27                  FP = 8
Actual Match (1)           FN = 7                   TP = 30
```
- **True Negatives (TN = 27)**: Unqualified candidates correctly classified.
- **True Positives (TP = 30)**: Qualified candidates correctly identified.
- **False Positives (FP = 8)**: Unqualified candidates misclassified as matches.
- **False Negatives (FN = 7)**: Qualified candidates misclassified as mismatches.

#### 5-Fold Stratified Cross-Validation on Training Set (288 rows)
- **CV Accuracy**: `0.7258 (+/- 0.0363)`
- **CV Precision**: `0.7174 (+/- 0.0478)`
- **CV Recall**: `0.7756 (+/- 0.0761)`
- **CV F1-Score**: `0.7420 (+/- 0.0348)`
- **CV ROC-AUC**: `0.8380 (+/- 0.0295)`

---

## Marauder ML Studio

The **Marauder ML Studio** interface (`/careerfit-ml`) provides an interactive environment for real-time inference and analysis:

- **Similarity Scoring**: Computes TF-IDF Cosine Similarity score reflecting vocabulary and contextual overlap.
- **Role Qualification Inference**: Predicts binary qualification (`Qualified Match` vs. `Skill Mismatch`) with model confidence probability.
- **Skill Gap Discovery Engine**: Automatically compares candidate skills against target requirements:
  - **Matched Skills**: Technical keywords present in both resume and job description.
  - **Missing Target Competencies**: High-priority keywords from the job description missing from candidate experience.
  - **Candidate Unique Strengths**: Differentiating technologies present on the resume beyond standard job requirements.

---

## Core Application Features

### Master Resume & Tailoring Dashboard

Maintain a comprehensive master resume containing your full experience, projects, and skills.

![Dashboard with a master resume and tailored resumes](assets/step_2.webp)

### Resume Builder & Multi-Template PDF Export

Tailor resumes for individual job applications with real-time preview and export:

![Resume Builder with template picker and live preview](assets/step_5.webp)

| Template Name           | Preview                                                           | Description                                                                                                                      |
| ----------------------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Single Column**       | ![Single Column](assets/pdf-templates/single-column.webp)         | Traditional full-width layout with maximum content density. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/single-column.pdf)                   |
| **Two Column**          | ![Two Column](assets/pdf-templates/two-column.webp)               | Experience-focused main column with sidebar for skills. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/two-column.pdf)                          |
| **Modern**              | ![Modern](assets/pdf-templates/modern-single-column.webp)         | Colorful accents with customizable theme colors. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/modern-single-column.pdf)                       |
| **Modern Two Column**   | ![Modern Two Column](assets/pdf-templates/modern-two-column.webp) | Two-column layout with modern colorful accents and themes. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/modern-two-column.pdf)                |
| **LaTeX**               | ![LaTeX](assets/pdf-templates/latex.webp)                         | Classic serif academic layout with ruled section headers. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/latex.pdf)                             |
| **Clean**               | ![Clean](assets/pdf-templates/clean.webp)                         | Minimal sans layout with large understated section headers. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/clean.pdf)                           |
| **Vivid**               | ![Vivid](assets/pdf-templates/vivid.webp)                         | Colorful two-column layout with accent headers and arrow bullets. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/vivid.pdf)                     |
| **Vivid Single Column** | ![Vivid Single Column](assets/pdf-templates/vivid-single.webp)    | Colorful single-column layout with accent headers and arrow bullets. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/vivid-single.pdf)           |
| **Modern Deedy**        | ![Modern Deedy](assets/pdf-templates/modern-deedy.webp)           | Dense one-page layout in Lato with light grey uppercase section headings. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/modern-deedy.pdf)      |
| **Deedy**               | ![Deedy](assets/pdf-templates/deedy.webp)                         | Two-column layout with a hairline name, a narrow left column and light gray headings. [𝐕𝐢𝐞𝐰 𝐏𝐃𝐅](assets/pdf-templates/deedy.pdf) |

### Deterministic ATS Score & Parsing Check

- **ATS Score (0–100)**: Evaluates keyword match (40%), skills coverage (20%), job-title match (15%), section completeness (15%), and date consistency (10%).
- **ATS Check**: Runs PDF through pdfminer to verify machine parseability, reading order, and heading structure without making external API calls.

![JD Match tab with ATS score breakdown and keyword highlighting](assets/keyword_highlighter.webp)

---

## Quick Start & Installation

For full configuration guidelines, refer to **[SETUP.md](SETUP.md)**.

### Prerequisites

| Tool    | Minimum Version              | Purpose                                   |
| ------- | ---------------------------- | ----------------------------------------- |
| Python  | 3.13+                        | FastAPI backend, ML pipeline & ATS parser |
| Node.js | 24.22.1 (pinned in `.nvmrc`) | Next.js 16 frontend                       |
| uv      | Latest                       | Python package & environment manager      |

### Setup Commands

```bash
# 1. Backend Service (Terminal 1)
cd apps/backend
cp .env.example .env        # Configure local settings
uv sync                      # Install dependencies
uv run app

# 2. Frontend Application (Terminal 2)
cd apps/frontend
npm install
npm run dev
```

### Access Endpoints

- **Marauder ML Studio**: <http://localhost:3000/careerfit-ml>
- **Resume Dashboard**: <http://localhost:3000/dashboard>
- **API Documentation**: <http://localhost:8000/docs>
- **API Health Check**: <http://localhost:8000/api/v1/health>

---

## Machine Learning CLI Commands

The ML pipeline can be trained, tested, and evaluated directly from the command line:

```powershell
# Validate dataset integrity and class balance
uv run --directory apps/backend python -m app.ml.data_loader --validate data/ml_dataset/expanded_dataset.csv

# Train supervised Logistic Regression model with 5-fold cross-validation
uv run --directory apps/backend python -m app.ml.train --data-path data/ml_dataset/train.csv --output-model data/ml_models/careerfit_logistic_regression.joblib --cv-folds 5

# Evaluate trained model on held-out test split against Dummy baseline
uv run --directory apps/backend python -m app.ml.evaluate --model-path data/ml_models/careerfit_logistic_regression.joblib --test-data data/ml_dataset/test.csv --train-data data/ml_dataset/train.csv --output-dir data/ml_evaluation

# Run ML unit and integration test suite
uv run --directory apps/backend pytest tests/unit/test_ml_*.py tests/integration/test_careerfit_ml_api.py
```

---

## Supported AI Providers

Marauder ML supports both local (free, offline) and cloud AI backends for resume tailoring:

| Provider                                  | Mode                  | Details                                                                                                     |
| ----------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Ollama**                                | Local                 | Free, runs fully on your machine with zero external network transmission.                                   |
| **Your coding agent**                     | Local                 | Works with Claude Code, Cursor, or Copilot via MCP with no API key needed.                                  |
| **Claude subscription (via Claude Code)** | Cloud (via local CLI) | Uses installed `claude` CLI authentication without storing credentials in the app.                          |
| **ChatGPT plan (Sign in with ChatGPT)**   | Cloud                 | Uses OpenAI token-sharing with ChatGPT Plus/Pro; no manual API key required.                                |
| **OpenAI**                                | Cloud                 | GPT-4o, GPT-5 Nano with an API key.                                                                         |
| **Anthropic**                             | Cloud                 | Claude 3.5 Sonnet, Claude Haiku 4.5 with an API key.                                                        |
| **Google Gemini**                         | Cloud                 | Gemini 2.0 / 3.0 Flash with an API key.                                                                     |
| **DeepSeek**                              | Cloud                 | DeepSeek Chat / Reasoner with an API key.                                                                   |
| **Groq**                                  | Cloud                 | High-speed inference for Llama 3.3 70B with an API key.                                                      |
| **OpenRouter**                            | Cloud                 | Unified access to multiple open and proprietary LLM architectures.                                          |

---

## Docker Deployment

Build and run with Docker on a single port (`3000`):

```bash
# Build Docker image
docker build -t marauder-ml .

# Run container
docker run --name marauder-ml \
  -p 3000:3000 \
  -v marauder-data:/app/backend/data \
  marauder-ml:latest
```

---

## Tech Stack

| Component         | Technologies                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------- |
| **Machine Learning Engine** | Scikit-Learn, NumPy, TF-IDF Vectorizer, Logistic Regression, 5-Fold Stratified Cross-Validation |
| **Backend API**   | FastAPI, Python 3.13+, Pydantic v2, LiteLLM, SQLAlchemy 2.0                                 |
| **Frontend UI**   | Next.js 16 (App Router), React 19, TypeScript, Phosphor Icons                               |
| **Styling**       | Tailwind CSS 4, Swiss Design System, Custom Marauder ML Theme                               |
| **Database**      | SQLite (default local file storage), optional PostgreSQL                                    |
| **PDF Rendering** | Headless Chromium via Playwright, pdfminer text extraction                                  |

---

## Contributing

Contributions, enhancements, and feature suggestions are welcome! Feel free to open an issue or submit a pull request.
