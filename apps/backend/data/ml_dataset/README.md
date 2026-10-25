# Marauder ML — Training Dataset & Evaluation Documentation

## 1. Overview

This directory contains dataset files and documentation for training and evaluating supervised classification models within **Marauder ML — Resume Matching and Skill Gap Analysis**.

The dataset enables binary classification of candidate resumes against job descriptions, supporting academic evaluation, model baseline comparison, and skill gap discovery.

---

## 2. Dataset Files

| File                   |  Rows   | Positives (`1`) | Negatives (`0`) | Purpose                                                            |
| :--------------------- | :-----: | :-------------: | :-------------: | :----------------------------------------------------------------- |
| `expanded_dataset.csv` | **360** |   184 (51.1%)   |   176 (48.9%)   | Full curated multi-domain dataset (8 domains, 45 pairs each).      |
| `train.csv`            | **288** |   147 (51.0%)   |   141 (49.0%)   | 80% stratified training split used to fit vocabulary and model.    |
| `test.csv`             | **72**  |   37 (51.4%)    |   35 (48.6%)    | 20% held-out test split reserved exclusively for final evaluation. |
| `train_sample.csv`     | **12**  |    6 (50.0%)    |    6 (50.0%)    | Lightweight starter benchmark for fast CI/CD unit testing.         |
| `test_sample.csv`      |  **8**  |    4 (50.0%)    |    4 (50.0%)    | Lightweight starter benchmark for fast CI/CD unit testing.         |

---

## 3. CSV Schema Specification

Every dataset file conforms to standard RFC 4180 CSV formatting:

| Column Name       | Type    | Required | Description                                                                                                                                                                                                                                    | Example                                                                                                     |
| :---------------- | :------ | :------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| `resume_text`     | String  | **Yes**  | Cleaned, anonymized text representation of candidate experience, education, and technical skills. Must not be empty.                                                                                                                           | `"Full Stack Developer with 4 years experience in Python, FastAPI, React, PostgreSQL..."`                   |
| `job_description` | String  | **Yes**  | Full text of target job posting detailing role expectations, technical requirements, and responsibilities. Must not be empty.                                                                                                                  | `"Seeking Senior Backend Engineer proficient in Python, Docker, microservices and relational databases..."` |
| `label`           | Integer | **Yes**  | Binary classification label: <br>• `1`: **Match / Qualified** (Candidate fulfills core domain, skills, and experience criteria)<br>• `0`: **Mismatch / Unqualified** (Candidate lacks critical tools, domain alignment, or required seniority) | `1` or `0`                                                                                                  |
| `category`        | String  | Optional | Target technical domain for subgroup reporting.                                                                                                                                                                                                | `"AI & Machine Learning"`, `"Cloud & DevOps"`                                                               |

---

## 4. Domain Coverage (8 Domains)

The expanded dataset (`expanded_dataset.csv`) contains exactly 45 unique resume–job pairs across each of the following 8 domains:

1. **Artificial Intelligence and Machine Learning** (45 pairs): Deep Learning, NLP, Computer Vision, MLOps, PyTorch, Hugging Face, Reinforcement Learning.
2. **Data Science and Data Analytics** (45 pairs): Exploratory data analysis, statistical modeling, A/B testing, SQL, Pandas, Tableau, Power BI, Scikit-Learn.
3. **Frontend and Backend Web Development** (45 pairs): React, TypeScript, Next.js, Node.js, Go, Java Spring Boot, REST APIs, Microservices.
4. **Python and Software Development** (45 pairs): Core Python, FastAPI, Django, Celery, asyncio, Pytest, automated testing, system tooling.
5. **Cybersecurity** (45 pairs): SOC tier 1/2, penetration testing, SIEM (Splunk), Wireshark, Burp Suite, ISO 27001, GRC, AppSec.
6. **Cloud Computing and DevOps** (45 pairs): Kubernetes, Docker, Terraform, AWS, Azure, GCP, CI/CD, GitHub Actions, Prometheus, SRE.
7. **Database and Business Intelligence** (45 pairs): PostgreSQL, MySQL, Snowflake, dbt, SQL Server, query optimization, ETL, DAX, data modeling.
8. **General IT and Entry-Level Technology Roles** (45 pairs): Desktop support, Active Directory, Jira/ServiceNow, hardware imaging, entry-level QA testing, network cabling.

### Positive vs. Negative Pair Design

- **Positives (`label = 1`)**: Resumes genuinely exhibit the necessary technical stack, domain competence, and level of experience demanded by the job posting. Covers fresher, junior, mid-level, and senior candidate archetypes.
- **Negatives (`label = 0`)**:
  - _Cross-domain mismatches_: Candidate from an unrelated vocational or non-technical background (e.g., Commercial Roofer, Real Estate Agent, Culinary Pastry Chef, Accountant).
  - _Subtle within-domain near-misses_: Candidate possesses technical skills but lacks essential core requirements (e.g., IT Desktop Support applying for Senior Cloud DevOps Engineer; Manual QA tester applying for Deep Learning NLP Specialist).

---

## 5. Data Provenance & Academic Integrity Notice

> [!IMPORTANT]
> **Synthetic Dataset Acknowledgment**:
>
> 1. All records in this dataset are **synthetically generated and reviewed** for educational and demonstration purposes in an undergraduate Machine Learning mini-project.
> 2. **No personal data (PII)**: Resumes do not contain real candidate names, personal phone numbers, emails, or proprietary corporate data.
> 3. **Academic Honesty**: Synthetic benchmarks demonstrate pipeline correctness, model mechanics, and metric calculation; they do **not** claim to reflect real-world recruitment bias, noisy CV layouts, or production hiring distributions.
> 4. **Local Processing**: Model training and evaluation run 100% locally using scikit-learn. No text is sent to third-party commercial LLM APIs.

---

## 6. Train/Test Split & Data Leakage Prevention

- **Split Ratio**: 80% Training (`train.csv`, 288 rows), 20% Held-out Testing (`test.csv`, 72 rows).
- **Stratification**: Class proportions are strictly preserved across both splits (~51% positive, ~49% negative).
- **Leakage Prevention**:
  - The TF-IDF vectorizer vocabulary is fitted **only on the training split**.
  - Held-out test examples are transformed using the pre-fitted vectorizer and are never seen during training.
  - Cross-validation within the training script fits transformations independently inside each fold via `sklearn.pipeline.Pipeline`.

---

## 7. Model Architecture & Parameters

- **Pipeline**: `scikit-learn.pipeline.Pipeline`
  1. `TfidfVectorizer`:
     - Preprocessor: Local whitespace and case cleaner (`clean_text`)
     - Token pattern: Preserves C++, C#, .NET, and alphanumeric tech terms (`(?u)\b[a-zA-Z0-9+#.-]+\b`)
     - Stop words: English stop words removed
     - Sublinear TF scaling: `sublinear_tf=True` ($1 + \log(\text{tf})$)
     - N-gram range: Unigrams and bigrams `(1, 2)`
     - Max features: 5,000 vocabulary tokens
  2. `LogisticRegression`:
     - Regularization: L2 penalty, inverse strength $C = 1.0$
     - Class weight: `"balanced"`
     - Max iterations: 1,000
     - Random seed: 42

---

## 8. Empirical Evaluation Results

The model was trained on `train.csv` (288 rows) and evaluated on the held-out `test.csv` (72 rows). It was compared against a naive `DummyClassifier` baseline predicting the majority class (`strategy="most_frequent"`):

| Metric                  | Supervised Logistic Regression | Dummy Baseline (Most Frequent) |                Empirical Lift                |
| :---------------------- | :----------------------------: | :----------------------------: | :------------------------------------------: |
| **Accuracy**            |       **0.7917 (79.2%)**       |         0.5139 (51.4%)         |             **+0.2778 (+27.8%)**             |
| **Precision (Class 1)** |       **0.7895 (79.0%)**       |         0.5139 (51.4%)         |                   +0.2756                    |
| **Recall (Class 1)**    |       **0.8108 (81.1%)**       |        1.0000 (100.0%)         | -0.1892 (Baseline trivially predicts all 1s) |
| **F1-Score (Class 1)**  |       **0.8000 (80.0%)**       |         0.6789 (67.9%)         |             **+0.1211 (+12.1%)**             |
| **Macro F1-Score**      |       **0.7913 (79.1%)**       |         0.3394 (33.9%)         |             **+0.4519 (+45.2%)**             |
| **ROC-AUC**             |       **0.8479 (84.8%)**       |        0.5000 (Random)         |                 **+0.3479**                  |

### Confusion Matrix on Held-Out Test Set (72 rows)

```
                    Predicted Mismatch (0)    Predicted Match (1)
Actual Mismatch (0)        TN = 27                  FP = 8
Actual Match (1)           FN = 7                   TP = 30
```

- **True Negatives (TN = 27)**: 27 unqualified candidates correctly rejected.
- **True Positives (TP = 30)**: 30 qualified candidates correctly identified.
- **False Positives (FP = 8)**: 8 unqualified candidates misclassified as matches.
- **False Negatives (FN = 7)**: 7 qualified candidates misclassified as mismatches.

### 5-Fold Stratified Cross-Validation on Training Data (288 rows)

- **CV Accuracy**: `0.7258 (+/- 0.0363)`
- **CV Precision**: `0.7174 (+/- 0.0478)`
- **CV Recall**: `0.7756 (+/- 0.0761)`
- **CV F1-Score**: `0.7420 (+/- 0.0348)`
- **CV ROC-AUC**: `0.8380 (+/- 0.0295)`

---

## 9. Exact Windows PowerShell Commands

### Step 1: Validate Dataset

```powershell
& "$HOME\.local\bin\uv.exe" run --directory apps/backend python -m app.ml.data_loader --validate data/ml_dataset/expanded_dataset.csv
```

### Step 2: Generate Stratified Train/Test Split

```powershell
& "$HOME\.local\bin\uv.exe" run --directory apps/backend python -m app.ml.data_loader --split data/ml_dataset/expanded_dataset.csv --train-out data/ml_dataset/train.csv --test-out data/ml_dataset/test.csv --test-size 0.2 --seed 42
```

### Step 3: Train Model with 5-Fold Cross-Validation

```powershell
& "$HOME\.local\bin\uv.exe" run --directory apps/backend python -m app.ml.train --data-path data/ml_dataset/train.csv --output-model data/ml_models/careerfit_logistic_regression.joblib --cv-folds 5
```

### Step 4: Evaluate Model on Held-Out Test Data with Baseline Comparison

```powershell
& "$HOME\.local\bin\uv.exe" run --directory apps/backend python -m app.ml.evaluate --model-path data/ml_models/careerfit_logistic_regression.joblib --test-data data/ml_dataset/test.csv --train-data data/ml_dataset/train.csv --output-dir data/ml_evaluation
```

### Step 5: Run Automated ML and Integration Test Suite

```powershell
& "$HOME\.local\bin\uv.exe" run --directory apps/backend pytest tests/unit/test_ml_*.py tests/integration/test_careerfit_ml_api.py
```

---

## 10. Evaluation Reports & Artifact Output Locations

- **Model Binary**: `apps/backend/data/ml_models/careerfit_logistic_regression.joblib`
- **Training Metadata**: `apps/backend/data/ml_models/training_metadata.json`
- **Machine-Readable Evaluation JSON**: `apps/backend/data/ml_evaluation/evaluation_report.json`
- **Human-Readable Evaluation Markdown**: `apps/backend/data/ml_evaluation/evaluation_report.md`
