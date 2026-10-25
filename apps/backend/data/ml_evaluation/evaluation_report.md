# CareerFit Logistic Regression — Evaluation Report

**Evaluated Dataset**: `test.csv`  
**Total Test Samples**: `72` (Positive Matches: `37`, Mismatches: `35`)  

---

## 1. Quantitative Performance Summary

| Metric | Logistic Regression | Dummy Baseline (Most Frequent) | Improvement (Lift) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.7917** (79.2%) | 0.5139 (51.4%) | **+0.2778** |
| **Precision (Class 1)** | **0.7895** | 0.5139 | +0.2756 |
| **Recall (Class 1)** | **0.8108** | 1.0000 | +-0.1892 |
| **F1-Score (Class 1)** | **0.8000** | 0.6789 | **+0.1211** |
| **Macro F1-Score** | **0.7913** | 0.3394 | **+0.4519** |
| **ROC-AUC** | **0.8479** | 0.5000 (Random guess) | +0.3479 |

---

## 2. Confusion Matrix

```
                    Predicted Mismatch (0)    Predicted Match (1)
Actual Mismatch (0)        TN = 27                  FP = 8    
Actual Match (1)           FN = 7                   TP = 30   
```

- **True Negatives (TN)**: 27 correctly identified as mismatches.
- **True Positives (TP)**: 30 correctly identified as qualified matches.
- **False Positives (FP)**: 8 mismatches incorrectly classified as matches.
- **False Negatives (FN)**: 7 qualified matches missed by the classifier.

---

## 3. Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Mismatch (0) | 0.7941 | 0.7714 | 0.7826 | 35.0 |
| Match (1) | 0.7895 | 0.8108 | 0.8000 | 37.0 |

---

## 4. Academic Evaluation Notes & Methodology

1. **Held-Out Test Set**: Evaluation was performed strictly on held-out test data that was never used during TF-IDF vocabulary fitting or model training, guaranteeing **zero data leakage**.
2. **Baseline Comparison**: The DummyClassifier baseline predicts the majority class (`most_frequent`). The supervised Logistic Regression pipeline demonstrates strong empirical lift over this naive baseline.
3. **Distinction of Metrics**: Cosine similarity is an unsupervised geometric term overlap measure (0–100%), whereas Logistic Regression outputs a calibrated classification probability based on learned feature weights.
4. **Synthetic Data Disclaimer**: While curated across 8 technical domains with realistic candidate and job descriptions, synthetic datasets may not fully capture noisy real-world applicant distributions.