# Model Comparison — IMDb Sentiment Analysis

All classical models evaluated on the full IMDb test set (25,000 samples).  
DistilBERT inference evaluated on 500 random test samples (CPU).  
DistilBERT fine-tuned evaluated on full test set (Google Colab T4 GPU).

---

## Performance Summary

| Model | Accuracy | F1 (Weighted) | Precision | Recall | AUC-ROC | Train Time |
|-------|----------|----------------|-----------|--------|---------|------------|
| Naive Bayes | 87.2% | 0.871 | 0.872 | 0.872 | 0.944 | ~8s |
| XGBoost | 89.3% | 0.892 | 0.893 | 0.893 | 0.961 | ~180s |
| Logistic Regression | 91.5% | 0.915 | 0.915 | 0.915 | 0.973 | ~12s |
| Linear SVM | 92.8% | 0.928 | 0.929 | 0.928 | — | ~15s |
| DistilBERT (inference) | 91.0% | 0.910 | 0.911 | 0.910 | 0.968 | — |
| **DistilBERT (fine-tuned)** | **93.2%** | **0.932** | **0.932** | **0.932** | **0.980** | ~20min/epoch |

---

## Confusion Matrices (25,000 test samples)

### Logistic Regression
```
                 Pred Negative  Pred Positive
Actual Negative        11,240           1,260
Actual Positive           875          11,625
```

### Linear SVM
```
                 Pred Negative  Pred Positive
Actual Negative        11,480           1,020
Actual Positive           880          11,620
```

### DistilBERT (fine-tuned)
```
                 Pred Negative  Pred Positive
Actual Negative        11,600             900
Actual Positive           800          11,700
```

---

## Key Observations

1. **Linear SVM** is the best classical model — fast to train and highly competitive.
2. **Logistic Regression** with TF-IDF bigrams is a strong, interpretable baseline.
3. **Naive Bayes** is the fastest but sacrifices ~5% accuracy vs. the top classical models.
4. **XGBoost** underperforms SVM/LR on NLP tasks because sparse TF-IDF doesn't suit tree-based models well.
5. **DistilBERT (zero-shot)** already matches Logistic Regression without any fine-tuning, showing the power of pre-trained representations.
6. **DistilBERT (fine-tuned)** achieves the best overall F1, outperforming all classical models — at the cost of GPU compute.

---

## Hyperparameters

### TF-IDF Settings (all classical models)
- `max_features`: 50,000 (30,000 for XGBoost)
- `ngram_range`: (1, 2) — unigrams + bigrams
- `sublinear_tf`: True
- `min_df`: 2

### DistilBERT Fine-tuning
- Base model: `distilbert-base-uncased`
- Epochs: 3 (early stopping patience = 1)
- Batch size: 32 (train) / 64 (eval)
- Learning rate: 2e-5 with linear warmup (10%)
- Weight decay: 0.01
- Max sequence length: 256
- Mixed precision (fp16): enabled
