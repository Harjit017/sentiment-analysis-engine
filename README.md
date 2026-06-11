# 🎭 Sentiment Analysis Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

End-to-end sentiment analysis on the IMDb Movie Reviews dataset. Compares classical ML models (TF-IDF + Logistic Regression, Naive Bayes, SVM, XGBoost) against transformer-based models (DistilBERT). Fine-tuning notebook is designed to run on **Google Colab (free T4 GPU)**.

---

## 📊 Results

| Model | Accuracy | F1 (Weighted) | Precision | Recall | AUC-ROC |
|-------|----------|----------------|-----------|--------|---------|
| Naive Bayes | 87.2% | 0.871 | 0.872 | 0.872 | 0.944 |
| XGBoost | 89.3% | 0.892 | 0.893 | 0.893 | 0.961 |
| Logistic Regression | 91.5% | 0.915 | 0.915 | 0.915 | 0.973 |
| Linear SVM | 92.8% | 0.928 | 0.929 | 0.928 | — |
| DistilBERT (inference) | 91.0% | 0.910 | 0.911 | 0.910 | 0.968 |
| **DistilBERT (fine-tuned)** | **93.2%** | **0.932** | **0.932** | **0.932** | **0.980** |

> Fine-tuned results produced on Google Colab T4 GPU. See `notebooks/04_FineTuning_Colab.ipynb`.

---

## 📁 Project Structure

```
sentiment-analysis-engine/
│
├── src/
│   ├── preprocessing.py       # Text cleaning pipeline
│   ├── models.py              # Classical ML model builders
│   ├── evaluate.py            # Metrics and reporting
│   └── bert_utils.py          # HuggingFace inference + fine-tuning helpers
│
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Classical_ML_Pipeline.ipynb
│   ├── 03_DistilBERT_Inference.ipynb
│   └── 04_FineTuning_Colab.ipynb        ← Run on Google Colab
│
├── scripts/
│   ├── train_classical.py     # Train + evaluate all classical models
│   └── bert_inference.py      # Run DistilBERT inference pipeline
│
├── results/
│   └── model_comparison.md    # Detailed metrics + confusion matrix analysis
│
├── data/
│   └── README.md              # Dataset download instructions
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-engine
cd sentiment-analysis-engine
pip install -r requirements.txt
```

> **No GPU required** for notebooks 01–03. Notebook 04 (fine-tuning) runs on free Google Colab.

---

## 🚀 Quick Start

**Train all classical models and print metrics:**
```bash
python scripts/train_classical.py
```

**Run DistilBERT inference on test samples:**
```bash
python scripts/bert_inference.py
```

**Run notebooks in order** (Jupyter / VS Code / Colab):
```
01 → EDA → 02 → Classical ML → 03 → DistilBERT Inference → 04 (Colab) → Fine-tuning
```

---

## 🔑 Key Concepts Covered

- **Text Preprocessing**: HTML stripping, contraction expansion, lowercasing, regex cleaning
- **Feature Engineering**: TF-IDF with unigrams + bigrams, sublinear TF scaling
- **Classical Models**: Logistic Regression, Naive Bayes, Linear SVM, XGBoost
- **Evaluation**: Accuracy, F1 (macro/weighted), Precision, Recall, AUC-ROC, Confusion Matrix
- **Transformers**: DistilBERT zero-shot inference via HuggingFace pipeline
- **Fine-tuning**: DistilBERT on IMDb with AdamW, linear LR warmup, mixed-precision
- **Regularization**: Dropout (p=0.3), L2 weight decay, early stopping

