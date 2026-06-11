import numpy as np
import torch
from tqdm import tqdm
from transformers import pipeline

INFERENCE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

def get_sentiment_pipeline(model_name=INFERENCE_MODEL):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device,
        truncation=True,
        max_length=512,
    )

def batch_predict(texts, pipe, batch_size=32):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="DistilBERT inference"):
        batch = list(texts[i : i + batch_size])
        results.extend(pipe(batch))
    return results

def parse_pipeline_output(results):
    labels = np.array([1 if r["label"] == "POSITIVE" else 0 for r in results])
    scores = np.array([
        r["score"] if r["label"] == "POSITIVE" else 1 - r["score"]
        for r in results
    ])
    return labels, scores


def get_finetuning_trainer(model_name, train_dataset, val_dataset, output_dir="saved_models/distilbert-imdb"):
    """
    Returns a HuggingFace Trainer ready for fine-tuning.
    Designed to run on Google Colab (T4 GPU). Estimated time: ~20 min / epoch.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
    )
    from sklearn.metrics import accuracy_score, f1_score

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_tok = train_dataset.map(tokenize, batched=True)
    val_tok   = val_dataset.map(tokenize, batched=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1":       f1_score(labels, preds, average="weighted"),
        }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    return trainer, tokenizer
