import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import load_imdb_data, get_sample
from src.bert_utils import get_sentiment_pipeline, batch_predict, parse_pipeline_output
from src.evaluate import compute_metrics, print_report, confusion_matrix_df

SAMPLE_SIZE = 500

def main():
    print("Loading IMDb dataset...")
    _, test_df = load_imdb_data()
    sample = get_sample(test_df, n=SAMPLE_SIZE)

    print(f"\nRunning DistilBERT inference on {SAMPLE_SIZE} samples...")
    pipe = get_sentiment_pipeline()

    results = batch_predict(sample["text"].tolist(), pipe, batch_size=32)
    y_pred, y_scores = parse_pipeline_output(results)
    y_true = sample["label"].values

    print_report("DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)", y_true, y_pred)
    print(confusion_matrix_df(y_true, y_pred).to_string())

    metrics = compute_metrics(y_true, y_pred, y_scores)
    print("\nMetrics:", metrics)

    print("\n--- Sample Predictions ---")
    for i in range(5):
        label = "POSITIVE" if y_pred[i] == 1 else "NEGATIVE"
        true  = "POSITIVE" if y_true[i] == 1 else "NEGATIVE"
        print(f"\nReview : {sample['text'].iloc[i][:120]}...")
        print(f"  True : {true}  |  Predicted : {label}  (confidence: {y_scores[i]:.3f})")

if __name__ == "__main__":
    main()
