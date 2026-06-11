import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import load_imdb_data
from src.models import MODELS
from src.evaluate import compute_metrics, print_report, confusion_matrix_df, results_table

def main():
    print("Loading IMDb dataset...")
    train_df, test_df = load_imdb_data()

    X_train = train_df["clean_text"]
    y_train = train_df["label"]
    X_test  = test_df["clean_text"]
    y_test  = test_df["label"]

    all_metrics = {}

    for name, builder in MODELS.items():
        print(f"\nTraining: {name}")
        model = builder()

        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"  Train time: {elapsed:.1f}s")

        y_pred = model.predict(X_test)

        y_prob = None
        if hasattr(model.named_steps["clf"], "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]

        print_report(name, y_test, y_pred)
        print(confusion_matrix_df(y_test, y_pred).to_string())

        all_metrics[name] = compute_metrics(y_test, y_pred, y_prob)

    print("\n\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(results_table(all_metrics).to_string())

if __name__ == "__main__":
    main()
