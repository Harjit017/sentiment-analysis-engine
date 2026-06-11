# Dataset

This project uses the **IMDb Large Movie Review Dataset**.

The dataset is automatically downloaded via the HuggingFace `datasets` library when you run any notebook or script. No manual download needed.

```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

| Split | Samples | Positive | Negative |
|-------|---------|----------|----------|
| Train | 25,000 | 12,500 | 12,500 |
| Test | 25,000 | 12,500 | 12,500 |

> Raw data files (`.csv`, `.parquet`) are excluded from version control via `.gitignore`.
