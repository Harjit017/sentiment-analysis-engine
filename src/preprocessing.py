import re
import pandas as pd

CONTRACTION_MAP = {
    "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "isn't": "is not", "it's": "it is", "i'm": "i am",
    "i've": "i have", "i'll": "i will", "i'd": "i would",
    "shouldn't": "should not", "wasn't": "was not", "weren't": "were not",
    "won't": "will not", "wouldn't": "would not", "you're": "you are",
    "you've": "you have", "you'll": "you will", "you'd": "you would",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "that's": "that is", "there's": "there is", "what's": "what is",
    "let's": "let us", "who's": "who is", "he's": "he is", "she's": "she is",
}

_contraction_pattern = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in CONTRACTION_MAP) + r')\b',
    re.IGNORECASE
)

def _expand_contractions(text):
    return _contraction_pattern.sub(lambda m: CONTRACTION_MAP[m.group().lower()], text)

def clean_text(text):
    text = text.lower()
    text = _expand_contractions(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_imdb_data():
    from datasets import load_dataset
    dataset = load_dataset("imdb")

    train_df = pd.DataFrame({"text": dataset["train"]["text"], "label": dataset["train"]["label"]})
    test_df  = pd.DataFrame({"text": dataset["test"]["text"],  "label": dataset["test"]["label"]})

    print("Cleaning train set...")
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    print("Cleaning test set...")
    test_df["clean_text"]  = test_df["text"].apply(clean_text)

    return train_df, test_df

def get_sample(df, n=5000, random_state=42):
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)
