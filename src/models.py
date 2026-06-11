from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

_tfidf_base = dict(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)

def build_logistic_regression():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_tfidf_base)),
        ("clf",   LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])

def build_naive_bayes():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_tfidf_base)),
        ("clf",   MultinomialNB(alpha=0.1)),
    ])

def build_linear_svm():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**_tfidf_base)),
        ("clf",   LinearSVC(C=1.0, max_iter=2000)),
    ])

def build_xgboost():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)),
        ("clf",   XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )),
    ])

MODELS = {
    "Naive Bayes":          build_naive_bayes,
    "Logistic Regression":  build_logistic_regression,
    "Linear SVM":           build_linear_svm,
    "XGBoost":              build_xgboost,
}
