from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from qa_ftopsis.config import AppConfig
from qa_ftopsis.data import PREPARED_SPLITS, load_prepared_split, load_queue_lookup
from qa_ftopsis.features import (
    build_feature_frame,
    compute_complexity_raw_features,
    fit_complexity_stats,
    probability_column_names,
)


@dataclass(slots=True)
class ModelBundle:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    classifier: Any
    queue_ids: list[int]
    queue_id_to_name: dict[int, str]
    probability_columns: list[str]
    complexity_stats: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_bundle_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "model_bundle.joblib"


def feature_split_path(model_dir: str | Path, split_name: str) -> Path:
    return Path(model_dir) / "features" / f"{split_name}.parquet"


def classifier_metrics_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "classifier_metrics.json"


def _feature_matrix(
    texts: pd.Series,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
):
    word_matrix = word_vectorizer.transform(texts)
    char_matrix = char_vectorizer.transform(texts)
    return hstack([word_matrix, char_matrix]).tocsr()


def _model_text(frame: pd.DataFrame, prepend_language_token: bool) -> pd.Series:
    text = frame["text"].fillna("").astype(str)
    if not prepend_language_token:
        return text
    language = frame["language"].fillna("unknown").astype(str).str.lower()
    return "__lang__" + language + " " + text


def load_model_bundle(model_dir: str | Path) -> ModelBundle:
    payload = joblib.load(model_bundle_path(model_dir))
    return ModelBundle(**payload)


def train_classifier(config: AppConfig) -> dict[str, Any]:
    model_dir = config.paths.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = model_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    queue_lookup = load_queue_lookup(config.paths.prepared_dir)
    queue_ids = queue_lookup["queue_id"].astype(int).tolist()
    queue_id_to_name = dict(
        zip(queue_lookup["queue_id"].astype(int), queue_lookup["queue_name"].astype(str))
    )
    probability_columns = probability_column_names(queue_ids)

    train_df = load_prepared_split(config.paths.prepared_dir, "train")
    val_cal_df = load_prepared_split(config.paths.prepared_dir, "val_cal")

    train_model_text = _model_text(train_df, config.model.prepend_language_token)
    val_cal_model_text = _model_text(val_cal_df, config.model.prepend_language_token)

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=config.model.word_ngram_range,
        max_features=config.model.word_max_features,
        sublinear_tf=config.model.sublinear_tf,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=config.model.char_ngram_range,
        max_features=config.model.char_max_features,
        sublinear_tf=config.model.sublinear_tf,
    )

    word_vectorizer.fit(train_model_text)
    char_vectorizer.fit(train_model_text)

    x_train = _feature_matrix(train_model_text, word_vectorizer, char_vectorizer)
    x_val_cal = _feature_matrix(val_cal_model_text, word_vectorizer, char_vectorizer)
    y_train = train_df["true_queue_id"].astype(int).to_numpy()
    y_val_cal = val_cal_df["true_queue_id"].astype(int).to_numpy()

    base_classifier = LogisticRegression(
        max_iter=config.model.max_iter,
        class_weight="balanced",
        random_state=config.data.random_state,
    )
    base_classifier.fit(x_train, y_train)

    class_counts = np.bincount(y_val_cal)
    non_zero_class_counts = class_counts[class_counts > 0]
    min_class_count = int(non_zero_class_counts.min()) if len(non_zero_class_counts) else 1
    calibration_cv = "prefit" if min_class_count < 2 else min(5, min_class_count)
    calibrated_classifier = CalibratedClassifierCV(
        FrozenEstimator(base_classifier),
        method="sigmoid",
        cv=calibration_cv,
    )
    calibrated_classifier.fit(x_val_cal, y_val_cal)

    train_probabilities = calibrated_classifier.predict_proba(x_train)
    train_raw_complexity = compute_complexity_raw_features(train_df["text"], train_probabilities)
    complexity_stats = fit_complexity_stats(train_raw_complexity)

    classifier_metrics: dict[str, dict[str, float]] = {}
    for split_name in PREPARED_SPLITS:
        split_df = load_prepared_split(config.paths.prepared_dir, split_name)
        split_model_text = _model_text(split_df, config.model.prepend_language_token)
        x_split = _feature_matrix(split_model_text, word_vectorizer, char_vectorizer)
        probabilities = calibrated_classifier.predict_proba(x_split)
        enriched = build_feature_frame(
            split_df,
            prob_matrix=probabilities,
            stats=complexity_stats,
            queue_ids=queue_ids,
        )
        enriched.to_parquet(feature_split_path(model_dir, split_name), index=False)

        y_true = enriched["true_queue_id"].astype(int).to_numpy()
        y_pred = enriched["predicted_queue_id"].astype(int).to_numpy()
        classifier_metrics[split_name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }

    bundle = ModelBundle(
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        classifier=calibrated_classifier,
        queue_ids=queue_ids,
        queue_id_to_name=queue_id_to_name,
        probability_columns=probability_columns,
        complexity_stats=complexity_stats.to_dict(),
        metadata={
            "random_state": config.data.random_state,
            "model_config": {
                "word_ngram_range": list(config.model.word_ngram_range),
                "char_ngram_range": list(config.model.char_ngram_range),
                "word_max_features": config.model.word_max_features,
                "char_max_features": config.model.char_max_features,
                "max_iter": config.model.max_iter,
                "sublinear_tf": config.model.sublinear_tf,
                "prepend_language_token": config.model.prepend_language_token,
            },
        },
    )
    joblib.dump(bundle.to_dict(), model_bundle_path(model_dir))

    with classifier_metrics_path(model_dir).open("w", encoding="utf-8") as handle:
        json.dump(classifier_metrics, handle, indent=2)

    return {
        "model_dir": str(model_dir),
        "feature_dir": str(feature_dir),
        "classifier_metrics": classifier_metrics,
    }


def load_feature_split(model_dir: str | Path, split_name: str) -> pd.DataFrame:
    return pd.read_parquet(feature_split_path(model_dir, split_name))
