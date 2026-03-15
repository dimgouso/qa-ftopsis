from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from qa_ftopsis.config import AppConfig
from qa_ftopsis.jira_benchmark import benchmark_macro_groups_path, benchmark_transition_graph_path
from qa_ftopsis.models import feature_split_path
from qa_ftopsis.types import QueueSkillArtifacts


def skill_features_dir(model_dir: str | Path) -> Path:
    return Path(model_dir) / "skill_features"


def queue_centroids_path(model_dir: str | Path) -> Path:
    return skill_features_dir(model_dir) / "queue_centroids.npy"


def embedding_metadata_path(model_dir: str | Path) -> Path:
    return skill_features_dir(model_dir) / "embedding_metadata.json"


def similarity_columns(queue_ids: list[int]) -> list[str]:
    return [f"sim_q_{queue_id}" for queue_id in queue_ids]


def kappa_columns(queue_ids: list[int]) -> list[str]:
    return [f"kappa_q_{queue_id}" for queue_id in queue_ids]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _encode_hashing_mock(texts: pd.Series) -> np.ndarray:
    vectorizer = HashingVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        n_features=256,
        alternate_sign=False,
        norm=None,
    )
    matrix = vectorizer.transform(texts.fillna("").astype(str)).toarray().astype(np.float32)
    return _normalize_rows(matrix)


def _encode_transformer(
    texts: pd.Series,
    model_name: str,
    max_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = device
    if resolved_device != "cpu" and not torch.cuda.is_available():
        resolved_device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts.iloc[start : start + batch_size].fillna("").astype(str).tolist()
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            batch_embeddings = pooled.cpu().numpy().astype(np.float32)
            batches.append(batch_embeddings)
    embeddings = np.vstack(batches)
    return _normalize_rows(embeddings)


def encode_texts(config: AppConfig, texts: pd.Series) -> np.ndarray:
    model_name = config.skill_features.model_name
    if model_name == "hashing-mock":
        return _encode_hashing_mock(texts)
    return _encode_transformer(
        texts=texts,
        model_name=model_name,
        max_length=config.skill_features.max_length,
        batch_size=config.skill_features.batch_size,
        device=config.skill_features.device,
    )


def _artifacts_exist(model_dir: str | Path, queue_ids: list[int]) -> bool:
    metadata_path = embedding_metadata_path(model_dir)
    centroids_path = queue_centroids_path(model_dir)
    if not metadata_path.exists() or not centroids_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    expected_sim = similarity_columns(queue_ids)
    expected_kappa = kappa_columns(queue_ids)
    return (
        metadata.get("similarity_columns") == expected_sim
        and metadata.get("kappa_columns") == expected_kappa
        and metadata.get("centroid_source_split") == "train"
        and metadata.get("queue_ids") == queue_ids
        and int(metadata.get("num_centroids", 0)) == len(queue_ids)
    )


def ensure_skill_features(config: AppConfig, feature_splits: dict[str, pd.DataFrame]) -> None:
    train_df = feature_splits["train"]
    queue_ids = sorted(train_df["true_queue_id"].astype(int).unique().tolist())
    if _artifacts_exist(config.paths.model_dir, queue_ids):
        sample_df = pd.read_parquet(feature_split_path(config.paths.model_dir, "train"))
        if all(column in sample_df.columns for column in kappa_columns(queue_ids)):
            return
    build_skill_features(config, feature_splits)


def load_queue_compatibility(model_dir: str | Path) -> dict[int, list[int]]:
    metadata = json.loads(embedding_metadata_path(model_dir).read_text())
    compatible = metadata.get("compatible_queues", {})
    return {
        int(queue_id): [int(value) for value in neighbors]
        for queue_id, neighbors in compatible.items()
    }


def load_macro_groups(model_dir: str | Path) -> dict[int, int]:
    metadata = json.loads(embedding_metadata_path(model_dir).read_text())
    macro_groups = metadata.get("macro_groups", {})
    return {int(queue_id): int(group_id) for queue_id, group_id in macro_groups.items()}


def _history_neighbors(config: AppConfig, queue_ids: list[int]) -> tuple[dict[int, list[int]], dict[int, int]]:
    transition_path = benchmark_transition_graph_path(config)
    macro_groups_path = benchmark_macro_groups_path(config)
    history_neighbors: dict[int, list[int]] = {}
    macro_groups: dict[int, int] = {}
    if transition_path.exists():
        try:
            graph = pd.read_csv(transition_path)
        except pd.errors.EmptyDataError:
            graph = pd.DataFrame()
        if not graph.empty:
            for queue_id in queue_ids:
                sort_columns = [
                    column
                    for column in [
                        "symmetric_count",
                        "issue_overlap_count",
                        "token_overlap_count",
                        "count",
                        "target_queue_id",
                    ]
                    if column in graph.columns
                ]
                ascending = [False] * max(len(sort_columns) - 1, 0) + [True]
                outgoing = graph[graph["source_queue_id"] == queue_id].sort_values(
                    sort_columns,
                    ascending=ascending,
                )
                neighbors = [int(queue_id)] + [
                    int(value)
                    for value in outgoing["target_queue_id"].astype(int).tolist()[
                        : max(0, config.skill_features.compatibility_top_k)
                    ]
                ]
                history_neighbors[int(queue_id)] = neighbors
    if macro_groups_path.exists():
        try:
            macro_frame = pd.read_csv(macro_groups_path)
        except pd.errors.EmptyDataError:
            macro_frame = pd.DataFrame()
        macro_groups = {
            int(row["queue_id"]): int(row["macro_group_id"])
            for row in macro_frame.to_dict(orient="records")
        }
    return history_neighbors, macro_groups


def build_skill_features(
    config: AppConfig,
    feature_splits: dict[str, pd.DataFrame] | None = None,
) -> dict[str, str]:
    if feature_splits is None:
        feature_splits = {
            split_name: pd.read_parquet(feature_split_path(config.paths.model_dir, split_name))
            for split_name in ["train", "val_cal", "val_sim", "test"]
        }

    skill_dir = skill_features_dir(config.paths.model_dir)
    skill_dir.mkdir(parents=True, exist_ok=True)

    train_df = feature_splits["train"]
    queue_ids = sorted(train_df["true_queue_id"].astype(int).unique().tolist())
    sim_columns = similarity_columns(queue_ids)
    kappas = kappa_columns(queue_ids)

    embeddings_by_split: dict[str, np.ndarray] = {}
    for split_name, split_df in feature_splits.items():
        embeddings_by_split[split_name] = encode_texts(config, split_df["text"])

    train_embeddings = embeddings_by_split["train"]
    centroids = []
    for queue_id in queue_ids:
        mask = train_df["true_queue_id"].astype(int).to_numpy() == queue_id
        centroid = train_embeddings[mask].mean(axis=0)
        centroid = _normalize_rows(np.asarray([centroid], dtype=np.float32))[0]
        centroids.append(centroid.astype(np.float32))
    centroid_matrix = np.vstack(centroids).astype(np.float32)
    np.save(queue_centroids_path(config.paths.model_dir), centroid_matrix)
    centroid_similarity = np.clip(centroid_matrix @ centroid_matrix.T, -1.0, 1.0)

    for split_name, split_df in feature_splits.items():
        embeddings = embeddings_by_split[split_name]
        similarity = np.clip(embeddings @ centroid_matrix.T, -1.0, 1.0)
        similarity01 = np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)
        kappa = np.clip(
            1.4 - 0.6 * similarity01,
            config.skill_features.kappa_min,
            config.skill_features.kappa_max,
        )

        enriched = split_df.copy()
        for index, column in enumerate(sim_columns):
            enriched[column] = similarity[:, index]
        for index, column in enumerate(kappas):
            enriched[column] = kappa[:, index]
        enriched.to_parquet(feature_split_path(config.paths.model_dir, split_name), index=False)

    metadata = QueueSkillArtifacts(
        embedding_model_name=config.skill_features.model_name,
        queue_centroids=centroid_matrix.round(8).tolist(),
        similarity_columns=sim_columns,
        kappa_columns=kappas,
    )
    metadata_payload = metadata.to_dict()
    history_neighbors, macro_groups = _history_neighbors(config, queue_ids)
    compatible_queues: dict[str, list[int]] = {}
    for queue_index, queue_id in enumerate(queue_ids):
        ranked = sorted(
            [
                (other_queue_id, float(centroid_similarity[queue_index, other_index]))
                for other_index, other_queue_id in enumerate(queue_ids)
                if other_queue_id != queue_id
            ],
            key=lambda item: (-item[1], item[0]),
        )
        neighbors = list(history_neighbors.get(int(queue_id), [int(queue_id)]))
        for other_queue_id, _ in ranked:
            if len(neighbors) >= config.skill_features.compatibility_top_k + 1:
                break
            if int(other_queue_id) not in neighbors:
                neighbors.append(int(other_queue_id))
        compatible_queues[str(queue_id)] = neighbors
    metadata_payload.update(
        {
            "centroid_source_split": "train",
            "queue_ids": queue_ids,
            "num_centroids": len(queue_ids),
            "compatible_queues": compatible_queues,
            "compatible_queues_source": "history_plus_centroid_fallback",
            "macro_groups": {str(queue_id): int(macro_groups.get(queue_id, queue_id)) for queue_id in queue_ids},
        }
    )
    with embedding_metadata_path(config.paths.model_dir).open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)

    return {
        "skill_dir": str(skill_dir),
        "metadata_path": str(embedding_metadata_path(config.paths.model_dir)),
        "queue_centroids_path": str(queue_centroids_path(config.paths.model_dir)),
    }
