from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_VERSION = "phase4-logreg-v1"
UNCERTAIN_THRESHOLD = 0.60

METADATA_COLUMNS = {
    "source",
    "media_type",
    "relative_path",
    "identity_id",
    "contract_label",
    "training_role",
    "split_group",
    "is_smile_subtype",
    "extraction_status",
    "extractor_version",
    "notes",
}


def load_media_features(path: Path = Path("data/features/phase3_media_features.csv")) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No media feature rows found in {path}")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in METADATA_COLUMNS:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    if not columns:
        raise ValueError("No numeric feature columns were found.")
    return columns


def make_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def usable_feature_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    usable: list[str] = []
    for column in candidates:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            usable.append(column)
    if not usable:
        raise ValueError("No usable numeric features were found for this modeling subset.")
    return usable


def _probability_rows(
    probabilities: np.ndarray,
    classes: np.ndarray,
    prefix: str,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in probabilities:
        rows.append({f"{prefix}_prob_{label}": float(prob) for label, prob in zip(classes, row)})
    return rows


def _evaluate_predictions(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    return {
        "rows": len(y_true),
        "labels": labels,
        "class_counts": dict(sorted(Counter(y_true).items())),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _coverage_stats(confidences: list[float], threshold: float) -> dict[str, float]:
    uncertain = [confidence < threshold for confidence in confidences]
    return {
        "uncertain_threshold": threshold,
        "uncertain_rows": int(sum(uncertain)),
        "coverage": float(1.0 - (sum(uncertain) / len(confidences))) if confidences else 0.0,
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
    }


def _cross_validate(
    df: pd.DataFrame,
    target_column: str,
    mode: str,
    classifier_mode: str,
    cv: Any,
    features: list[str],
    probability_prefix: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[target_column].astype(str).to_numpy()
    groups = df["split_group"].astype(str).to_numpy()

    for fold_index, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"Group leakage detected in fold {fold_index}: {sorted(overlap)[:5]}")

        pipeline = make_pipeline()
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        probabilities = pipeline.predict_proba(X.iloc[test_idx])
        classes = pipeline.classes_
        best_indices = probabilities.argmax(axis=1)
        predictions = classes[best_indices]
        confidences = probabilities[np.arange(len(best_indices)), best_indices]

        fold = df.iloc[test_idx].copy()
        fold["fold"] = fold_index
        fold["true_label"] = y[test_idx]
        fold["predicted_label"] = predictions
        fold["prediction_confidence"] = confidences
        fold["uncertain"] = confidences < UNCERTAIN_THRESHOLD
        fold["classifier_mode"] = classifier_mode
        fold["classifier_version"] = MODEL_VERSION
        fold["labelConfidence"] = confidences if mode == "basic_expression" else np.nan
        fold["smileTypeConfidence"] = confidences if mode == "smile_subtype" else np.nan
        fold["smileType"] = predictions if mode == "smile_subtype" else ""
        probability_df = pd.DataFrame(_probability_rows(probabilities, classes, probability_prefix), index=fold.index)
        rows.append(pd.concat([fold, probability_df], axis=1))

    predictions_df = pd.concat(rows).sort_index()
    return predictions_df


def _train_final_model(df: pd.DataFrame, target_column: str, features: list[str]) -> Pipeline:
    X = df[features].apply(pd.to_numeric, errors="coerce")
    y = df[target_column].astype(str)
    pipeline = make_pipeline()
    pipeline.fit(X, y)
    return pipeline


def _write_predictions(path: Path, predictions: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keep_first = [
        "relative_path",
        "source",
        "identity_id",
        "split_group",
        "contract_label",
        "true_label",
        "predicted_label",
        "prediction_confidence",
        "uncertain",
        "classifier_mode",
        "classifier_version",
        "labelConfidence",
        "smileType",
        "smileTypeConfidence",
        "fold",
    ]
    ordered = keep_first + [column for column in predictions.columns if column not in keep_first]
    predictions[ordered].to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_pickle(path: Path, model: Pipeline, features: list[str], labels: list[str], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "model_version": MODEL_VERSION,
        "uncertain_threshold": UNCERTAIN_THRESHOLD,
        "features": features,
        "labels": labels,
        "pipeline": model,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def run_baselines(
    feature_path: Path = Path("data/features/phase3_media_features.csv"),
    out_dir: Path = Path("data/model_outputs"),
    model_dir: Path = Path("models/phase4"),
) -> dict[str, Any]:
    df = load_media_features(feature_path)
    features = feature_columns(df)

    basic_df = df[
        (df["source"] == "fei_face_database")
        & (df["contract_label"].isin(["neutral", "generic_smile"]))
    ].copy()
    basic_df["basic_target"] = basic_df["contract_label"].map({"neutral": "neutral", "generic_smile": "smile"})

    subtype_df = df[
        (df["source"] == "osf_smile_types")
        & (df["contract_label"].isin(["reward", "affiliative", "dominance"]))
    ].copy()
    subtype_df["subtype_target"] = subtype_df["contract_label"]

    if len(basic_df) != 400:
        raise AssertionError(f"Expected 400 FEI basic rows, found {len(basic_df)}")
    if len(subtype_df) != 45:
        raise AssertionError(f"Expected 45 OSF smile subtype rows, found {len(subtype_df)}")

    basic_features = usable_feature_columns(basic_df, features)
    subtype_features = usable_feature_columns(subtype_df, features)

    basic_predictions = _cross_validate(
        basic_df,
        target_column="basic_target",
        mode="basic_expression",
        classifier_mode="basic",
        cv=GroupKFold(n_splits=5),
        features=basic_features,
        probability_prefix="basic",
    )
    subtype_predictions = _cross_validate(
        subtype_df,
        target_column="subtype_target",
        mode="smile_subtype",
        classifier_mode="model-subtype",
        cv=LeaveOneGroupOut(),
        features=subtype_features,
        probability_prefix="subtype",
    )

    basic_labels = ["neutral", "smile"]
    subtype_labels = ["affiliative", "dominance", "reward"]
    metrics = {
        "model_version": MODEL_VERSION,
        "feature_source": str(feature_path),
        "feature_columns": {
            "all_numeric_features": features,
            "basic_expression": basic_features,
            "smile_subtype": subtype_features,
        },
        "uncertain_threshold": UNCERTAIN_THRESHOLD,
        "basic_expression": {
            **_evaluate_predictions(
                basic_predictions["true_label"].tolist(),
                basic_predictions["predicted_label"].tolist(),
                basic_labels,
            ),
            **_coverage_stats(basic_predictions["prediction_confidence"].astype(float).tolist(), UNCERTAIN_THRESHOLD),
            "cross_validation": "GroupKFold(n_splits=5) by FEI subject",
        },
        "smile_subtype": {
            **_evaluate_predictions(
                subtype_predictions["true_label"].tolist(),
                subtype_predictions["predicted_label"].tolist(),
                subtype_labels,
            ),
            **_coverage_stats(subtype_predictions["prediction_confidence"].astype(float).tolist(), UNCERTAIN_THRESHOLD),
            "cross_validation": "LeaveOneGroupOut by OSF actor",
        },
        "not_modeled_yet": {
            "frown": "Only OSF Sadness is available as frown_candidate; it is not treated as confirmed app frown ground truth.",
            "unknown_uncertain": "Uncertain is thresholded from confidence; unknown remains an app-level fallback, not a trained class yet.",
        },
    }

    _write_predictions(out_dir / "phase4_basic_expression_predictions.csv", basic_predictions)
    _write_predictions(out_dir / "phase4_smile_subtype_predictions.csv", subtype_predictions)
    _write_json(out_dir / "phase4_metrics.json", metrics)
    _train_and_save_final_models(basic_df, subtype_df, basic_features, subtype_features, model_dir)
    return metrics


def _train_and_save_final_models(
    basic_df: pd.DataFrame,
    subtype_df: pd.DataFrame,
    basic_features: list[str],
    subtype_features: list[str],
    model_dir: Path,
) -> None:
    basic_model = _train_final_model(basic_df, "basic_target", basic_features)
    subtype_model = _train_final_model(subtype_df, "subtype_target", subtype_features)
    _write_pickle(
        model_dir / "basic_expression_baseline.pkl",
        basic_model,
        basic_features,
        ["neutral", "smile"],
        "basic",
    )
    _write_pickle(
        model_dir / "smile_subtype_baseline.pkl",
        subtype_model,
        subtype_features,
        ["affiliative", "dominance", "reward"],
        "model-subtype",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 baseline models.")
    parser.add_argument("--features", type=Path, default=Path("data/features/phase3_media_features.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/model_outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/phase4"))
    args = parser.parse_args()
    metrics = run_baselines(args.features, args.out_dir, args.model_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
