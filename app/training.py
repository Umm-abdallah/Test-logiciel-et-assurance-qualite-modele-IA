"""Partition des données, entraînement, réglage et sauvegarde du pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app.config import (
    ACCEPTANCE_CRITERIA,
    MIN_RECALL_FOR_THRESHOLD,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)
from app.preprocessing import build_preprocessor


class ModelLoadError(RuntimeError):
    """Levée lorsqu'un modèle sauvegardé est absent, corrompu ou altéré."""


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Partition stratifiée entraînement / validation / test.

    La partition est réalisée AVANT tout ajustement de transformation.
    ``val_size`` est exprimé en proportion du jeu complet.
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=random_state
    )
    return DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)


MODEL_FACTORIES = {
    "logistic_regression": lambda **p: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **p),
    "decision_tree": lambda **p: DecisionTreeClassifier(random_state=RANDOM_STATE, **p),
}

# Grilles volontairement petites : modèles légers, réglage sur la validation
PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logistic_regression": [
        {"C": c, "class_weight": cw} for c in (0.01, 0.1, 1.0, 10.0) for cw in (None, "balanced")
    ],
    "decision_tree": [
        {"max_depth": d, "min_samples_leaf": leaf, "class_weight": cw}
        for d in (4, 6, 8, 12)
        for leaf in (1, 20)
        for cw in (None, "balanced")
    ],
}


def build_pipeline(model_name: str = "logistic_regression", **params: Any) -> Pipeline:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Modèle inconnu : {model_name!r}. Choix : {sorted(MODEL_FACTORIES)}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", MODEL_FACTORIES[model_name](**params))])


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "logistic_regression",
    **params: Any,
) -> Pipeline:
    """Ajuste le pipeline complet (transformations + modèle) sur ``X, y``.

    Appeler cette fonction uniquement avec les données d'entraînement.
    """
    if len(X) != len(y):
        raise ValueError(f"Dimensions incohérentes : X={len(X)} lignes, y={len(y)}")
    classes = set(pd.unique(y))
    if classes != {0, 1}:
        raise ValueError(f"L'entraînement exige les deux classes 0 et 1, reçu {sorted(classes)}")
    pipe = build_pipeline(model_name, **params)
    pipe.fit(X, y)
    return pipe


def tune_on_validation(
    splits: DataSplits,
    model_name: str,
    grid: list[dict[str, Any]] | None = None,
) -> tuple[Pipeline, dict[str, Any], list[dict[str, Any]]]:
    """Sélectionne les hyperparamètres maximisant l'Average Precision en validation.

    Le jeu de test n'est jamais utilisé ici.
    """
    grid = grid if grid is not None else PARAM_GRIDS[model_name]
    results = []
    best = (-np.inf, None, None)
    for params in grid:
        pipe = train_model(splits.X_train, splits.y_train, model_name, **params)
        proba = pipe.predict_proba(splits.X_val)[:, 1]
        ap = float(average_precision_score(splits.y_val, proba))
        results.append({"params": params, "val_pr_auc": ap})
        if ap > best[0]:
            best = (ap, pipe, params)
    return best[1], best[2], results


def select_threshold(
    y_val: pd.Series | np.ndarray,
    proba_val: np.ndarray,
    min_recall: float = MIN_RECALL_FOR_THRESHOLD,
    max_fpr: float = ACCEPTANCE_CRITERIA["fpr_max"],
) -> dict[str, float]:
    """Choisit le seuil de décision sur la VALIDATION.

    Règle (fixée a priori à partir des exigences) : parmi les seuils dont le
    rappel est au moins ``min_recall`` ET le taux de faux positifs au plus
    ``max_fpr``, on retient celui de plus grande précision (le plus haut en
    cas d'égalité). Si aucun seuil ne satisfait les deux contraintes, on
    retient le seuil de F1 maximal et ``constraint_met`` vaut faux, ce qui
    doit être signalé dans le rapport.
    """
    y = np.asarray(y_val).astype(int)
    proba = np.asarray(proba_val, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y, proba)
    # precision/recall ont une valeur de plus que thresholds (point final)
    p, r = precision[:-1], recall[:-1]
    neg = np.sort(proba[y == 0])
    fp = neg.size - np.searchsorted(neg, thresholds, side="left")
    fpr = fp / max(neg.size, 1)
    ok = (r >= min_recall) & (fpr <= max_fpr)
    if ok.any():
        cand = np.where(ok)[0]
        best_p = p[cand].max()
        idx = int(cand[np.isclose(p[cand], best_p)].max())
        met = True
    else:
        f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
        idx = int(np.argmax(f1))
        met = False
    return {
        "threshold": float(thresholds[idx]),
        "val_precision": float(p[idx]),
        "val_recall": float(r[idx]),
        "val_fpr": float(fpr[idx]),
        "min_recall_target": float(min_recall),
        "max_fpr_target": float(max_fpr),
        "constraint_met": bool(met),
    }


def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
    }


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Empreinte SHA-256 stable d'un DataFrame (contenu et colonnes)."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(pipeline: Pipeline, threshold: float, path: str | Path, metadata: dict | None = None) -> Path:
    """Sauvegarde le pipeline, son seuil et ses métadonnées.

    Un fichier ``.sha256`` est écrit à côté : il permet de détecter une
    altération avant chargement (joblib/pickle exécute du code au chargement,
    on ne charge donc qu'un fichier dont l'empreinte est attendue).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "metadata": {**(metadata or {}), "environment": environment_info()},
    }
    joblib.dump(bundle, path)
    path.with_suffix(path.suffix + ".sha256").write_text(_sha256_file(path))
    return path


def load_model(path: str | Path, verify_hash: bool = True) -> dict[str, Any]:
    """Recharge un modèle sauvegardé avec contrôles explicites."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"Fichier modèle introuvable : {path}")
    if verify_hash:
        hash_file = path.with_suffix(path.suffix + ".sha256")
        if not hash_file.exists():
            raise ModelLoadError(f"Empreinte absente : {hash_file}")
        if hash_file.read_text().strip() != _sha256_file(path):
            raise ModelLoadError("Empreinte SHA-256 non conforme : fichier modèle altéré")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # fichier tronqué, format inconnu, etc.
        raise ModelLoadError(f"Chargement impossible : {exc}") from exc
    if not isinstance(bundle, dict) or not {"pipeline", "threshold"} <= set(bundle):
        raise ModelLoadError("Structure du fichier modèle inattendue")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ModelLoadError("L'objet chargé n'expose pas predict_proba")
    return bundle


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


__all__ = [
    "DataSplits",
    "ModelLoadError",
    "TARGET",
    "build_pipeline",
    "dataframe_fingerprint",
    "environment_info",
    "load_model",
    "save_model",
    "select_threshold",
    "split_data",
    "train_model",
    "tune_on_validation",
]
