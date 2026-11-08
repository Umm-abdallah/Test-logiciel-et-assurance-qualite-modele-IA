"""Procédure de référence partagée par la baseline et les tests de non-régression."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app import __version__
from app.config import FEATURES, MIN_RECALL_FOR_THRESHOLD, RANDOM_STATE, TARGET
from app.evaluation import evaluate_model
from app.preprocessing import drop_exact_duplicates, preprocess_data
from app.training import (
    dataframe_fingerprint,
    environment_info,
    select_threshold,
    split_data,
    train_model,
)

# Configuration retenue sur la validation (scripts/run_experiments.py). Toute modification
# de ces paramètres constitue une « nouvelle version » à comparer à la baseline figée.
REFERENCE_MODEL = "decision_tree"
REFERENCE_PARAMS: dict[str, Any] = {"max_depth": 6, "min_samples_leaf": 20, "class_weight": None}


def git_commit() -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "non versionné"


def run_reference(
    df: pd.DataFrame,
    model_name: str = REFERENCE_MODEL,
    params: dict[str, Any] | None = None,
    min_recall: float = MIN_RECALL_FOR_THRESHOLD,
) -> dict[str, Any]:
    """Entraîne la configuration et mesure ses performances sur la VALIDATION fixe."""
    params = REFERENCE_PARAMS if params is None else params
    df, n_dup = drop_exact_duplicates(df)
    X, y = preprocess_data(df)
    s = split_data(X, y)
    pipe = train_model(s.X_train, s.y_train, model_name, **params)
    proba = pipe.predict_proba(s.X_val)[:, 1]
    thr = select_threshold(s.y_val, proba, min_recall=min_recall)
    metrics = evaluate_model(s.y_val, (proba >= thr["threshold"]).astype(int), proba)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": __version__,
        "git_commit": git_commit(),
        "model": model_name,
        "params": params,
        "random_state": RANDOM_STATE,
        "threshold": thr,
        "environment": environment_info(),
        "dataset": {
            "fingerprint": dataframe_fingerprint(df[[*FEATURES, TARGET]]),
            "n_rows_after_dedup": int(len(df)),
            "n_duplicates_removed": int(n_dup),
            "n_fraud": int(df[TARGET].sum()),
            "fraud_rate": float(df[TARGET].mean()),
            "n_validation": int(len(s.y_val)),
            "n_fraud_validation": int(s.y_val.sum()),
        },
        "validation_metrics": metrics,
    }
