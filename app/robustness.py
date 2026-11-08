"""Perturbations réalistes et mesure de sensibilité.

Les perturbations sont différenciées par nature de variable :

* ``V1`` à ``V28`` (composantes PCA, sans unité métier) : bruit gaussien dont
  l'écart type est une fraction ``rel`` de l'écart type de la variable,
  estimé sur l'entraînement ;
* ``Amount`` : variation multiplicative (erreur d'arrondi, conversion de
  devise, frais), bornée à zéro ;
* ``Time`` : décalage additif en secondes (horloge désynchronisée).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.config import PCA_FEATURES, RANDOM_STATE
from app.evaluation import evaluate_model


def perturb_pca(X: pd.DataFrame, train_std: pd.Series, rel: float, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = X.copy()
    noise = rng.normal(0.0, 1.0, size=(len(X), len(PCA_FEATURES))) * (
        train_std[PCA_FEATURES].to_numpy() * rel
    )
    out[PCA_FEATURES] = out[PCA_FEATURES].to_numpy() + noise
    return out


def perturb_amount(X: pd.DataFrame, rel: float, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = X.copy()
    factor = 1.0 + rng.uniform(-rel, rel, size=len(X))
    out["Amount"] = np.clip(out["Amount"].to_numpy() * factor, 0, None)
    return out


def shift_time(X: pd.DataFrame, seconds: float) -> pd.DataFrame:
    out = X.copy()
    out["Time"] = np.clip(out["Time"].to_numpy() + seconds, 0, None)
    return out


def sensitivity(
    pipeline: Any,
    X_ref: pd.DataFrame,
    X_pert: pd.DataFrame,
    y: pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Mesure l'effet d'une perturbation sur les probabilités et les métriques."""
    p0 = pipeline.predict_proba(X_ref)[:, 1]
    p1 = pipeline.predict_proba(X_pert)[:, 1]
    valid = np.isfinite(p1) & (p1 >= 0) & (p1 <= 1)
    d = np.abs(p1 - p0)
    c0, c1 = p0 >= threshold, p1 >= threshold
    m0 = evaluate_model(y, c0.astype(int), p0)
    m1 = evaluate_model(y, c1.astype(int), p1)
    y_arr = np.asarray(y)
    return {
        "mean_abs_dproba": float(d.mean()),
        "p95_abs_dproba": float(np.percentile(d, 95)),
        "max_abs_dproba": float(d.max()),
        "flip_rate": float((c0 != c1).mean()),
        "n_flips": int((c0 != c1).sum()),
        "n_flips_fraud": int(((c0 != c1) & (y_arr == 1)).sum()),
        "n_invalid": int((~valid).sum()),
        "delta_recall": m1["recall"] - m0["recall"],
        "delta_precision": m1["precision"] - m0["precision"],
        "delta_fpr": m1["fpr"] - m0["fpr"],
    }
