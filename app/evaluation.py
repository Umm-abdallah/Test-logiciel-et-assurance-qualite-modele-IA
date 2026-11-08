"""Calcul des métriques, critères d'acceptation et comparaison à la baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.config import ACCEPTANCE_CRITERIA, REGRESSION_TOLERANCES


def evaluate_model(y_true, y_pred, y_proba=None) -> dict[str, Any]:
    """Métriques adaptées à une classe positive rare.

    Retourne la matrice de confusion détaillée (TP, TN, FP, FN), la précision,
    le rappel, le F1, les taux FPR et FNR, l'accuracy (à titre informatif
    seulement) et, si ``y_proba`` est fourni, ROC-AUC et PR-AUC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Dimensions incohérentes : {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Aucune observation à évaluer")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if not set(np.unique(arr)) <= {0, 1}:
            raise ValueError(f"{name} doit être binaire (0/1)")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.shape != y_true.shape:
            raise ValueError("y_proba doit avoir la même dimension que y_true")
        if not np.isfinite(y_proba).all() or (y_proba < 0).any() or (y_proba > 1).any():
            raise ValueError("y_proba doit contenir des probabilités finies dans [0, 1]")
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None
    return metrics


def bootstrap_ci(
    y_true, y_proba, threshold: float, n_boot: int = 500, alpha: float = 0.05, seed: int = 42
) -> dict[str, tuple[float, float]]:
    """Intervalles de confiance bootstrap (percentiles) des métriques principales.

    Avec moins d'une centaine de fraudes en test, l'incertitude sur le rappel
    est de plusieurs points : elle doit accompagner toute décision.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    keys = ("precision", "recall", "fpr", "pr_auc")
    samples: dict[str, list[float]] = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_proba[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        m = evaluate_model(yt, (yp >= threshold).astype(int), yp)
        for k in keys:
            samples[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in samples.items()}


def check_acceptance(metrics: dict[str, Any], criteria: dict[str, float] | None = None) -> dict[str, Any]:
    """Confronte les métriques de test aux critères définis a priori."""
    c = criteria or ACCEPTANCE_CRITERIA
    checks = {
        "pr_auc": (metrics.get("pr_auc"), ">=", c["pr_auc_min"]),
        "recall": (metrics["recall"], ">=", c["recall_min"]),
        "precision": (metrics["precision"], ">=", c["precision_min"]),
        "fpr": (metrics["fpr"], "<=", c["fpr_max"]),
    }
    out = {}
    for name, (value, op, limit) in checks.items():
        passed = value is not None and (value >= limit if op == ">=" else value <= limit)
        out[name] = {"value": value, "rule": f"{op} {limit}", "passed": bool(passed)}
    out["all_passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    return out


def compare_to_baseline(
    candidate: dict[str, float],
    baseline: dict[str, float],
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare une nouvelle version à la référence figée.

    Décision : ``accepter`` si aucune dégradation ne dépasse la tolérance,
    ``rejeter`` si une métrique se dégrade au-delà de la tolérance,
    ``investiguer`` si une métrique se dégrade sans franchir la tolérance
    de plus de la moitié de celle-ci (zone grise), ou si elle s'améliore
    fortement (une amélioration inattendue peut trahir une fuite).
    """
    tol = tolerances or REGRESSION_TOLERANCES
    lower_is_better = {"fpr", "fnr"}
    details = {}
    decision = "accepter"
    for metric, t in tol.items():
        if metric not in candidate or metric not in baseline:
            continue
        new, ref = float(candidate[metric]), float(baseline[metric])
        degradation = (new - ref) if metric in lower_is_better else (ref - new)
        if degradation > t:
            status = "rejeter"
        elif degradation > t / 2 or degradation < -3 * t:
            status = "investiguer"
        else:
            status = "ok"
        details[metric] = {
            "baseline": ref,
            "candidate": new,
            "degradation": degradation,
            "tolerance": t,
            "status": status,
        }
        if status == "rejeter":
            decision = "rejeter"
        elif status == "investiguer" and decision == "accepter":
            decision = "investiguer"
    return {"decision": decision, "details": details}
