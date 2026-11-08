"""Production des prédictions et point d'entrée de type « service »."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from app.config import FEATURES
from app.preprocessing import DataValidationError, preprocess_data, range_alerts

logger = logging.getLogger("aitestguard.prediction")


def predict_fraud(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Retourne, pour chaque transaction, la probabilité de fraude et la décision.

    ``model`` est un pipeline exposant ``predict_proba``. ``X`` est validé
    selon le contrat (sans cible). La décision vaut 1 si ``proba >= threshold``.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"Seuil hors [0, 1] : {threshold}")
    X_clean, _ = preprocess_data(X, require_target=False)
    proba = model.predict_proba(X_clean)[:, 1]
    if not np.isfinite(proba).all():
        # Ne devrait jamais arriver avec une entrée validée : anomalie bloquante
        raise RuntimeError("Le modèle a produit des probabilités non finies")
    return pd.DataFrame(
        {"fraud_probability": proba, "is_fraud": (proba >= threshold).astype(int)},
        index=X.index,
    )


class FraudScoringService:
    """Simule l'endpoint d'une API de scoring transaction par transaction.

    * validation stricte : champ manquant, supplémentaire ou mal typé = rejet ;
    * aucun contenu de transaction n'est écrit dans les journaux (seuls un
      identifiant opaque, la décision et la latence le sont) ;
    * les montants hors domaine d'entraînement sont traités mais signalés.
    """

    def __init__(self, bundle: dict[str, Any]):
        self.pipeline = bundle["pipeline"]
        self.threshold = float(bundle["threshold"])
        self.reference_stats = bundle.get("metadata", {}).get("reference_stats")

    @staticmethod
    def _record_to_frame(record: dict[str, Any]) -> pd.DataFrame:
        if not isinstance(record, dict):
            raise DataValidationError([f"Enregistrement attendu dict, reçu {type(record).__name__}"])
        bad_types = [
            k
            for k, v in record.items()
            if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating))
        ]
        if bad_types:
            raise DataValidationError([f"Types non numériques : {sorted(bad_types)}"])
        return pd.DataFrame([record])

    def score(self, record: dict[str, Any], request_id: str = "n/a") -> dict[str, Any]:
        start = time.perf_counter()
        try:
            df = self._record_to_frame(record)
            X, _ = preprocess_data(df, require_target=False, strict=True)
        except DataValidationError as exc:
            logger.warning("request=%s status=rejected n_issues=%d", request_id, len(exc.issues))
            return {"status": "rejected", "errors": exc.issues}
        proba = float(self.pipeline.predict_proba(X)[:, 1][0])
        alert = bool(range_alerts(X, self.reference_stats).iloc[0]) if self.reference_stats else False
        decision = int(proba >= self.threshold)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request=%s status=scored decision=%d out_of_range=%s latency_ms=%.2f",
            request_id,
            decision,
            alert,
            latency_ms,
        )
        return {
            "status": "scored",
            "fraud_probability": proba,
            "is_fraud": decision,
            "out_of_range_alert": alert,
        }


__all__ = ["FEATURES", "FraudScoringService", "predict_fraud"]
