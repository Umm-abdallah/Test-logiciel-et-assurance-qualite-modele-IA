"""Contrôles de performance rapides (T22).

Les mesures de référence complètes (P95 sur 2 000 appels, débit par taille de
lot, mémoire) sont produites par ``scripts/benchmark.py``. Ici, on vérifie
seulement que les ordres de grandeur restent compatibles avec les critères,
avec une marge pour la variabilité des machines d'intégration continue.
"""

from __future__ import annotations

import time

import numpy as np

from app.config import ACCEPTANCE_CRITERIA
from app.prediction import FraudScoringService
from app.training import load_model


def test_model_file_size_within_limit(saved_bundle_path):
    size_mb = saved_bundle_path.stat().st_size / 1e6
    assert size_mb < ACCEPTANCE_CRITERIA["model_size_mb_max"]


def test_single_prediction_latency_p95(saved_bundle_path, splits):
    pipe = load_model(saved_bundle_path)["pipeline"]
    row = splits.X_test.iloc[[0]]
    for _ in range(20):  # échauffement
        pipe.predict_proba(row)
    lat = []
    for _ in range(200):
        t0 = time.perf_counter()
        pipe.predict_proba(row)
        lat.append((time.perf_counter() - t0) * 1000)
    assert np.percentile(lat, 95) < ACCEPTANCE_CRITERIA["latency_p95_ms_max"]


def test_service_latency_includes_validation(saved_bundle_path, splits):
    service = FraudScoringService(load_model(saved_bundle_path))
    record = {k: float(v) for k, v in splits.X_test.iloc[0].items()}
    lat = []
    for _ in range(100):
        t0 = time.perf_counter()
        assert service.score(record)["status"] == "scored"
        lat.append((time.perf_counter() - t0) * 1000)
    assert np.percentile(lat, 95) < ACCEPTANCE_CRITERIA["latency_p95_ms_max"]


def test_batch_throughput(saved_bundle_path, splits):
    pipe = load_model(saved_bundle_path)["pipeline"]
    X = splits.X_test.sample(10_000, replace=True, random_state=0)
    t0 = time.perf_counter()
    pipe.predict_proba(X)
    throughput = len(X) / (time.perf_counter() - t0)
    assert throughput > 10_000  # transactions par seconde
