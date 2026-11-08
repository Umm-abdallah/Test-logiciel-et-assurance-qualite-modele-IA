"""Protocole de mesure des performances d'exécution.

* matériel et logiciel : relevés automatiquement (CPU, RAM, versions) ;
* échauffement : 50 appels non mesurés ;
* latence unitaire : 2 000 appels predict_proba sur 1 ligne (lignes tirées
  du jeu de test), horloge time.perf_counter ;
* latence « service » : idem via FraudScoringService (conversion dict ->
  DataFrame + validation stricte + prédiction + alerte), hors HTTP ;
* débit : lots de 1, 100, 1 000, 10 000 et 56 000 lignes, 5 répétitions,
  médiane retenue ;
* mémoire : RSS du processus (psutil) avant et après chargement, et pic
  pendant la prédiction d'un lot de 56 000 lignes.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import psutil  # noqa: E402

from app.config import ACCEPTANCE_CRITERIA, DEFAULT_DATA_PATH, MODELS_DIR, REPORTS_DIR  # noqa: E402
from app.prediction import FraudScoringService  # noqa: E402
from app.preprocessing import drop_exact_duplicates, load_data, preprocess_data  # noqa: E402
from app.training import environment_info, load_model, split_data, write_json  # noqa: E402


def pct(a, q):
    return float(np.percentile(a, q))


def main(data: Path, model_path: Path, n_calls: int = 2000) -> None:
    proc = psutil.Process(os.getpid())
    df, _ = drop_exact_duplicates(load_data(data))
    X, y = preprocess_data(df)
    Xt = split_data(X, y).X_test
    rss_before = proc.memory_info().rss / 1e6

    t0 = time.perf_counter()
    bundle = load_model(model_path)
    load_ms = (time.perf_counter() - t0) * 1000
    rss_after_load = proc.memory_info().rss / 1e6
    pipe = bundle["pipeline"]
    service = FraudScoringService(bundle)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(Xt), n_calls)
    rows = [Xt.iloc[[i]] for i in idx]
    records = [{k: float(v) for k, v in r.iloc[0].items()} for r in rows]

    for r in rows[:50]:
        pipe.predict_proba(r)
    lat = []
    for r in rows:
        t = time.perf_counter()
        pipe.predict_proba(r)
        lat.append((time.perf_counter() - t) * 1000)
    for r in records[:50]:
        service.score(r)
    lat_s = []
    for r in records:
        t = time.perf_counter()
        service.score(r)
        lat_s.append((time.perf_counter() - t) * 1000)

    throughput = {}
    peak = rss_after_load
    for size in (1, 100, 1000, 10_000, len(Xt)):
        batch = Xt.sample(size, replace=size > len(Xt), random_state=1)
        times = []
        for _ in range(5):
            t = time.perf_counter()
            pipe.predict_proba(batch)
            times.append(time.perf_counter() - t)
            peak = max(peak, proc.memory_info().rss / 1e6)
        med = float(np.median(times))
        throughput[str(size)] = {"median_s": med, "tx_per_s": size / med}

    res = {
        "protocol": __doc__,
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cpus": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        },
        "environment": environment_info(),
        "model_file_mb": model_path.stat().st_size / 1e6,
        "model_load_ms": load_ms,
        "single_prediction_ms": {
            "p50": pct(lat, 50),
            "p95": pct(lat, 95),
            "p99": pct(lat, 99),
            "max": max(lat),
            "n": n_calls,
        },
        "service_call_ms": {
            "p50": pct(lat_s, 50),
            "p95": pct(lat_s, 95),
            "p99": pct(lat_s, 99),
            "max": max(lat_s),
            "n": n_calls,
        },
        "throughput": throughput,
        "memory_mb": {
            "rss_before_load": rss_before,
            "rss_after_load": rss_after_load,
            "model_load_delta": rss_after_load - rss_before,
            "peak_during_batches": peak,
        },
        "criteria": {
            "latency_p95_ms_max": ACCEPTANCE_CRITERIA["latency_p95_ms_max"],
            "model_size_mb_max": ACCEPTANCE_CRITERIA["model_size_mb_max"],
            "latency_ok": pct(lat_s, 95) <= ACCEPTANCE_CRITERIA["latency_p95_ms_max"],
            "size_ok": model_path.stat().st_size / 1e6 <= ACCEPTANCE_CRITERIA["model_size_mb_max"],
        },
    }
    write_json(res, REPORTS_DIR / "benchmark.json")
    print({k: res[k] for k in ("model_file_mb", "single_prediction_ms", "service_call_ms", "memory_mb")})
    print({k: round(v["tx_per_s"]) for k, v in throughput.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "model.joblib")
    a = ap.parse_args()
    main(a.data, a.model)
