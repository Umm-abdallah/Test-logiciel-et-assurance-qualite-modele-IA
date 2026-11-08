"""Tests de reproductibilité et de non-régression (T06, T19).

Trois niveaux :

1. ``synthetic`` : baseline figée sur le jeu synthétique déterministe,
   exécutée à chaque Pull Request (rapide, sans données externes) ;
2. ``sample`` : baseline figée sur l'échantillon réel versionné
   ``data/sample_creditcard.csv`` (ignoré si l'échantillon est absent) ;
3. ``slow`` : baseline complète ``reports/baseline.json`` sur le jeu Kaggle
   intégral, exécutée manuellement ou sur calendrier.

La référence n'est JAMAIS recalculée par les tests : elle est régénérée
uniquement par ``scripts/make_baseline.py`` après une décision explicite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.config import DEFAULT_DATA_PATH, REGRESSION_TOLERANCES, REPORTS_DIR, SAMPLE_DATA_PATH
from app.evaluation import compare_to_baseline
from app.experiment import run_reference
from app.preprocessing import load_data
from tests.conftest import make_synthetic

BASELINE_DIR = Path(__file__).parent / "baselines"


def _load(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Baseline absente : {path}")
    return json.loads(path.read_text())


def _candidate(baseline: dict) -> tuple[str, dict]:
    """Version candidate : celle de la baseline, sauf injection via AITG_CANDIDATE.

    ``AITG_CANDIDATE='{"model": "decision_tree", "params": {"max_depth": 1}}'``
    sert à démontrer qu'une régression volontaire fait échouer la CI.
    """
    raw = os.environ.get("AITG_CANDIDATE")
    if raw:
        cand = json.loads(raw)
        return cand.get("model", baseline["model"]), cand.get("params", baseline["params"])
    return baseline["model"], baseline["params"]


def _assert_no_regression(baseline: dict, candidate: dict) -> None:
    assert candidate["dataset"]["fingerprint"] == baseline["dataset"]["fingerprint"], (
        "Le jeu de validation a changé : la comparaison n'est pas valide"
    )
    res = compare_to_baseline(candidate["validation_metrics"], baseline["validation_metrics"])
    rejected = {k: v for k, v in res["details"].items() if v["status"] == "rejeter"}
    assert res["decision"] != "rejeter", f"Régression détectée : {json.dumps(rejected, indent=2)}"


# Reproductibilité


def test_t06_same_seed_same_results():
    df = make_synthetic()
    a, b = run_reference(df), run_reference(df)
    assert a["validation_metrics"] == b["validation_metrics"]
    assert a["threshold"] == b["threshold"]
    assert a["dataset"]["fingerprint"] == b["dataset"]["fingerprint"]


# Logique de comparaison


BASE = {"pr_auc": 0.80, "roc_auc": 0.97, "recall": 0.82, "precision": 0.60, "fpr": 0.0004}


def test_compare_accepts_identical():
    assert compare_to_baseline(BASE, BASE)["decision"] == "accepter"


def test_compare_rejects_degradation_beyond_tolerance():
    cand = {**BASE, "recall": BASE["recall"] - REGRESSION_TOLERANCES["recall"] - 0.01}
    res = compare_to_baseline(cand, BASE)
    assert res["decision"] == "rejeter"
    assert res["details"]["recall"]["status"] == "rejeter"


def test_compare_rejects_fpr_increase():
    cand = {**BASE, "fpr": BASE["fpr"] + 0.001}
    assert compare_to_baseline(cand, BASE)["decision"] == "rejeter"


def test_compare_investigates_grey_zone():
    cand = {**BASE, "pr_auc": BASE["pr_auc"] - 0.75 * REGRESSION_TOLERANCES["pr_auc"]}
    assert compare_to_baseline(cand, BASE)["decision"] == "investiguer"


def test_compare_investigates_suspicious_improvement():
    cand = {**BASE, "pr_auc": BASE["pr_auc"] + 0.10}
    assert compare_to_baseline(cand, BASE)["decision"] == "investiguer"


# Non-régression contre les baselines figées


def test_no_regression_synthetic():
    baseline = _load(BASELINE_DIR / "synthetic_baseline.json")
    cand = run_reference(make_synthetic(), *_candidate(baseline))
    _assert_no_regression(baseline, cand)


def test_no_regression_real_sample():
    if not SAMPLE_DATA_PATH.exists():
        pytest.skip("Échantillon réel absent")
    baseline = _load(BASELINE_DIR / "sample_baseline.json")
    cand = run_reference(load_data(SAMPLE_DATA_PATH), *_candidate(baseline))
    _assert_no_regression(baseline, cand)


@pytest.mark.slow
def test_no_regression_full_dataset():
    if not DEFAULT_DATA_PATH.exists():
        pytest.skip("Jeu Kaggle complet absent (data/creditcard.csv)")
    baseline = _load(REPORTS_DIR / "baseline.json")
    cand = run_reference(load_data(DEFAULT_DATA_PATH), *_candidate(baseline))
    _assert_no_regression(baseline, cand)
