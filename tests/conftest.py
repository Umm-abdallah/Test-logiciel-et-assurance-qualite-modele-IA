"""Fixtures partagées.

Les tests rapides s'appuient sur des données SYNTHÉTIQUES respectant le
schéma Kaggle (Time, V1 à V28, Amount, Class) : ils vérifient le
comportement du code, pas la performance réelle du modèle. Les contrôles sur
les vraies données sont dans scripts/ et dans les tests marqués ``slow``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import FEATURES, PCA_FEATURES, TARGET  # noqa: E402
from app.preprocessing import compute_reference_stats  # noqa: E402
from app.training import save_model, select_threshold, split_data, train_model  # noqa: E402


def make_synthetic(n: int = 4000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Génère un jeu au schéma Kaggle avec un signal de fraude apprenable."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    y[:2] = [0, 1]  # garantit la présence des deux classes
    V = rng.normal(0, 1, size=(n, 28))
    # Les fraudes sont décalées sur quelques composantes, avec recouvrement
    shift = np.zeros(28)
    shift[[0, 2, 3, 9, 11, 13]] = [-1.5, -1.8, 1.6, -1.2, -1.7, -2.0]
    V[y == 1] += shift
    df = pd.DataFrame(V, columns=PCA_FEATURES)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_792, n)))
    df["Amount"] = np.round(rng.lognormal(3.5, 1.3, n), 2)
    df[TARGET] = y
    return df[[*FEATURES, TARGET]]


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()


@pytest.fixture
def valid_df(synthetic_df) -> pd.DataFrame:
    """Petite copie modifiable, pour les tests de validation."""
    return synthetic_df.head(50).copy()


@pytest.fixture(scope="session")
def splits(synthetic_df):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET]
    return split_data(X, y)


@pytest.fixture(scope="session")
def trained(splits):
    """Pipeline de régression logistique et seuil choisi sur la validation."""
    pipe = train_model(splits.X_train, splits.y_train, "logistic_regression", C=1.0)
    proba_val = pipe.predict_proba(splits.X_val)[:, 1]
    thr = select_threshold(splits.y_val, proba_val, min_recall=0.8)
    return pipe, thr["threshold"]


@pytest.fixture(scope="session")
def saved_bundle_path(trained, splits, tmp_path_factory):
    pipe, threshold = trained
    path = tmp_path_factory.mktemp("models") / "model.joblib"
    save_model(
        pipe,
        threshold,
        path,
        metadata={"reference_stats": compute_reference_stats(splits.X_train)},
    )
    return path
