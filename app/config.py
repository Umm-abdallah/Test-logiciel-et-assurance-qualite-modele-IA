"""Configuration centrale du projet AI-TestGuard.

Toutes les constantes qui conditionnent la reproductibilité (graines, proportions
de partition, contrat de données, critères d'acceptation) sont regroupées ici
afin d'être versionnées avec le code et tracées dans la baseline.
"""

from __future__ import annotations

from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_DATA_PATH = DATA_DIR / "creditcard.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_creditcard.csv"

# Contrat de données
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = ["Time", *PCA_FEATURES, "Amount"]
TARGET = "Class"
REQUIRED_COLUMNS = [*FEATURES, TARGET]
ALLOWED_TARGET_VALUES = (0, 1)

# Reproductibilité
RANDOM_STATE = 42

# Partition stratifiée : 60 % entraînement, 20 % validation, 20 % test
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # proportion du jeu complet

# Politique de seuil, fixée AVANT toute observation du jeu de test.
# On retient le seuil qui maximise la précision sous contrainte de rappel
# minimal sur le jeu de VALIDATION.
MIN_RECALL_FOR_THRESHOLD = 0.80

# Critères d'acceptation (définis à partir des exigences, voir docs/plan_de_test.md)
ACCEPTANCE_CRITERIA = {
    "pr_auc_min": 0.70,
    "recall_min": 0.75,
    "precision_min": 0.50,
    "fpr_max": 0.001,
    "latency_p95_ms_max": 50.0,
    "model_size_mb_max": 10.0,
}

# Tolérances de non-régression (dégradation absolue maximale admise)
REGRESSION_TOLERANCES = {
    "pr_auc": 0.02,
    "roc_auc": 0.01,
    "recall": 0.05,
    "precision": 0.05,
    "fpr": 0.0005,
}

# Tolérance numérique pour la cohérence avant/après rechargement
RELOAD_ATOL = 1e-12

# Seuil d'alerte de dérive (Population Stability Index)
PSI_ALERT_THRESHOLD = 0.20
PSI_WARNING_THRESHOLD = 0.10
