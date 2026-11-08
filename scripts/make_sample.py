"""Extrait un échantillon stratifié et reproductible du jeu Kaggle.

L'échantillon (environ 1 Mo) est versionné pour que la CI puisse exécuter
un contrôle de non-régression sur de vraies données sans télécharger le jeu
complet. Il sur-représente les fraudes (100 fraudes, 4 900 normales) : ses
métriques ne sont PAS représentatives de la production et ne servent qu'à
détecter une variation entre versions du code.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.config import DEFAULT_DATA_PATH, RANDOM_STATE, SAMPLE_DATA_PATH, TARGET  # noqa: E402
from app.preprocessing import load_data  # noqa: E402


def main() -> None:
    df = load_data(DEFAULT_DATA_PATH).drop_duplicates()
    fraud = df[df[TARGET] == 1].sample(100, random_state=RANDOM_STATE)
    normal = df[df[TARGET] == 0].sample(4900, random_state=RANDOM_STATE)
    sample = pd.concat([fraud, normal]).sort_values("Time").reset_index(drop=True)
    sample.to_csv(SAMPLE_DATA_PATH, index=False, float_format="%.6f")
    print(f"{SAMPLE_DATA_PATH} : {len(sample)} lignes, {SAMPLE_DATA_PATH.stat().st_size / 1e6:.2f} Mo")


if __name__ == "__main__":
    main()
