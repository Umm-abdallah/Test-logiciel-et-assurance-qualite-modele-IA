"""Contrôle du contrat de données et préparation des variables.

Deux comportements sont distingués, conformément au plan de test :

* REJET explicite (``DataValidationError``) lorsque le contrat est violé :
  colonne absente ou dupliquée, type non numérique, valeur manquante ou
  infinie, cible hors {0, 1}, montant ou temps négatif, jeu vide.
* TRAITEMENT documenté lorsque la règle le prévoit : colonnes réordonnées
  selon le contrat, colonnes supplémentaires ignorées en mode lot (avec
  avertissement) mais rejetées en mode strict (entrées API), montants très
  élevés traités mais signalés par une alerte (``range_alerts``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app.config import ALLOWED_TARGET_VALUES, FEATURES, PCA_FEATURES, TARGET

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Levée lorsque des données violent le contrat d'entrée."""

    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        super().__init__("Données invalides : " + " | ".join(self.issues))


@dataclass
class ValidationReport:
    """Résultat détaillé d'une validation."""

    n_rows: int
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _is_numeric_dtype(series: pd.Series) -> bool:
    """Numérique au sens du contrat : int ou float, booléens exclus."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def validate_data(
    df: pd.DataFrame,
    require_target: bool = True,
    strict: bool = False,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Vérifie le schéma, les types et les valeurs d'un DataFrame.

    Paramètres
    ----------
    df : données à contrôler.
    require_target : exige la colonne ``Class`` (vrai pour l'entraînement,
        faux pour l'inférence).
    strict : si vrai, toute colonne non prévue par le contrat est une erreur
        (mode API). Sinon elle est ignorée avec un avertissement (mode lot).
    raise_on_error : lève ``DataValidationError`` si au moins une anomalie.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError([f"Type d'entrée attendu DataFrame, reçu {type(df).__name__}"])

    report = ValidationReport(n_rows=len(df))
    expected = FEATURES + ([TARGET] if require_target else [])

    # 1. Données vides
    if df.shape[0] == 0:
        report.issues.append("Jeu de données vide (0 ligne)")

    # 2. Colonnes dupliquées
    duplicated = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated:
        report.issues.append(f"Colonnes dupliquées : {duplicated}")

    # 3. Colonnes manquantes / supplémentaires
    missing = [c for c in expected if c not in df.columns]
    if missing:
        report.issues.append(f"Colonnes manquantes : {missing}")
    allowed = set(FEATURES) | {TARGET}
    extra = [c for c in df.columns if c not in allowed]
    if extra:
        msg = f"Colonnes non prévues par le contrat : {extra}"
        if strict:
            report.issues.append(msg)
        else:
            report.warnings.append(msg + " (ignorées)")

    # Les contrôles de valeurs n'ont de sens que sur des colonnes uniques présentes
    present = [c for c in expected if c in df.columns and c not in duplicated]

    # 4. Types
    non_numeric = [c for c in present if not _is_numeric_dtype(df[c])]
    if non_numeric:
        report.issues.append(f"Types non numériques : {non_numeric}")
    numeric_cols = [c for c in present if c not in non_numeric]

    if df.shape[0] > 0 and numeric_cols:
        values = df[numeric_cols]
        # 5. Valeurs manquantes
        na_counts = values.isna().sum()
        na_cols = na_counts[na_counts > 0]
        if not na_cols.empty:
            report.issues.append(f"Valeurs manquantes : {na_cols.to_dict()}")
        # 6. Valeurs infinies
        inf_counts = np.isinf(values.to_numpy(dtype=float)).sum(axis=0)
        inf_cols = {c: int(n) for c, n in zip(numeric_cols, inf_counts, strict=True) if n > 0}
        if inf_cols:
            report.issues.append(f"Valeurs infinies : {inf_cols}")
        # 7. Domaines
        for col in ("Amount", "Time"):
            if col in numeric_cols and (df[col] < 0).any():
                report.issues.append(f"Valeurs négatives dans {col} : {int((df[col] < 0).sum())}")
        if TARGET in numeric_cols and require_target:
            target = df[TARGET].dropna()
            invalid = sorted(set(target.unique()) - set(ALLOWED_TARGET_VALUES))
            if invalid:
                report.issues.append(f"Valeurs de cible hors {{0, 1}} : {invalid[:10]}")

    if report.warnings:
        for w in report.warnings:
            logger.warning(w)
    if raise_on_error and not report.is_valid:
        raise DataValidationError(report.issues)
    return report


def preprocess_data(
    df: pd.DataFrame,
    require_target: bool = True,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Valide puis met les données au format attendu par le modèle.

    Retourne ``(X, y)`` avec ``X`` limité aux variables du contrat, dans
    l'ordre du contrat, en ``float64`` ; ``y`` est ``None`` en inférence.
    Aucune transformation apprise n'est appliquée ici : la standardisation
    est réalisée dans le Pipeline afin d'être ajustée sur l'entraînement seul.
    """
    validate_data(df, require_target=require_target, strict=strict)
    X = df.loc[:, FEATURES].astype("float64")
    y = df[TARGET].astype("int64") if require_target else None
    return X, y


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les lignes strictement identiques (règle documentée).

    Des doublons répartis entre entraînement et test gonflent artificiellement
    les performances (fuite par mémorisation). On les retire donc avant la
    partition.
    """
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


def build_preprocessor() -> ColumnTransformer:
    """Transformations apprises, destinées à être placées dans un Pipeline.

    * ``Amount`` : log1p (forte asymétrie, valeurs extrêmes) puis standardisation.
    * ``Time`` et ``V1`` à ``V28`` : standardisation.
    """
    amount_pipe = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("amount", amount_pipe, ["Amount"]),
            ("num", StandardScaler(), ["Time", *PCA_FEATURES]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def compute_reference_stats(X_train: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Statistiques de référence calculées sur l'entraînement uniquement."""
    stats = {}
    for col in FEATURES:
        s = X_train[col]
        stats[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "p001": float(s.quantile(0.001)),
            "p999": float(s.quantile(0.999)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
        }
    return stats


def range_alerts(X: pd.DataFrame, reference_stats: dict) -> pd.Series:
    """Signale les lignes hors du domaine observé à l'entraînement.

    Ces lignes ne sont pas rejetées (le modèle peut produire une probabilité
    valide) mais doivent être marquées pour revue, car le modèle extrapole.
    Retourne une série booléenne indexée comme ``X``.
    """
    flags = pd.Series(False, index=X.index)
    for col, st in reference_stats.items():
        if col in X.columns:
            flags |= (X[col] > st["max"]) | (X[col] < st["min"])
    return flags


class DataLoadError(RuntimeError):
    """Levée lorsque le fichier de données est absent ou illisible."""


def load_data(path) -> pd.DataFrame:
    """Charge le CSV Kaggle (ou l'archive zip qui le contient)."""
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        raise DataLoadError(
            f"Fichier introuvable : {path}. Télécharger creditcard.csv depuis Kaggle (voir data/README.md)."
        )
    if path.suffix.lower() not in {".csv", ".zip"}:
        raise DataLoadError(f"Extension non prise en charge : {path.suffix}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Lecture impossible de {path} : {exc}") from exc
    if df.shape[1] < 2:
        raise DataLoadError(f"Contenu inattendu dans {path} (séparateur ou format incorrect)")
    return df
