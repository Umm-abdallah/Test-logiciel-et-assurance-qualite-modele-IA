"""Indicateurs de surveillance post-déploiement (prototype)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import PSI_ALERT_THRESHOLD, PSI_WARNING_THRESHOLD


def population_stability_index(reference, current, n_bins: int = 10) -> float:
    """PSI entre une distribution de référence et une distribution courante.

    Les bornes sont les quantiles de la référence ; une petite constante évite
    les logarithmes de zéro. Usage courant : < 0,10 stable ; 0,10 à 0,20 à
    surveiller ; > 0,20 dérive significative.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("Distributions vides")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / ref.size
    cur_pct = np.histogram(cur, edges)[0] / cur.size
    eps = 1e-6
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_status(psi: float) -> str:
    if psi >= PSI_ALERT_THRESHOLD:
        return "alerte"
    if psi >= PSI_WARNING_THRESHOLD:
        return "surveillance"
    return "stable"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns=None) -> pd.DataFrame:
    """PSI par variable, trié par ordre décroissant."""
    cols = columns or [c for c in reference.columns if c in current.columns]
    rows = []
    for c in cols:
        psi = population_stability_index(reference[c], current[c])
        rows.append({"variable": c, "psi": psi, "statut": psi_status(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def flag_rate_alert(
    flag_rate: float,
    expected_rate: float,
    n: int | None = None,
    rel_tolerance: float = 0.5,
    z: float = 3.0,
) -> str:
    """Alerte si le taux de transactions signalées s'écarte de l'attendu.

    Indicateur disponible immédiatement, sans attendre les labels. La bande
    tolérée est la plus large de ``rel_tolerance`` x attendu et, si le volume
    ``n`` de la fenêtre est fourni, de ``z`` écarts types binomiaux : sur une
    petite fenêtre, quelques signalements de plus ne sont pas significatifs.
    """
    if expected_rate <= 0:
        raise ValueError("expected_rate doit être positif")
    band = rel_tolerance * expected_rate
    if n:
        band = max(band, z * float(np.sqrt(expected_rate * (1 - expected_rate) / n)))
    return "alerte" if abs(flag_rate - expected_rate) > band else "normal"
