"""Écarts de performance par segment opérationnel (jeu de test, modèle final).

Le jeu ne contient aucun attribut démographique (âge, sexe, localisation) :
cette analyse porte sur des segments OPÉRATIONNELS observables (tranche de
montant, période de la journée). Elle illustre la méthode d'audit, avec
intervalles de Wilson, sans permettre de conclusion sur l'équité envers des
groupes de personnes.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import DEFAULT_DATA_PATH, MODELS_DIR, REPORTS_DIR  # noqa: E402
from app.preprocessing import drop_exact_duplicates, load_data, preprocess_data  # noqa: E402
from app.training import load_model, split_data  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    b = load_model(MODELS_DIR / "model.joblib")
    df, _ = drop_exact_duplicates(load_data(DEFAULT_DATA_PATH))
    X, y = preprocess_data(df)
    s = split_data(X, y)
    t = s.X_test.assign(y=s.y_test.to_numpy())
    t["pred"] = (b["pipeline"].predict_proba(s.X_test)[:, 1] >= b["threshold"]).astype(int)
    t["tranche_montant"] = pd.cut(
        t["Amount"], [-0.01, 10, 100, np.inf], labels=["0 à 10", "10 à 100", "plus de 100"]
    )
    hour = (t["Time"] % 86_400) // 3600
    t["periode"] = np.where(hour < 7, "nuit (0 h à 7 h)", "jour (7 h à 24 h)")

    rows = []
    for dim in ("tranche_montant", "periode"):
        for g, d in t.groupby(dim, observed=True):
            tp = int(((d.y == 1) & (d.pred == 1)).sum())
            fn = int(((d.y == 1) & (d.pred == 0)).sum())
            fp = int(((d.y == 0) & (d.pred == 1)).sum())
            tn = int(((d.y == 0) & (d.pred == 0)).sum())
            r_lo, r_hi = wilson(tp, tp + fn)
            f_lo, f_hi = wilson(fp, fp + tn)
            rows.append(
                {
                    "dimension": dim,
                    "segment": str(g),
                    "n": len(d),
                    "fraudes": tp + fn,
                    "rappel": tp / (tp + fn) if tp + fn else np.nan,
                    "rappel_ic95": f"[{r_lo:.2f} ; {r_hi:.2f}]",
                    "precision": tp / (tp + fp) if tp + fp else np.nan,
                    "fpr": fp / (fp + tn) if fp + tn else np.nan,
                    "fpr_ic95": f"[{f_lo:.5f} ; {f_hi:.5f}]",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "segment_analysis.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
