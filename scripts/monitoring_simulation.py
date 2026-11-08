"""Simulation de surveillance post-déploiement sur le jeu de test ordonné dans le temps.

Le jeu de test est découpé en fenêtres de 4 heures (champ Time) jouant le
rôle d'un flux de production. Pour chaque fenêtre :
* indicateurs IMMÉDIATS (sans labels) : volume, taux de signalement,
  PSI des scores et des variables par rapport à la validation, alertes de domaine ;
* indicateurs DIFFÉRÉS (labels connus après enquête / rétrofacturation) :
  rappel, précision, FPR.
Une fenêtre est ensuite perturbée artificiellement pour vérifier le
déclenchement des alertes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import DEFAULT_DATA_PATH, FIGURES_DIR, MODELS_DIR, REPORTS_DIR  # noqa: E402
from app.evaluation import evaluate_model  # noqa: E402
from app.monitoring import flag_rate_alert, population_stability_index, psi_status  # noqa: E402
from app.preprocessing import drop_exact_duplicates, load_data, preprocess_data, range_alerts  # noqa: E402
from app.training import load_model, split_data  # noqa: E402

WINDOW_S = 4 * 3600


def main(data: Path, model_path: Path) -> None:
    bundle = load_model(model_path)
    pipe, thr, stats = bundle["pipeline"], bundle["threshold"], bundle["metadata"]["reference_stats"]
    df, _ = drop_exact_duplicates(load_data(data))
    X, y = preprocess_data(df)
    s = split_data(X, y)
    p_val = pipe.predict_proba(s.X_val)[:, 1]
    expected_flag = float((p_val >= thr).mean())

    test = s.X_test.assign(y=s.y_test).sort_values("Time")
    test["window"] = (test["Time"] // WINDOW_S).astype(int)
    # fenêtre 9 dégradée volontairement : dérive sur V14 (variable très discriminante)
    test.loc[test.window == 9, "V14"] -= 1.5 * s.X_train["V14"].std()

    rows = []
    for w, g in test.groupby("window"):
        Xg = g.drop(columns=["y", "window"])
        p = pipe.predict_proba(Xg)[:, 1]
        flag = float((p >= thr).mean())
        psi_score = population_stability_index(p_val, p)
        psi_v14 = population_stability_index(s.X_val["V14"], Xg["V14"])
        m = evaluate_model(g["y"], (p >= thr).astype(int), p) if g["y"].sum() > 0 else None
        rows.append(
            {
                "fenetre": int(w),
                "debut_h": int(w * 4),
                "n": len(g),
                "n_fraudes": int(g["y"].sum()),
                "taux_signalement": flag,
                "alerte_signalement": flag_rate_alert(flag, expected_flag, n=len(g)),
                "psi_scores": psi_score,
                "statut_psi_scores": psi_status(psi_score),
                "psi_V14": psi_v14,
                "statut_psi_V14": psi_status(psi_v14),
                "part_hors_domaine": float(range_alerts(Xg, stats).mean()),
                "rappel_differe": m["recall"] if m else None,
                "precision_differee": m["precision"] if m else None,
                "fpr_differe": m["fpr"] if m else None,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS_DIR / "monitoring_simulation.csv", index=False)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.plot(out.debut_h, out.taux_signalement * 100, color="#2a78d6", marker="o", ms=4)
    a1.axhline(expected_flag * 100, color="#595959", ls="--", lw=1, label="Taux attendu (validation)")
    alerts = out[out.alerte_signalement == "alerte"]
    a1.scatter(
        alerts.debut_h,
        alerts.taux_signalement * 100,
        s=70,
        facecolor="none",
        edgecolor="#e34948",
        linewidth=1.5,
        zorder=5,
        label="Alerte sur le taux de signalement",
    )
    a1.set(ylabel="Transactions signalées (%)", title="Indicateurs immédiats par fenêtre de 4 h (test)")
    a1.legend(frameon=False, loc="upper right")
    a2.plot(out.debut_h, out.psi_V14, color="#eb6834", marker="o", ms=4, label="PSI V14")
    a2.plot(out.debut_h, out.psi_scores, color="#6250d6", marker="o", ms=4, label="PSI des scores")
    a2.axhline(0.2, color="#595959", ls="--", lw=1, label="Seuil d'alerte 0,20")
    a2.set(xlabel="Heure de début de fenêtre (depuis la 1re transaction)", ylabel="PSI")
    a2.legend(frameon=False, loc="upper left")
    for a in (a1, a2):
        a.axvline(36, color="#e34948", lw=0.8, ls=":")
    a2.annotate(
        "dérive injectée",
        xy=(36, out.psi_V14.max()),
        xytext=(38, out.psi_V14.max() * 0.85),
        color="#52514e",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "monitoring_simulation.png")
    print(f"Taux de signalement attendu : {expected_flag:.4%}")
    print(out.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "model.joblib")
    a = ap.parse_args()
    main(a.data, a.model)
