"""Scénarios de robustesse sur le modèle final et le jeu de test.

Pour chaque scénario : comportement attendu (rejet, traitement, alerte),
comportement observé, et mesures de sensibilité.
Usage : python scripts/run_robustness.py [--data ...] [--model models/model.joblib]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.config import DEFAULT_DATA_PATH, FEATURES, MODELS_DIR, REPORTS_DIR  # noqa: E402
from app.monitoring import drift_report  # noqa: E402
from app.prediction import FraudScoringService, predict_fraud  # noqa: E402
from app.preprocessing import (  # noqa: E402
    DataValidationError,
    drop_exact_duplicates,
    load_data,
    preprocess_data,
    range_alerts,
    validate_data,
)
from app.robustness import perturb_amount, perturb_pca, sensitivity, shift_time  # noqa: E402
from app.training import load_model, split_data, write_json  # noqa: E402


def try_predict(pipe, X, thr) -> str:
    try:
        predict_fraud(pipe, X, thr)
        return "traité"
    except DataValidationError as e:
        return f"rejeté ({len(e.issues)} anomalie(s))"


def main(data: Path, model_path: Path) -> None:
    bundle = load_model(model_path)
    pipe, thr = bundle["pipeline"], bundle["threshold"]
    stats = bundle["metadata"]["reference_stats"]
    df, _ = drop_exact_duplicates(load_data(data))
    X, y = preprocess_data(df)
    s = split_data(X, y)
    Xt, yt = s.X_test, s.y_test
    train_std = s.X_train.std(ddof=0)

    rows: list[dict] = []
    sens: list[dict] = []

    def add(sc, attendu, observe, conforme, detail=""):
        rows.append(
            {"scenario": sc, "attendu": attendu, "observé": observe, "conforme": conforme, "détail": detail}
        )

    # 1. Données manquantes
    for cols, label in ((["V14"], "1 variable"), (["V1", "V4", "Amount"], "3 variables")):
        Xm = Xt.head(1000).copy()
        Xm.loc[Xm.index[:50], cols] = np.nan
        obs = try_predict(pipe, Xm, thr)
        rep = validate_data(Xm, require_target=False, raise_on_error=False)
        add(
            f"Valeurs manquantes ({label}, 50 lignes sur 1000)",
            "rejet du lot",
            obs,
            obs.startswith("rejeté"),
            "; ".join(rep.issues),
        )

    # 2. Infinies, non numériques
    Xi = Xt.head(1000).copy()
    Xi.iloc[:5, Xi.columns.get_loc("V3")] = np.inf
    obs = try_predict(pipe, Xi, thr)
    add("Valeurs infinies (5 lignes)", "rejet", obs, obs.startswith("rejeté"))
    Xs = Xt.head(1000).copy().astype({"Amount": object})
    Xs.iloc[0, Xs.columns.get_loc("Amount")] = "12,50"
    obs = try_predict(pipe, Xs, thr)
    add("Valeur non numérique (Amount = '12,50')", "rejet", obs, obs.startswith("rejeté"))

    service = FraudScoringService(bundle)
    rec = {k: float(v) for k, v in Xt.iloc[0].items()}
    for label, r in (
        ("Champ manquant (API)", {k: v for k, v in rec.items() if k != "V7"}),
        ("Champ supplémentaire (API)", {**rec, "merchant": 3.0}),
        ("Type booléen (API)", {**rec, "Amount": True}),
    ):
        out = service.score(r)
        add(label, "rejet", out["status"], out["status"] == "rejected", "; ".join(out.get("errors", [])))

    # 3. Montants extrêmes
    amax = stats["Amount"]["max"]
    for factor in (2, 10, 1000):
        Xa = Xt.copy()
        Xa["Amount"] = amax * factor
        out = predict_fraud(pipe, Xa, thr)
        flagged = range_alerts(Xa, stats).mean()
        p0 = pipe.predict_proba(Xt)[:, 1]
        valid = out["fraud_probability"].between(0, 1).all()
        add(
            f"Amount = {factor} x max entraînement ({amax * factor:,.0f})".replace(",", " "),
            "traitement + alerte",
            f"traité, {flagged:.0%} alertés, taux de signalement {out['is_fraud'].mean():.2%} "
            f"(référence {(p0 >= thr).mean():.2%})",
            bool(valid and flagged == 1.0),
            f"variation moyenne de probabilité {np.abs(out['fraud_probability'] - p0).mean():.4f}",
        )

    # 4. Perturbations limitées
    for rel in (0.01, 0.05, 0.10, 0.25):
        r = sensitivity(pipe, Xt, perturb_pca(Xt, train_std, rel), yt, thr)
        sens.append({"perturbation": f"Bruit gaussien V1 à V28, sigma = {rel:.0%} de l'écart type", **r})
    for rel in (0.01, 0.05, 0.20):
        r = sensitivity(pipe, Xt, perturb_amount(Xt, rel), yt, thr)
        sens.append({"perturbation": f"Amount multiplié par U(1 ± {rel:.0%})", **r})
    for sec in (60, 3600):
        r = sensitivity(pipe, Xt, shift_time(Xt, sec), yt, thr)
        sens.append({"perturbation": f"Time décalé de +{sec} s", **r})

    # 5. Distribution différente
    shifted = Xt.copy()
    for c in ("V14", "V12", "V10"):
        shifted[c] = shifted[c] + 0.5 * train_std[c]
    rep = drift_report(s.X_train, shifted, columns=FEATURES)
    alerts = rep[rep.statut == "alerte"].variable.tolist()
    r = sensitivity(pipe, Xt, shifted, yt, thr)
    sens.append({"perturbation": "Décalage de +0,5 écart type sur V10, V12, V14", **r})
    add(
        "Dérive simulée (+0,5 sigma sur V10, V12, V14)",
        "alerte de dérive (PSI > 0,20)",
        f"alerte sur {alerts}",
        set(alerts) == {"V10", "V12", "V14"},
        f"rappel {r['delta_recall']:+.3f}, précision {r['delta_precision']:+.3f}, FPR {r['delta_fpr']:+.5f}",
    )

    d1, d2 = df[df["Time"] < 86_400], df[df["Time"] >= 86_400]
    cols_no_time = [c for c in FEATURES if c != "Time"]
    rep_t = drift_report(d1[cols_no_time], d2[cols_no_time], columns=cols_no_time)
    top = rep_t.head(3)
    n_alert = int((rep_t.statut == "alerte").sum())
    add(
        "Dérive naturelle jour 1 vs jour 2 (Time < 86 400 s)",
        "surveillance (PSI)",
        f"{n_alert} variables en alerte ; "
        + ", ".join(f"{v} PSI={p:.2f}" for v, p in zip(top.variable, top.psi, strict=True)),
        True,
        "Time exclu : sa dérive est structurelle (compteur croissant)",
    )
    rep_t.to_csv(REPORTS_DIR / "drift_day1_vs_day2.csv", index=False)

    # 6. Schéma invalide
    for label, Xb in (
        ("Colonne absente (V20)", Xt.head(100).drop(columns=["V20"])),
        ("Colonne renommée (amount)", Xt.head(100).rename(columns={"Amount": "amount"})),
        ("Colonne dupliquée (V1)", pd.concat([Xt.head(100), Xt.head(100)[["V1"]]], axis=1)),
        ("Lot vide", Xt.head(0)),
    ):
        obs = try_predict(pipe, Xb, thr)
        add(label, "rejet", obs, obs.startswith("rejeté"))
    obs = try_predict(pipe, Xt.head(100)[list(reversed(FEATURES))], thr)
    add("Colonnes dans un ordre différent", "traitement (réordonnancement)", obs, obs == "traité")

    pd.DataFrame(rows).to_csv(REPORTS_DIR / "robustness_scenarios.csv", index=False)
    pd.DataFrame(sens).to_csv(REPORTS_DIR / "robustness_sensitivity.csv", index=False)
    write_json({"scenarios": rows, "sensitivity": sens}, REPORTS_DIR / "robustness.json")
    print(pd.DataFrame(rows)[["scenario", "observé", "conforme"]].to_string())
    print(pd.DataFrame(sens).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "model.joblib")
    a = ap.parse_args()
    main(a.data, a.model)
