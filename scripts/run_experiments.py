"""Évaluation complète : comparaison des modèles, seuil, test final, figures.

Ordre imposé (traçable dans reports/experiment_log.json) :
1. validation du schéma, suppression des doublons, partition stratifiée ;
2. réglage des hyperparamètres et du seuil sur la VALIDATION ;
3. choix du modèle final sur la VALIDATION, écrit sur disque ;
4. évaluation UNIQUE de chaque modèle sur le TEST ;
5. sauvegarde du pipeline final.

Usage : python scripts/run_experiments.py [--data data/creditcard.csv]
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    precision_recall_curve,
    roc_curve,
)

from app.config import (  # noqa: E402
    ACCEPTANCE_CRITERIA,
    DEFAULT_DATA_PATH,
    FEATURES,
    FIGURES_DIR,
    MIN_RECALL_FOR_THRESHOLD,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET,
)
from app.evaluation import bootstrap_ci, check_acceptance, evaluate_model  # noqa: E402
from app.preprocessing import (  # noqa: E402
    compute_reference_stats,
    drop_exact_duplicates,
    load_data,
    preprocess_data,
    validate_data,
)
from app.training import (  # noqa: E402
    dataframe_fingerprint,
    save_model,
    select_threshold,
    split_data,
    train_model,
    tune_on_validation,
    write_json,
)

COLORS = {"logistic_regression": "#2a78d6", "decision_tree": "#eb6834", "hasard": "#8f8f8f"}
LABELS = {"logistic_regression": "Régression logistique", "decision_tree": "Arbre de décision"}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#8f8f8f",
        "font.size": 9,
        "lines.linewidth": 2,
    }
)


def fmt(m: dict) -> dict:
    keys = [
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    return {k: m.get(k) for k in keys}


def main(data_path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    log: dict = {"data_path": str(data_path)}

    # 1. Données
    raw = load_data(data_path)
    validate_data(raw)
    df, n_dup = drop_exact_duplicates(raw)
    X, y = preprocess_data(df)
    s = split_data(X, y)
    log["dataset"] = {
        "n_rows_raw": int(len(raw)),
        "n_fraud_raw": int(raw[TARGET].sum()),
        "n_duplicates_removed": int(n_dup),
        "n_fraud_duplicates_removed": int(raw[TARGET].sum() - df[TARGET].sum()),
        "n_rows": int(len(df)),
        "n_fraud": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "fingerprint": dataframe_fingerprint(df[[*FEATURES, TARGET]]),
        "amount_describe": {k: float(v) for k, v in df["Amount"].describe().items()},
        "splits": {
            name: {"n": int(len(yy)), "n_fraud": int(yy.sum()), "fraud_rate": float(yy.mean())}
            for name, yy in (("train", s.y_train), ("validation", s.y_val), ("test", s.y_test))
        },
    }
    print("Données :", log["dataset"]["splits"])

    # 2. Réglage sur la validation
    models = {}
    for name in ("logistic_regression", "decision_tree"):
        pipe, params, grid_results = tune_on_validation(s, name)
        p_val = pipe.predict_proba(s.X_val)[:, 1]
        thr = select_threshold(s.y_val, p_val, MIN_RECALL_FOR_THRESHOLD)
        val_m = evaluate_model(s.y_val, (p_val >= thr["threshold"]).astype(int), p_val)
        val_default = evaluate_model(s.y_val, (p_val >= 0.5).astype(int), p_val)
        models[name] = {"pipe": pipe, "params": params, "threshold": thr, "p_val": p_val}
        log.setdefault("validation", {})[name] = {
            "best_params": params,
            "grid": grid_results,
            "threshold": thr,
            "metrics_at_threshold": fmt(val_m),
            "metrics_at_0_5": fmt(val_default),
        }
        print(f"{name}: {params} PR-AUC val={val_m['pr_auc']:.4f} seuil={thr['threshold']:.4f}")

    # 3. Choix du modèle final sur la validation, AVANT toute lecture du test
    final_name = max(models, key=lambda n: log["validation"][n]["metrics_at_threshold"]["pr_auc"])
    log["selection"] = {
        "criterion": "PR-AUC maximale sur la validation",
        "final_model": final_name,
        "decided_before_test": True,
    }
    write_json(log, REPORTS_DIR / "experiment_log.json")

    # Tableau de sensibilité au seuil (validation uniquement)
    fm = models[final_name]
    rows = []
    for t in sorted({0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, round(fm["threshold"]["threshold"], 4)}):
        m = evaluate_model(s.y_val, (fm["p_val"] >= t).astype(int), fm["p_val"])
        rows.append({"seuil": t, **{k: m[k] for k in ("precision", "recall", "fpr", "tp", "fp", "fn")}})
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "threshold_sensitivity_validation.csv", index=False)

    # 4. Évaluation unique sur le test
    results = {}
    fig_cm, axes_cm = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, (name, md) in zip(axes_cm, models.items(), strict=True):
        pipe, t = md["pipe"], md["threshold"]["threshold"]
        p_tr = pipe.predict_proba(s.X_train)[:, 1]
        p_te = pipe.predict_proba(s.X_test)[:, 1]
        test_m = evaluate_model(s.y_test, (p_te >= t).astype(int), p_te)
        train_m = evaluate_model(s.y_train, (p_tr >= t).astype(int), p_tr)
        ci = bootstrap_ci(s.y_test, p_te, t, n_boot=1000)

        amounts = s.X_test["Amount"].to_numpy()
        is_f, pred = s.y_test.to_numpy() == 1, p_te >= t
        cost = {
            "fraud_amount_total": float(amounts[is_f].sum()),
            "fraud_amount_detected": float(amounts[is_f & pred].sum()),
            "fraud_amount_missed": float(amounts[is_f & ~pred].sum()),
            "fp_amount_median": float(np.median(amounts[~is_f & pred])) if (~is_f & pred).any() else None,
            "fn_amount_median": float(np.median(amounts[is_f & ~pred])) if (is_f & ~pred).any() else None,
            "tp_amount_median": float(np.median(amounts[is_f & pred])) if (is_f & pred).any() else None,
            "fn_proba_quantiles": [float(q) for q in np.quantile(p_te[is_f & ~pred], [0.25, 0.5, 0.75])]
            if (is_f & ~pred).any()
            else None,
            "fp_proba_quantiles": [float(q) for q in np.quantile(p_te[~is_f & pred], [0.25, 0.5, 0.75])]
            if (~is_f & pred).any()
            else None,
        }
        results[name] = {
            "threshold": t,
            "train": fmt(train_m),
            "validation": log["validation"][name]["metrics_at_threshold"],
            "test": fmt(test_m),
            "test_at_0_5": fmt(evaluate_model(s.y_test, (p_te >= 0.5).astype(int), p_te)),
            "test_ci95": ci,
            "acceptance": check_acceptance(test_m),
            "error_analysis": cost,
        }
        md["p_te"] = p_te

        cm = np.array([[test_m["tn"], test_m["fp"]], [test_m["fn"], test_m["tp"]]])
        ax.imshow(np.log1p(cm), cmap="Blues", vmin=0, vmax=np.log1p(cm.max()) * 1.3)
        lab = [["VN", "FP"], ["FN", "VP"]]
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{lab[i][j]}\n{cm[i, j]:,}".replace(",", " "),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#0b0b0b",
                )
        ax.set_xticks([0, 1], ["Prédit normal", "Prédit fraude"])
        ax.set_yticks([0, 1], ["Réel normal", "Réel fraude"])
        ax.set_title(f"{LABELS[name]} (seuil {t:.3f})".replace(".", ","))
        ax.grid(False)
    fig_cm.suptitle("Matrices de confusion sur le jeu de test")
    fig_cm.tight_layout()
    fig_cm.savefig(FIGURES_DIR / "confusion_matrices.png")

    # Courbes ROC et précision-rappel (test)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    base_rate = float(s.y_test.mean())
    for name, md in models.items():
        fpr, tpr, _ = roc_curve(s.y_test, md["p_te"])
        prec, rec, _ = precision_recall_curve(s.y_test, md["p_te"])
        r = results[name]["test"]
        a1.plot(fpr, tpr, color=COLORS[name], label=f"{LABELS[name]} (AUC {r['roc_auc']:.3f})")
        a2.plot(rec, prec, color=COLORS[name], label=f"{LABELS[name]} (AP {r['pr_auc']:.3f})")
        a2.scatter(
            [r["recall"]],
            [r["precision"]],
            color=COLORS[name],
            s=40,
            zorder=5,
            edgecolor="white",
            linewidth=1.5,
        )
    a1.plot([0, 1], [0, 1], ls="--", lw=1, color=COLORS["hasard"], label="Hasard")
    a2.axhline(base_rate, ls="--", lw=1, color=COLORS["hasard"], label=f"Hasard ({base_rate:.4f})")
    a1.set(xlabel="Taux de faux positifs", ylabel="Rappel (taux de vrais positifs)", title="Courbe ROC")
    a2.set(xlabel="Rappel", ylabel="Précision", title="Courbe précision-rappel (point = seuil retenu)")
    a2.axvline(MIN_RECALL_FOR_THRESHOLD, lw=0.8, color="#d6d6d6")
    for a in (a1, a2):
        a.legend(loc="lower right" if a is a1 else "lower left", frameon=False)
        a.set_xlim(-0.01, 1.01)
        a.set_ylim(-0.01, 1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png")

    # Zoom ROC sur la zone utile (FPR < 1 %)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for name, md in models.items():
        fpr, tpr, _ = roc_curve(s.y_test, md["p_te"])
        ax.plot(fpr * 100, tpr, color=COLORS[name], label=LABELS[name])
    ax.axvline(
        ACCEPTANCE_CRITERIA["fpr_max"] * 100,
        ls="--",
        lw=1,
        color="#8f8f8f",
        label=f"FPR max admis ({ACCEPTANCE_CRITERIA['fpr_max']:.1%})",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="Taux de faux positifs (%)",
        ylabel="Rappel",
        title="ROC, zone opérationnelle (FPR < 1 %)",
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_zoom.png")

    # Compromis seuil (validation, modèle final)
    prec, rec, thr = precision_recall_curve(s.y_val, fm["p_val"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(thr, prec[:-1], color="#2a78d6", label="Précision")
    ax.plot(thr, rec[:-1], color="#eb6834", label="Rappel")
    ax.axvline(
        fm["threshold"]["threshold"],
        color="#595959",
        lw=1,
        ls="--",
        label=f"Seuil retenu {fm['threshold']['threshold']:.3f}".replace(".", ","),
    )
    ax.axvline(0.5, color="#d6d6d6", lw=1, label="Seuil par défaut 0,5")
    ax.set(
        xlabel="Seuil de décision",
        ylabel="Valeur",
        title=f"Compromis précision/rappel ({LABELS[final_name]}, validation)",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_tradeoff_validation.png")

    # Distribution des probabilités par classe (test, modèle final)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(0, 1, 41)
    p = models[final_name]["p_te"]
    ax.hist(p[s.y_test == 0], bins=bins, color="#8f8f8f", alpha=0.8, label="Normales", log=True)
    ax.hist(p[s.y_test == 1], bins=bins, color="#e34948", alpha=0.85, label="Fraudes", log=True)
    ax.axvline(models[final_name]["threshold"]["threshold"], color="#0b0b0b", lw=1, ls="--", label="Seuil")
    ax.set(
        xlabel="Probabilité de fraude prédite",
        ylabel="Effectif (échelle log)",
        title=f"Séparation des classes, {LABELS[final_name]} (test)",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "proba_distribution.png")
    plt.close("all")

    # Modèle trivial pour illustrer l'accuracy
    trivial = evaluate_model(s.y_test, np.zeros(len(s.y_test), dtype=int))
    log["trivial_all_normal_test"] = fmt(trivial)

    # Contrôle hors temps (complément, non utilisé pour la décision)
    order = df.sort_values("Time")
    cut1, cut2 = int(0.6 * len(order)), int(0.8 * len(order))
    tr, va, te = order.iloc[:cut1], order.iloc[cut1:cut2], order.iloc[cut2:]
    pipe_t = train_model(tr[FEATURES], tr[TARGET], final_name, **models[final_name]["params"])
    t_t = select_threshold(va[TARGET], pipe_t.predict_proba(va[FEATURES])[:, 1])["threshold"]
    p_t = pipe_t.predict_proba(te[FEATURES])[:, 1]
    log["out_of_time"] = {
        "description": (
            "Entraînement sur les 60 % les plus anciens, seuil sur les 20 % suivants, "
            "test sur les 20 % les plus récents"
        ),
        "threshold": t_t,
        "test": fmt(evaluate_model(te[TARGET], (p_t >= t_t).astype(int), p_t)),
        "n_fraud_test": int(te[TARGET].sum()),
    }

    # 5. Sauvegarde du modèle final
    MODELS_DIR.mkdir(exist_ok=True)
    save_model(
        models[final_name]["pipe"],
        models[final_name]["threshold"]["threshold"],
        MODELS_DIR / "model.joblib",
        metadata={
            "model": final_name,
            "params": models[final_name]["params"],
            "dataset_fingerprint": log["dataset"]["fingerprint"],
            "reference_stats": compute_reference_stats(s.X_train),
        },
    )
    log["results"] = results
    write_json(log, REPORTS_DIR / "experiment_log.json")

    table = pd.DataFrame(
        {
            LABELS[n]: {
                "Hyperparamètres": str(models[n]["params"]),
                "Seuil (validation)": round(r["threshold"], 4),
                **{
                    f"{k} test": r["test"][k]
                    for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "fpr", "fnr", "accuracy")
                },
                **{k.upper(): r["test"][k] for k in ("tp", "fp", "fn", "tn")},
                "PR-AUC entraînement": r["train"]["pr_auc"],
                "PR-AUC validation": r["validation"]["pr_auc"],
                "Critères acceptés": r["acceptance"]["all_passed"],
            }
            for n, r in results.items()
        }
    )
    table.to_csv(REPORTS_DIR / "model_comparison.csv")
    print(table.to_string())
    print("Hors temps :", log["out_of_time"]["test"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    main(ap.parse_args().data)
