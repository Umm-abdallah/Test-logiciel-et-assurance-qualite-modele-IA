"""Compare des versions candidates à la baseline figée (reports/baseline.json).

Chaque candidate est entraînée avec la même procédure et la même graine ;
ses métriques de VALIDATION sont comparées à la référence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.config import DEFAULT_DATA_PATH, REPORTS_DIR  # noqa: E402
from app.evaluation import compare_to_baseline  # noqa: E402
from app.experiment import run_reference  # noqa: E402
from app.preprocessing import load_data  # noqa: E402
from app.training import write_json  # noqa: E402

CANDIDATES = {
    "v1.1 arbre profondeur 8": (
        "decision_tree",
        {"max_depth": 8, "min_samples_leaf": 20, "class_weight": None},
    ),
    "v1.2 arbre pondéré (class_weight balanced)": (
        "decision_tree",
        {"max_depth": 6, "min_samples_leaf": 20, "class_weight": "balanced"},
    ),
    "v1.3 régression logistique C=1": ("logistic_regression", {"C": 1.0, "class_weight": None}),
    "v1.4 arbre profondeur 1 (régression injectée)": ("decision_tree", {"max_depth": 1}),
}


def main() -> None:
    baseline = json.loads((REPORTS_DIR / "baseline.json").read_text())
    df = load_data(DEFAULT_DATA_PATH)
    rows, details = [], {}
    bm = baseline["validation_metrics"]
    rows.append(
        {
            "version": "v1.0 référence",
            **{k: bm[k] for k in ("pr_auc", "roc_auc", "recall", "precision", "fpr")},
            "décision": "référence",
        }
    )
    for label, (model, params) in CANDIDATES.items():
        cand = run_reference(df, model, params)
        res = compare_to_baseline(cand["validation_metrics"], bm)
        m = cand["validation_metrics"]
        rows.append(
            {
                "version": label,
                **{k: m[k] for k in ("pr_auc", "roc_auc", "recall", "precision", "fpr")},
                "décision": res["decision"],
            }
        )
        details[label] = {"model": model, "params": params, "threshold": cand["threshold"], "comparison": res}
        print(label, res["decision"])
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "regression_comparison.csv", index=False)
    write_json(details, REPORTS_DIR / "regression_comparison.json")
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
