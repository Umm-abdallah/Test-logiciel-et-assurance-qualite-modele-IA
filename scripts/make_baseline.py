"""Génère (ou régénère, sur décision explicite) une baseline de non-régression.

Usage :
    python scripts/make_baseline.py synthetic   # tests/baselines/synthetic_baseline.json
    python scripts/make_baseline.py sample      # tests/baselines/sample_baseline.json
    python scripts/make_baseline.py full        # reports/baseline.json

La baseline n'est jamais régénérée par la CI : la régénérer revient à
accepter une nouvelle référence, décision à tracer dans le rapport.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_DATA_PATH, REPORTS_DIR, SAMPLE_DATA_PATH  # noqa: E402
from app.experiment import run_reference  # noqa: E402
from app.preprocessing import load_data  # noqa: E402
from app.training import write_json  # noqa: E402


def main(kind: str) -> None:
    if kind == "synthetic":
        from tests.conftest import make_synthetic

        df, out = make_synthetic(), ROOT / "tests/baselines/synthetic_baseline.json"
    elif kind == "sample":
        df, out = load_data(SAMPLE_DATA_PATH), ROOT / "tests/baselines/sample_baseline.json"
    elif kind == "full":
        df, out = load_data(DEFAULT_DATA_PATH), REPORTS_DIR / "baseline.json"
    else:
        raise SystemExit(f"Type inconnu : {kind}")
    baseline = run_reference(df)
    write_json(baseline, out)
    m = baseline["validation_metrics"]
    print(f"Baseline écrite : {out}")
    print(f"  PR-AUC={m['pr_auc']:.4f} recall={m['recall']:.4f}")
    print(f"  precision={m['precision']:.4f} fpr={m['fpr']:.5f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
