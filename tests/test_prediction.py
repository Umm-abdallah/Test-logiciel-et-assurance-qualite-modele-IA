"""Tests unitaires : entraînement, prédiction, métriques et service (T04, T11, T12, T14)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from app.config import FEATURES, TARGET
from app.evaluation import check_acceptance, evaluate_model
from app.prediction import FraudScoringService, predict_fraud
from app.preprocessing import DataValidationError
from app.training import build_pipeline, load_model, select_threshold, train_model

# Contrat de sortie (T04)


def test_t04_probabilities_are_valid(trained, splits):
    pipe, _ = trained
    proba = pipe.predict_proba(splits.X_test)[:, 1]
    assert proba.shape == (len(splits.X_test),)
    assert np.isfinite(proba).all()
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_fraud_output_contract(trained, splits):
    pipe, thr = trained
    out = predict_fraud(pipe, splits.X_test, threshold=thr)
    assert list(out.columns) == ["fraud_probability", "is_fraud"]
    assert out.index.equals(splits.X_test.index)
    assert set(out["is_fraud"].unique()) <= {0, 1}
    expected = (out["fraud_probability"] >= thr).astype(int)
    pd.testing.assert_series_equal(out["is_fraud"], expected, check_names=False)


def test_predict_fraud_threshold_extremes(trained, splits):
    pipe, _ = trained
    assert predict_fraud(pipe, splits.X_test, threshold=0.0)["is_fraud"].all()
    high = predict_fraud(pipe, splits.X_test, threshold=1.0)["is_fraud"]
    assert high.sum() <= (pipe.predict_proba(splits.X_test)[:, 1] >= 1.0).sum()


@pytest.mark.parametrize("thr", [-0.1, 1.5])
def test_predict_fraud_rejects_invalid_threshold(trained, splits, thr):
    with pytest.raises(ValueError, match="Seuil"):
        predict_fraud(trained[0], splits.X_test, threshold=thr)


def test_predict_fraud_rejects_invalid_input(trained, splits):
    X = splits.X_test.head(5).copy()
    X.iloc[0, 3] = np.nan
    with pytest.raises(DataValidationError):
        predict_fraud(trained[0], X)


def test_predict_fraud_ignores_target_column_if_present(trained, splits):
    X = splits.X_test.head(10)
    with_target = X.assign(**{TARGET: 0})
    a = predict_fraud(trained[0], X)
    b = predict_fraud(trained[0], with_target)
    pd.testing.assert_frame_equal(a, b)


# Entraînement


def test_train_model_rejects_single_class(splits):
    y = pd.Series(0, index=splits.X_train.index)
    with pytest.raises(ValueError, match="deux classes"):
        train_model(splits.X_train, y)


def test_train_model_rejects_dimension_mismatch(splits):
    with pytest.raises(ValueError, match="Dimensions"):
        train_model(splits.X_train, splits.y_train.iloc[:-1])


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        build_pipeline("xgboost")


@pytest.mark.parametrize("name", ["logistic_regression", "decision_tree"])
def test_training_is_deterministic(splits, name):
    a = train_model(splits.X_train, splits.y_train, name).predict_proba(splits.X_val)
    b = train_model(splits.X_train, splits.y_train, name).predict_proba(splits.X_val)
    np.testing.assert_array_equal(a, b)


def test_model_beats_random_on_learnable_signal(trained, splits):
    """Sur des données synthétiques séparables, le modèle doit apprendre."""
    pipe, _ = trained
    proba = pipe.predict_proba(splits.X_test)[:, 1]
    m = evaluate_model(splits.y_test, (proba >= 0.5).astype(int), proba)
    assert m["roc_auc"] > 0.9
    assert m["pr_auc"] > 5 * splits.y_test.mean()


# Métriques (valeurs calculées à la main)


def test_evaluate_model_known_values():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    m = evaluate_model(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(1 / 6)
    assert m["fnr"] == pytest.approx(0.25)
    assert m["accuracy"] == pytest.approx(0.8)


def test_high_accuracy_can_hide_zero_recall():
    """Illustration du piège de l'accuracy sur classe rare (0,2 % de fraudes)."""
    y_true = np.array([1] * 2 + [0] * 998)
    y_pred = np.zeros(1000, dtype=int)  # modèle qui ne détecte jamais rien
    m = evaluate_model(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0.0
    assert m["fnr"] == 1.0


def test_evaluate_model_perfect_proba():
    y = np.array([0, 0, 1, 1])
    m = evaluate_model(y, y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0


def test_evaluate_model_single_class_has_no_auc():
    m = evaluate_model([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3])
    assert m["roc_auc"] is None and m["pr_auc"] is None
    assert m["precision"] == 0.0  # zero_division explicite


@pytest.mark.parametrize(
    "y_true, y_pred, y_proba, msg",
    [
        ([0, 1], [0], None, "Dimensions"),
        ([], [], None, "Aucune"),
        ([0, 2], [0, 1], None, "binaire"),
        ([0, 1], [0, 1], [0.2, 1.3], "probabilités"),
        ([0, 1], [0, 1], [0.2, np.nan], "probabilités"),
    ],
)
def test_evaluate_model_rejects_invalid_inputs(y_true, y_pred, y_proba, msg):
    with pytest.raises(ValueError, match=msg):
        evaluate_model(y_true, y_pred, y_proba)


def test_check_acceptance():
    ok = {"pr_auc": 0.8, "recall": 0.8, "precision": 0.6, "fpr": 0.0005}
    assert check_acceptance(ok)["all_passed"]
    ko = {**ok, "fpr": 0.01}
    res = check_acceptance(ko)
    assert not res["all_passed"] and not res["fpr"]["passed"] and res["recall"]["passed"]


# Seuil (T12)


def test_select_threshold_respects_recall_constraint():
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(50), np.zeros(950)].astype(int)
    p = np.r_[rng.uniform(0.3, 1, 50), rng.uniform(0, 0.6, 950)]
    res = select_threshold(y, p, min_recall=0.8, max_fpr=1.0)
    assert res["constraint_met"]
    achieved = ((p >= res["threshold"]) & (y == 1)).sum() / 50
    assert achieved >= 0.8
    assert achieved == pytest.approx(res["val_recall"])


def test_select_threshold_fallback_when_constraint_unreachable():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.1, 0.05])
    res = select_threshold(y, p, min_recall=1.01)
    assert not res["constraint_met"]
    assert 0 <= res["threshold"] <= 1


# Service de scoring et sécurité opérationnelle (T11, T14)


@pytest.fixture
def service(saved_bundle_path):
    return FraudScoringService(load_model(saved_bundle_path))


@pytest.fixture
def record(splits):
    return {k: float(v) for k, v in splits.X_test.iloc[0].items()}


def test_service_scores_valid_record(service, record):
    out = service.score(record)
    assert out["status"] == "scored"
    assert 0 <= out["fraud_probability"] <= 1
    assert out["is_fraud"] in (0, 1)
    assert out["out_of_range_alert"] is False


def test_service_rejects_missing_field(service, record):
    record.pop("V10")
    out = service.score(record)
    assert out["status"] == "rejected"
    assert any("V10" in e for e in out["errors"])


def test_service_rejects_extra_field(service, record):
    out = service.score({**record, "card_number": 1234})
    assert out["status"] == "rejected"
    assert any("card_number" in e for e in out["errors"])


@pytest.mark.parametrize("bad", ["12.5", None, True, [1.0]])
def test_service_rejects_wrong_type(service, record, bad):
    record["Amount"] = bad
    assert service.score(record)["status"] == "rejected"


def test_service_rejects_non_dict(service):
    assert service.score([1, 2, 3])["status"] == "rejected"


def test_service_flags_out_of_range_amount(service, record):
    record["Amount"] = 5e7
    out = service.score(record)
    assert out["status"] == "scored"
    assert out["out_of_range_alert"] is True


def test_logs_do_not_leak_transaction_content(service, record, caplog):
    record["Amount"] = 9876.54
    with caplog.at_level(logging.DEBUG, logger="aitestguard"):
        service.score(record, request_id="req-1")
        service.score({**record, "V3": "texte"}, request_id="req-2")
    text = caplog.text
    assert "req-1" in text and "req-2" in text
    assert "9876" not in text
    for name in FEATURES:
        assert f"{record[name]:.4f}" not in text


def test_bootstrap_ci_brackets_point_estimate(trained, splits):
    from app.evaluation import bootstrap_ci

    pipe, thr = trained
    p = pipe.predict_proba(splits.X_test)[:, 1]
    point = evaluate_model(splits.y_test, (p >= thr).astype(int), p)
    ci = bootstrap_ci(splits.y_test, p, thr, n_boot=200)
    for k in ("precision", "recall", "fpr", "pr_auc"):
        lo, hi = ci[k]
        assert lo <= point[k] <= hi
        assert 0 <= lo <= hi <= 1


def test_select_threshold_respects_fpr_constraint():
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(100), np.zeros(9900)].astype(int)
    p = np.r_[rng.uniform(0.2, 1, 100), rng.beta(1, 30, 9900)]
    res = select_threshold(y, p, min_recall=0.5, max_fpr=0.001)
    fpr = ((p >= res["threshold"]) & (y == 0)).sum() / 9900
    assert res["constraint_met"] and fpr <= 0.001
    assert fpr == pytest.approx(res["val_fpr"])


def test_select_threshold_never_returns_flag_everything_when_constraint_fails():
    """Un arbre aux probabilités discrètes ne doit pas aboutir à « tout signaler »."""
    y = np.r_[np.ones(10), np.zeros(990)].astype(int)
    p = np.r_[np.full(6, 0.9), np.full(4, 0.0), np.full(5, 0.9), np.zeros(985)]
    res = select_threshold(y, p, min_recall=0.8, max_fpr=0.001)
    assert not res["constraint_met"]
    assert res["threshold"] > 0
