"""Tests d'intégration : pipeline de bout en bout et prévention des fuites (T05, T09, T13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import FEATURES, TARGET
from app.evaluation import evaluate_model
from app.prediction import predict_fraud
from app.preprocessing import (
    drop_exact_duplicates,
    load_data,
    preprocess_data,
    validate_data,
)
from app.training import (
    DataSplits,
    ModelLoadError,
    load_model,
    save_model,
    select_threshold,
    split_data,
    tune_on_validation,
)


def test_end_to_end_scenario(tmp_path, synthetic_df):
    """Charger, valider, séparer, entraîner, régler, sauver, recharger, prédire, rapporter."""
    csv = tmp_path / "creditcard.csv"
    synthetic_df.to_csv(csv, index=False)

    df = load_data(csv)
    validate_data(df)
    df, _ = drop_exact_duplicates(df)
    X, y = preprocess_data(df)
    s = split_data(X, y)

    pipe, params, _ = tune_on_validation(s, "logistic_regression", grid=[{"C": 0.1}, {"C": 1.0}])
    thr = select_threshold(s.y_val, pipe.predict_proba(s.X_val)[:, 1])

    path = save_model(pipe, thr["threshold"], tmp_path / "model.joblib", {"params": params})
    bundle = load_model(path)
    preds = predict_fraud(bundle["pipeline"], s.X_test, bundle["threshold"])
    report = evaluate_model(s.y_test, preds["is_fraud"], preds["fraud_probability"])

    assert len(preds) == len(s.X_test)
    assert report["tp"] + report["fn"] == int(s.y_test.sum())
    assert report["tp"] + report["tn"] + report["fp"] + report["fn"] == len(s.X_test)
    assert bundle["threshold"] == pytest.approx(thr["threshold"])
    assert bundle["metadata"]["params"] == params
    assert "sklearn" in bundle["metadata"]["environment"]


# Partition


def test_splits_are_disjoint_and_complete(splits, synthetic_df):
    idx = [set(splits.X_train.index), set(splits.X_val.index), set(splits.X_test.index)]
    assert idx[0].isdisjoint(idx[1]) and idx[0].isdisjoint(idx[2]) and idx[1].isdisjoint(idx[2])
    assert sum(len(i) for i in idx) == len(synthetic_df)


def test_split_proportions(splits, synthetic_df):
    n = len(synthetic_df)
    assert len(splits.X_test) / n == pytest.approx(0.20, abs=0.01)
    assert len(splits.X_val) / n == pytest.approx(0.20, abs=0.01)


def test_split_preserves_class_ratio(splits, synthetic_df):
    global_rate = synthetic_df[TARGET].mean()
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert y.mean() == pytest.approx(global_rate, rel=0.15)
        assert y.sum() > 0


def test_split_is_reproducible(synthetic_df):
    X, y = synthetic_df[FEATURES], synthetic_df[TARGET]
    a, b = split_data(X, y), split_data(X, y)
    assert a.X_test.index.equals(b.X_test.index)
    assert not split_data(X, y, random_state=7).X_test.index.equals(a.X_test.index)


def test_duplicates_removed_before_split_do_not_cross_sets(synthetic_df):
    dup = pd.concat([synthetic_df, synthetic_df.sample(300, random_state=0)], ignore_index=True)
    clean, n = drop_exact_duplicates(dup)
    assert n == 300
    s = split_data(clean[FEATURES], clean[TARGET])
    key = lambda d: set(map(tuple, d.round(10).to_numpy()))  # noqa: E731
    assert key(s.X_train).isdisjoint(key(s.X_test))


# Fuite de données


def test_scaler_is_fitted_on_train_only(trained, splits):
    """Les moyennes apprises sont celles de l'entraînement, pas du test."""
    pipe, _ = trained
    scaler = pipe.named_steps["preprocess"].named_transformers_["num"]
    cols = list(scaler.feature_names_in_)
    np.testing.assert_allclose(scaler.mean_, splits.X_train[cols].mean().to_numpy(), rtol=1e-10)
    assert not np.allclose(scaler.mean_, splits.X_test[cols].mean().to_numpy(), rtol=1e-6)
    assert scaler.n_samples_seen_ == len(splits.X_train)


class _PoisonedFrame:
    """Lève une erreur à tout accès : prouve que le test n'est jamais lu."""

    def __getattr__(self, name):
        raise AssertionError(f"Accès interdit au jeu de test ({name})")

    def __len__(self):
        raise AssertionError("Accès interdit au jeu de test (len)")


def test_t05_tuning_and_threshold_never_touch_test_set(splits):
    poisoned = DataSplits(
        splits.X_train,
        splits.X_val,
        _PoisonedFrame(),
        splits.y_train,
        splits.y_val,
        _PoisonedFrame(),
    )
    pipe, params, results = tune_on_validation(
        poisoned, "decision_tree", grid=[{"max_depth": 3}, {"max_depth": 5}]
    )
    assert len(results) == 2 and params in ({"max_depth": 3}, {"max_depth": 5})
    select_threshold(poisoned.y_val, pipe.predict_proba(poisoned.X_val)[:, 1])


# Sauvegarde et rechargement (T13)


def test_predictions_identical_after_reload(trained, splits, saved_bundle_path):
    pipe, thr = trained
    bundle = load_model(saved_bundle_path)
    before = pipe.predict_proba(splits.X_test)[:, 1]
    after = bundle["pipeline"].predict_proba(splits.X_test)[:, 1]
    np.testing.assert_allclose(before, after, rtol=0, atol=1e-12)
    assert bundle["threshold"] == thr


def test_dimensions_in_and_out(saved_bundle_path, splits):
    bundle = load_model(saved_bundle_path)
    pre = bundle["pipeline"].named_steps["preprocess"]
    out = pre.transform(splits.X_test)
    assert out.shape == (len(splits.X_test), len(FEATURES))
    assert np.isfinite(out).all()
    assert list(pre.feature_names_in_) == FEATURES


def test_load_missing_model(tmp_path):
    with pytest.raises(ModelLoadError, match="introuvable"):
        load_model(tmp_path / "nope.joblib")


def test_load_tampered_model_is_refused(tmp_path, trained):
    path = save_model(trained[0], 0.5, tmp_path / "m.joblib")
    with open(path, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ModelLoadError, match="altéré"):
        load_model(path)


def test_load_model_without_hash_is_refused(tmp_path, trained):
    path = save_model(trained[0], 0.5, tmp_path / "m.joblib")
    path.with_suffix(".joblib.sha256").unlink()
    with pytest.raises(ModelLoadError, match="Empreinte absente"):
        load_model(path)


def test_load_corrupted_file(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"not a pickle")
    with pytest.raises(ModelLoadError, match="Chargement impossible"):
        load_model(path, verify_hash=False)


def test_load_unexpected_structure(tmp_path):
    import joblib

    path = tmp_path / "wrong.joblib"
    joblib.dump({"foo": 1}, path)
    with pytest.raises(ModelLoadError, match="Structure"):
        load_model(path, verify_hash=False)
