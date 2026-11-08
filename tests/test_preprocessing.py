"""Tests unitaires : contrat de données et prétraitement (T01, T02, T03, T08 à T10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from app.config import FEATURES, TARGET
from app.preprocessing import (
    DataLoadError,
    DataValidationError,
    build_preprocessor,
    compute_reference_stats,
    drop_exact_duplicates,
    load_data,
    preprocess_data,
    range_alerts,
    validate_data,
)

# Contrat : colonnes requises


def test_t01_valid_data_has_all_required_columns(valid_df):
    report = validate_data(valid_df)
    assert report.is_valid
    assert report.issues == []
    assert report.n_rows == 50


@pytest.mark.parametrize("col", ["Time", "V1", "V14", "V28", "Amount", "Class"])
def test_missing_column_is_rejected(valid_df, col):
    with pytest.raises(DataValidationError) as exc:
        validate_data(valid_df.drop(columns=[col]))
    assert any(col in issue and "manquantes" in issue for issue in exc.value.issues)


def test_target_not_required_at_inference(valid_df):
    report = validate_data(valid_df.drop(columns=[TARGET]), require_target=False)
    assert report.is_valid


def test_duplicated_column_is_rejected(valid_df):
    dup = pd.concat([valid_df, valid_df[["Amount"]]], axis=1)
    with pytest.raises(DataValidationError, match="dupliquées"):
        validate_data(dup)


def test_extra_column_warns_in_batch_mode_and_fails_in_strict_mode(valid_df):
    df = valid_df.assign(customer_id=1)
    report = validate_data(df, strict=False)
    assert report.is_valid
    assert any("customer_id" in w for w in report.warnings)
    with pytest.raises(DataValidationError, match="customer_id"):
        validate_data(df, strict=True)


def test_columns_are_reordered_to_contract(valid_df):
    shuffled = valid_df[list(reversed(valid_df.columns))]
    X, y = preprocess_data(shuffled)
    assert list(X.columns) == FEATURES
    pd.testing.assert_series_equal(X["Amount"], valid_df["Amount"].astype(float))


# Types (T02)


def test_t02_types_are_float_after_preprocessing(valid_df):
    X, y = preprocess_data(valid_df)
    assert all(dt == np.float64 for dt in X.dtypes)
    assert y.dtype == np.int64


def test_string_column_is_rejected(valid_df):
    df = valid_df.astype({"V3": str})
    with pytest.raises(DataValidationError, match="non numériques.*V3"):
        validate_data(df)


def test_boolean_column_is_rejected(valid_df):
    df = valid_df.assign(Amount=valid_df["Amount"] > 10)
    with pytest.raises(DataValidationError, match="Amount"):
        validate_data(df)


def test_integer_columns_are_accepted(valid_df):
    df = valid_df.assign(Time=valid_df["Time"].astype(int))
    assert validate_data(df).is_valid


def test_non_dataframe_input_is_rejected():
    with pytest.raises(DataValidationError, match="DataFrame"):
        validate_data(np.zeros((3, 31)))


# Valeurs manquantes, infinies, vides


def test_missing_values_are_rejected_with_count(valid_df):
    df = valid_df.copy()
    df.loc[[0, 1, 2], "V7"] = np.nan
    with pytest.raises(DataValidationError) as exc:
        validate_data(df)
    assert "manquantes" in str(exc.value) and "'V7': 3" in str(exc.value)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_are_rejected(valid_df, value):
    df = valid_df.copy()
    df.loc[5, "Amount"] = value
    with pytest.raises(DataValidationError, match="infinies"):
        validate_data(df)


def test_empty_dataframe_is_rejected():
    empty = pd.DataFrame(columns=[*FEATURES, TARGET], dtype=float)
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(empty)


def test_all_issues_are_reported_together(valid_df):
    df = valid_df.drop(columns=["V1"]).copy()
    df.loc[0, "V2"] = np.nan
    df.loc[1, TARGET] = 2
    report = validate_data(df, raise_on_error=False)
    assert not report.is_valid
    assert len(report.issues) == 3


# Cible et domaines


@pytest.mark.parametrize("bad", [2, -1, 0.5])
def test_target_outside_0_1_is_rejected(valid_df, bad):
    df = valid_df.astype({TARGET: float})
    df.loc[3, TARGET] = bad
    with pytest.raises(DataValidationError, match="cible"):
        validate_data(df)


@pytest.mark.parametrize("col", ["Amount", "Time"])
def test_negative_amount_or_time_is_rejected(valid_df, col):
    df = valid_df.copy()
    df.loc[0, col] = -1.0
    with pytest.raises(DataValidationError, match=f"négatives dans {col}"):
        validate_data(df)


def test_amount_zero_is_valid(valid_df):
    df = valid_df.copy()
    df.loc[0, "Amount"] = 0.0
    assert validate_data(df).is_valid


# Une seule observation et dimensions (T03, T10)


def test_single_observation_is_processed(valid_df, splits):
    one = valid_df.head(1)
    X, y = preprocess_data(one)
    assert X.shape == (1, 30)
    pre = build_preprocessor().fit(splits.X_train)
    out = pre.transform(X)
    assert out.shape == (1, 30)
    assert np.isfinite(out).all()


def test_t03_preprocessor_outputs_only_finite_values(splits):
    pre = build_preprocessor().fit(splits.X_train)
    for X in (splits.X_train, splits.X_val, splits.X_test):
        out = pre.transform(X)
        assert out.shape == (len(X), len(FEATURES))
        assert np.isfinite(out).all()


def test_preprocessor_handles_extreme_amount(splits):
    pre = build_preprocessor().fit(splits.X_train)
    X = splits.X_test.head(3).copy()
    X["Amount"] = [0.0, 1e6, 1e12]
    out = pre.transform(X)
    assert np.isfinite(out).all()
    # log1p rend la relation monotone : un montant plus grand reste plus grand
    amount_col = list(pre.get_feature_names_out()).index("Amount")
    assert out[0, amount_col] < out[1, amount_col] < out[2, amount_col]


def test_preprocessor_feature_names_cover_contract(splits):
    pre = build_preprocessor().fit(splits.X_train)
    assert sorted(pre.get_feature_names_out()) == sorted(FEATURES)


def test_preprocess_does_not_mutate_input(valid_df):
    before = valid_df.copy()
    preprocess_data(valid_df)
    pd.testing.assert_frame_equal(valid_df, before)


# Scaler : exemple du sujet et ajustement sur l'entraînement


def test_standard_scaler():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


def test_scaler_on_constant_column_stays_finite():
    data = pd.DataFrame({"Amount": [5.0, 5.0, 5.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert np.allclose(scaled, 0.0)


# Doublons, statistiques de référence, alertes


def test_drop_exact_duplicates(valid_df):
    df = pd.concat([valid_df, valid_df.head(5)], ignore_index=True)
    out, n_removed = drop_exact_duplicates(df)
    assert n_removed == 5
    assert len(out) == len(valid_df)
    assert not out.duplicated().any()


def test_range_alerts_flag_only_out_of_domain_rows(splits):
    stats = compute_reference_stats(splits.X_train)
    typical = {c: stats[c]["mean"] for c in FEATURES}
    X = pd.DataFrame([typical, typical, typical])
    X.loc[1, "Amount"] = stats["Amount"]["max"] * 10
    X.loc[2, "V4"] = stats["V4"]["min"] - 1
    flags = range_alerts(X, stats)
    assert flags.tolist() == [False, True, True]


# Chargement


def test_load_data_missing_file(tmp_path):
    with pytest.raises(DataLoadError, match="introuvable"):
        load_data(tmp_path / "absent.csv")


def test_load_data_wrong_extension(tmp_path):
    p = tmp_path / "data.parquet"
    p.write_text("x")
    with pytest.raises(DataLoadError, match="Extension"):
        load_data(p)


def test_load_data_wrong_separator(tmp_path, valid_df):
    p = tmp_path / "data.csv"
    valid_df.to_csv(p, sep=";", index=False)
    with pytest.raises(DataLoadError, match="format"):
        load_data(p)


def test_load_data_roundtrip(tmp_path, valid_df):
    p = tmp_path / "data.csv"
    valid_df.to_csv(p, index=False)
    df = load_data(p)
    assert validate_data(df).is_valid
    assert df.shape == valid_df.shape
