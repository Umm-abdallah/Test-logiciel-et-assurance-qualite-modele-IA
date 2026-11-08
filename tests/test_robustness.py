"""Tests de robustesse et de sensibilité (T15 à T18).

Ces tests mesurent la sensibilité au lieu d'imposer une invariance
universelle : une petite perturbation peut légitimement changer la classe
d'une transaction proche du seuil.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.config import FEATURES
from app.monitoring import drift_report, flag_rate_alert, population_stability_index, psi_status
from app.prediction import predict_fraud
from app.preprocessing import DataValidationError, compute_reference_stats, range_alerts, validate_data
from app.robustness import perturb_amount, perturb_pca, sensitivity, shift_time

# Scénario 1 et 2 : données manquantes, infinies, non numériques => REJET


@pytest.mark.parametrize("cols", [["V1"], ["V1", "V2", "Amount"], FEATURES])
def test_missing_values_in_one_or_more_columns_are_rejected(trained, splits, cols):
    X = splits.X_test.head(20).copy()
    X.loc[X.index[:3], cols] = np.nan
    with pytest.raises(DataValidationError, match="manquantes"):
        predict_fraud(trained[0], X)


def test_non_numeric_value_is_rejected(trained, splits):
    X = splits.X_test.head(5).copy().astype({"V5": object})
    X.iloc[2, X.columns.get_loc("V5")] = "abc"
    with pytest.raises(DataValidationError, match="non numériques"):
        predict_fraud(trained[0], X)


# Scénario 3 : montants extrêmes => TRAITEMENT + ALERTE


@pytest.mark.parametrize("amount", [1e4, 1e6, 1e9])
def test_extreme_amount_is_scored_and_flagged(trained, splits, amount):
    X = splits.X_test.head(10).copy()
    X["Amount"] = amount
    out = predict_fraud(trained[0], X, trained[1])
    p = out["fraud_probability"]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    stats = compute_reference_stats(splits.X_train)
    if amount > stats["Amount"]["max"]:
        assert range_alerts(X, stats).all()


# Scénario 4 : perturbation limitée => SENSIBILITÉ MESURÉE


@pytest.fixture(scope="module")
def train_std(splits):
    return splits.X_train.std(ddof=0)


def test_small_pca_noise_has_small_effect(trained, splits, train_std):
    pipe, thr = trained
    Xp = perturb_pca(splits.X_test, train_std, rel=0.01)
    s = sensitivity(pipe, splits.X_test, Xp, splits.y_test, thr)
    assert s["n_invalid"] == 0
    assert s["mean_abs_dproba"] < 0.01
    assert s["flip_rate"] < 0.01


def test_sensitivity_grows_with_noise_amplitude(trained, splits, train_std):
    pipe, thr = trained
    res = [
        sensitivity(pipe, splits.X_test, perturb_pca(splits.X_test, train_std, r), splits.y_test, thr)
        for r in (0.01, 0.05, 0.20)
    ]
    d = [r["mean_abs_dproba"] for r in res]
    assert d[0] < d[1] < d[2]


def test_flips_concentrate_near_decision_boundary(trained, splits, train_std):
    """Les changements de classe concernent surtout les cas proches du seuil."""
    pipe, _ = trained
    thr = 0.5
    p0 = pipe.predict_proba(splits.X_test)[:, 1]
    p1 = pipe.predict_proba(perturb_pca(splits.X_test, train_std, 0.25))[:, 1]
    flipped = (p0 >= thr) != (p1 >= thr)
    assert flipped.sum() > 0
    dist = np.abs(p0 - thr)
    assert np.median(dist[flipped]) < np.median(dist[~flipped])


def test_amount_and_time_perturbations_keep_valid_domain(trained, splits):
    Xa = perturb_amount(splits.X_test, rel=0.05)
    Xt = shift_time(splits.X_test, seconds=-1e9)
    assert (Xa["Amount"] >= 0).all() and (Xt["Time"] >= 0).all()
    validate_data(Xa, require_target=False)
    validate_data(Xt, require_target=False)
    ratio = Xa["Amount"] / splits.X_test["Amount"].replace(0, np.nan)
    assert ratio.dropna().between(0.95 - 1e-9, 1.05 + 1e-9).all()


def test_perturbations_do_not_mutate_input(splits, train_std):
    before = splits.X_test.copy()
    perturb_pca(splits.X_test, train_std, 0.1)
    perturb_amount(splits.X_test, 0.1)
    shift_time(splits.X_test, 60)
    pd.testing.assert_frame_equal(splits.X_test, before)


# Scénario 5 : distribution différente => ALERTE DE DÉRIVE


def test_psi_is_stable_on_same_distribution():
    rng = np.random.default_rng(0)
    assert population_stability_index(rng.normal(size=5000), rng.normal(size=5000)) < 0.05


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    psi = population_stability_index(rng.normal(size=5000), rng.normal(0.8, 1, size=5000))
    assert psi_status(psi) == "alerte"


def test_psi_rejects_empty():
    with pytest.raises(ValueError):
        population_stability_index([], [1.0])


def test_drift_report_ranks_shifted_variable_first(splits, train_std):
    cur = splits.X_test.copy()
    cur["V14"] = cur["V14"] + 1.0 * train_std["V14"]
    rep = drift_report(splits.X_train, cur, columns=FEATURES)
    assert rep.iloc[0]["variable"] == "V14"
    assert rep.iloc[0]["statut"] == "alerte"
    assert (rep.loc[rep.variable != "V14", "statut"] != "alerte").all()


def test_flag_rate_alert():
    assert flag_rate_alert(0.002, 0.002) == "normal"
    assert flag_rate_alert(0.010, 0.002) == "alerte"
    assert flag_rate_alert(0.0005, 0.002) == "alerte"
    # sur 500 transactions, 3 signalements au lieu de 1 ne sont pas significatifs
    assert flag_rate_alert(0.006, 0.002, n=500) == "normal"
    assert flag_rate_alert(0.006, 0.002, n=200_000) == "alerte"
    with pytest.raises(ValueError):
        flag_rate_alert(0.01, 0)


# Scénario 6 : schéma invalide => REJET


@pytest.mark.parametrize(
    "mutate",
    [
        lambda X: X.drop(columns=["V20"]),
        lambda X: X.rename(columns={"Amount": "amount"}),
        lambda X: pd.concat([X, X[["V1"]]], axis=1),
    ],
    ids=["colonne_absente", "colonne_renommee", "colonne_dupliquee"],
)
def test_invalid_schema_is_rejected(trained, splits, mutate):
    with pytest.raises(DataValidationError):
        predict_fraud(trained[0], mutate(splits.X_test.head(5)))


# Tests par propriétés (Hypothesis)

finite = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    v=st.lists(finite, min_size=28, max_size=28),
    amount=st.floats(min_value=0, max_value=1e8, allow_nan=False),
    time=st.floats(min_value=0, max_value=1e7, allow_nan=False),
)
def test_property_any_valid_row_gets_valid_probability(trained, v, amount, time):
    row = {"Time": time, **{f"V{i + 1}": x for i, x in enumerate(v)}, "Amount": amount}
    out = predict_fraud(trained[0], pd.DataFrame([row]))
    p = out["fraud_probability"].iloc[0]
    assert np.isfinite(p) and 0.0 <= p <= 1.0


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    row=st.integers(min_value=0, max_value=9),
    col=st.sampled_from(FEATURES),
    bad=st.sampled_from([np.nan, np.inf, -np.inf]),
)
def test_property_any_non_finite_value_is_rejected(splits, row, col, bad):
    X = splits.X_test.head(10).copy()
    X.iloc[row, X.columns.get_loc(col)] = bad
    with pytest.raises(DataValidationError):
        validate_data(X, require_target=False)
