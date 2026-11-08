# Test logiciel et assurance qualité d'un modèle IA (AI-TestGuard)

Stratégie de test et qualification d'un modèle de détection de fraude par carte bancaire (jeu Kaggle *Credit Card Fraud Detection*).

L'objectif n'est pas seulement d'obtenir un bon score : il s'agit de démontrer, par des tests reproductibles et des résultats mesurables, dans quelles conditions le modèle peut être considéré comme acceptable pour une phase pilote.

## Livrables

| Livrable | Emplacement |
|---|---|
| Plan de test, matrice de traçabilité (T01 à T24), critères d'acceptation | `docs/plan_de_test.md` |
| Code du pipeline | `app/` |
| Tests automatisés | `tests/` |
| Rapport d'évaluation ML (métriques, matrices, courbes, analyse des erreurs) | `reports/rapport_evaluation_ml.md`, `reports/figures/` |
| Rapport de robustesse et de non-régression | `reports/rapport_robustesse_regression.md` |
| Rapport de performance et de sécurité opérationnelle | `reports/rapport_performance_securite.md` |
| Note d'audit des biais | `reports/note_audit_biais.md` |
| Stratégie de surveillance | `reports/plan_surveillance.md` |
| Rapport final de qualification et question de synthèse | `reports/rapport_final_qualification.md` |
| Preuves d'exécution (tests, couverture, échec injecté) | `reports/test_execution.txt`, `reports/test_slow_full_dataset.txt`, `reports/evidence_failure_injection.txt` |
| Workflows GitHub Actions | `.github/workflows/tests.yml`, `.github/workflows/full-evaluation.yml` |

## Architecture

Le traitement suit toujours le même ordre :

1. les données brutes sont contrôlées (`validate_data`) puis débarrassées des doublons exacts ;
2. elles sont partagées en entraînement, validation et test (60 / 20 / 20, stratifié, graine 42) ;
3. un `Pipeline` scikit-learn est ajusté sur l'entraînement seul : log1p puis standardisation du montant, standardisation de `Time` et de `V1` à `V28`, puis régression logistique ou arbre de décision ;
4. la validation sert à choisir les hyperparamètres, le modèle et le seuil de décision ;
5. le modèle retenu est sauvegardé avec joblib et une empreinte SHA-256 ;
6. le test n'intervient qu'une fois, pour l'évaluation finale du modèle rechargé (`load_model`, puis `predict_fraud` ou `FraudScoringService`).

| Module | Rôle |
|---|---|
| `app/config.py` | contrat de données, graines, critères d'acceptation, tolérances |
| `app/preprocessing.py` | `load_data`, `validate_data`, `preprocess_data`, `build_preprocessor`, alertes de domaine |
| `app/training.py` | `split_data`, `train_model`, `tune_on_validation`, `select_threshold`, `save_model`, `load_model` |
| `app/prediction.py` | `predict_fraud`, `FraudScoringService` (validation stricte, journaux sans données) |
| `app/evaluation.py` | `evaluate_model`, `bootstrap_ci`, `check_acceptance`, `compare_to_baseline` |
| `app/robustness.py` | perturbations par type de variable, mesure de sensibilité |
| `app/monitoring.py` | PSI, rapport de dérive, alerte sur le taux de signalement |
| `app/experiment.py` | procédure de référence commune à la baseline et aux tests de non-régression |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Placer `creditcard.csv` dans `data/` (voir `data/README.md`).

## Exécution

```bash
# Contrôles rapides (identiques à la CI)
ruff check . && ruff format --check .
pytest -m "not slow" --cov=app --cov-report=term-missing

# Évaluation complète sur le jeu Kaggle
python scripts/run_experiments.py        # comparaison des modèles, seuil, test final, figures
python scripts/run_robustness.py         # scénarios de robustesse et sensibilité
python scripts/benchmark.py              # latence, débit, mémoire, taille
python scripts/monitoring_simulation.py  # surveillance simulée
python scripts/compare_versions.py       # versions candidates comparées à la baseline
python scripts/segment_analysis.py       # écarts de performance par segment
pytest -m slow                           # non-régression sur le jeu complet

# Baselines (uniquement sur décision explicite, jamais en CI)
python scripts/make_baseline.py full
```

## Intégration continue

* `tests.yml` : à chaque push sur `main` et chaque Pull Request. Job `lint` (Ruff) puis job `tests` (Python 3.12 et 3.13) avec couverture (seuil 85 %), rapport JUnit, résumé dans l'onglet Actions et artefacts.
* `full-evaluation.yml` : manuel ou chaque lundi ; télécharge le jeu Kaggle via les secrets du dépôt, exécute les tests lents et les scripts d'évaluation, archive `reports/`.

### Prouver la détection d'un échec

1. Onglet **Actions** > workflow **tests** > **Run workflow**, cocher « Injecter une régression volontaire ».
2. Le job remplace le modèle de référence par un arbre de profondeur 1 (`AITG_CANDIDATE`) : `test_no_regression_*` échoue et le workflow passe au rouge avec le détail des métriques dégradées.

Remarque issue des essais : réduire seulement la régularisation (`C = 0,00001`) ne provoque pas de régression mesurable. La direction des coefficients est conservée, donc le classement des transactions aussi, et le seuil se réajuste sur la validation. Les métriques de rang (PR-AUC, ROC-AUC) sont insensibles à ce changement, d'où le choix d'une dégradation structurelle pour la démonstration.

En local : `AITG_CANDIDATE='{"model": "decision_tree", "params": {"max_depth": 1}}' pytest tests/test_regression.py`.

### Protéger la branche principale

Settings > Branches > Add branch protection rule pour `main` : exiger une Pull Request, exiger le succès des statuts `lint` et `tests (3.12)`, interdire les push forcés.

## Publication sur GitHub

```bash
git init && git add . && git commit -m "AI-TestGuard : pipeline, tests et CI"
git branch -M main
git remote add origin https://github.com/Umm-abdallah/Test-logiciel-et-assurance-qualite-modele-IA.git
git push -u origin main
```

## Résultats en bref

| | Arbre de décision retenu (test) |
|---|---|
| PR-AUC | 0,724 |
| Précision | 0,852 |
| Rappel | 0,726 (critère : 0,75) |
| Faux positifs | 12 sur 56 651 transactions normales (0,021 %) |
| Latence P95 du service | 13,4 ms |
| Tests | 136 réussis, couverture 97 % |

Décision : pilote en mode observation uniquement, après correction de la dépendance à la variable `Time`. Le raisonnement complet est dans `reports/rapport_final_qualification.md`.

## Limites

Le jeu de données date de 2013, ne couvre que deux jours et repose sur des variables anonymisées. Les résultats qualifient une démarche, ils ne prouvent pas la performance du modèle sur des transactions actuelles.
