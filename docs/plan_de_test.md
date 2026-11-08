# Plan de test AI-TestGuard

Qualification du modèle de détection de fraude par carte bancaire

| Élément | Valeur |
|---|---|
| Système sous test | Pipeline `app/` : validation, prétraitement, modèle, prédiction, service de scoring |
| Version | 1.0.0 |
| Jeu de données | Credit Card Fraud Detection (Kaggle, ULB), 284 807 transactions dont 492 fraudes |
| Rôle | Ingénierie QA des systèmes d'IA |
| Référentiel | Sujet « Validation, robustesse et fiabilité d'un modèle de détection de fraude bancaire » |

## 1. Périmètre

| Composant | Fichier | Dans le périmètre | Hors périmètre |
|---|---|---|---|
| Données | `app/preprocessing.py` (`load_data`, `validate_data`) | schéma, types, valeurs manquantes ou infinies, domaines, doublons | qualité de la collecte en amont, exactitude des labels Kaggle |
| Prétraitement | `app/preprocessing.py` (`preprocess_data`, `build_preprocessor`) | ordre des colonnes, typage, log1p sur `Amount`, standardisation | ingénierie de variables métier (impossible : variables anonymisées) |
| Modèle | `app/training.py` | partition, réglage, seuil, sauvegarde et rechargement | modèles lourds (boosting, réseaux), réentraînement en ligne |
| Prédiction | `app/prediction.py` | contrat de sortie, seuil, service de scoring, journalisation | API HTTP réelle, authentification, chiffrement réseau |
| Évaluation | `app/evaluation.py` | métriques, intervalles de confiance, acceptation, non-régression | validation métier par les équipes fraude |
| Intégration | `tests/test_integration.py`, `.github/workflows/` | chaîne complète, CI | déploiement, orchestration planifiée (Airflow) |
| Surveillance | `app/monitoring.py` | PSI, taux de signalement, alertes simulées | Prometheus, Grafana |

## 2. Exigences

### 2.1 Exigences fonctionnelles

| Code | Exigence |
|---|---|
| EF1 | Le système refuse toute donnée qui viole le contrat (colonne absente ou dupliquée, type non numérique, valeur manquante ou infinie, cible hors {0, 1}, montant ou temps négatif, lot vide). |
| EF2 | Le système produit, pour chaque transaction valide, une probabilité finie dans [0, 1] et une décision binaire cohérente avec le seuil. |
| EF3 | Le seuil de décision est choisi sur la validation selon une règle fixée à l'avance. |
| EF4 | Le pipeline sauvegardé et rechargé produit les mêmes prédictions. |
| EF5 | Les montants hors du domaine d'entraînement sont traités mais signalés. |

### 2.2 Exigences non fonctionnelles

| Code | Exigence |
|---|---|
| ENF1 | Reproductibilité : mêmes données et même graine donnent les mêmes résultats. |
| ENF2 | Performance prédictive : critères d'acceptation de la section 5. |
| ENF3 | Latence P95 d'un appel de scoring (validation incluse, hors HTTP) au plus 50 ms ; modèle au plus 10 Mo. |
| ENF4 | Aucune donnée de transaction dans les journaux ; aucun secret dans le dépôt. |
| ENF5 | Toute modification du code est testée automatiquement ; une régression fait échouer la CI. |

## 3. Matrice de traçabilité

Les lignes T01 à T07 reprennent la matrice du sujet ; T08 à T24 sont les exigences ajoutées.

| ID | Exigence | Type de test | Risque maîtrisé | Cas de test (fichier) | Résultat attendu |
|---|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | Données | Prédiction sur un schéma incomplet, erreur silencieuse | `test_preprocessing.py::test_t01_*`, `test_missing_column_is_rejected` | Aucune colonne indispensable manquante ; sinon rejet nommant la colonne |
| T02 | Les données respectent les types attendus | Unitaire | Conversion implicite (texte, booléen) faussant les scores | `test_t02_*`, `test_string_column_is_rejected`, `test_boolean_column_is_rejected` | Types conformes au contrat ; sinon rejet |
| T03 | Le prétraitement ne produit pas de valeurs non finies | Unitaire | NaN propagé jusqu'au modèle, décision arbitraire | `test_t03_*`, `test_preprocessor_handles_extreme_amount` | Aucune valeur NaN ou infinie en sortie |
| T04 | Le modèle produit une probabilité valide | Fonctionnel ML | Score inexploitable par les règles en aval | `test_prediction.py::test_t04_*`, tests Hypothesis | Probabilité comprise entre 0 et 1 |
| T05 | Les performances sont évaluées sur des données indépendantes | ML | Performances surestimées, sélection biaisée | `test_integration.py::test_t05_*` (jeu de test « empoisonné ») | Jeu de test non utilisé pour le réglage |
| T06 | Les résultats sont reproductibles | Non-régression | Résultats impossibles à auditer | `test_regression.py::test_t06_*`, `test_training_is_deterministic` | Écart nul à graine identique (tolérance 0) |
| T07 | Les tests s'exécutent sur GitHub | CI | Régression introduite sans détection | `.github/workflows/tests.yml` | Workflow réussi ou échec correctement signalé |
| T08 | Les valeurs manquantes, infinies, la cible invalide et les doublons de colonnes sont rejetés | Données | Corruption silencieuse du lot | `test_preprocessing.py` (14 cas paramétrés) | Rejet explicite listant toutes les anomalies |
| T09 | Pas de fuite de données : partition avant ajustement, scaler ajusté sur l'entraînement seul, doublons retirés avant partition | Intégration | Performance gonflée, modèle non généralisable | `test_scaler_is_fitted_on_train_only`, `test_duplicates_removed_before_split_do_not_cross_sets`, `test_splits_are_disjoint_and_complete` | Moyennes du scaler égales à celles de l'entraînement ; ensembles disjoints |
| T10 | La partition préserve la proportion de fraudes | Intégration | Jeu de test sans fraude ou non représentatif | `test_split_preserves_class_ratio` | Taux de fraude par partition à 15 % près du taux global |
| T11 | Les entrées de l'API sont validées strictement | Sécurité | Injection de champs, mauvais typage | `test_service_rejects_*` | Champ manquant, supplémentaire ou mal typé rejeté |
| T12 | Le seuil respecte la règle fixée sur la validation | ML | Seuil choisi a posteriori sur le test | `test_select_threshold_*` | Rappel et FPR de validation conformes à la contrainte, ou contrainte signalée non tenue |
| T13 | Les prédictions sont identiques après sauvegarde et rechargement ; un fichier altéré est refusé | Intégration | Modèle déployé différent du modèle qualifié ; exécution de code via pickle | `test_predictions_identical_after_reload`, `test_load_tampered_model_is_refused` | Écart au plus 1e-12 ; `ModelLoadError` sur empreinte non conforme |
| T14 | Les journaux ne contiennent aucune donnée de transaction | Sécurité | Fuite de données personnelles ou bancaires | `test_logs_do_not_leak_transaction_content` | Seuls identifiant de requête, décision, latence |
| T15 | Les montants extrêmes sont traités et signalés | Robustesse | Extrapolation silencieuse du modèle | `test_extreme_amount_is_scored_and_flagged` | Probabilité valide et alerte de domaine |
| T16 | Une perturbation limitée a un effet limité et mesuré | Robustesse | Instabilité des décisions | `test_small_pca_noise_has_small_effect`, `test_sensitivity_grows_with_noise_amplitude`, `scripts/run_robustness.py` | Variation moyenne de probabilité < 0,01 à 1 % de bruit ; sensibilité croissante avec l'amplitude |
| T17 | Les changements de classe sous perturbation concernent les cas proches du seuil | Robustesse | Décisions erratiques loin de la frontière | `test_flips_concentrate_near_decision_boundary` | Distance médiane au seuil plus faible pour les cas basculés |
| T18 | Une dérive de distribution déclenche une alerte | Surveillance | Dégradation non détectée en production | `test_drift_report_ranks_shifted_variable_first`, `test_psi_detects_shift` | PSI > 0,20 sur la variable décalée uniquement |
| T19 | Une nouvelle version ne dégrade pas les métriques au-delà de la tolérance | Non-régression | Régression de performance après modification | `test_no_regression_*`, `compare_to_baseline` | Décision « accepter » ; sinon échec de la CI |
| T20 | Aucun secret ni fichier d'identifiants dans le dépôt | Sécurité | Compromission des comptes (Kaggle, GitHub) | `test_security.py` | Aucune correspondance aux motifs de secrets |
| T21 | Les données brutes et les modèles ne sont pas versionnés | Sécurité, conformité | Diffusion non autorisée du jeu, dépôt volumineux | `test_gitignore_protects_*`, `test_no_large_files_in_repository` | `.gitignore` conforme, aucun fichier > 5 Mo |
| T22 | Latence, débit et taille du modèle respectent les limites | Performance | Délai d'autorisation de paiement dépassé | `test_performance.py`, `scripts/benchmark.py` | P95 au plus 50 ms ; modèle au plus 10 Mo |
| T23 | Les métriques adaptées au déséquilibre sont calculées correctement | Unitaire ML | Mauvaise interprétation (accuracy trompeuse) | `test_evaluate_model_known_values`, `test_high_accuracy_can_hide_zero_recall` | Valeurs égales au calcul manuel |
| T24 | Les erreurs de chargement de données et de modèle sont explicites | Intégration | Échec opaque en exploitation | `test_load_data_*`, `test_load_missing_model`, `test_load_corrupted_file` | Exceptions typées `DataLoadError`, `ModelLoadError` avec message clair |

## 4. Stratégie de test

| Niveau | Objet | Données | Déclenchement |
|---|---|---|---|
| Unitaire | fonctions de `app/` isolément | synthétiques au schéma Kaggle (`tests/conftest.py`) | chaque push et chaque Pull Request |
| Propriétés | invariants (probabilité valide, rejet des non finis) | générées par Hypothesis | chaque Pull Request |
| Intégration | chaîne charger, valider, séparer, entraîner, régler, sauver, recharger, prédire, rapporter | synthétiques | chaque Pull Request |
| Non-régression rapide | métriques contre baselines figées | synthétiques et échantillon réel versionné | chaque Pull Request |
| Évaluation complète, non-régression lente, robustesse, benchmark | modèle final | jeu Kaggle complet | manuel ou hebdomadaire (`full-evaluation.yml`) |

Règle de comportement face à une entrée invalide : **rejet** quand le contrat est violé (aucune correction silencieuse) ; **traitement avec alerte** quand la donnée est valide mais hors du domaine d'entraînement ; **traitement** quand seule la forme diffère sans ambiguïté (ordre des colonnes).

## 5. Critères d'acceptation et de rejet

Les critères sont fixés à partir des exigences avant toute observation du jeu de test (`app/config.py`).

| Critère | Seuil | Justification |
|---|---|---|
| PR-AUC (test) | au moins 0,70 | La classe positive représente 0,17 % des transactions : un classifieur aléatoire a une PR-AUC d'environ 0,0017. 0,70 garantit un classement nettement exploitable sur toute la plage de seuils. |
| Rappel (test) | au moins 0,75 | Le coût d'une fraude manquée (montant perdu, rétrofacturation, atteinte à la confiance) dépasse largement celui d'une vérification. Le seuil de réglage vise 0,80 en validation ; 0,75 laisse une marge pour l'incertitude d'échantillonnage (environ 95 fraudes en test). |
| Précision (test) | au moins 0,50 | Au plus une fausse alerte par fraude détectée : charge d'analyse soutenable et expérience client préservée. |
| FPR (test) | au plus 0,1 % | Au plus une transaction légitime bloquée sur 1 000 ; au-delà, le volume de réclamations devient significatif pour une banque traitant des millions de paiements. |
| Latence P95 | au plus 50 ms | Le budget total d'une autorisation carte est de l'ordre de quelques centaines de millisecondes ; le scoring ne doit en consommer qu'une fraction. |
| Taille du modèle | au plus 10 Mo | Déploiement simple en conteneur, chargement rapide. |
| Non-régression | dégradations au plus : PR-AUC 0,02 ; ROC-AUC 0,01 ; rappel 0,05 ; précision 0,05 ; FPR 0,0005 | Ordres de grandeur de la variabilité d'échantillonnage sur la validation. |

Règle de seuil : sur la validation, parmi les seuils de rappel au moins 0,80 et de FPR au plus 0,1 %, retenir celui de plus grande précision ; si aucun ne convient, retenir le seuil de F1 maximal et le signaler.

Décision :
* **Acceptation technique pour pilote** : tous les tests automatisés passent et tous les critères ci-dessus sont tenus sur le test, sans anomalie bloquante ouverte.
* **Rejet** : un critère de performance est manqué au-delà de son intervalle de confiance, ou une anomalie bloquante (fuite, non reproductibilité, probabilité invalide) est ouverte.
* **Investigation** : critère manqué mais dans l'intervalle de confiance, ou amélioration suspecte par rapport à la baseline.

## 6. Outils et preuves conservées

| Outil | Usage | Preuve |
|---|---|---|
| pytest, pytest-cov | tests unitaires, intégration, couverture (seuil 85 %) | `reports/test_execution.txt`, `reports/coverage.txt`, `junit.xml` et `htmlcov/` en artefact CI |
| Hypothesis | tests par propriétés | inclus dans l'exécution pytest |
| scikit-learn, pandas, NumPy, joblib | pipeline, sauvegarde | `models/model.joblib` + empreinte SHA-256 |
| matplotlib | courbes ROC, PR, matrices | `reports/figures/` |
| psutil, time.perf_counter | performance | `reports/benchmark.json` |
| Ruff | style et règles de sécurité (règles S de Bandit) | journal du job `lint` |
| GitHub Actions | CI | historique des exécutions, résumé de job, artefacts |
| Baselines JSON | non-régression | `reports/baseline.json`, `tests/baselines/*.json` |
