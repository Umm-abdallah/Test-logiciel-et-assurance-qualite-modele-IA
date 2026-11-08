# Rapport de robustesse et de non-régression

Ce rapport couvre les parties IV (prévention des fuites), V (robustesse) et VI (validation hors échantillon et non-régression). Le modèle étudié est l'arbre de décision retenu (profondeur 6, 20 observations minimum par feuille, seuil 0,30), sauvegardé dans `models/model.joblib`. Les résultats bruts sont dans `robustness_scenarios.csv`, `robustness_sensitivity.csv`, `drift_day1_vs_day2.csv` et `regression_comparison.csv`.

## 1. Intégration et fuites de données

Le scénario complet du sujet est rejoué par `test_end_to_end_scenario` : chargement, contrôle du schéma, partition, ajustement, choix du seuil, sauvegarde, rechargement, prédiction et rapport de métriques. Les tests d'intégration vérifient en plus les points suivants.

| Contrôle | Méthode | Résultat |
|---|---|---|
| Scaler ajusté sur l'entraînement seul | Les moyennes apprises sont comparées à celles de l'entraînement (égalité à 1e-10) puis à celles du test (différence exigée), et le nombre d'observations vues doit égaler la taille de l'entraînement. | conforme |
| Jeu de test jamais lu pendant le réglage | Le test est remplacé par un objet qui lève une erreur au premier accès. | conforme |
| Partitions disjointes et complètes | Intersection des index vide, somme des tailles égale au total. | conforme |
| Doublons exacts | Retirés avant la partition (1 081 sur le jeu réel), puis vérification qu'aucune ligne ne figure à la fois dans l'entraînement et dans le test. | conforme |
| Rechargement | Écart maximal entre prédictions avant et après rechargement au plus 1e-12. Écart observé nul. | conforme |
| Intégrité du fichier modèle | Une empreinte SHA-256 est écrite à la sauvegarde. Un fichier modifié d'un seul octet, ou sans empreinte, est refusé avant d'être désérialisé. | conforme |
| Erreurs de chargement | Fichier absent, corrompu ou de structure inattendue : exception `ModelLoadError` avec un message explicite. | conforme |

Le contrôle d'intégrité mérite un mot. Joblib repose sur pickle, qui peut exécuter du code arbitraire au chargement. Vérifier l'empreinte avant d'ouvrir le fichier ne remplace pas une signature cryptographique, mais cela empêche au moins de charger un modèle remplacé ou abîmé par erreur.

## 2. Scénarios de robustesse

Pour chaque scénario du sujet, nous avons d'abord écrit le comportement attendu, puis nous l'avons confronté au comportement observé sur le jeu de test réel.

| Scénario | Comportement attendu | Observé | Conforme |
|---|---|---|---|
| Valeurs manquantes dans 1 variable (V14, 50 lignes sur 1 000) | rejet du lot | rejet, anomalie nommée avec son effectif | oui |
| Valeurs manquantes dans 3 variables (V1, V4, Amount) | rejet du lot | rejet | oui |
| Valeurs infinies (5 lignes) | rejet | rejet | oui |
| Valeur non numérique (`Amount = "12,50"`) | rejet | rejet | oui |
| Champ manquant dans une requête de l'API | rejet | rejet, champ nommé | oui |
| Champ supplémentaire dans une requête (`merchant`) | rejet | rejet | oui |
| Type booléen à la place d'un nombre | rejet | rejet | oui |
| `Amount` égal à 2, 10 puis 1 000 fois le maximum d'entraînement (19 657 €) | traitement et alerte | traité, 100 % des lignes alertées, taux de signalement inchangé (0,14 %) | oui, voir remarque |
| Décalage de 0,5 écart type sur V10, V12 et V14 | alerte de dérive | PSI supérieur à 0,20 sur exactement ces trois variables | oui |
| Jour 1 comparé au jour 2 (données réelles) | surveillance | 11 variables au-delà de 0,20 (V1 : 1,57 ; V3 : 1,43 ; V28 : 1,05) | voir section 4 |
| Colonne absente, renommée ou dupliquée, lot vide | rejet | rejet | oui |
| Colonnes dans un autre ordre | traitement | réordonnées selon le contrat, prédictions identiques | oui |

Pourquoi rejeter plutôt qu'imputer ? Dans un système de paiement, une transaction dont une variable manque n'est pas une transaction « presque normale ». Elle signale un problème en amont : capteur, flux ou format. Lui attribuer une valeur moyenne reviendrait à fabriquer un score sur une donnée inventée. Nous préférons un rejet explicite, qui laisse le circuit de repli de la banque (règles déterministes, revue manuelle) prendre le relais. Les montants hors norme sont un cas différent : ce sont des données valides, le modèle peut les scorer, mais il extrapole. Ils sont donc traités et marqués d'une alerte.

Le scénario des montants extrêmes révèle une anomalie. Multiplier tous les montants par 1 000 ne change strictement rien aux décisions. Ce n'est pas une preuve de robustesse : l'arbre n'utilise tout simplement pas la variable `Amount`. Une transaction de 19 millions d'euros est jugée exactement comme une transaction de 10 €. L'alerte de domaine joue donc ici un rôle indispensable, puisque le modèle seul ne réagirait pas.

## 3. Sensibilité aux petites perturbations

Les variables n'ont pas toutes la même nature, et nous ne leur avons pas appliqué le même bruit.

* Pour `V1` à `V28`, composantes PCA sans unité, nous avons ajouté un bruit gaussien dont l'écart type vaut 1 %, 5 %, 10 % puis 25 % de l'écart type de chaque variable, mesuré sur l'entraînement.
* Pour `Amount`, nous avons appliqué un facteur multiplicatif aléatoire entre 1 moins x et 1 plus x, avec x égal à 1 %, 5 % puis 20 %. Cela simule un arrondi, une conversion de devise ou des frais. Le résultat est borné à zéro.
* Pour `Time`, nous avons appliqué un décalage de 60 secondes puis d'une heure, comme le ferait une horloge désynchronisée.

Mesures sur les 56 746 transactions de test :

| Perturbation | Variation moyenne du score | Variation maximale | Changements de décision | dont fraudes | Effet sur le rappel | Effet sur la précision |
|---|---:|---:|---:|---:|---:|---:|
| Bruit 1 % sur V1 à V28 | < 0,0001 | 0,148 | 0 | 0 | 0 | 0 |
| Bruit 5 % | 0,0001 | 0,900 | 3 | 1 | −0,011 | −0,023 |
| Bruit 10 % | 0,0001 | 0,900 | 4 | 1 | −0,011 | −0,033 |
| Bruit 25 % | 0,0002 | 0,998 | 5 | 2 | 0 | −0,030 |
| Amount ± 1 %, 5 % ou 20 % | 0 | 0 | 0 | 0 | 0 | 0 |
| Time + 60 s | 0 | 0 | 0 | 0 | 0 | 0 |
| Time + 1 h | < 0,0001 | 0,850 | 1 | 0 | 0 | −0,010 |
| Décalage de 0,5 écart type sur V10, V12, V14 | 0,0001 | 0,849 | 7 | 3 | −0,032 | +0,040 |

Aucune probabilité invalide n'a été produite, quel que soit le scénario.

Deux lectures sont possibles et toutes deux sont vraies. En moyenne, le modèle est très stable : même avec un bruit de 25 %, seules 5 décisions sur près de 57 000 changent. Mais quand une décision change, elle change brutalement, avec un score qui peut passer de 0 à presque 1. C'est la signature d'un arbre : ses 19 feuilles découpent l'espace en zones aux frontières nettes, et une transaction posée sur une frontière bascule d'un bloc. Une régression logistique aurait des variations plus douces mais plus nombreuses. Le test `test_flips_concentrate_near_decision_boundary` vérifie d'ailleurs que les basculements concernent des transactions proches du seuil, pas des cas tranchés.

Le décalage d'une heure sur `Time` suffit à faire basculer une transaction. C'est un signal clair : le modèle a appris quelque chose de la position temporelle dans le jeu Kaggle, une information qui n'aura aucun équivalent en production.

## 4. Une dérive réelle entre les deux jours

En comparant les transactions du premier jour (`Time` inférieur à 86 400 s) à celles du second, 11 des 29 variables dépassent le seuil d'alerte de 0,20 sur l'indice de stabilité de population (PSI). Le phénomène n'est pas cosmétique : la médiane de V1 passe de −0,28 à 0,11, et celle de V3 de 0,75 à −0,70. Le taux de fraude varie lui aussi, de 0,194 % le premier jour à 0,151 % le second.

Cette observation a deux conséquences.

* La partition aléatoire stratifiée mélange les deux jours et masque ce changement. Le contrôle hors temps du rapport d'évaluation (précision 0,743 au lieu de 0,852) en donne une idée plus réaliste.
* Un seuil d'alerte PSI unique à 0,20, appliqué sans tenir compte de l'heure ni du jour, déclencherait des alertes en permanence. Le plan de surveillance en tient compte.

## 5. Validation hors échantillon

| PR-AUC | Entraînement | Validation | Test | Hors temps |
|---|---:|---:|---:|---:|
| Arbre de décision | 0,825 | 0,833 | 0,724 | 0,699 |

Le détail figure dans le rapport d'évaluation. Retenons ici que l'écart entre l'entraînement et la validation est nul (pas de surapprentissage au sens strict), que l'écart avec le test reste dans la marge d'incertitude liée aux 95 fraudes, et que l'évaluation hors temps est la plus pessimiste des quatre.

## 6. Non-régression

### 6.1 La référence

La baseline `reports/baseline.json` a été produite une seule fois par `scripts/make_baseline.py full`. Elle enregistre tout ce qui est nécessaire pour rejouer l'expérience et comparer une future version :

| Élément | Valeur |
|---|---|
| Version du code | 1.0.0 (commit Git ajouté automatiquement lorsque le dépôt est versionné) |
| Modèle et paramètres | arbre de décision, profondeur 6, 20 observations minimum par feuille, sans pondération |
| Bibliothèques | Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3, joblib 1.6.0 |
| Graine | 42 |
| Seuil | 0,30 (rappel 0,851, précision 0,870, FPR 0,021 % en validation) |
| Jeu de données | empreinte SHA-256 `d175ada8…`, 283 726 lignes après dédoublonnage, 473 fraudes |
| Métriques de validation | PR-AUC 0,833 ; ROC-AUC 0,953 ; rappel 0,851 ; précision 0,870 ; FPR 0,021 % |

L'empreinte du jeu de données est vérifiée avant toute comparaison : si les données changent, le test échoue avec un message explicite au lieu de comparer des résultats qui ne sont pas comparables.

La référence n'est jamais recalculée par les tests ni par la CI. La régénérer revient à accepter une nouvelle version comme référence, et cette décision doit être tracée. Deux baselines légères complètent le dispositif pour la CI : l'une sur les données synthétiques, l'autre sur l'échantillon réel versionné (5 000 lignes dont 100 fraudes).

### 6.2 Tolérances et règle de décision

Une version candidate est **rejetée** si une métrique se dégrade au-delà de sa tolérance : 0,02 pour la PR-AUC, 0,01 pour la ROC-AUC, 0,05 pour le rappel et la précision, 0,0005 pour le FPR. Elle est **à investiguer** si la dégradation dépasse la moitié de la tolérance, ou si une métrique s'améliore de plus de trois fois la tolérance : une amélioration trop belle peut trahir une fuite. Dans tous les autres cas, elle est **acceptée**.

### 6.3 Comparaison de quatre versions candidates

| Version | PR-AUC | ROC-AUC | Rappel | Précision | FPR | Décision |
|---|---:|---:|---:|---:|---:|---|
| v1.0 référence | 0,833 | 0,953 | 0,851 | 0,870 | 0,021 % | |
| v1.1 arbre de profondeur 8 | 0,832 | 0,953 | 0,851 | 0,870 | 0,021 % | **accepter** |
| v1.2 arbre pondéré (`balanced`) | 0,466 | 0,942 | 0,777 | 0,579 | 0,094 % | **rejeter** |
| v1.3 régression logistique, C = 1 | 0,798 | 0,975 | 0,809 | 0,776 | 0,039 % | **rejeter** |
| v1.4 arbre de profondeur 1 (régression injectée) | 0,556 | 0,856 | 0,713 | 0,779 | 0,034 % | **rejeter** |

La version 1.1 est pratiquement identique à la référence. La contrainte de 20 observations par feuille empêche l'arbre de se ramifier davantage, augmenter la profondeur autorisée ne change presque rien. La version 1.2 montre à quel point la pondération des classes dégrade la précision sur ces données. La version 1.3 est intéressante : sa ROC-AUC est meilleure, mais sa PR-AUC et sa précision reculent au-delà de la tolérance. Selon la métrique regardée, on conclurait à une amélioration ou à une régression, ce qui justifie de suivre plusieurs indicateurs à la fois et de privilégier ceux qui sont adaptés à la classe rare.

### 6.4 Preuve qu'un échec est bien détecté

Le workflow `tests` peut être lancé manuellement avec l'option « Injecter une régression volontaire ». La variable `AITG_CANDIDATE` remplace alors le modèle de référence par un arbre de profondeur 1. En local, cette injection fait échouer les trois tests de non-régression (synthétique, échantillon réel, jeu complet) avec le détail des métriques dégradées. La trace est dans `evidence_failure_injection.txt`. Sans injection, les trois tests passent (`test_execution.txt` et `test_slow_full_dataset.txt`).

Un premier essai d'injection nous a appris quelque chose. Nous avions d'abord dégradé la régression logistique en la régularisant très fortement (C = 0,00001), et aucun test n'a échoué. L'explication est mathématique : une régularisation plus forte réduit l'amplitude des coefficients sans changer leur direction, donc l'ordre des transactions reste le même, et le seuil, recalculé sur la validation, s'ajuste en conséquence. Les métriques de classement ne voient rien. La leçon pour la qualification est qu'un test de non-régression construit uniquement sur la PR-AUC et la ROC-AUC peut laisser passer une modification qui change pourtant les probabilités brutes. Si des règles métier exploitent directement la valeur du score, et pas seulement la décision, il faudra ajouter un contrôle de calibration (score de Brier, par exemple).

## 7. Limites de ces tests

Les perturbations restent synthétiques : elles testent la stabilité mathématique du modèle, pas sa résistance à un fraudeur qui adapterait son comportement. La seule dérive « naturelle » disponible est celle qui sépare deux jours de septembre 2013, ce qui est bien peu pour juger un système appelé à fonctionner des années. Enfin, chaque mesure de rappel repose sur moins de 100 fraudes, d'où des intervalles de confiance larges d'une quinzaine de points.
