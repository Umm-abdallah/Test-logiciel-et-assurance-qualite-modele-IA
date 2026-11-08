# Rapport final de qualification

**Système :** AI-TestGuard, modèle de détection de fraude par carte bancaire
**Version qualifiée :** 1.0.0, arbre de décision (profondeur 6, 20 observations minimum par feuille), seuil 0,30
**Données :** Credit Card Fraud Detection (Kaggle, ULB), transactions de porteurs européens, septembre 2013
**Date :** novembre 2026

## Synthèse

Le pipeline est techniquement solide. Les 136 tests automatisés passent et couvrent 97 % du code. Aucune fuite de données n'a été détectée, les résultats sont reproductibles à l'identique, les entrées invalides sont rejetées explicitement et le temps de réponse reste très inférieur aux limites fixées.

Le modèle, lui, ne remplit pas entièrement ses critères. Il est remarquablement sobre en fausses alertes (12 clients bloqués à tort sur près de 57 000 transactions, soit un taux de 0,021 %) et précis (85 % des alertes sont de vraies fraudes). En revanche, il laisse passer 27 % des fraudes, pour un rappel de 0,726 contre 0,75 exigé. L'écart est faible et compris dans la marge d'incertitude, mais l'analyse révèle trois faiblesses de fond : le modèle ignore le montant, il s'appuie en partie sur une variable temporelle qui n'a pas de sens hors du jeu Kaggle, et ses performances baissent quand on le teste sur des transactions postérieures à son entraînement.

**Décision : le modèle n'est pas autorisé à bloquer des paiements. Il peut entrer en phase pilote en mode observation**, c'est-à-dire scorer les transactions réelles en parallèle du dispositif existant, sans effet sur les clients, après correction de deux anomalies identifiées ci-dessous.

## 1. Contexte et périmètre

Une banque souhaite s'appuyer sur un modèle d'apprentissage automatique pour repérer les paiements frauduleux. Deux types d'erreurs ont un coût : bloquer un client honnête (faux positif) et laisser passer une fraude (faux négatif). Notre mission était de qualifier le système, c'est-à-dire de démontrer par des tests reproductibles dans quelles conditions il peut être jugé acceptable, plutôt que de chercher le meilleur score possible.

Le périmètre couvre la chaîne complète : contrôle des données, prétraitement, entraînement, choix du seuil, sauvegarde, rechargement, prédiction, service de scoring, intégration continue et surveillance. Il exclut l'API HTTP réelle, l'infrastructure de production et la validation par les équipes métier, qui restent à mener.

## 2. Plan de test et traçabilité

Le plan complet figure dans `docs/plan_de_test.md`. Il définit 10 exigences (5 fonctionnelles, 5 non fonctionnelles) et une matrice de 24 exigences testables : les 7 du sujet et 17 ajoutées. Chaque ligne de la matrice indique le risque maîtrisé, le ou les tests qui la vérifient et le résultat attendu. Parmi les exigences ajoutées figurent la prévention des fuites (scaler, doublons, jeu de test), la validation stricte des entrées de l'API, l'intégrité du fichier modèle, l'absence de données dans les journaux et de secrets dans le dépôt, la sensibilité aux perturbations, la détection de dérive et la non-régression contre une référence figée.

Les critères d'acceptation ont été écrits avant toute lecture du jeu de test : PR-AUC d'au moins 0,70, rappel d'au moins 0,75, précision d'au moins 0,50, taux de faux positifs d'au plus 0,1 %, latence P95 d'au plus 50 ms, modèle d'au plus 10 Mo.

## 3. Architecture et choix méthodologiques

Le code est organisé en modules (`app/`) avec un fichier de configuration unique qui regroupe le contrat de données, la graine, les critères et les tolérances. La préparation et le modèle forment un `Pipeline` scikit-learn : il est impossible d'appliquer la standardisation avec des paramètres autres que ceux appris sur l'entraînement.

Les principaux choix, et leur raison :

* **Dédoublonnage avant partition.** 1 081 lignes identiques, dont 19 fraudes, auraient pu se retrouver à la fois dans l'entraînement et dans le test.
* **Partition stratifiée 60 / 20 / 20 avec graine fixe.** Chaque jeu garde 0,167 % de fraudes, environ 95 par jeu.
* **Transformation log1p sur le montant.** Sa distribution est très asymétrique (médiane 22 €, maximum 25 691 €).
* **Deux modèles légers comparés sur la validation.** L'arbre de décision (PR-AUC 0,833) l'emporte sur la régression logistique (0,798). La PR-AUC est préférée à la ROC-AUC parce qu'elle reste informative quand la classe positive est rare.
* **Seuil choisi par une règle écrite à l'avance.** Parmi les seuils qui assurent au moins 80 % de rappel et au plus 0,1 % de faux positifs en validation, on retient celui qui maximise la précision.
* **Rejet plutôt qu'imputation.** Une donnée manquante, infinie ou mal typée est rejetée, jamais devinée.
* **Sauvegarde contrôlée.** Le modèle est sauvegardé avec une empreinte SHA-256 vérifiée au chargement.

## 4. Tests unitaires et d'intégration

| Fichier | Tests | Ce qui est vérifié |
|---|---:|---|
| `test_preprocessing.py` | 40 | colonnes requises, absentes ou dupliquées, types, valeurs manquantes et infinies, données vides, cible hors {0, 1}, une seule observation, dimensions, montants extrêmes, chargement |
| `test_prediction.py` | 38 | contrat de sortie, seuil, entraînement, métriques comparées à un calcul manuel, piège de l'accuracy, règle de seuil, service de scoring, journaux |
| `test_integration.py` | 15 | scénario de bout en bout, partition, fuite du scaler, jeu de test jamais lu, rechargement, fichiers corrompus ou altérés |
| `test_robustness.py` | 22 | les six scénarios du sujet, sensibilité au bruit, propriétés générées par Hypothesis, dérive |
| `test_regression.py` | 9 | reproductibilité, règle de décision, non-régression sur trois niveaux de données |
| `test_security.py` | 8 | secrets, fichiers d'identifiants, `.gitignore`, fichiers volumineux |
| `test_performance.py` | 4 | taille, latence P95 du modèle et du service, débit |

Résultat : **136 tests réussis, aucun échec**, dont un test lent exécuté sur le jeu complet. Couverture : **97 %** (seuil imposé en CI : 85 %). Ruff ne signale aucune anomalie. Les traces sont dans `test_execution.txt`, `test_slow_full_dataset.txt` et `junit.xml`.

Les tests vérifient des comportements, pas seulement l'absence d'exception. Par exemple, un rejet doit nommer la colonne fautive et le nombre de valeurs concernées, et une prédiction rechargée doit coïncider à 10⁻¹² près avec l'originale.

## 5. Performances du modèle

| Jeu de test (95 fraudes, 56 651 normales) | Régression logistique | Arbre de décision retenu | IC 95 % (arbre) |
|---|---:|---:|---|
| PR-AUC | 0,698 | 0,724 | 0,63 à 0,81 |
| ROC-AUC | 0,949 | 0,939 | |
| Précision | 0,843 | 0,852 | 0,77 à 0,92 |
| Rappel | 0,737 | 0,726 | 0,64 à 0,81 |
| Taux de faux positifs | 0,023 % | 0,021 % | 0,011 % à 0,034 % |
| Matrice (VP / FP / FN / VN) | 70 / 13 / 25 / 56 638 | 69 / 12 / 26 / 56 639 | |

En montant, l'arbre intercepte 68 % de la fraude du jeu de test (10 080 € sur 14 766 €). Un modèle qui ne signalerait jamais rien aurait une accuracy de 99,83 %, contre 99,93 % pour le nôtre : cet indicateur n'a joué aucun rôle dans nos décisions. Les fraudes manquées obtiennent pour la plupart un score proche de zéro. Abaisser le seuil ne les rattraperait pas, il faudrait d'autres informations. Le rapport `rapport_evaluation_ml.md` détaille courbes, matrices et analyse des erreurs.

## 6. Robustesse et non-régression

Les 17 scénarios de robustesse se comportent comme prévu : rejet pour les données manquantes, infinies, mal typées ou mal structurées, traitement avec alerte pour les montants hors domaine, alerte de dérive pour les distributions décalées. Sous un bruit de 25 % de l'écart type sur toutes les composantes, seules 5 décisions sur 56 746 changent. Quand une décision bascule, elle le fait toutefois de façon franche, ce qui est typique d'un arbre.

Le dispositif de non-régression compare chaque version à une référence figée. Sur quatre versions candidates, il accepte un arbre plus profond (résultats identiques) et rejette un arbre pondéré, une régression logistique et un arbre réduit à un seul niveau. Cette dernière dégradation, injectée volontairement, fait échouer les tests et donc la CI. Détails dans `rapport_robustesse_regression.md`.

## 7. Latence et ressources

Le modèle pèse 11 Ko et se charge en 1,8 ms. Un appel complet au service de scoring (validation incluse, réseau exclu) prend 13,4 ms au 95e percentile, pour une limite fixée à 50 ms. En traitement par lots, le débit dépasse 4 millions de transactions par seconde sur 2 processeurs. Détails et protocole dans `rapport_performance_securite.md`.

## 8. Anomalies

| N° | Anomalie | Gravité | Statut |
|---|---|---|---|
| A1 | 1 081 transactions en double, dont 19 fraudes, risquant de se retrouver dans l'entraînement et le test | majeure | **corrigée** : dédoublonnage avant partition, couvert par un test |
| A2 | Rappel de 0,726 en test, sous le critère de 0,75 | majeure | **ouverte** : écart dans l'intervalle de confiance, à confirmer sur données réelles |
| A3 | Rappel de 0,851 en validation contre 0,726 en test : la validation, ayant servi à choisir, est optimiste | majeure | **ouverte** : recommander une validation croisée stratifiée et une validation temporelle pour la prochaine version |
| A4 | Le modèle utilise `Time` (2 % d'importance), un compteur relatif au début du jeu Kaggle, sans équivalent en production. Un décalage d'une heure change une décision. | **bloquante pour le pilote** | **ouverte** : retirer `Time` ou le remplacer par l'heure de la journée, puis requalifier |
| A5 | Le modèle n'utilise pas le montant (importance nulle) : il ne peut pas prioriser les alertes à fort enjeu, et son rappel est de 0,65 sur les fraudes de plus de 100 € | majeure | **ouverte** : tester des variables dérivées du montant et un critère pondéré par la valeur |
| A6 | Dérive marquée entre les deux jours (11 variables avec un PSI supérieur à 0,20) et performances plus faibles hors temps (précision 0,743) | majeure | **ouverte** : surveillance par tranche horaire, réentraînement planifié |
| A7 | Le PSI des scores et le taux de signalement ne détectent pas une dérive forte de V14 | moyenne | **traitée dans le plan de surveillance** : surveillance obligatoire des entrées |
| A8 | Les alertes sur le taux de signalement se déclenchent chaque nuit (effet de rythme, pas de panne) | mineure | **ouverte** : référence par tranche horaire |
| A9 | Une variation de régularisation de la régression logistique échappe aux tests de non-régression fondés sur les métriques de rang | mineure | **close** : documentée ; injection de démonstration remplacée ; contrôle de calibration recommandé si le score brut est exploité |
| A10 | Pic isolé de 101 ms sur une prédiction | mineure | **ouverte** : suivre P99 et maximum en production |

## 9. Limites et risques résiduels

**Limites des données.** Les données datent de 2013, couvrent deux jours seulement et proviennent d'une seule source. Les fraudes d'aujourd'hui (paiement mobile, hameçonnage, prise de contrôle de compte) n'y figurent probablement pas. Les variables anonymisées interdisent toute interprétation métier et toute ingénierie de variables fondée sur la connaissance du domaine. Aucune donnée ne permet d'évaluer l'équité envers des groupes de clients (voir `note_audit_biais.md`).

**Limites de l'évaluation.** Le jeu de test ne contient que 95 fraudes : le rappel est connu à plus ou moins 9 points environ. La partition aléatoire mélange les deux jours et donne une image plus favorable qu'une évaluation chronologique. Les perturbations testées sont synthétiques et ne reproduisent pas un fraudeur qui adapterait sa stratégie. La latence n'inclut pas le réseau ni une API réelle.

**Risques résiduels si le modèle était déployé tel quel.** Environ une fraude sur quatre passerait, surtout les très petits montants (souvent des tests de carte avant un achat plus important) et les montants élevés. La dépendance à `Time` produirait des décisions imprévisibles. Sans surveillance des entrées, une dérive pourrait dégrader le modèle pendant des semaines avant que les labels ne la révèlent.

## 10. Décision

Il faut distinguer deux décisions. La première porte sur la qualité technique du système, et le système réussit les tests techniques : contrat de données, absence de fuite, reproductibilité, robustesse aux entrées invalides, performance d'exécution, sécurité du dépôt, intégration continue. La seconde porte sur l'aptitude du modèle à agir sur des paiements réels. Celle-ci ne relève pas de nous seuls, et elle est **défavorable en l'état**.

Nous recommandons un passage en **pilote en mode observation**, aux conditions suivantes :

1. corriger l'anomalie A4 (retrait ou transformation de `Time`) puis rejouer l'intégralité de la qualification ;
2. réentraîner sur des données récentes de la banque, labellisées, couvrant plusieurs semaines, et réévaluer sur une période postérieure à l'entraînement ;
3. faire valider les critères d'acceptation, en particulier le rappel minimal et le coût d'une fausse alerte, par l'équipe de lutte contre la fraude et le service client ;
4. activer la surveillance décrite dans `plan_surveillance.md`, avec références par tranche horaire ;
5. conserver les règles existantes de la banque comme seul mécanisme de blocage pendant toute la durée du pilote.

L'autorisation de déployer le modèle avec un effet sur les clients nécessitera en plus une validation métier, opérationnelle, sécuritaire et réglementaire, qui dépasse le cadre de ce rapport.

## 11. Question de synthèse

> Dans quelles conditions votre modèle peut-il être considéré comme suffisamment fiable pour être testé dans un environnement pilote, et quelles preuves supplémentaires seraient nécessaires avant une mise en production réelle ?

Notre modèle peut entrer en pilote à une condition principale : que le pilote ne puisse nuire à personne. Il doit fonctionner en parallèle du dispositif actuel, scorer les transactions sans les bloquer, et ses alertes doivent être comparées après coup aux décisions des analystes. Dans ce cadre, nos résultats suffisent. Le pipeline est fiable au sens technique : il rejette ce qu'il ne comprend pas, donne toujours la même réponse à la même question, ne laisse fuir ni données ni secrets et répond en moins de 15 ms. Le modèle, lui, a montré sur des données qu'il n'avait jamais vues qu'il signale très peu de clients honnêtes (0,021 %) et qu'il se trompe rarement quand il signale (85 % de précision). Ce profil convient à un pilote, où une fausse alerte coûte une vérification et rien de plus.

Deux réserves conditionnent toutefois ce feu vert. La dépendance à la variable `Time` doit être supprimée avant le pilote : sans cela, ses décisions dépendraient d'un compteur qui n'existe pas en production. Ensuite, les critères d'acceptation doivent être confirmés par le métier. Nous avons fixé un rappel minimal de 75 % par raisonnement, et c'est précisément le critère que le modèle manque, de peu.

Avant une mise en production réelle, il faudra des preuves que ce projet ne pouvait pas fournir. La première est une **performance démontrée sur les transactions de la banque**, récentes et sur plusieurs mois, évaluée sur une période postérieure à l'entraînement. Nos propres essais montrent que la précision chute de 0,85 à 0,74 dès qu'on évalue sur une période postérieure. Il faudra aussi **assez de fraudes pour que les intervalles de confiance soient étroits** : avec 95 fraudes, notre rappel est connu à plus ou moins 9 points, alors qu'une décision de production en demande plutôt 2 ou 3, soit plusieurs milliers de fraudes labellisées. Viendront ensuite une **mesure du coût réel des erreurs**, faite avec les équipes : montant moyen d'une fraude manquée, coût d'une vérification, taux de réclamation après un blocage. Ces chiffres permettront de choisir le seuil sur une base économique plutôt que sur nos hypothèses. Il faudra encore un **audit d'équité** sur des attributs clients autorisés (âge, région, type de carte), conduit dans le respect du RGPD, ainsi que des **tests de bout en bout sur l'API réelle**, sous charge et avec authentification. Enfin, la **surveillance doit avoir été éprouvée pendant le pilote**, avec des seuils d'alerte étalonnés sur l'historique, une procédure de retour arrière testée, et l'accord des fonctions risques, conformité et sécurité, puisqu'un tel système relève vraisemblablement des usages à haut risque du règlement européen sur l'IA.

Autrement dit, ce projet prouve que le système est **testable, et testé**. Il ne prouve pas encore que le modèle est **bon pour la banque**. C'est l'objet du pilote.
