# Rapport de performance et de sécurité opérationnelle

Ce rapport couvre la partie VIII. Les mesures viennent de `scripts/benchmark.py` (`benchmark.json`), les contrôles de sécurité de `tests/test_security.py` et `tests/test_prediction.py`.

## 1. Protocole de mesure

| Paramètre | Choix |
|---|---|
| Matériel | machine virtuelle Linux x86_64, 2 processeurs logiques, 8,4 Go de mémoire |
| Logiciel | Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3 |
| Horloge | `time.perf_counter` |
| Échauffement | 50 appels non mesurés avant chaque série |
| Latence unitaire | 2 000 appels de `predict_proba` sur une seule transaction, tirée au hasard dans le jeu de test |
| Latence « service » | 2 000 appels de `FraudScoringService.score` : conversion de la requête en tableau, validation stricte du schéma et des types, prédiction, contrôle de domaine et journalisation |
| Débit | lots de 1, 100, 1 000, 10 000 et 56 746 transactions, 5 répétitions, médiane retenue |
| Mémoire | mémoire résidente du processus (psutil) avant et après chargement du modèle, et maximum pendant les lots |

La latence « service » n'est pas celle d'une vraie API. Elle exclut le réseau, la sérialisation HTTP, l'authentification et la file d'attente du serveur. Nous la présentons comme une borne basse de ce que verrait le système d'autorisation. Une mesure de bout en bout demanderait de déployer l'API puis d'utiliser Locust ou JMeter, ce qui dépasse le cadre du prototype.

## 2. Résultats

| Mesure | Valeur | Critère | Verdict |
|---|---:|---:|---|
| Taille du modèle sauvegardé | 11 Ko | au plus 10 Mo | conforme |
| Temps de chargement | 1,8 ms | | |
| Prédiction seule, médiane | 3,2 ms | | |
| Prédiction seule, P95 | 4,8 ms | | |
| Prédiction seule, P99 | 5,3 ms | | |
| Appel de service, médiane | 10,3 ms | | |
| Appel de service, P95 | 13,4 ms | au plus 50 ms | conforme |
| Appel de service, P99 | 15,9 ms | | |
| Mémoire ajoutée par le chargement du modèle | 0,25 Mo | | |
| Mémoire maximale du processus | 461 Mo | | voir remarque |

| Taille du lot | Transactions par seconde |
|---:|---:|
| 1 | 288 |
| 100 | 30 800 |
| 1 000 | 289 000 |
| 10 000 | 2 016 000 |
| 56 746 | 4 727 000 |

## 3. Lecture des résultats

Le modèle lui-même ne coûte presque rien. Un arbre de 19 feuilles parcourt au plus six comparaisons par transaction. Ce qui coûte, c'est tout ce qui l'entoure. Sur les 3,2 ms d'une prédiction isolée, l'essentiel part dans la construction des structures pandas et scikit-learn. La validation stricte triple ensuite le temps (10,3 ms en médiane), parce qu'elle vérifie chaque colonne une par une. Le débit par lot le montre bien : il passe de 288 transactions par seconde, une à une, à plus de 4 millions par seconde en traitement groupé.

Les 461 Mo de mémoire maximale correspondent surtout au jeu de données chargé pour le banc d'essai, pas au modèle, qui n'ajoute que 0,25 Mo.

Le critère de 50 ms au P95 est tenu avec une large marge. Nous l'avions fixé ainsi parce qu'une autorisation de paiement par carte doit aboutir en quelques centaines de millisecondes au total, et que le scoring de fraude n'est qu'une étape parmi d'autres (réseau, émetteur, contrôles réglementaires). Si l'on voulait descendre sous la milliseconde, il suffirait de valider les requêtes avec un schéma léger (pydantic, par exemple) et d'appeler l'arbre sur un tableau NumPy plutôt que sur un DataFrame. Rien ne l'impose aujourd'hui.

Une anomalie mineure : sur 2 000 prédictions isolées, une a pris 101 ms. Ce pic ponctuel, absent des mesures du service, vient vraisemblablement du système (ramasse-miettes, ordonnanceur de la machine virtuelle). Il justifie de suivre le P99 et le maximum en production, et pas seulement la médiane.

## 4. Sécurité opérationnelle

| Contrôle | Comment il est vérifié | Résultat |
|---|---|---|
| Champs manquants | `test_service_rejects_missing_field` et le scénario de robustesse correspondant | requête rejetée, champ nommé |
| Champs supplémentaires | `test_service_rejects_extra_field` (un champ `card_number` est envoyé) | requête rejetée : aucun champ inattendu n'entre dans le système |
| Types incorrects | chaînes, `None`, booléens, listes (`test_service_rejects_wrong_type`) | rejet. Les booléens sont refusés explicitement, car Python les considère comme des entiers. |
| Données sensibles dans les journaux | `test_logs_do_not_leak_transaction_content` capture les journaux et vérifie qu'aucune valeur de la transaction n'y figure | seuls l'identifiant de requête, le statut, la décision, l'alerte de domaine et la latence sont écrits |
| Secrets dans le dépôt | `test_no_secrets_in_repository` cherche des clés AWS, des jetons GitHub, des clés privées, des clés Kaggle et des mots de passe en clair | aucun secret trouvé |
| Fichiers d'identifiants | `kaggle.json`, `.env` et clés SSH absents du dépôt et listés dans `.gitignore` | conforme |
| Données brutes et modèles | `data/creditcard.csv`, `*.zip` et `models/` ignorés, aucun fichier de plus de 5 Mo versionné | conforme |
| Intégrité du modèle | empreinte SHA-256 contrôlée avant désérialisation | fichier altéré refusé |
| Analyse statique | Ruff avec les règles de sécurité S (dérivées de Bandit) en CI | aucune alerte |
| Identifiants de la CI | les clés Kaggle sont lues dans les secrets GitHub, jamais dans le code | conforme |

Même rejetée, une requête ne laisse qu'une trace minimale : le nombre d'anomalies, pas leur contenu. On évite ainsi qu'une erreur de format copie dans les journaux des données qu'on n'aurait jamais dû y voir.

## 5. Ce qui reste à faire avant une exposition réelle

Ce prototype n'a pas d'API HTTP, donc rien n'a été testé sur l'authentification des appelants, le chiffrement TLS, la limitation de débit ni la résistance à la saturation. Le contrôle des secrets repose sur une liste de motifs : un outil spécialisé comme gitleaks, branché sur l'historique Git complet, serait plus fiable. Enfin, l'empreinte SHA-256 prouve qu'un fichier n'a pas changé, pas qui l'a produit. Une chaîne de confiance (signature du modèle par la plateforme d'entraînement, registre de modèles) serait nécessaire en production.
