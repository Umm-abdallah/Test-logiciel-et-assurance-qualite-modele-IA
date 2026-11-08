# Données

Le jeu complet n'est pas versionné (taille d'environ 150 Mo, conditions d'utilisation de la source).

## Obtenir le jeu Kaggle

1. Créer un compte sur kaggle.com, puis télécharger « Credit Card Fraud Detection » (éditeur : Machine Learning Group, ULB) :
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Décompresser l'archive et placer `creditcard.csv` dans ce dossier : `data/creditcard.csv`.

En ligne de commande (jeton API dans `~/.kaggle/kaggle.json`, jamais dans le dépôt) :

```bash
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

Pour la CI, les identifiants sont stockés dans les secrets GitHub `KAGGLE_USERNAME` et `KAGGLE_KEY`.

## Contrat de données

| Colonne | Type | Domaine | Remarque |
|---|---|---|---|
| `Time` | float | au moins 0 | secondes écoulées depuis la première transaction du jeu (2 jours) |
| `V1` à `V28` | float | réels finis | composantes PCA anonymisées : aucune interprétation métier possible |
| `Amount` | float | au moins 0 | montant de la transaction (euros) |
| `Class` | int | {0, 1} | 1 = fraude ; absente en inférence |

Toute violation entraîne un rejet explicite (`DataValidationError`). Les doublons exacts sont retirés avant la partition.

## Échantillon versionné

`sample_creditcard.csv` : 5 000 lignes (100 fraudes, 4 900 normales), tirage aléatoire à graine fixe (`scripts/make_sample.py`). Il sert uniquement au contrôle de non-régression en CI. Les fraudes y sont sur-représentées : ses métriques ne reflètent pas la performance réelle.

La licence applicable est celle indiquée sur la page Kaggle du jeu : la vérifier avant de versionner ou rediffuser l'échantillon. Les données sont des transactions de porteurs européens de septembre 2013 : elles ne prouvent pas la performance sur des transactions actuelles.
