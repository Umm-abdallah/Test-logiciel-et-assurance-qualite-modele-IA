# Note d'audit des biais

Cette note couvre la partie VII du sujet.

## 1. Ce que les données permettent, et ce qu'elles ne permettent pas

Le jeu Kaggle ne contient aucune caractéristique des porteurs de carte : ni âge, ni sexe, ni lieu de résidence, ni nationalité. En dehors du temps et du montant, il ne propose que 28 composantes issues d'une analyse en composantes principales, dont les variables d'origine n'ont pas été publiées. Il est donc impossible, à partir de ce seul jeu, d'affirmer que le modèle traite équitablement des groupes de personnes, comme d'affirmer le contraire.

Il serait tentant de chercher dans V1 à V28 un « proxy » démographique, par exemple en regroupant les transactions selon une composante pour y voir un groupe. Je ne l'ai pas fait. Ces composantes mélangent des informations inconnues : leur attribuer un sens social reviendrait à inventer des groupes, et toute conclusion tirée de groupes inventés serait sans valeur, voire trompeuse.

## 2. Ce que j'ai pu mesurer : des écarts entre segments opérationnels

J'ai appliqué la méthode d'audit demandée (rappel, taux de faux positifs et précision par groupe, avec leurs incertitudes) à deux découpages que les données permettent d'observer honnêtement : la tranche de montant et la période de la journée. Ces segments ne sont pas des groupes de personnes. L'exercice montre comment l'analyse se mènerait, et il révèle déjà des disparités de service.

Arbre de décision, jeu de test, intervalles de Wilson à 95 % (`segment_analysis.csv`) :

| Segment | Transactions | Fraudes | Rappel | IC 95 % du rappel | Précision | FPR |
|---|---:|---:|---:|---|---:|---:|
| Montant de 0 à 10 € | 20 176 | 47 | 0,68 | 0,54 à 0,80 | 0,82 | 0,035 % |
| Montant de 10 à 100 € | 25 367 | 22 | 0,91 | 0,72 à 0,97 | 0,95 | 0,004 % |
| Montant supérieur à 100 € | 11 203 | 26 | 0,65 | 0,46 à 0,81 | 0,81 | 0,036 % |
| Jour (7 h à 24 h) | 50 999 | 66 | 0,68 | 0,56 à 0,78 | 0,82 | 0,020 % |
| Nuit (0 h à 7 h) | 5 747 | 29 | 0,83 | 0,65 à 0,92 | 0,92 | 0,035 % |

L'heure est reconstituée à partir de `Time` modulo 24 heures. L'heure réelle de début de collecte n'étant pas publiée, la frontière entre le jour et la nuit est approximative.

Ce que l'on peut en dire avec prudence :

* Les fraudes de montant intermédiaire sont mieux détectées (91 %) que les très petites (68 %) et que les plus élevées (65 %). Les fraudes de plus de 100 € sont celles qui coûtent le plus, et ce sont aussi celles que le modèle rate le plus souvent en proportion. Le constat rejoint celui du rapport d'évaluation : l'arbre ignore le montant.
* Les très petits et les gros montants subissent un taux de fausses alertes environ neuf fois plus élevé que la tranche intermédiaire (0,035 % et 0,036 % contre 0,004 %). En valeur absolue, cela reste peu : 7 et 4 transactions bloquées à tort, contre 1.
* Les intervalles de confiance se chevauchent presque tous. Avec 22 à 47 fraudes par segment, aucun de ces écarts n'est statistiquement établi. Ils désignent des pistes à surveiller, pas des conclusions.

## 3. Les données qu'il faudrait pour un véritable audit d'équité

| Donnée | Pourquoi | Précaution |
|---|---|---|
| Tranche d'âge du porteur | Les personnes âgées sont des cibles fréquentes de fraude, et un taux de fausses alertes plus élevé peut les exclure de certains paiements. | utilisation encadrée, finalité d'audit uniquement |
| Pays ou région de résidence, pays de la transaction | Risque que les paiements à l'étranger ou depuis certaines régions soient bloqués de manière disproportionnée. | attention aux discriminations indirectes liées à l'origine |
| Type de carte et segment de clientèle (particulier, professionnel, carte prépayée) | Une carte prépayée peut servir de proxy de revenu modeste. | analyser l'effet, ne pas l'utiliser comme variable de décision sans justification |
| Canal (paiement en ligne, sans contact, retrait) | Écarts de performance techniques entre canaux. | aucune |
| Issue réelle des alertes (fraude confirmée, client légitime, réclamation) | Mesurer le coût réel des faux positifs par segment. | délai d'obtention des labels |

Le RGPD encadre strictement ces données, et certaines catégories (origine, santé) ne peuvent être traitées que dans des conditions très limitées. Mener un audit d'équité suppose donc en amont une analyse d'impact relative à la protection des données (AIPD), un accord du délégué à la protection des données et, idéalement, un accès à ces attributs restreint à l'équipe d'audit, séparé du modèle de production. Par ailleurs, un système de notation susceptible de refuser l'accès à un service financier relève vraisemblablement des usages à haut risque visés par le règlement européen sur l'IA. Ce point reste à confirmer par le service juridique, mais il renforce l'exigence d'un audit documenté.

## 4. Recommandations

1. Ne pas présenter ce modèle comme « équitable » : sur ce jeu, la question ne peut pas être tranchée.
2. Dès la phase pilote, collecter avec les alertes les attributs de segment autorisés et l'issue réelle de chaque alerte, afin de calculer rappel et taux de faux positifs par segment avec leurs intervalles. Fairlearn (`MetricFrame`) produit directement ces tableaux.
3. Se fixer à l'avance un écart maximal acceptable entre segments, par exemple un rapport des taux de faux positifs inférieur à 2 lorsque les intervalles sont disjoints, et le traiter comme un critère d'acceptation à part entière.
4. Traiter en priorité la faiblesse sur les fraudes de plus de 100 €. C'est à la fois une question de pertes et d'égalité de protection entre clients.
