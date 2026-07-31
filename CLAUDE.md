# CLAUDE.md

## Projet

Détection d'anomalies visuelles non supervisée sur pièces industrielles (dataset MVTec AD),
avec PatchCore via la librairie `anomalib`. Déploiement AWS end-to-end à terme.

Ce projet est un portfolio technique. L'objectif n'est pas de livrer vite, mais que je
comprenne et sache défendre chaque décision en entretien.

## Contexte technique

- Python 3.10+
- Développement local sur CPU d'abord, SageMaker ensuite
- Catégories cibles : `bottle` (défauts nets), `screw` (petits défauts, cas difficile),
  `carpet` (texture)
- Backbones à comparer : WideResNet50 (défaut), DINOv2 ViT-S
- Ablation prévue sur `coreset_sampling_ratio` : 0.01 / 0.1 / 0.25

## Architecture cible

Local (semaine 1) : anomalib → artefact modèle → classe AnomalyDetector → Docker

AWS (semaine 2) : S3 → SageMaker Training Job → ECR → SageMaker Serverless Inference
→ Lambda → API Gateway, le tout en Terraform, monitoring CloudWatch

## Conventions de code

- Aucune valeur en dur : chemins, hyperparamètres et configs vont dans des YAML
- Tout paramétrable par catégorie et par backbone dès le départ
- Type hints sur les signatures publiques
- Docstrings courtes, une ligne quand c'est suffisant
- Pas de sur-ingénierie : pas d'abstraction avant qu'un deuxième cas d'usage existe
- Les résultats d'expérience sont écrits en CSV avec la config complète en colonnes,
  pour être reproductibles et comparables

## Méthode de travail

1. Poser des questions avant d'implémenter si quelque chose est ambigu
2. Proposer l'approche et attendre validation avant d'écrire du code
3. Implémenter par petits incréments, s'arrêter après chaque élément
4. Ne jamais générer plusieurs modules d'un coup

## Ce que je fais moi, pas toi

Ne pas rédiger ni décider à ma place sur ces points :

- Le choix du seuil de décision et sa justification métier
- La section "Décisions d'architecture" du README
- Le contenu de `docs/troubleshooting.md` (mon journal de debug AWS)
- L'interprétation des résultats d'ablation

Tu peux générer les courbes, les tableaux et les fonctions de calcul.
L'analyse et l'argumentaire sont à moi.

## Garde-fous AWS (à partir de la semaine 2)

- Région unique : `eu-west-1`
- Toujours rappeler de détruire les endpoints après une session de test
- Principe du moindre privilège sur tous les rôles IAM, jamais de wildcard `*`
- Build Docker avec `--platform linux/amd64` (Mac Apple Silicon)
- Timeout Lambda à 30s minimum (cold start Serverless Inference)
- Signaler toute ressource facturée à l'heure avant de la créer

## État d'avancement

Mettre à jour cette section au fil du projet.

- [x] Setup repo et environnement
- [x] Dataset téléchargé et exploré
- [x] PatchCore fonctionnel sur `bottle`
- [x] Ablation backbone et coreset ratio
- [x] Calibration du seuil
- [x] Packaging inférence local
- [x] S3 et compte AWS
- [x] Entraînement SageMaker
- [ ] Endpoint Serverless
- [ ] Lambda + API Gateway
- [ ] Terraform complet
- [ ] README et benchmark final
